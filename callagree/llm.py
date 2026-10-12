"""Бэкенды LLM: mock (детерминированный, без модели) и OpenAI-совместимый.

Продукт не привязан к поставщику: любой сервер с OpenAI-совместимым интерфейсом (Ollama,
llama.cpp, vLLM) подключается строкой в .env. Модель только ПРЕДЛАГАЕТ: её ответ проходит
валидацию здесь (ask_json) и пост-обработку кодом в extract.py / closing.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import dates
from .config import DATA_DIR, Settings, load_settings
from .contracts import Commitment, GoldCall, Transcript
from .quotes import normalize

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Модель недоступна или дважды вернула ответ, не прошедший валидацию."""


# ---------- разбор ответа модели ----------

def extract_json(text: str) -> dict:
    """JSON-объект из ответа модели: без <think>…</think>, без ```-обёртки и текста вокруг."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("ожидался JSON-объект")
    return data


def strict_schema(model: Type[BaseModel]) -> dict:
    """JSON-схема для response_format: все поля обязательны, лишние запрещены (требование
    strict-режима). Сами модели мягче — пропущенное поле у маленькой модели не ошибка."""
    def walk(node):
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        return node
    return walk(model.model_json_schema())


def ask_json(backend, messages: list[dict], model: Type[T], schema_name: str) -> T:
    """Запрос → JSON → pydantic. При ошибке — ОДИН повтор с текстом ошибки."""
    messages = list(messages)
    schema = strict_schema(model)
    error = None
    for attempt in range(2):
        raw = backend.chat_json(messages, schema, schema_name)
        try:
            return model.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:  # JSONDecodeError — тоже ValueError
            error = e
            messages += [
                {"role": "assistant", "content": raw or ""},
                {"role": "user", "content":
                    "Ответ не прошёл проверку:\n" + str(e)[:1500] +
                    "\nИсправь и верни ТОЛЬКО корректный JSON по схеме, без пояснений."},
            ]
    raise LLMError(f"ответ модели не прошёл валидацию после повтора: {error}")


# ---------- OpenAI-совместимый бэкенд ----------

class OpenAIBackend:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed",
                 timeout: float = 300.0, client=None):
        self.model = model
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                            timeout=timeout, max_retries=1)
        self.client = client
        # от строгого к простому; запоминаем первый режим, который сервер принял
        self._formats = ["json_schema", "json_object", None]

    def _response_format(self, kind, schema, schema_name):
        if kind == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
        if kind == "json_object":
            return {"type": "json_object"}
        return None

    def chat_json(self, messages: list[dict], schema: dict, schema_name: str) -> str:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        while True:
            kind = self._formats[0]
            kwargs = {"model": self.model, "messages": messages, "temperature": 0}
            fmt = self._response_format(kind, schema, schema_name)
            if fmt:
                kwargs["response_format"] = fmt
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except (APIConnectionError, APITimeoutError) as e:
                raise LLMError(f"LLM недоступна ({self.model}): {e}") from e
            except APIStatusError as e:
                # 400/422 обычно значит «такой response_format не поддерживается»
                if e.status_code in (400, 404, 422, 501) and len(self._formats) > 1:
                    self._formats.pop(0)
                    continue
                raise LLMError(f"ошибка LLM ({self.model}): {e}") from e
            return resp.choices[0].message.content or ""


# ---------- mock: эталон для синтетики, эвристика для остального ----------

_WD = r"(?:понедельник\w*|вторник\w*|сред[аыуе]|четверг\w*|пятниц\w*|суббот\w*|воскресень\w*)"
_MONTH = (r"(?:январ\w+|феврал\w+|март\w*|апрел\w+|ма[яйе]|июн\w+|июл\w+|август\w*|сентябр\w+|"
          r"октябр\w+|ноябр\w+|декабр\w+)")
_DEADLINE_RES = [re.compile(p) for p in (
    rf"\b(?:на|в) (?:эт\w+|следующ\w+) недел\w+(?: во? {_WD})?",
    r"\bдо конца (?:следующ\w+ )?(?:недели|месяца|дня|года)",
    r"\bчерез (?:\w+ )?(?:дн\w+|день|недел\w+)",
    rf"\b(?:до|к|в|во|ко) (?:следующ\w+ )?{_WD}",
    r"\b(?:послезавтра|завтра|сегодня)\b",
    rf"\b(?:до|к|ко) (?:\d{{1,2}}(?:-?го|-?му)?|\w+(?:ого|его|ому|ему)(?: \w+(?:ого|его|ому|ему))?)"
    rf"(?: числа| {_MONTH})?",
    rf"\b\d{{1,2}}(?:-?го)? (?:числа|{_MONTH})",
    r"\b(?:на днях|в ближайшее время|как можно скорее)",
)]

# форма будущего времени → инфинитив; (1-е лицо: должник = говорящий)
_VERBS_SELF = {
    r"пришл(?:ю|ем)": "прислать", r"отправл?(?:ю|им)": "отправить", r"вышл(?:ю|ем)": "выслать",
    r"скин(?:у|ем)": "скинуть", r"перезвон(?:ю|им)": "перезвонить", r"позвон(?:ю|им)": "позвонить",
    r"набер(?:у|ем)": "набрать", r"подготовл?(?:ю|им)": "подготовить",
    r"выставл?(?:ю|им)": "выставить", r"перевед(?:у|ем)": "перевести",
    r"оплач?(?:у|тим)": "оплатить", r"заплач?(?:у|тим)": "заплатить",
    r"подпиш(?:у|ем)": "подписать", r"привез(?:у|ем)": "привезти", r"завез(?:у|ем)": "завезти",
    r"доставл?(?:ю|им)": "доставить", r"уточн(?:ю|им)": "уточнить",
    r"соглас(?:ую|уем)": "согласовать", r"переда(?:м|дим)": "передать",
    r"сдела(?:ю|ем)": "сделать", r"провер(?:ю|им)": "проверить",
    r"посчита(?:ю|ем)": "посчитать", r"внес(?:у|ем)": "внести", r"отгру(?:жу|зим)": "отгрузить",
}
# 3-е лицо: обязательство третьего лица, только если в фразе назван, кто это
_VERBS_THIRD = {
    r"перезвонит": "перезвонить", r"позвонит": "позвонить", r"приедет": "приехать",
    r"привезет": "привезти", r"завезет": "завезти", r"пришлет": "прислать",
    r"отправит": "отправить", r"подпишет": "подписать", r"оплатит": "оплатить",
}
_ROLES = r"(?:бухгалтер\w*|мастер\w*|курьер\w*|директор\w*|менеджер\w*|водител\w+|инженер\w*|юрист\w*)"
_OBJECTS = ["коммерческое предложение", "кп", "счет", "договор", "реквизиты", "прайс", "накладную",
            "документы", "акт сверки", "акт", "образцы", "предоплату", "оплату", "смету",
            "спецификацию"]
_TENTATIVE = ["наверное", "может быть", "возможно", "не точно", "постара", "попробу",
              "скорее всего", "если получится"]

# закрытие: слово о выполнении + общий предмет с обязательством
_DONE = (r"\b(?:получил\w*|пришл[оиа]|дош[елоа]\w*|оплатил\w*|перевел\w*|отправил\w*|выслал\w*|"
         r"прислал\w*|звонил\w*|перезвонил\w*|приезжал\w*|приехал\w*|подписал\w*|сделал\w*|"
         r"привез\w*|завез\w*|доставил\w*|выставил\w*|согласовал\w*)")
_SYNONYMS = {"коммерческое": "кп", "предложение": "кп"}
_STOP = {"прислать", "отправить", "выслать", "сделать", "первую", "часть", "вторую", "сразу",
         "месте", "около", "новый", "следующий"}


def _text(s: str) -> str:
    return s.lower().replace("ё", "е")


def _stems(text: str) -> set[str]:
    out = set()
    for w in normalize(text).split():
        if len(w) < 4 and w != "кп" or w in _STOP:
            continue
        out.add(_SYNONYMS.get(w, w[:5]))
    return out


def find_deadline_phrase(text: str, call_date, start: int = 0) -> Optional[str]:
    """Первая фраза-срок, которую понимает dates.py (сначала после позиции start)."""
    text = _text(text)
    found = []
    for rx in _DEADLINE_RES:
        for m in rx.finditer(text):
            if "unparsed" not in dates.resolve(m.group(), call_date).flags:
                found.append((m.start() < start, m.start(), -len(m.group()), m.group()))
    return min(found)[3] if found else None


def heuristic_extract(transcript: Transcript) -> dict:
    """Простая эвристика по ключевым фразам — чтобы конвейер и интерфейс работали без LLM
    на незнакомом звонке. Качество низкое намеренно: это заглушка, не модель."""
    items = []
    for seg in transcript.segments:
        text = _text(seg.text)
        role = re.search(rf"\b{_ROLES}\b", text)
        for table, third in ((_VERBS_SELF, False), (_VERBS_THIRD, True)):
            for rx, infinitive in table.items():
                m = re.search(rf"\b{rx}\b", text)
                if not m or text[:m.start()].rstrip().endswith(" не") or (third and not role):
                    continue
                obj = next((o for o in _OBJECTS if re.search(rf"\b{o}\b", text)), None)
                cond = re.search(r"\bесли ((?:\S+ ?){1,8}?)(?= то\b|$)", text[:m.start()].rstrip()) \
                    or re.search(r"\bесли ((?:\S+ ?){1,8})", text[m.end():])
                tentative = any(t in text for t in _TENTATIVE)
                raw = find_deadline_phrase(text, transcript.call_date, m.end())
                items.append({
                    "ref": f"k{len(items) + 1}",
                    "debtor": "third_party" if third else seg.speaker,
                    "debtor_name": role.group() if third else None,
                    "action": infinitive + (f" {obj}" if obj else ""),
                    "deadline": {"raw": raw, "type": None, "value": None},
                    "condition": f"если {cond.group(1).strip()}" if cond and not tentative else None,
                    "status": "tentative" if tentative else "conditional" if cond else "firm",
                    "quotes": [seg.text],
                })
    return {"commitments": items}


def heuristic_closures(transcript: Transcript, open_commitments: Sequence[Commitment]) -> dict:
    out = []
    for seg in transcript.segments:
        text = _text(seg.text)
        if not re.search(_DONE, text):
            continue
        stems = _stems(text)
        for c in open_commitments:
            if _stems(c.action) & stems:
                out.append({"commitment_id": c.id, "quote": seg.text})
    return {"closures": out}


class MockBackend:
    """Имитация идеальной LLM: для синтетических звонков отдаёт эталон из data/gold в том же
    «сыром» виде, в каком ответила бы модель (без дат, времени и авторов цитат — их считает
    код), для незнакомых звонков — эвристика."""
    name = "mock"

    def __init__(self, gold_dir: Optional[Path] = None):
        self.gold_dir = Path(gold_dir) if gold_dir else DATA_DIR / "gold"

    def _gold(self, call_id: str) -> Optional[GoldCall]:
        path = self.gold_dir / f"{call_id}.json"
        if not re.fullmatch(r"[\w.-]+", call_id) or not path.exists():
            return None
        return GoldCall.model_validate_json(path.read_text(encoding="utf-8"))

    def _gold_for(self, transcript: Transcript) -> Optional[GoldCall]:
        """Эталон подходит, только если его цитаты дословно есть в этой расшифровке
        (чужой звонок с тем же call_id уйдёт в эвристику)."""
        gold = self._gold(transcript.call_id)
        if gold is None:
            return None
        text = " | ".join(normalize(s.text) for s in transcript.segments)
        quotes = [e.quote for c in gold.commitments for e in c.evidence]
        quotes += [x.quote for x in gold.closures]
        return gold if all(normalize(q) in text for q in quotes) else None

    def extract(self, transcript: Transcript) -> dict:
        gold = self._gold_for(transcript)
        if gold is None:
            return heuristic_extract(transcript)
        return {"commitments": [{
            "ref": c.id, "debtor": c.debtor, "debtor_name": c.debtor_name, "creditor": c.creditor,
            "action": c.action,
            "deadline": {"raw": c.deadline.raw, "type": c.deadline.type, "value": c.deadline.value},
            "amount": c.amount.model_dump() if c.amount else None,
            "condition": c.condition,
            # «replaced» выставляет код по ссылке replaces, модель его не ставит
            "status": "firm" if c.status == "replaced" else c.status,
            "replaces": c.replaces,
            "quotes": [e.quote for e in c.evidence],
        } for c in gold.commitments]}

    def closures(self, transcript: Transcript, open_commitments: Sequence[Commitment]) -> dict:
        gold = self._gold_for(transcript)
        if gold is None:
            return heuristic_closures(transcript, open_commitments)
        out = []
        for closure in gold.closures:
            source = self._gold(closure.commitment_id.rsplit("_c", 1)[0])
            ref = next((c for c in (source.commitments if source else [])
                        if c.id == closure.commitment_id), None)
            if ref is None:
                continue
            for c in open_commitments:  # id в хранилище другие — сопоставляем по содержанию
                if (c.call_id, c.debtor, c.action, c.deadline.raw) == (
                        ref.call_id, ref.debtor, ref.action, ref.deadline.raw):
                    out.append({"commitment_id": c.id, "quote": closure.quote})
                    break
        return {"closures": out}


def get_backend(settings: Optional[Settings] = None):
    settings = settings or load_settings()
    if settings.llm_backend == "mock":
        return MockBackend()
    if settings.llm_backend == "openai":
        return OpenAIBackend(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                             timeout=settings.llm_timeout)
    raise LLMError(f"неизвестный LLM_BACKEND={settings.llm_backend!r} (нужно mock или openai)")
