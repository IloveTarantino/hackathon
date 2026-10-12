"""Извлечение обязательств из расшифровки: модель предлагает, код проверяет.

От модели берётся смысл (кто, что, срок как сказан, сумма, условие, статус, цитаты).
Код в postprocess() считает даты (dates.py), находит цитаты в расшифровке и берёт оттуда
автора и время (quotes.py), проставляет «replaced» и id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import dates, quotes
from .contracts import Amount, Commitment, DeadlineType, Party, Transcript
from .llm import ask_json, get_backend

PROMPTS = Path(__file__).parent / "prompts"
SPEAKER_RU = {"employee": "сотрудник", "client": "клиент"}
_DEADLINE_TYPES = set(DeadlineType.__args__)


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


# ---------- что отдаёт модель (мягче контракта: мелочи чиним кодом, а не повтором) ----------

class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _blank_to_none(v):
    if v is None or isinstance(v, str) and v.strip().lower() in ("", "null", "none", "нет"):
        return None
    return v


class LLMDeadline(_LLMModel):
    raw: Optional[str] = None
    type: Optional[DeadlineType] = None
    value: Optional[str] = None

    @field_validator("raw", "value", mode="before")
    @classmethod
    def _text(cls, v):
        v = _blank_to_none(v)
        return str(v) if isinstance(v, (int, float)) else v

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, v):  # незнакомый тип — не повод для повтора: срок разберёт dates.py
        return v if v in _DEADLINE_TYPES else None


class LLMAmount(_LLMModel):
    value: float
    currency: str = "BYN"
    approximate: bool = False
    raw: Optional[str] = None

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v):
        return _blank_to_none(v) or "BYN"


class LLMCommitment(_LLMModel):
    ref: Optional[str] = None
    debtor: Party
    debtor_name: Optional[str] = None
    creditor: Optional[Party] = None
    action: str = ""
    deadline: Optional[LLMDeadline] = None
    amount: Optional[LLMAmount] = None
    condition: Optional[str] = None
    status: Literal["firm", "tentative", "conditional", "declined", "replaced"] = "firm"
    replaces: Optional[str] = None
    quotes: list[str] = Field(default_factory=list)

    _blank = field_validator("debtor_name", "creditor", "condition", "replaces", "ref",
                             mode="before")(_blank_to_none)

    @field_validator("amount", mode="before")
    @classmethod
    def _no_amount(cls, v):  # {"value": null, ...} = суммы нет
        return None if isinstance(v, dict) and _blank_to_none(v.get("value")) is None else v

    @field_validator("quotes", mode="before")
    @classmethod
    def _quotes(cls, v):
        if isinstance(v, str):
            v = [v]
        return [q for q in (v or []) if isinstance(q, str) and q.strip()]


class LLMExtraction(_LLMModel):
    commitments: list[LLMCommitment] = Field(default_factory=list)


# ---------- запрос ----------

def render_transcript(transcript: Transcript) -> str:
    return "\n".join(f"{SPEAKER_RU[s.speaker]}: {s.text}" for s in transcript.segments)


def build_messages(transcript: Transcript) -> list[dict]:
    who = "сотрудник звонит клиенту" if transcript.direction == "outbound" else "клиент звонит в компанию"
    user = (f"Разговор ({who}). Расшифровка:\n{render_transcript(transcript)}\n\n"
            "Выпиши обязательства в JSON.")
    return [{"role": "system", "content": load_prompt("extract_system.md")},
            {"role": "user", "content": user}]


def ask_model(transcript: Transcript, backend) -> LLMExtraction:
    if hasattr(backend, "extract"):  # mock отвечает сразу «сырым» JSON, без промпта
        return LLMExtraction.model_validate(backend.extract(transcript))
    return ask_json(backend, build_messages(transcript), LLMExtraction, "commitments")


# ---------- пост-обработка (детерминированная) ----------

def _default_creditor(debtor: str, evidence) -> str:
    if debtor == "employee":
        return "client"
    if debtor == "client":
        return "employee"
    # третье лицо: обещают собеседнику того, кто это сказал
    speaker = next((e.speaker for e in evidence if e.speaker), None)
    return "client" if speaker == "employee" else "employee"


def postprocess(extraction: LLMExtraction, transcript: Transcript, id_start: int = 1
                ) -> list[Commitment]:
    """Ответ модели → обязательства по контракту 2. id — c_0001… начиная с id_start."""
    call_date = transcript.call_date
    out: list[Commitment] = []
    refs: dict[str, Commitment] = {}
    pending_replaces: list[tuple[Commitment, str]] = []
    seen = set()

    for item in extraction.commitments:
        action = " ".join(item.action.split())
        if not action:
            continue
        d = item.deadline or LLMDeadline()
        key = (item.debtor, action.lower(), (d.raw or "").lower(), tuple(item.quotes))
        if key in seen:  # модель повторила то же обязательство
            continue
        seen.add(key)

        evidence, verified = quotes.verify_evidence(item.quotes, transcript)
        creditor = item.creditor
        if creditor in (None, "third_party", item.debtor):
            creditor = _default_creditor(item.debtor, evidence)
        amount = None
        if item.amount and item.amount.value > 0:
            amount = Amount(value=item.amount.value, currency=item.amount.currency.upper(),
                            approximate=item.amount.approximate, raw=item.amount.raw)
        condition = item.condition.strip() if item.condition else None
        status = "firm" if item.status == "replaced" else item.status
        if condition and status == "firm":
            status = "conditional"

        c = Commitment(
            id=f"c_{id_start + len(out):04d}", client_phone=transcript.client_phone,
            call_id=transcript.call_id, debtor=item.debtor,
            debtor_name=(item.debtor_name or None) if item.debtor == "third_party" else None,
            creditor=creditor, action=action,
            deadline=dates.resolve(d.raw, call_date, d.type, d.value),
            amount=amount, condition=condition, status=status,
            evidence=evidence, verified=verified)
        out.append(c)
        if item.ref and item.ref not in refs:
            refs[item.ref] = c
        if item.replaces:
            pending_replaces.append((c, item.replaces))

    for new, ref in pending_replaces:
        old = refs.get(ref)
        if old is None or old is new:
            continue
        new.replaces = old.id
        if old.status != "declined":
            old.status = "replaced"
    return out


def extract_commitments(transcript: Transcript, backend=None, id_start: int = 1
                        ) -> list[Commitment]:
    """Расшифровка → проверенные обязательства (review=pending, lifecycle=open)."""
    backend = backend or get_backend()
    return postprocess(ask_model(transcript, backend), transcript, id_start)
