import json
from datetime import date
from pathlib import Path

import pytest

from callagree import llm
from callagree.contracts import GoldCall, Transcript
from callagree.extract import (LLMExtraction, build_messages, extract_commitments, postprocess)
from callagree.llm import LLMError, MockBackend, OpenAIBackend, ask_json, extract_json
from callagree.store import load_transcript

ROOT = Path(__file__).resolve().parent.parent
CALLS = [f"call_{i:03d}" for i in range(1, 9)]


def transcript(*segments, when="2026-10-13T14:05:00+03:00", call_id="t1"):
    """segments: (speaker, text) — время расставляется по 5 секунд на фразу."""
    return Transcript(call_id=call_id, call_datetime=when, client_phone="+375290000000",
                      segments=[{"speaker": s, "start": i * 5.0, "end": i * 5.0 + 4, "text": t}
                                for i, (s, t) in enumerate(segments)])


def item(**kw):
    base = {"ref": "k1", "debtor": "employee", "action": "прислать кп",
            "deadline": {"raw": "до пятницы", "type": "weekday", "value": "пятница"},
            "quotes": ["кп пришлю до пятницы"]}
    return {**base, **kw}


def process(items, tr=None, **kw):
    tr = tr or transcript(("employee", "кп пришлю до пятницы"),
                          ("client", "хорошо оплатим до двадцатого тысяч пять"),
                          ("employee", "нет давайте лучше к среде"))
    return postprocess(LLMExtraction.model_validate({"commitments": items}), tr, **kw)


# ---------- сквозной прогон в mock-режиме ----------

@pytest.mark.parametrize("call_id", CALLS)
def test_mock_end_to_end_equals_gold(call_id):
    """Эталон проходит через «сырой» ответ модели и пост-обработку и собирается обратно:
    даты, флаги, авторы и время цитат вычислены кодом и совпали с рукописными."""
    tr = load_transcript(ROOT / "data/transcripts" / f"{call_id}.json")
    gold = GoldCall.model_validate_json((ROOT / "data/gold" / f"{call_id}.json").read_text("utf-8"))
    got = extract_commitments(tr, MockBackend())
    assert len(got) == len(gold.commitments)
    ids = {g.id: c.id for g, c in zip(gold.commitments, got)}
    for g, c in zip(gold.commitments, got):
        expected = g.model_dump()
        expected["id"] = c.id
        expected["replaces"] = ids.get(g.replaces)
        assert c.model_dump() == expected
        assert c.verified and c.review == "pending" and c.lifecycle == "open"


def test_mock_raw_answer_has_no_computed_fields():
    raw = MockBackend().extract(load_transcript(ROOT / "data/transcripts/call_001.json"))
    text = json.dumps(raw, ensure_ascii=False)
    assert "resolved" not in text and "2026" not in text and '"t"' not in text
    assert all(c["status"] != "replaced" for c in raw["commitments"])


def test_mock_unknown_call_uses_heuristic():
    tr = transcript(("employee", "счет я вам пришлю до пятницы"),
                    ("client", "хорошо а мы наверное оплатим на следующей неделе"),
                    ("client", "наш бухгалтер вам перезвонит завтра"),
                    ("employee", "я вам не перезвоню"),
                    ("client", "до свидания"))
    got = extract_commitments(tr, MockBackend())
    assert [(c.debtor, c.action, c.status) for c in got] == [
        ("employee", "прислать счет", "firm"), ("client", "оплатить", "tentative"),
        ("third_party", "перезвонить", "firm")]
    assert got[0].deadline.resolved_end == date(2026, 10, 16)
    assert got[1].deadline.type == "next_week"
    assert got[2].debtor_name == "бухгалтер" and got[2].creditor == "employee"
    assert all(c.verified for c in got)
    assert [c.evidence[0].t for c in got] == [0.0, 5.0, 10.0]


def test_mock_ignores_gold_for_foreign_call_with_same_id():
    tr = transcript(("employee", "договор отправлю завтра"), call_id="call_001")
    got = extract_commitments(tr, MockBackend())
    assert [c.action for c in got] == ["отправить договор"]


def test_default_backend_is_mock(monkeypatch):
    monkeypatch.setattr(llm, "load_settings", lambda: llm.Settings())
    assert extract_commitments(load_transcript(ROOT / "data/transcripts/call_008.json")) == []
    with pytest.raises(LLMError):
        llm.get_backend(llm.Settings(llm_backend="claude"))


# ---------- пост-обработка ----------

def test_dates_time_and_speaker_come_from_code():
    (c,) = process([item(deadline={"raw": "до пятницы", "type": "tomorrow", "value": None})])
    assert c.deadline.type == "weekday"  # сказанное важнее типа от модели
    assert c.deadline.resolved_start == date(2026, 10, 16)
    assert c.evidence[0].speaker == "employee" and c.evidence[0].t == 0.0
    assert c.evidence[0].match_score == 100 and c.verified
    assert (c.id, c.creditor, c.client_phone, c.call_id) == ("c_0001", "client", "+375290000000", "t1")


def test_ids_continue_from_id_start():
    got = process([item(), item(ref="k2", action="позвонить")], id_start=41)
    assert [c.id for c in got] == ["c_0041", "c_0042"]


def test_invented_quote_is_not_verified():
    (c,) = process([item(quotes=["договор подпишем в понедельник утром"])])
    assert not c.verified and c.evidence[0].t is None and c.evidence[0].speaker is None


def test_no_quotes_is_not_verified():
    (c,) = process([item(quotes=[])])
    assert not c.verified and c.evidence == []


def test_replaces_marks_old_commitment():
    old, new = process([item(), item(ref="k2", replaces="k1", quotes=["нет давайте лучше к среде"],
                                     deadline={"raw": "к среде"})])
    assert (old.status, new.status, new.replaces) == ("replaced", "firm", old.id)
    assert new.deadline.resolved_start == date(2026, 10, 14) and new.evidence[0].t == 10.0


def test_bad_replaces_is_dropped():
    a, b = process([item(replaces="k1"), item(ref="k2", action="позвонить", replaces="k9")])
    assert a.replaces is None and b.replaces is None and a.status == "firm"


def test_model_cannot_set_replaced_itself():
    (c,) = process([item(status="replaced")])
    assert c.status == "firm"


def test_condition_makes_conditional():
    firm, tentative = process([item(condition="если будет на складе"),
                               item(ref="k2", action="позвонить", status="tentative",
                                    condition="если успею")])
    assert firm.status == "conditional" and firm.condition == "если будет на складе"
    assert tentative.status == "tentative"


def test_amount_and_missing_deadline():
    (c,) = process([item(debtor="client", action="оплатить", deadline=None,
                         amount={"value": "5000", "currency": "byn", "approximate": True,
                                 "raw": "тысяч пять"},
                         quotes=["оплатим до двадцатого тысяч пять"])])
    assert c.amount.value == 5000 and c.amount.currency == "BYN" and c.amount.approximate
    assert c.deadline.flags == ["no_deadline"] and c.creditor == "employee"
    (c,) = process([item(amount={"value": None, "currency": None}), ])
    assert c.amount is None
    (c,) = process([item(amount={"value": 0})])
    assert c.amount is None


def test_unparsed_deadline_keeps_raw():
    (c,) = process([item(deadline={"raw": "после дождичка", "type": "чепуха", "value": 3})])
    assert c.deadline.raw == "после дождичка" and c.deadline.flags == ["unparsed"]
    assert c.deadline.resolved_start is None


def test_third_party_creditor_and_name():
    tr = transcript(("employee", "сергей со склада завезет накладную послезавтра"),
                    ("client", "наш бухгалтер перезвонит завтра"))
    ours, theirs, plain = process([
        item(debtor="third_party", debtor_name="Сергей со склада", action="завезти накладную",
             quotes=["сергей со склада завезет накладную послезавтра"], deadline={"raw": "послезавтра"}),
        item(ref="k2", debtor="third_party", debtor_name="бухгалтер", creditor="third_party",
             action="перезвонить", quotes=["наш бухгалтер перезвонит завтра"], deadline={"raw": "завтра"}),
        item(ref="k3", debtor="employee", debtor_name="Андрей", creditor="employee",
             action="ждать", quotes=[]),
    ], tr)
    assert (ours.creditor, theirs.creditor) == ("client", "employee")
    assert ours.debtor_name == "Сергей со склада"
    assert plain.debtor_name is None and plain.creditor == "client"


def test_duplicates_and_empty_actions_are_dropped():
    got = process([item(), item(ref="k2"), item(ref="k3", action="  ")])
    assert len(got) == 1


def test_lenient_parsing_of_model_quirks():
    ex = LLMExtraction.model_validate({"commitments": [{
        "debtor": "client", "action": "оплатить", "quotes": "оплатим до двадцатого тысяч пять",
        "deadline": {"raw": "", "type": "soon", "value": 20}, "condition": "null",
        "comment": "лишнее поле"}]})
    c = ex.commitments[0]
    assert c.quotes == ["оплатим до двадцатого тысяч пять"] and c.condition is None
    assert (c.deadline.raw, c.deadline.type, c.deadline.value) == (None, None, "20")
    with pytest.raises(ValueError):
        LLMExtraction.model_validate({"commitments": [{"debtor": "boss", "action": "x"}]})


# ---------- OpenAI-совместимый бэкенд (без сети) ----------

class ScriptedBackend:
    """Отдаёт заранее заданные ответы и запоминает запросы."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def chat_json(self, messages, schema, schema_name):
        self.calls.append(list(messages))
        return self.answers.pop(0)


GOOD = json.dumps({"commitments": [item()]}, ensure_ascii=False)


def test_extract_through_chat_backend_and_prompt():
    tr = transcript(("employee", "кп пришлю до пятницы"), ("client", "хорошо"))
    backend = ScriptedBackend("<think>так…</think>```json\n" + GOOD + "\n```")
    (c,) = extract_commitments(tr, backend)
    assert c.action == "прислать кп" and c.verified
    system, user = backend.calls[0]
    assert "ДОСЛОВНЫЕ" in system["content"] and "tentative" in system["content"]
    assert "сотрудник: кп пришлю до пятницы\nклиент: хорошо" in user["content"]
    assert "2026" not in user["content"]  # дату модели не даём — считать её не её дело


def test_one_retry_with_error_message():
    backend = ScriptedBackend('{"commitments": [{"debtor": "boss"}]}', GOOD)
    out = ask_json(backend, [{"role": "user", "content": "x"}], LLMExtraction, "commitments")
    assert len(out.commitments) == 1 and len(backend.calls) == 2
    retry = backend.calls[1]
    assert retry[-2]["role"] == "assistant" and "boss" in retry[-2]["content"]
    assert "не прошёл проверку" in retry[-1]["content"]


def test_two_bad_answers_raise():
    backend = ScriptedBackend("не знаю", "{oops")
    with pytest.raises(LLMError):
        ask_json(backend, [{"role": "user", "content": "x"}], LLMExtraction, "commitments")
    assert len(backend.calls) == 2


def test_extract_json_variants():
    assert extract_json('Вот: {"a": {"b": 1}} конец') == {"a": {"b": 1}}
    for bad in ("", "[1, 2]", "<think>{}</think>"):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_strict_schema_requires_everything():
    schema = llm.strict_schema(LLMExtraction)
    c = schema["$defs"]["LLMCommitment"]
    assert set(c["required"]) == set(c["properties"]) and c["additionalProperties"] is False
    assert "weekday" in json.dumps(schema) and "default" not in json.dumps(schema)


class FakeCompletions:
    def __init__(self, unsupported):
        self.unsupported, self.seen = unsupported, []

    def create(self, **kw):
        from types import SimpleNamespace
        from openai import BadRequestError
        kind = (kw.get("response_format") or {}).get("type")
        self.seen.append(kind)
        if kind in self.unsupported:
            resp = SimpleNamespace(status_code=400, headers={}, request=None)
            raise BadRequestError("unsupported response_format", response=resp, body=None)
        msg = type("M", (), {"content": GOOD})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def fake_client(unsupported=()):
    completions = FakeCompletions(unsupported)
    return type("Cl", (), {"chat": type("Ch", (), {"completions": completions})}), completions


def test_openai_backend_uses_json_schema():
    client, completions = fake_client()
    backend = OpenAIBackend("http://x", "qwen3", client=client)
    assert backend.chat_json([{"role": "user", "content": "x"}], {"type": "object"}, "s") == GOOD
    assert completions.seen == ["json_schema"]


def test_openai_backend_falls_back_and_remembers():
    client, completions = fake_client(unsupported=("json_schema", "json_object"))
    backend = OpenAIBackend("http://x", "qwen3", client=client)
    for _ in range(2):
        assert backend.chat_json([{"role": "user", "content": "x"}], {}, "s") == GOOD
    assert completions.seen == ["json_schema", "json_object", None, None]


def test_openai_backend_unreachable_server_is_llm_error():
    backend = OpenAIBackend("http://127.0.0.1:9/v1", "qwen3", timeout=2)
    with pytest.raises(LLMError):
        backend.chat_json([{"role": "user", "content": "x"}], {}, "s")


def test_build_messages_direction():
    tr = transcript(("client", "алло"))
    tr.direction = "inbound"
    assert "клиент звонит в компанию" in build_messages(tr)[1]["content"]
