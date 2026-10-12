from pathlib import Path

import pytest

from callagree import store as st
from callagree.closing import (LLMClosures, build_messages, candidates, postprocess,
                               suggest_closures)
from callagree.contracts import GoldCall, Store, Transcript
from callagree.extract import extract_commitments
from callagree.llm import MockBackend
from callagree.store import load_transcript

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("call_001", "call_002"), ("call_003", "call_004")]


def load(call_id):
    return load_transcript(ROOT / "data/transcripts" / f"{call_id}.json")


def gold(call_id):
    return GoldCall.model_validate_json((ROOT / "data/gold" / f"{call_id}.json").read_text("utf-8"))


def first_call(call_id="call_001"):
    return extract_commitments(load(call_id), MockBackend())


def second(text="да кп получили спасибо", phone="+375291234567", call_id="t2"):
    return Transcript(call_id=call_id, call_datetime="2026-10-15T10:00:00+03:00", client_phone=phone,
                      segments=[{"speaker": "employee", "start": 0, "end": 3, "text": "добрый день"},
                                {"speaker": "client", "start": 4, "end": 8, "text": text},
                                {"speaker": "employee", "start": 9, "end": 11, "text": "хорошо спасибо"}])


class Scripted:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def chat_json(self, messages, schema, schema_name):
        self.calls.append(messages)
        return self.answer


@pytest.mark.parametrize("first, then", PAIRS)
def test_synthetic_pairs_give_gold_suggestions(first, then):
    commitments = first_call(first)
    expected = gold(then).closures
    by_gold_id = {g.id: c for g, c in zip(gold(first).commitments, commitments)}
    got = suggest_closures(load(then), commitments, MockBackend())
    assert len(got) == len(expected) == 2
    assert {(s.commitment_id, s.speaker, s.t, s.quote) for s in got} == {
        (by_gold_id[x.commitment_id].id, x.speaker, x.t, x.quote) for x in expected}
    assert all(s.status == "pending" and s.call_id == then and s.match_score == 100 for s in got)


def test_never_closes_automatically():
    commitments = first_call()
    before = [c.model_dump() for c in commitments]
    assert suggest_closures(load("call_002"), commitments, MockBackend())
    assert [c.model_dump() for c in commitments] == before
    assert all(c.lifecycle == "open" for c in commitments)


@pytest.mark.parametrize("call_id", ["call_005", "call_006", "call_007", "call_008"])
def test_no_suggestions_for_other_clients_calls(call_id):
    assert suggest_closures(load(call_id), first_call(), MockBackend()) == []


def test_no_suggestions_when_nothing_is_mentioned():
    tr = second("я просто хотел узнать вы в субботу работаете")
    assert suggest_closures(tr, first_call(), MockBackend()) == []


def test_first_call_alone_and_same_call_give_nothing():
    assert suggest_closures(load("call_001"), [], MockBackend()) == []
    assert suggest_closures(load("call_001"), first_call(), MockBackend()) == []


def test_heuristic_for_unknown_call():
    commitments = first_call()
    got = suggest_closures(second("значит да кп получили вчера все отлично"), commitments, MockBackend())
    assert [(s.commitment_id, s.speaker, s.t) for s in got] == [(commitments[1].id, "client", 4.0)]


def test_candidates_skip_replaced_done_rejected_and_other_clients():
    commitments = first_call()
    tr = second()
    assert [c.id for c in candidates(commitments, tr)] == ["c_0002", "c_0003", "c_0004"]  # c_0001 заменён
    commitments[1].lifecycle = "done"
    commitments[2].review = "rejected"
    assert [c.id for c in candidates(commitments, tr)] == ["c_0004"]
    assert candidates(commitments, second(phone="+375440000000")) == []


def test_model_is_not_asked_when_nothing_is_open():
    backend = Scripted('{"closures": []}')
    assert suggest_closures(second(), [], backend) == []
    assert backend.calls == []


def test_llm_answer_is_checked_by_code():
    commitments = first_call()
    answer = ('{"closures": ['
              '{"commitment_id": "c_0002", "quote": "да кп получили спасибо"},'
              '{"commitment_id": "c_0003", "quote": "оплату мы перевели еще вчера"},'   # выдумано
              '{"commitment_id": "c_0004", "quote": "спасибо"},'                        # слишком общо
              '{"commitment_id": "c_0001", "quote": "да кп получили спасибо"},'         # заменённое
              '{"commitment_id": "c_0777", "quote": "да кп получили спасибо"}]}')       # нет такого
    backend = Scripted(answer)
    got = suggest_closures(second(), commitments, backend, id_start=5)
    assert [(s.id, s.commitment_id, s.speaker, s.t) for s in got] == [("s_0005", "c_0002", "client", 4.0)]
    system, user = backend.calls[0]
    assert "c_0002 | должен: сотрудник | прислать коммерческое предложение | срок: к среде → ср, 14.10" \
        in user["content"]
    assert "c_0004 | должен: бухгалтер Светлана" in user["content"]
    assert "c_0001" not in user["content"]
    assert "клиент: да кп получили спасибо" in user["content"]


def test_one_suggestion_per_commitment_best_quote_wins():
    commitments = first_call()
    answer = LLMClosures.model_validate({"closures": [
        {"commitment_id": "c_0002", "quote": "да кп палучили спасиба"},
        {"commitment_id": "c_0002", "quote": "да кп получили спасибо"}]})
    (s,) = postprocess(answer, second(), candidates(commitments, second()))
    assert s.quote == "да кп получили спасибо" and s.match_score == 100


def test_suggestion_accepted_by_human_closes(tmp_path):
    commitments = first_call()
    store = Store(commitments=commitments)
    store.close_suggestions = suggest_closures(load("call_002"), commitments, MockBackend())
    done = st.accept_suggestion(store, store.close_suggestions[0].id)
    assert done.lifecycle == "done"
    st.save_store(store, tmp_path / "store.json")
    assert st.load_store(tmp_path / "store.json").close_suggestions[0].status == "accepted"


def test_build_messages_lists_amount():
    commitments = first_call()
    text = build_messages(second(), candidates(commitments, second()))[1]["content"]
    assert "сумма: тысяч пять" in text
