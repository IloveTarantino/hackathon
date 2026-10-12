import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from callagree import store as st
from callagree.contracts import (Commitment, Deadline, Segment, Store, Transcript)

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def _load(name):
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


@pytest.fixture
def store():
    return Store.model_validate(_load("store_example.json"))


# ---------- валидация ----------

def test_transcript_sample_valid():
    t = Transcript.model_validate(_load("transcript_example.json"))
    assert t.call_date == date(2026, 10, 13)
    assert t.segments[0].speaker == "employee"
    assert t.call_datetime.utcoffset().total_seconds() == 3 * 3600


def test_commitment_sample_valid():
    c = Commitment.model_validate(_load("commitment_example.json"))
    assert c.deadline.resolved_start == date(2026, 10, 14)
    assert c.replaces == "c_0001"
    assert c.is_active


def test_naive_datetime_is_minsk():
    t = Transcript(call_id="x", call_datetime="2026-10-13T23:30:00", client_phone="+375291234567")
    assert t.call_datetime.utcoffset().total_seconds() == 3 * 3600
    assert t.call_date == date(2026, 10, 13)


def test_call_date_uses_minsk_not_utc():
    t = Transcript(call_id="x", call_datetime="2026-10-13T22:30:00+00:00", client_phone="+375291234567")
    assert t.call_date == date(2026, 10, 14)


def test_commitment_defaults():
    c = Commitment(id="c_0001", client_phone="+375", call_id="call_001", debtor="employee",
                   creditor="client", action="перезвонить")
    assert (c.status, c.review, c.lifecycle, c.verified) == ("firm", "pending", "open", False)
    assert c.deadline.type == "none" and c.amount is None and c.evidence == []


@pytest.mark.parametrize("patch", [
    {"debtor": "boss"},
    {"status": "maybe"},
    {"review": "ok"},
    {"lifecycle": "closed"},
    {"action": ""},
    {"deadline": {"type": "someday"}},
    {"deadline": {"flags": ["weird"]}},
    {"deadline": {"resolved_start": "2026-10-20", "resolved_end": "2026-10-14"}},
    {"amount": {"value": "много"}},
    {"unknown_field": 1},
    {"evidence": [{"speaker": "robot", "quote": "x"}]},
])
def test_commitment_rejects_bad(patch):
    data = _load("commitment_example.json")
    data.update(patch)
    with pytest.raises(ValidationError):
        Commitment.model_validate(data)


@pytest.mark.parametrize("seg", [
    {"speaker": "employee", "start": 5, "end": 4, "text": "а"},
    {"speaker": "operator", "start": 0, "end": 1, "text": "а"},
    {"speaker": "client", "start": -1, "end": 1, "text": "а"},
    {"speaker": "client", "start": 0, "end": 1},
])
def test_segment_rejects_bad(seg):
    with pytest.raises(ValidationError):
        Segment.model_validate(seg)


def test_transcript_rejects_bad_direction():
    data = _load("transcript_example.json")
    data["direction"] = "sideways"
    with pytest.raises(ValidationError):
        Transcript.model_validate(data)


def test_store_rejects_duplicate_ids():
    data = _load("store_example.json")
    data["commitments"].append(data["commitments"][0])
    with pytest.raises(ValidationError):
        Store.model_validate(data)


def test_is_active():
    base = dict(id="c_1", client_phone="+375", call_id="c", debtor="client", creditor="employee",
                action="оплатить")
    assert Commitment(**base).is_active
    assert Commitment(**base, status="tentative").is_active
    assert not Commitment(**base, status="replaced").is_active
    assert not Commitment(**base, status="declined").is_active
    assert not Commitment(**base, review="rejected").is_active
    assert not Commitment(**base, lifecycle="done").is_active


# ---------- round-trip ----------

@pytest.mark.parametrize("name,model", [
    ("transcript_example.json", Transcript),
    ("commitment_example.json", Commitment),
    ("store_example.json", Store),
])
def test_roundtrip_json(name, model):
    raw = _load(name)
    obj = model.model_validate(raw)
    again = json.loads(obj.model_dump_json())
    assert again == raw
    assert model.model_validate(again) == obj


def test_store_save_load(tmp_path, store):
    path = tmp_path / "sub" / "store.json"
    st.save_store(store, path)
    assert st.load_store(path) == store
    assert "Иван Петрович" in path.read_text(encoding="utf-8")  # кириллица не экранируется
    assert not list(path.parent.glob("*.tmp"))


def test_load_missing_store_is_empty(tmp_path):
    s = st.load_store(tmp_path / "nope.json")
    assert s == Store() and s.commitments == []


def test_load_broken_store_raises(tmp_path):
    p = tmp_path / "store.json"
    p.write_text('{"commitments": [{"id": "c_1"}]}', encoding="utf-8")
    with pytest.raises(ValidationError):
        st.load_store(p)


# ---------- операции ----------

def test_next_ids(store):
    assert st.next_commitment_id(store) == "c_0004"
    assert st.next_suggestion_id(store) == "s_0002"
    assert st.next_commitment_id(Store()) == "c_0001"


def test_add_call_and_client():
    s = Store()
    t = Transcript.model_validate(_load("transcript_example.json"))
    st.add_call(s, t, "data/transcripts/call_001.json")
    st.add_call(s, t, "data/transcripts/call_001.json")  # повтор не дублирует
    assert len(s.calls) == 1 and s.calls[0].transcript_path.endswith("call_001.json")
    assert s.clients["+375291234567"].name == "Иван Петрович"


def test_upsert_client_keeps_known_name():
    s = Store()
    st.upsert_client(s, "+375", None)
    st.upsert_client(s, "+375", "Ольга")
    st.upsert_client(s, "+375", "Другая")
    assert s.clients["+375"].name == "Ольга"


def test_open_commitments_excludes_replaced(store):
    assert [c.id for c in st.open_commitments(store, "+375291234567")] == ["c_0002", "c_0003"]
    assert st.open_commitments(store, "+000") == []


def test_review_and_edit(store):
    st.set_review(store, "c_0002", "confirmed")
    assert st.get_commitment(store, "c_0002").review == "confirmed"
    with pytest.raises(ValidationError):
        st.set_review(store, "c_0002", "whatever")
    c = st.edit_commitment(store, "c_0002", action="оплатить аванс")
    assert c.review == "edited" and st.get_commitment(store, "c_0002").action == "оплатить аванс"
    with pytest.raises(KeyError):
        st.set_review(store, "c_9999", "confirmed")


def test_close_only_by_human(store):
    assert [s.id for s in st.pending_suggestions(store, "+375291234567")] == ["s_0001"]
    assert st.get_commitment(store, "c_0003").lifecycle == "open"  # предложение само не закрывает
    c = st.accept_suggestion(store, "s_0001")
    assert c.lifecycle == "done" and st.get_suggestion(store, "s_0001").status == "accepted"
    assert st.pending_suggestions(store, "+375291234567") == []


def test_reject_suggestion_keeps_open(store):
    st.reject_suggestion(store, "s_0001")
    assert st.get_commitment(store, "c_0003").lifecycle == "open"
    assert st.pending_suggestions(store, "+375291234567") == []


def test_remove_call_results(store):
    st.remove_call_results(store, "call_001")
    assert store.commitments == [] and store.close_suggestions == []


def test_deadline_default_none():
    d = Deadline()
    assert d.type == "none" and d.flags == [] and d.raw is None
