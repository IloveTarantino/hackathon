import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from callagree import pipeline, run, store as st
from callagree.contracts import GoldCall, Store, Transcript
from callagree.llm import MockBackend

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "data/transcripts"
CALLS = [f"call_{i:03d}" for i in range(1, 9)]


def path(call_id):
    return TRANSCRIPTS / f"{call_id}.json"


def gold(call_id):
    return GoldCall.model_validate_json((ROOT / "data/gold" / f"{call_id}.json").read_text("utf-8"))


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "store.json"


@pytest.fixture
def full_store(store_path):
    pipeline.process_transcript_files([path(c) for c in CALLS], store_path, MockBackend())
    return st.load_store(store_path)


def test_all_synthetic_calls_give_correct_store(full_store):
    store = full_store
    assert [c.call_id for c in store.calls] == CALLS  # по времени звонка
    assert all(c.transcript_path == f"data/transcripts/{c.call_id}.json" for c in store.calls)
    assert len(store.clients) == 6
    assert store.clients["+375291234567"].name == "Иван Петрович"

    expected = [g for c in CALLS for g in gold(c).commitments]
    assert [c.id for c in store.commitments] == [f"c_{i:04d}" for i in range(1, len(expected) + 1)]
    ids = {g.id: c.id for g, c in zip(expected, store.commitments)}
    for g, c in zip(expected, store.commitments):
        want = {**g.model_dump(), "id": c.id, "replaces": ids.get(g.replaces)}
        assert c.model_dump() == want

    closures = [(ids[x.commitment_id], c, x.speaker, x.t, x.quote)
                for c in CALLS for x in gold(c).closures]
    assert [(s.commitment_id, s.call_id, s.speaker, s.t, s.quote)
            for s in store.close_suggestions] == closures
    assert [s.id for s in store.close_suggestions] == ["s_0001", "s_0002", "s_0003", "s_0004"]
    # предложено, но не закрыто: закрывает только человек
    assert all(s.status == "pending" for s in store.close_suggestions)
    assert all(c.lifecycle == "open" and c.review == "pending" for c in store.commitments)


def test_order_of_arguments_does_not_matter(store_path, full_store):
    other = store_path.with_name("other.json")
    pipeline.process_transcript_files([path(c) for c in reversed(CALLS)], other, MockBackend())
    assert st.load_store(other) == full_store


def test_one_by_one_equals_batch(tmp_path, full_store):
    p = tmp_path / "one.json"
    for c in CALLS:
        pipeline.process_transcript_file(path(c), p, MockBackend())
    assert st.load_store(p) == full_store


def test_reprocessing_last_call_is_idempotent(store_path, full_store):
    r = pipeline.process_transcript_file(path("call_008"), store_path, MockBackend())
    assert r.commitments == [] and st.load_store(store_path) == full_store


def test_reprocessing_replaces_results_of_that_call(store_path):
    for c in ("call_001", "call_002", "call_002"):
        result = pipeline.process_transcript_file(path(c), store_path, MockBackend())
    store = st.load_store(store_path)
    assert len(store.calls) == 2 and len(store.commitments) == 7
    assert len(store.close_suggestions) == 2 == len(result.suggestions)
    assert len({c.id for c in store.commitments}) == 7


def test_second_call_before_first_gives_no_suggestions(store_path):
    """Закрывать можно только то, что обещано РАНЬШЕ этого звонка."""
    pipeline.process_transcript_file(path("call_002"), store_path, MockBackend())
    pipeline.process_transcript_file(path("call_001"), store_path, MockBackend())
    assert st.load_store(store_path).close_suggestions == []


def test_no_second_suggestion_while_one_is_pending_and_none_after_close(store_path):
    for c in ("call_001", "call_002"):
        pipeline.process_transcript_file(path(c), store_path, MockBackend())
    store = st.load_store(store_path)
    third = st.load_transcript(path("call_002")).model_copy(update={
        "call_id": "call_002b", "call_datetime": datetime.fromisoformat("2026-10-16T10:00:00+03:00")})
    assert pipeline.earlier_open_commitments(store, third) != []
    assert {"c_0002", "c_0004"}.isdisjoint(c.id for c in pipeline.earlier_open_commitments(store, third))
    st.accept_suggestion(store, "s_0001")
    st.reject_suggestion(store, "s_0002")
    ids = [c.id for c in pipeline.earlier_open_commitments(store, third)]
    assert "c_0002" not in ids and "c_0004" in ids  # отклонённое предложение не закрывает


def test_process_audio_uses_asr_and_saves_transcript(tmp_path, store_path, monkeypatch):
    wav = tmp_path / "Звонок Ивану (1).wav"
    wav.write_bytes(b"RIFF")
    seen = {}

    def fake_transcribe(audio_path, **kw):
        seen.update(kw, audio_path=audio_path)
        return Transcript(call_id=kw["call_id"], call_datetime=kw["call_datetime"],
                          client_phone=kw["client_phone"], client_name=kw["client_name"],
                          segments=[{"speaker": "employee", "start": 1.5, "end": 4,
                                     "text": "договор отправлю завтра"}])

    monkeypatch.setattr(pipeline.asr, "transcribe", fake_transcribe, raising=False)
    result = pipeline.process_audio(wav, "+375290000001", datetime(2026, 10, 13, 14, 5), store_path,
                                    MockBackend(), client_name="Иван", transcripts_dir=tmp_path / "tr")
    assert result.call_id == seen["call_id"] == "Звонок_Ивану_1"
    saved = st.load_transcript(tmp_path / "tr" / "Звонок_Ивану_1.json")
    assert saved.audio_path == pipeline._relative(wav)
    store = st.load_store(store_path)
    (c,) = store.commitments
    assert c.action == "отправить договор" and c.evidence[0].t == 1.5
    assert str(c.deadline.resolved_start) == "2026-10-14"
    assert store.calls[0].audio_path == pipeline._relative(wav) and store.clients["+375290000001"].name == "Иван"


def test_process_audio_errors(tmp_path, store_path, monkeypatch):
    with pytest.raises(pipeline.PipelineError, match="нет файла"):
        pipeline.process_audio(tmp_path / "no.wav", "+375", datetime(2026, 10, 13), store_path)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.delattr(pipeline.asr, "transcribe", raising=False)
    with pytest.raises(pipeline.PipelineError, match="ASR"):
        pipeline.process_audio(wav, "+375", datetime(2026, 10, 13), store_path)


# ---------- CLI ----------

def test_cli_main_all_and_reset(store_path, full_store, capsys):
    assert run.main(["--store", str(store_path), "--reset", "--all", "--backend", "mock"]) == 0
    out = capsys.readouterr().out
    assert "хранилище очищено" in out and "call_002: обязательств 3, предложений закрыть 2" in out
    assert "к среде → ср, 14.10" in out and "«да кп получили спасибо»" in out
    assert "звонков 8, обязательств 29, предложений закрыть 4 (LLM: mock)" in out
    assert st.load_store(store_path) == full_store

    assert run.main(["--store", str(store_path), "--reset"]) == 0
    assert st.load_store(store_path) == Store()


def test_cli_transcripts_and_errors(store_path, capsys):
    args = ["--store", str(store_path), "--backend", "mock"]
    assert run.main(args + ["--transcript", str(path("call_002")), str(path("call_001"))]) == 0
    assert len(st.load_store(store_path).close_suggestions) == 2
    assert run.main(args + ["--transcript", "нет_такого.json"]) == 1
    assert run.main(args + ["--audio", "нет.wav", "--phone", "+375", "--datetime", "2026-10-13T14:05"]) == 1
    assert "ошибка" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run.main(args)
    with pytest.raises(SystemExit):
        run.main(args + ["--audio", "a.wav"])


def test_cli_as_module(store_path):
    cmd = [sys.executable, "-m", "callagree.run", "--store", str(store_path), "--backend", "mock",
           "--transcript", "data/transcripts/call_001.json"]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "call_001: обязательств 4" in done.stdout
    assert len(st.load_store(store_path).commitments) == 4
