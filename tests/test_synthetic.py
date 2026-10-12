"""Синтетический датасет (T03): файлы соответствуют контрактам, покрывают нужные случаи,
эталон согласован с расшифровками и со скриптом сборки."""
import array
import importlib.util
import shutil
import subprocess
from datetime import date, timedelta

import pytest

from callagree.config import DATA_DIR, ROOT
from callagree.contracts import GoldCall, Transcript
from callagree.store import load_json

CALL_IDS = [f"call_{i:03d}" for i in range(1, 9)]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def transcripts():
    return {c: Transcript.model_validate(load_json(DATA_DIR / "transcripts" / f"{c}.json"))
            for c in CALL_IDS}


@pytest.fixture(scope="module")
def gold():
    return {c: GoldCall.model_validate(load_json(DATA_DIR / "gold" / f"{c}.json"))
            for c in CALL_IDS}


@pytest.fixture(scope="module")
def commitments(gold):
    return [c for g in gold.values() for c in g.commitments]


def test_files_match_builder(transcripts, gold):
    """data/ не разошлась с scripts/build_synthetic.py (правили скрипт — пересоберите)."""
    builder = _load_script("build_synthetic")
    assert [c["call_id"] for c in builder.CALLS] == CALL_IDS
    for call in builder.CALLS:
        transcript, g = builder.build(call)
        assert transcript == transcripts[call["call_id"]]
        assert g == gold[call["call_id"]]


def test_segments_ordered_and_asr_like(transcripts):
    for t in transcripts.values():
        assert len(t.segments) >= 10
        assert {s.speaker for s in t.segments} == {"employee", "client"}
        for prev, cur in zip(t.segments, t.segments[1:]):
            assert cur.start >= prev.end
        for s in t.segments:  # как после ASR: нижний регистр, без пунктуации и цифр
            assert s.text == s.text.lower()
            assert not any(ch.isdigit() or ch in ".,!?:;" for ch in s.text)


def test_call_dates_cover_weekdays(transcripts):
    dates = [t.call_date for t in transcripts.values()]
    assert all(d.year == 2026 and d.month == 10 for d in dates)
    assert len({d.weekday() for d in dates}) >= 6


def test_evidence_is_verbatim(transcripts, gold):
    """Цитата дословно есть в расшифровке; автор и время — от этой фразы."""
    for cid, g in gold.items():
        segs = transcripts[cid].segments
        items = [(e.quote, e.speaker, e.t) for c in g.commitments for e in c.evidence]
        items += [(x.quote, x.speaker, x.t) for x in g.closures]
        for quote, speaker, t in items:
            hits = [s for s in segs if quote in s.text]
            assert len(hits) == 1, (cid, quote)
            assert (hits[0].speaker, hits[0].start) == (speaker, t)
        for c in g.commitments:
            assert c.evidence and c.verified
            assert c.call_id == cid and c.client_phone == transcripts[cid].client_phone
            if c.deadline.raw:  # срок «как сказано» звучит в самой цитате
                assert any(c.deadline.raw in e.quote for e in c.evidence), c.id


def test_ids_unique_and_replaces_consistent(gold, commitments):
    ids = [c.id for c in commitments]
    assert len(ids) == len(set(ids))
    by_id = {c.id: c for c in commitments}
    replaced = {c.replaces for c in commitments if c.replaces}
    assert len(replaced) >= 2
    for c in commitments:
        if c.replaces:
            assert by_id[c.replaces].call_id == c.call_id
            assert by_id[c.replaces].status == "replaced"
        if c.status == "replaced":
            assert c.id in replaced
        assert (c.debtor == "third_party") == (c.debtor_name is not None)
        assert (c.status == "conditional") == (c.condition is not None)


def test_required_cases_covered(gold, commitments):
    assert {c.status for c in commitments} == {"firm", "tentative", "conditional", "replaced",
                                               "declined"}
    assert {c.debtor for c in commitments} == {"employee", "client", "third_party"}
    assert any(c.amount and c.amount.approximate for c in commitments)
    assert any(c.amount and not c.amount.approximate for c in commitments)
    flags = {f for c in commitments for f in c.deadline.flags}
    assert {"same_weekday_ambiguous", "same_day_of_month", "interval_collapsed", "vague",
            "no_deadline"} <= flags
    types = {c.deadline.type for c in commitments}
    assert {"weekday", "day_of_month", "explicit_date", "today", "tomorrow",
            "day_after_tomorrow", "in_days", "in_weeks", "this_week", "next_week",
            "end_of_week", "end_of_month", "vague", "none"} <= types
    assert sum(1 for g in gold.values() if not g.commitments) == 1


def test_deadline_flags_and_dates_sane(transcripts, commitments):
    for c in commitments:
        d, call_date = c.deadline, transcripts[c.call_id].call_date
        assert ("no_deadline" in d.flags) == (d.type == "none")
        assert ("vague" in d.flags) == (d.type == "vague")
        if d.type in ("none", "vague"):
            assert d.resolved_start is None and d.resolved_end is None
        else:
            assert call_date <= d.resolved_start <= d.resolved_end <= call_date + timedelta(days=45)


def test_handwritten_dates_spot_check(commitments):
    """Несколько эталонных дат, пересчитанных по календарю независимо от dates.py."""
    by_id = {c.id: c.deadline for c in commitments}
    expected = {
        "call_001_c1": (date(2026, 10, 16), date(2026, 10, 16)),  # вт 13.10, «до пятницы»
        "call_001_c2": (date(2026, 10, 14), date(2026, 10, 14)),  # «к среде»
        "call_002_c2": (date(2026, 10, 19), date(2026, 10, 25)),  # чт 15.10, «на следующей неделе»
        "call_005_c1": (date(2026, 10, 30), date(2026, 10, 30)),  # пт 23.10, «в пятницу»
        "call_005_c2": (date(2026, 10, 23), date(2026, 10, 25)),  # пт, «на этой неделе»
        "call_006_c5": (date(2026, 11, 3), date(2026, 11, 3)),    # вт 27.10, «через неделю»
        "call_007_c3": (date(2026, 11, 2), date(2026, 11, 8)),    # чт 29.10, переход месяца
        "call_007_c5": (date(2026, 11, 29), date(2026, 11, 29)),  # 29-го сказано 29-го
    }
    for cid, (start, end) in expected.items():
        assert (by_id[cid].resolved_start, by_id[cid].resolved_end) == (start, end), cid
    assert date(2026, 10, 13).weekday() == 1 and date(2026, 10, 23).weekday() == 4


def test_closure_pairs(transcripts, gold, commitments):
    """Не меньше двух пар «звонок 1 → звонок 2»: закрывается активное обязательство из более
    раннего звонка того же клиента."""
    by_id = {c.id: c for c in commitments}
    pairs = set()
    for cid, g in gold.items():
        for x in g.closures:
            target = by_id[x.commitment_id]
            first, second = transcripts[target.call_id], transcripts[cid]
            assert first.client_phone == second.client_phone
            assert first.call_datetime < second.call_datetime
            assert target.is_active
            pairs.add((target.call_id, cid))
    assert len(pairs) >= 2
    assert sum(len(g.closures) for g in gold.values()) >= 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")
def test_make_stereo(tmp_path):
    """Две записи с хлопком в разное время → выровненное стерео 8 кГц A-law."""
    ms = _load_script("make_stereo")

    def record(path, clap_at, freq):
        # тихий тон (голос) + короткий громкий щелчок (хлопок) в момент clap_at
        graph = (f"sine=f={freq}:d=6,volume=0.1[v];"
                 f"sine=f=2000:d=0.03,volume=8,adelay={int(clap_at * 1000)}[c];"
                 f"[v][c]amix=inputs=2:duration=first:normalize=0[a]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-map", "[a]",
                        "-ar", "44100", str(path)], check=True)

    emp, cli, out = tmp_path / "emp.wav", tmp_path / "cli.wav", tmp_path / "out.wav"
    record(emp, 1.0, 440)
    record(cli, 2.5, 660)
    assert ms.clap_offset(str(emp)) == pytest.approx(1.0, abs=0.05)
    assert ms.clap_offset(str(cli)) == pytest.approx(2.5, abs=0.05)

    ms.main([str(emp), str(cli), str(out)])
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,sample_rate,channels:format=duration", "-of", "default=nw=1",
         str(out)], check=True, capture_output=True, text=True).stdout
    assert "codec_name=pcm_alaw" in info
    assert "sample_rate=8000" in info and "channels=2" in info
    duration = float(info.split("duration=")[1].split()[0])
    assert duration == pytest.approx(6 - 1.0 - 0.5, abs=0.1)  # по более длинному остатку
    # правый канал короче: его хвост — тишина, левый при этом не обрезан
    tail = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "3.5", "-i", str(out), "-f", "s16le", "-"],
        check=True, capture_output=True).stdout
    pcm = array.array("h")
    pcm.frombytes(tail)
    assert max(abs(v) for v in pcm[0::2]) > 200
    assert max(abs(v) for v in pcm[1::2]) < 50
