"""Сквозной конвейер: расшифровка → обязательства и предложения закрыть → хранилище.

Порядок для одного звонка: зарегистрировать звонок → по открытым обязательствам клиента из
ПРЕДЫДУЩИХ звонков спросить про закрытия → извлечь новые обязательства → дописать в store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import asr, store as st
from .closing import suggest_closures
from .config import DATA_DIR, ROOT, STORE_PATH
from .contracts import CloseSuggestion, Commitment, Store, Transcript
from .extract import extract_commitments
from .llm import get_backend

TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


class PipelineError(RuntimeError):
    pass


@dataclass
class CallResult:
    call_id: str
    commitments: list[Commitment] = field(default_factory=list)
    suggestions: list[CloseSuggestion] = field(default_factory=list)


def _next_number(prefix: str, ids: list[str]) -> int:
    return int(st._next_id(prefix, ids).split("_")[1])


def _relative(path) -> str:
    """Пути в store.json — относительно папки проекта (если файл внутри неё)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def earlier_open_commitments(store: Store, transcript: Transcript) -> list[Commitment]:
    """Кандидаты на закрытие: активные обязательства клиента из звонков РАНЬШЕ этого,
    по которым ещё нет предложения, ждущего решения человека."""
    when = {c.call_id: c.datetime for c in store.calls}
    waiting = {s.commitment_id for s in store.close_suggestions if s.status == "pending"}
    return [c for c in st.open_commitments(store, transcript.client_phone)
            if c.call_id in when and when[c.call_id] < transcript.call_datetime
            and c.id not in waiting]


def process_transcript(transcript: Transcript, store: Store, backend=None,
                       transcript_path=None) -> CallResult:
    """Обрабатывает один звонок и дописывает результат в store (в памяти). Повторная
    обработка того же call_id заменяет прежние результаты этого звонка."""
    backend = backend or get_backend()
    st.remove_call_results(store, transcript.call_id)
    st.add_call(store, transcript, _relative(transcript_path) if transcript_path else None)

    suggestions = suggest_closures(
        transcript, earlier_open_commitments(store, transcript), backend,
        id_start=_next_number("s", [s.id for s in store.close_suggestions]))
    commitments = extract_commitments(
        transcript, backend, id_start=_next_number("c", [c.id for c in store.commitments]))

    store.close_suggestions += suggestions
    store.commitments += commitments
    return CallResult(transcript.call_id, commitments, suggestions)


def process_transcript_file(path, store_path=STORE_PATH, backend=None) -> CallResult:
    transcript = st.load_transcript(path)
    store = st.load_store(store_path)
    result = process_transcript(transcript, store, backend, transcript_path=path)
    st.save_store(store, store_path)
    return result


def process_transcript_files(paths, store_path=STORE_PATH, backend=None) -> list[CallResult]:
    """Несколько звонков — в порядке времени звонка (важно для закрытий)."""
    backend = backend or get_backend()
    loaded = sorted(((st.load_transcript(p), p) for p in paths),
                    key=lambda x: x[0].call_datetime)
    store = st.load_store(store_path)
    results = [process_transcript(t, store, backend, transcript_path=p) for t, p in loaded]
    st.save_store(store, store_path)
    return results


def synthetic_transcripts() -> list[Path]:
    return sorted(TRANSCRIPTS_DIR.glob("call_0*.json"))


def call_id_from_audio(audio_path) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(audio_path).stem).strip("_.")
    return stem or "call"


def process_audio(audio_path, phone: str, call_datetime: datetime, store_path=STORE_PATH,
                  backend=None, client_name: Optional[str] = None, direction: str = "outbound",
                  call_id: Optional[str] = None, transcripts_dir=TRANSCRIPTS_DIR) -> CallResult:
    """wav → ASR → расшифровка (сохраняется в data/transcripts) → обычный конвейер."""
    if not Path(audio_path).exists():
        raise PipelineError(f"нет файла {audio_path}")
    if not hasattr(asr, "transcribe"):
        raise PipelineError("ASR ещё не подключён (задача T09): пока работает только --transcript")
    transcript = asr.transcribe(
        audio_path, call_id=call_id or call_id_from_audio(audio_path), call_datetime=call_datetime,
        client_phone=phone, client_name=client_name, direction=direction)
    transcript.audio_path = _relative(audio_path)
    path = Path(transcripts_dir) / f"{transcript.call_id}.json"
    st.dump_model(transcript, path)
    return process_transcript_file(path, store_path, backend)


def reset_store(store_path=STORE_PATH) -> None:
    st.save_store(Store(), store_path)
