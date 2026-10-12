"""Загрузка/сохранение data/store.json и операции над хранилищем.

Все изменения из интерфейса (подтвердить/исправить/отклонить/закрыть) идут через функции
этого модуля, чтобы их можно было тестировать без Streamlit.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Union

from .config import STORE_PATH
from .contracts import (CallRecord, Client, CloseSuggestion, Commitment, Store, Transcript)

PathLike = Union[str, Path]


def load_json(path: PathLike) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_model(model, path: PathLike) -> None:
    """Запись через временный файл: интерфейс не увидит наполовину записанный JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_transcript(path: PathLike) -> Transcript:
    return Transcript.model_validate(load_json(path))


def load_store(path: PathLike = STORE_PATH) -> Store:
    """Нет файла → пустое хранилище."""
    path = Path(path)
    if not path.exists():
        return Store()
    return Store.model_validate(load_json(path))


def save_store(store: Store, path: PathLike = STORE_PATH) -> None:
    dump_model(Store.model_validate(store.model_dump()), path)


def _next_id(prefix: str, ids: list[str]) -> str:
    nums = [int(m.group(1)) for i in ids if (m := re.fullmatch(rf"{prefix}_(\d+)", i))]
    return f"{prefix}_{max(nums, default=0) + 1:04d}"


def next_commitment_id(store: Store) -> str:
    return _next_id("c", [c.id for c in store.commitments])


def next_suggestion_id(store: Store) -> str:
    return _next_id("s", [s.id for s in store.close_suggestions])


def upsert_client(store: Store, phone: str, name: Optional[str] = None) -> Client:
    client = store.clients.get(phone)
    if client is None:
        client = store.clients[phone] = Client(phone=phone, name=name)
    elif name and not client.name:
        client.name = name
    return client


def add_call(store: Store, transcript: Transcript,
             transcript_path: Optional[PathLike] = None) -> CallRecord:
    """Регистрирует звонок; повторная обработка того же call_id заменяет запись."""
    upsert_client(store, transcript.client_phone, transcript.client_name)
    record = CallRecord(
        call_id=transcript.call_id, client_phone=transcript.client_phone,
        datetime=transcript.call_datetime, audio_path=transcript.audio_path,
        transcript_path=str(transcript_path) if transcript_path else None)
    store.calls = [c for c in store.calls if c.call_id != record.call_id] + [record]
    store.calls.sort(key=lambda c: c.datetime)
    return record


def remove_call_results(store: Store, call_id: str) -> None:
    """Убирает обязательства и предложения этого звонка (перед повторной обработкой)."""
    gone = {c.id for c in store.commitments if c.call_id == call_id}
    store.commitments = [c for c in store.commitments if c.call_id != call_id]
    store.close_suggestions = [s for s in store.close_suggestions
                               if s.call_id != call_id and s.commitment_id not in gone]


def get_commitment(store: Store, commitment_id: str) -> Commitment:
    for c in store.commitments:
        if c.id == commitment_id:
            return c
    raise KeyError(commitment_id)


def get_suggestion(store: Store, suggestion_id: str) -> CloseSuggestion:
    for s in store.close_suggestions:
        if s.id == suggestion_id:
            return s
    raise KeyError(suggestion_id)


def client_commitments(store: Store, phone: str) -> list[Commitment]:
    return [c for c in store.commitments if c.client_phone == phone]


def open_commitments(store: Store, phone: str) -> list[Commitment]:
    """Кандидаты на закрытие следующим звонком."""
    return [c for c in client_commitments(store, phone) if c.is_active]


# ---------- решения человека ----------

def set_review(store: Store, commitment_id: str, review: str) -> Commitment:
    c = get_commitment(store, commitment_id)
    c.review = review
    return c


def edit_commitment(store: Store, commitment_id: str, **fields) -> Commitment:
    """Правка человеком: меняет поля и ставит review=edited."""
    c = get_commitment(store, commitment_id)
    updated = Commitment.model_validate({**c.model_dump(), **fields, "review": "edited"})
    store.commitments[store.commitments.index(c)] = updated
    return updated


def accept_suggestion(store: Store, suggestion_id: str) -> Commitment:
    """Человек нажал «Закрыть»: единственный путь к lifecycle=done."""
    s = get_suggestion(store, suggestion_id)
    c = get_commitment(store, s.commitment_id)
    s.status = "accepted"
    c.lifecycle = "done"
    return c


def reject_suggestion(store: Store, suggestion_id: str) -> CloseSuggestion:
    s = get_suggestion(store, suggestion_id)
    s.status = "rejected"
    return s


def pending_suggestions(store: Store, phone: str) -> list[CloseSuggestion]:
    ids = {c.id for c in client_commitments(store, phone) if c.lifecycle == "open"}
    return [s for s in store.close_suggestions if s.status == "pending" and s.commitment_id in ids]
