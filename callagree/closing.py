"""Предложения закрыть обязательства по следующему звонку.

Открытые обязательства клиента + новая расшифровка → модель называет, что выполнено, и
цитату; код проверяет цитату по расшифровке. Результат — только ПРЕДЛОЖЕНИЯ (status=pending):
lifecycle=done ставит человек кнопкой «Закрыть» (store.accept_suggestion).
"""
from __future__ import annotations

from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from . import quotes
from .contracts import CloseSuggestion, Commitment, Transcript
from .dates import format_deadline
from .extract import SPEAKER_RU, load_prompt, render_transcript
from .llm import ask_json, get_backend

# «спасибо», «да хорошо» есть в любом звонке — такой цитатой закрывать нельзя
MIN_QUOTE_CHARS = 10
MIN_QUOTE_WORDS = 2


class LLMClosure(BaseModel):
    model_config = ConfigDict(extra="ignore")
    commitment_id: str
    quote: str


class LLMClosures(BaseModel):
    model_config = ConfigDict(extra="ignore")
    closures: list[LLMClosure] = Field(default_factory=list)


def candidates(open_commitments: Sequence[Commitment], transcript: Transcript) -> list[Commitment]:
    """Что можно закрыть этим звонком: активные обязательства этого клиента из ДРУГИХ звонков."""
    return [c for c in open_commitments
            if c.is_active and c.client_phone == transcript.client_phone
            and c.call_id != transcript.call_id]


def describe(c: Commitment) -> str:
    who = SPEAKER_RU.get(c.debtor) or c.debtor_name or "третье лицо"
    line = f"{c.id} | должен: {who} | {c.action} | срок: {format_deadline(c.deadline)}"
    if c.amount:
        line += f" | сумма: {c.amount.raw or c.amount.value}"
    return line


def build_messages(transcript: Transcript, open_commitments: Sequence[Commitment]) -> list[dict]:
    user = ("Открытые обязательства:\n" + "\n".join(describe(c) for c in open_commitments) +
            f"\n\nНовый разговор:\n{render_transcript(transcript)}\n\n"
            "Какие обязательства выполнены? Ответ в JSON.")
    return [{"role": "system", "content": load_prompt("close_system.md")},
            {"role": "user", "content": user}]


def ask_model(transcript: Transcript, open_commitments: Sequence[Commitment], backend
              ) -> LLMClosures:
    if hasattr(backend, "closures"):  # mock
        return LLMClosures.model_validate(backend.closures(transcript, open_commitments))
    return ask_json(backend, build_messages(transcript, open_commitments), LLMClosures, "closures")


def postprocess(answer: LLMClosures, transcript: Transcript,
                open_commitments: Sequence[Commitment], id_start: int = 1
                ) -> list[CloseSuggestion]:
    """Оставляет только предложения с известным id и цитатой, найденной в расшифровке;
    на одно обязательство — одно предложение (с лучшим совпадением цитаты)."""
    known = {c.id for c in open_commitments}
    best: dict[str, tuple[float, float, quotes.QuoteMatch, str]] = {}
    for item in answer.closures:
        if item.commitment_id not in known:
            continue
        q = quotes.normalize(item.quote)
        if len(q) < MIN_QUOTE_CHARS or len(q.split()) < MIN_QUOTE_WORDS:
            continue
        m = quotes.find_quote(item.quote, transcript.segments)
        if not m.verified:
            continue
        rank = (m.score, -m.t)
        if item.commitment_id not in best or rank > best[item.commitment_id][:2]:
            best[item.commitment_id] = (*rank, m, item.quote)

    out = []
    for c in open_commitments:  # порядок — как у обязательств
        if c.id in best:
            _, _, m, quote = best[c.id]
            out.append(CloseSuggestion(
                id=f"s_{id_start + len(out):04d}", commitment_id=c.id, call_id=transcript.call_id,
                speaker=m.speaker, t=m.t, quote=quote, match_score=m.score))
    return out


def suggest_closures(transcript: Transcript, open_commitments: Sequence[Commitment],
                     backend=None, id_start: int = 1) -> list[CloseSuggestion]:
    """Новый звонок + открытые обязательства клиента → предложения закрыть (ничего не меняет
    в самих обязательствах)."""
    open_commitments = candidates(open_commitments, transcript)
    if not open_commitments or not transcript.segments:
        return []  # закрывать нечего — модель не спрашиваем
    backend = backend or get_backend()
    return postprocess(ask_model(transcript, open_commitments, backend), transcript,
                       open_commitments, id_start)
