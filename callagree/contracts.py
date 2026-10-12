"""Контракты данных (pydantic): расшифровка, обязательство, предложение закрыть, хранилище.

Соответствуют контрактам 1–3 из CLAUDE.md. Лишние поля запрещены, чтобы опечатки в JSON
(в том числе от LLM) ловились на валидации, а не в интерфейсе.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MINSK = ZoneInfo("Europe/Minsk")

Speaker = Literal["employee", "client"]
Party = Literal["employee", "client", "third_party"]
Direction = Literal["outbound", "inbound"]
DeadlineType = Literal[
    "weekday", "day_of_month", "explicit_date", "today", "tomorrow", "day_after_tomorrow",
    "in_days", "in_weeks", "this_week", "next_week", "end_of_week", "end_of_month",
    "vague", "none",
]
DeadlineFlag = Literal[
    "same_weekday_ambiguous", "same_day_of_month", "interval_collapsed", "vague",
    "no_deadline", "unparsed",
]
Status = Literal["firm", "tentative", "conditional", "replaced", "declined"]
Review = Literal["pending", "confirmed", "edited", "rejected"]
Lifecycle = Literal["open", "done"]
SuggestionStatus = Literal["pending", "accepted", "rejected"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _aware(dt: datetime) -> datetime:
    """Время без пояса считаем минским."""
    return dt.replace(tzinfo=MINSK) if dt.tzinfo is None else dt


# ---------- Контракт 1: расшифровка ----------

class Segment(_Model):
    speaker: Speaker
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _order(self) -> "Segment":
        if self.end < self.start:
            raise ValueError("end раньше start")
        return self


class Transcript(_Model):
    call_id: str = Field(min_length=1)
    call_datetime: datetime
    client_phone: str = Field(min_length=3)
    client_name: Optional[str] = None
    direction: Direction = "outbound"
    audio_path: Optional[str] = None
    segments: list[Segment] = Field(default_factory=list)

    _tz = field_validator("call_datetime")(_aware)

    @property
    def call_date(self) -> date:
        """Опорная дата для относительных сроков (по Минску)."""
        return self.call_datetime.astimezone(MINSK).date()


# ---------- Контракт 2: обязательство ----------

class Deadline(_Model):
    raw: Optional[str] = None  # всегда как сказано; None, если срок не звучал
    type: DeadlineType = "none"
    value: Optional[str] = None
    resolved_start: Optional[date] = None  # вычисляет код (dates.py), не LLM
    resolved_end: Optional[date] = None
    flags: list[DeadlineFlag] = Field(default_factory=list)

    @model_validator(mode="after")
    def _interval(self) -> "Deadline":
        if self.resolved_start and self.resolved_end and self.resolved_end < self.resolved_start:
            raise ValueError("resolved_end раньше resolved_start")
        return self


class Amount(_Model):
    value: float
    currency: str = "BYN"
    approximate: bool = False
    raw: Optional[str] = None


class Evidence(_Model):
    speaker: Optional[Speaker] = None  # берётся из расшифровки; None, если цитата не найдена
    t: Optional[float] = None
    quote: str
    match_score: Optional[float] = None


class Commitment(_Model):
    id: str
    client_phone: str
    call_id: str
    debtor: Party
    debtor_name: Optional[str] = None
    creditor: Party
    action: str = Field(min_length=1)
    deadline: Deadline = Field(default_factory=Deadline)
    amount: Optional[Amount] = None
    condition: Optional[str] = None
    status: Status = "firm"
    replaces: Optional[str] = None
    review: Review = "pending"
    lifecycle: Lifecycle = "open"
    evidence: list[Evidence] = Field(default_factory=list)
    verified: bool = False

    @property
    def is_active(self) -> bool:
        """Открыто и по смыслу ещё должно быть выполнено."""
        return (self.lifecycle == "open" and self.review != "rejected"
                and self.status not in ("replaced", "declined"))


# ---------- Контракт 3: хранилище ----------

class CloseSuggestion(_Model):
    id: str
    commitment_id: str
    call_id: str
    speaker: Optional[Speaker] = None
    t: Optional[float] = None
    quote: str
    status: SuggestionStatus = "pending"
    match_score: Optional[float] = None


class Client(_Model):
    name: Optional[str] = None
    phone: str


class CallRecord(_Model):
    call_id: str
    client_phone: str
    datetime: datetime
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None

    _tz = field_validator("datetime")(_aware)


class Store(_Model):
    clients: dict[str, Client] = Field(default_factory=dict)
    calls: list[CallRecord] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    close_suggestions: list[CloseSuggestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Store":
        for name, items in (("commitments", self.commitments),
                            ("close_suggestions", self.close_suggestions)):
            ids = [x.id for x in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"повторяющиеся id в {name}")
        calls = [c.call_id for c in self.calls]
        if len(calls) != len(set(calls)):
            raise ValueError("повторяющиеся call_id в calls")
        return self
