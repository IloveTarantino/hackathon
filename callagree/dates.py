"""Нормализация сроков: сырой текст («к среде», «до двадцатого») + дата звонка → даты и флаги.

Правила — из CLAUDE.md («Правила дат»). Опорная дата — дата звонка по Минску. Модель даты
не считает: она отдаёт срок «как сказано» (raw) и, возможно, разложенный type/value. Код
сначала разбирает raw сам; type/value от модели используются, только если raw разобрать не
удалось. Праздники не учитываются.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Optional

from .contracts import Deadline

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
_WEEKDAY_RE = [r"понедельник\w*", r"вторник\w*", r"сред[аыуе]|средой", r"четверг\w*",
               r"пятниц\w*", r"суббот\w*", r"воскресень\w*"]
_MONTH_STEMS = ["январ", "феврал", "март", "апрел", "ма[яйею]", "июн", "июл", "август",
                "сентябр", "октябр", "ноябр", "декабр"]

_UNITS = {
    "ноль": 0, "один": 1, "одного": 1, "одну": 1, "одной": 1, "одна": 1, "два": 2, "две": 2,
    "двух": 2, "пару": 2, "пара": 2, "пары": 2, "три": 3, "трех": 3, "четыре": 4,
    "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11, "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13, "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15, "шестнадцать": 16, "шестнадцати": 16,
    "семнадцать": 17, "семнадцати": 17, "восемнадцать": 18, "восемнадцати": 18,
    "девятнадцать": 19, "девятнадцати": 19,
}
_TENS = {"двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40,
         "сорока": 40, "пятьдесят": 50, "пятидесяти": 50, "шестьдесят": 60, "семьдесят": 70,
         "восемьдесят": 80, "девяносто": 90}

# Порядковые: основа → число; окончания покрывают «двадцатое/-ого/-ому/-ым», «третье/-ьего».
_ORD_STEMS = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7,
    "восьм": 8, "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13,
    "четырнадцат": 14, "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17,
    "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20, "тридцат": 30,
}
_ORD_RE = re.compile(r"^(" + "|".join(sorted(_ORD_STEMS, key=len, reverse=True))
                     + r")(?:ое|ого|ому|ым|ом|ье|ьего|ьему|ьим|ьем)$")

_VAGUE = ["на днях", "ближайш", "как можно скорее", "как можно быстрее", "после праздник",
          "скоро", "в скором времени", "попозже", "позже", "потом", "как только",
          "при первой возможности", "при случае", "со временем", "оперативно", "когда-нибудь"]
_TODAY = ["сегодня", "сейчас", "к вечеру", "до вечера", "вечером", "до обеда", "после обеда",
          "в течение дня", "до конца дня", "в течение часа"]


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s./-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in (tok.strip("./-") for tok in text.split()) if t]


def _cardinal(tokens: list[str], i: int) -> tuple[Optional[int], int]:
    """Количественное числительное с позиции i: (число, сколько слов занято)."""
    if i >= len(tokens):
        return None, 0
    tok = tokens[i]
    if tok.isdigit():
        return int(tok), 1
    if tok in _TENS:
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if nxt in _UNITS and 1 <= _UNITS[nxt] <= 9:
            return _TENS[tok] + _UNITS[nxt], 2
        return _TENS[tok], 1
    if tok in _UNITS:
        return _UNITS[tok], 1
    return None, 0


def _ordinal(tokens: list[str], i: int) -> tuple[Optional[int], int]:
    """Порядковое («двадцатого», «тридцать первого») или цифры («15», «15-го») с позиции i."""
    if i >= len(tokens):
        return None, 0
    tok = tokens[i]
    m = re.fullmatch(r"(\d{1,2})(?:-?(?:го|е|му|ое|ого|ому|м|ым))?", tok)
    if m:
        return int(m.group(1)), 1
    if tok in ("двадцать", "тридцать") and i + 1 < len(tokens):
        m = _ORD_RE.match(tokens[i + 1])
        if m and _ORD_STEMS[m.group(1)] <= 9:
            return _TENS[tok] + _ORD_STEMS[m.group(1)], 2
    m = _ORD_RE.match(tok)
    if m:
        return _ORD_STEMS[m.group(1)], 1
    return None, 0


def _month(token: str) -> Optional[int]:
    for n, stem in enumerate(_MONTH_STEMS, start=1):
        if re.fullmatch(stem + r"\w{0,3}", token):
            return n
    return None


def _weekday(text: str) -> Optional[int]:
    for n, pattern in enumerate(_WEEKDAY_RE):
        if re.search(rf"\b(?:{pattern})\b", text):
            return n
    for n, short in enumerate(WEEKDAYS_SHORT):
        if re.fullmatch(short, text):
            return n
    return None


def _month_end(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def _add_months(d: date, n: int) -> date:
    """Первое число месяца через n месяцев."""
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


# ---------- вычисление по типу ----------

def _result(raw, type_, value, start, end=None, flags=()):
    return Deadline(raw=raw, type=type_, value=value, resolved_start=start,
                    resolved_end=end or start, flags=list(flags))


def _by_weekday(raw, call: date, wd: int, week: Optional[str] = None) -> Deadline:
    """week: None — ближайший будущий; "next" — на следующей неделе; "this" — на этой."""
    value = WEEKDAYS[wd]
    if week == "next":
        return _result(raw, "weekday", value, _monday(call) + timedelta(days=7 + wd))
    if week == "this" and wd >= call.weekday():
        return _result(raw, "weekday", value, _monday(call) + timedelta(days=wd))
    delta = (wd - call.weekday()) % 7
    if delta == 0:
        return _result(raw, "weekday", value, call + timedelta(days=7),
                       flags=["same_weekday_ambiguous"])
    return _result(raw, "weekday", value, call + timedelta(days=delta))


def _by_day_of_month(raw, call: date, day: int) -> Optional[Deadline]:
    if not 1 <= day <= 31:
        return None
    flags = ["same_day_of_month"] if day == call.day else []
    first = call.replace(day=1) if day > call.day else _add_months(call, 1)
    for n in range(4):  # «тридцать первого» в ноябре → ближайший месяц, где есть 31-е
        month = _add_months(first, n)
        if day <= calendar.monthrange(month.year, month.month)[1]:
            return _result(raw, "day_of_month", str(day), month.replace(day=day), flags=flags)
    return None


def _by_explicit(raw, call: date, day: int, month: int, year: Optional[int] = None
                 ) -> Optional[Deadline]:
    value = f"{day:02d}.{month:02d}"
    if year is not None:
        year += 2000 if year < 100 else 0
        try:
            return _result(raw, "explicit_date", f"{value}.{year}", date(year, month, day))
        except ValueError:
            return None
    for y in range(call.year, call.year + 5):  # 29.02 — до ближайшего високосного
        try:
            d = date(y, month, day)
        except ValueError:
            continue
        if d >= call:
            return _result(raw, "explicit_date", value, d)
    return None


def _by_type(raw, call: date, type_: str, value: Optional[str]) -> Optional[Deadline]:
    """Срок по разложенным type/value (от модели или после разбора текста)."""
    v = _norm(value or "")
    tokens = _tokens(v)
    week_end = _monday(call) + timedelta(days=6)
    if type_ == "none":
        return _result(raw, "none", None, None, flags=["no_deadline"])
    if type_ == "vague":
        return Deadline(raw=raw, type="vague", flags=["vague"])
    if type_ == "today":
        return _result(raw, "today", None, call)
    if type_ == "tomorrow":
        return _result(raw, "tomorrow", None, call + timedelta(days=1))
    if type_ == "day_after_tomorrow":
        return _result(raw, "day_after_tomorrow", None, call + timedelta(days=2))
    if type_ in ("in_days", "in_weeks"):
        n, _ = _cardinal(tokens, 0)
        if n is None or not 0 <= n <= 366:
            return None
        step = n if type_ == "in_days" else 7 * n
        return _result(raw, type_, str(n), call + timedelta(days=step))
    if type_ == "this_week":
        flags = ["interval_collapsed"] if call.weekday() >= 4 else []
        return _result(raw, "this_week", None, call, week_end, flags)
    if type_ == "next_week":
        return _result(raw, "next_week", None, week_end + timedelta(days=1),
                       week_end + timedelta(days=7))
    if type_ == "end_of_week":
        if v == "next":
            return _result(raw, "end_of_week", "next", week_end + timedelta(days=7))
        return _result(raw, "end_of_week", None, week_end)
    if type_ == "end_of_month":
        if v == "next":
            return _result(raw, "end_of_month", "next", _month_end(_add_months(call, 1)))
        return _result(raw, "end_of_month", None, _month_end(call))
    if type_ == "weekday":
        wd = int(v) if v in "0123456" and v else _weekday(v)
        return None if wd is None else _by_weekday(raw, call, wd)
    if type_ == "day_of_month":
        day, _ = _ordinal(tokens, 0)
        return None if day is None else _by_day_of_month(raw, call, day)
    if type_ == "explicit_date":
        return _explicit_from_text(raw, call, v)
    return None


# ---------- разбор сырого текста ----------

def _explicit_from_text(raw, call: date, text: str) -> Optional[Deadline]:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return _by_explicit(raw, call, int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{4}|\d{2}))?(?!\d)", text)
    if m and 1 <= int(m.group(2)) <= 12:
        return _by_explicit(raw, call, int(m.group(1)), int(m.group(2)),
                            int(m.group(3)) if m.group(3) else None)
    tokens = _tokens(text)
    for i in range(len(tokens)):
        day, used = _ordinal(tokens, i)
        if day is None or i + used >= len(tokens):
            continue
        month = _month(tokens[i + used])
        if month:
            nxt = tokens[i + used + 1] if i + used + 1 < len(tokens) else ""
            year = int(nxt) if re.fullmatch(r"\d{4}", nxt) else None
            return _by_explicit(raw, call, day, month, year)
    return None


def _relative_from_text(raw, call: date, text: str) -> Optional[Deadline]:
    """«через три дня», «через неделю», «в течение двух недель», «через час»."""
    tokens = _tokens(text)
    for i, tok in enumerate(tokens):
        within = tok == "течение" or tok == "течении"
        if tok != "через" and not within:
            continue
        n, used = _cardinal(tokens, i + 1)
        if n is None:
            n, used = 1, 0  # «через неделю», «через день»
        unit = tokens[i + 1 + used] if i + 1 + used < len(tokens) else ""
        if re.fullmatch(r"минут\w*|час\w*|полчаса", unit):
            return _result(raw, "today", None, call)
        if re.fullmatch(r"день|дня|дней|сут\w+", unit):
            type_ = "in_days"
        elif re.fullmatch(r"недел\w+", unit):
            type_ = "in_weeks"
        else:
            continue
        out = _by_type(raw, call, type_, str(n))
        if out and within:  # «в течение трёх дней» — интервал от дня звонка
            out.resolved_start = call
        return out
    return None


def _parse_raw(raw: str, call: date) -> Optional[Deadline]:
    text = _norm(raw)
    tokens = _tokens(text)
    if not text:
        return None
    for word, type_ in (("послезавтра", "day_after_tomorrow"), ("завтра", "tomorrow"),
                        ("сегодня", "today")):
        if word in tokens:
            return _by_type(raw, call, type_, None)
    out = _explicit_from_text(raw, call, text)
    if out is None and any(_month(t) for t in tokens):
        return None  # месяц назван, а даты такой нет («тридцать первого ноября»)
    out = out or _relative_from_text(raw, call, text)
    if out:
        return out

    next_week = bool(re.search(r"следующ\w+ недел", text))
    this_week = bool(re.search(r"\bэт\w+ недел", text))
    m = re.search(r"\bкон(?:ца|це|ец|цу)\s+(следующ\w+\s+)?(недел|месяц|дня|года)", text)
    if m:
        value = "next" if m.group(1) else None
        if m.group(2) == "дня":
            return _by_type(raw, call, "today", None)
        if m.group(2) == "года":
            return _by_explicit(raw, call, 31, 12)
        return _by_type(raw, call, "end_of_week" if m.group(2) == "недел" else "end_of_month",
                        value)
    wd = _weekday(text)
    if wd is not None:
        # «в следующую пятницу» трактуем как пятницу следующей недели
        next_wd = bool(re.search(r"\bследующ", text))
        return _by_weekday(raw, call, wd, "next" if next_wd else "this" if this_week else None)
    if next_week:
        return _by_type(raw, call, "next_week", None)
    if this_week:
        return _by_type(raw, call, "this_week", None)

    hours = bool(re.search(r"\bс \d{1,2}\S* до \d", text))  # «с 10 до 14» — часы, не числа
    for i, tok in enumerate(tokens):  # число месяца
        day, used = _ordinal(tokens, i)
        if day is None or (hours and tok[0].isdigit()):
            continue
        nxt = tokens[i + used] if i + used < len(tokens) else ""
        prev = tokens[i - 1] if i else ""
        marked = (not tok.isdigit() or nxt.startswith("числ") or prev in ("до", "к", "по")
                  or len(tokens) == 1)
        if marked:
            out = _by_day_of_month(raw, call, day)
            if out:
                return out
    if any(p in text for p in _TODAY):
        return _by_type(raw, call, "today", None)
    if any(p in text for p in _VAGUE):
        return _by_type(raw, call, "vague", None)
    return None


# ---------- внешний интерфейс ----------

def resolve(raw: Optional[str], call_date: date, type: Optional[str] = None,
            value: Optional[str] = None) -> Deadline:
    """Срок «как сказано» (+ необязательные type/value от модели) → Deadline с датами и флагами.

    Порядок: разбор raw кодом → type/value от модели → флаг unparsed (сырой текст сохраняется).
    """
    raw = raw.strip() if raw and raw.strip() else None
    out = _parse_raw(raw, call_date) if raw else None
    if out is None and type and not (raw and type == "none"):
        try:
            out = _by_type(raw, call_date, type, value)
        except (ValueError, OverflowError):
            out = None
    if out is None:
        if raw is None:
            return Deadline(raw=None, type="none", flags=["no_deadline"])
        return Deadline(raw=raw, type="none", value=value, flags=["unparsed"])
    return out


def resolve_deadline(deadline: Deadline, call_date: date) -> Deadline:
    """Пересчитать даты и флаги у срока, пришедшего от модели или из правки человека."""
    return resolve(deadline.raw, call_date, deadline.type, deadline.value)


def format_date(d: date) -> str:
    return f"{WEEKDAYS_SHORT[d.weekday()]}, {d:%d.%m}"


def format_deadline(deadline: Deadline) -> str:
    """Строка для интерфейса: «к среде → ср, 14.10», «на следующей неделе → пн, 19.10 – вс, 25.10»."""
    raw = deadline.raw or "без срока"
    if deadline.resolved_start is None:
        return raw
    if deadline.resolved_end and deadline.resolved_end != deadline.resolved_start:
        return f"{raw} → {format_date(deadline.resolved_start)} – {format_date(deadline.resolved_end)}"
    return f"{raw} → {format_date(deadline.resolved_start)}"
