"""Нормализатор сроков (T04). Ожидаемые даты посчитаны руками по календарю.

Опорная неделя: пн 12.10.2026 … вс 18.10.2026. 31.12.2026 — четверг.
"""
from datetime import date

import pytest

from callagree.config import DATA_DIR
from callagree.contracts import Deadline, GoldCall, Transcript
from callagree.dates import format_deadline, resolve, resolve_deadline
from callagree.store import load_json


def d(s):
    return date.fromisoformat(s)


WEEK = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17",
        "2026-10-18"]  # пн … вс

# (сказано, день звонка, ожидаемая дата, флаг)
WEEKDAY_MATRIX = (
    [("в пятницу", c, "2026-10-16", False) for c in WEEK[:4]]
    + [("в пятницу", WEEK[4], "2026-10-23", True)]
    + [("в пятницу", c, "2026-10-23", False) for c in WEEK[5:]]
    + [("до среды", c, "2026-10-14", False) for c in WEEK[:2]]
    + [("до среды", WEEK[2], "2026-10-21", True)]
    + [("до среды", c, "2026-10-21", False) for c in WEEK[3:]]
)


@pytest.mark.parametrize("raw,call,expected,ambiguous", WEEKDAY_MATRIX)
def test_weekday_from_every_call_day(raw, call, expected, ambiguous):
    out = resolve(raw, d(call))
    assert out.type == "weekday" and out.raw == raw
    assert out.resolved_start == out.resolved_end == d(expected)
    assert out.flags == (["same_weekday_ambiguous"] if ambiguous else [])


# (сказано, день звонка, тип, значение, начало, конец, флаги)
CASES = [
    # --- день недели ---
    ("к понедельнику", "2026-10-13", "weekday", "понедельник", "2026-10-19", None, []),
    ("во вторник", "2026-10-18", "weekday", "вторник", "2026-10-20", None, []),
    ("в воскресенье", "2026-10-12", "weekday", "воскресенье", "2026-10-18", None, []),
    ("к четвергу", "2026-10-13", "weekday", "четверг", "2026-10-15", None, []),
    ("в субботу", "2026-12-30", "weekday", "суббота", "2027-01-02", None, []),
    ("в среду до обеда", "2026-10-13", "weekday", "среда", "2026-10-14", None, []),
    ("До Пятницы!", "2026-10-13", "weekday", "пятница", "2026-10-16", None, []),
    ("в следующую пятницу", "2026-10-13", "weekday", "пятница", "2026-10-23", None, []),
    ("на следующей неделе в понедельник", "2026-10-13", "weekday", "понедельник",
     "2026-10-19", None, []),
    ("на этой неделе в пятницу", "2026-10-13", "weekday", "пятница", "2026-10-16", None, []),
    ("на этой неделе в пятницу", "2026-10-16", "weekday", "пятница", "2026-10-16", None, []),
    # --- число месяца ---
    ("до двадцатого", "2026-10-13", "day_of_month", "20", "2026-10-20", None, []),
    ("до двадцатого", "2026-10-25", "day_of_month", "20", "2026-11-20", None, []),
    ("к 15-му", "2026-10-13", "day_of_month", "15", "2026-10-15", None, []),
    ("15 числа", "2026-10-20", "day_of_month", "15", "2026-11-15", None, []),
    ("до 25", "2026-10-13", "day_of_month", "25", "2026-10-25", None, []),
    ("тридцать первого", "2026-10-13", "day_of_month", "31", "2026-10-31", None, []),
    ("до четвёртого", "2026-10-13", "day_of_month", "4", "2026-11-04", None, []),
    ("первого числа", "2026-10-13", "day_of_month", "1", "2026-11-01", None, []),
    ("к двадцать третьему", "2026-10-13", "day_of_month", "23", "2026-10-23", None, []),
    ("двадцать девятого числа", "2026-10-29", "day_of_month", "29", "2026-11-29", None,
     ["same_day_of_month"]),
    ("до пятого", "2026-12-20", "day_of_month", "5", "2027-01-05", None, []),
    ("до тридцать первого", "2026-10-31", "day_of_month", "31", "2026-12-31", None,
     ["same_day_of_month"]),  # в ноябре 31-го нет
    ("до тридцатого", "2027-02-10", "day_of_month", "30", "2027-03-30", None, []),
    # --- явная дата ---
    ("15 октября", "2026-10-13", "explicit_date", "15.10", "2026-10-15", None, []),
    ("15 октября", "2026-10-20", "explicit_date", "15.10", "2027-10-15", None, []),
    ("13 октября", "2026-10-13", "explicit_date", "13.10", "2026-10-13", None, []),
    ("тридцатого октября", "2026-10-29", "explicit_date", "30.10", "2026-10-30", None, []),
    ("до первого января", "2026-12-20", "explicit_date", "01.01", "2027-01-01", None, []),
    ("15.10", "2026-10-13", "explicit_date", "15.10", "2026-10-15", None, []),
    ("до 01.11.2026", "2026-10-13", "explicit_date", "01.11.2026", "2026-11-01", None, []),
    ("5 мая 2027", "2026-10-13", "explicit_date", "05.05.2027", "2027-05-05", None, []),
    ("двадцать девятого февраля", "2026-10-13", "explicit_date", "29.02", "2028-02-29",
     None, []),
    ("до конца года", "2026-10-13", "explicit_date", "31.12", "2026-12-31", None, []),
    # --- сегодня / завтра / послезавтра ---
    ("сегодня", "2026-10-13", "today", None, "2026-10-13", None, []),
    ("сегодня вечером", "2026-10-19", "today", None, "2026-10-19", None, []),
    ("к вечеру", "2026-10-13", "today", None, "2026-10-13", None, []),
    ("через час", "2026-10-13", "today", None, "2026-10-13", None, []),
    ("до конца дня", "2026-10-13", "today", None, "2026-10-13", None, []),
    ("завтра", "2026-10-13", "tomorrow", None, "2026-10-14", None, []),
    ("завтра с десяти до четырнадцати", "2026-10-19", "tomorrow", None, "2026-10-20", None, []),
    ("завтра с 10 до 14", "2026-10-19", "tomorrow", None, "2026-10-20", None, []),
    ("завтра", "2026-12-31", "tomorrow", None, "2027-01-01", None, []),
    ("послезавтра.", "2026-10-30", "day_after_tomorrow", None, "2026-11-01", None, []),
    # --- через N ---
    ("через три дня", "2026-10-21", "in_days", "3", "2026-10-24", None, []),
    ("через 10 дней", "2026-10-25", "in_days", "10", "2026-11-04", None, []),
    ("через день", "2026-10-13", "in_days", "1", "2026-10-14", None, []),
    ("через пару дней", "2026-10-13", "in_days", "2", "2026-10-15", None, []),
    ("через двадцать один день", "2026-10-13", "in_days", "21", "2026-11-03", None, []),
    ("через неделю", "2026-10-27", "in_weeks", "1", "2026-11-03", None, []),
    ("через две недели", "2026-12-25", "in_weeks", "2", "2027-01-08", None, []),
    ("в течение трёх дней", "2026-10-13", "in_days", "3", "2026-10-13", "2026-10-16", []),
    ("в течение недели", "2026-10-13", "in_weeks", "1", "2026-10-13", "2026-10-20", []),
    # --- недели и концы ---
    ("на следующей неделе", "2026-10-12", "next_week", None, "2026-10-19", "2026-10-25", []),
    ("на следующей неделе", "2026-10-15", "next_week", None, "2026-10-19", "2026-10-25", []),
    ("на следующей неделе", "2026-10-18", "next_week", None, "2026-10-19", "2026-10-25", []),
    ("на следующей неделе", "2026-12-30", "next_week", None, "2027-01-04", "2027-01-10", []),
    ("на этой неделе", "2026-10-13", "this_week", None, "2026-10-13", "2026-10-18", []),
    ("на этой неделе", "2026-10-15", "this_week", None, "2026-10-15", "2026-10-18", []),
    ("на этой неделе", "2026-10-16", "this_week", None, "2026-10-16", "2026-10-18",
     ["interval_collapsed"]),
    ("на этой неделе", "2026-10-18", "this_week", None, "2026-10-18", "2026-10-18",
     ["interval_collapsed"]),
    ("до конца недели", "2026-10-21", "end_of_week", None, "2026-10-25", None, []),
    ("до конца недели", "2026-10-18", "end_of_week", None, "2026-10-18", None, []),
    ("до конца следующей недели", "2026-10-13", "end_of_week", "next", "2026-10-25", None, []),
    ("до конца месяца", "2026-10-23", "end_of_month", None, "2026-10-31", None, []),
    ("в конце месяца", "2026-12-15", "end_of_month", None, "2026-12-31", None, []),
    ("до конца месяца", "2028-02-10", "end_of_month", None, "2028-02-29", None, []),
    ("до конца следующего месяца", "2026-10-13", "end_of_month", "next", "2026-11-30", None, []),
    # --- расплывчато ---
    ("на днях", "2026-10-13", "vague", None, None, None, ["vague"]),
    ("в ближайшее время", "2026-10-13", "vague", None, None, None, ["vague"]),
    ("как можно скорее", "2026-10-13", "vague", None, None, None, ["vague"]),
    ("после праздников", "2026-10-13", "vague", None, None, None, ["vague"]),
    # --- не распознано: сырой текст остаётся ---
    ("когда рак на горе свистнет", "2026-10-13", "none", None, None, None, ["unparsed"]),
    ("в следующем квартале", "2026-10-13", "none", None, None, None, ["unparsed"]),
    ("тридцать первого ноября", "2026-10-13", "none", None, None, None, ["unparsed"]),
    ("с десяти до четырнадцати", "2026-10-13", "none", None, None, None, ["unparsed"]),
    ("с 10 до 14", "2026-10-13", "none", None, None, None, ["unparsed"]),
]


@pytest.mark.parametrize("raw,call,type_,value,start,end,flags", CASES)
def test_resolve_raw(raw, call, type_, value, start, end, flags):
    out = resolve(raw, d(call))
    assert out.raw == raw
    assert (out.type, out.value) == (type_, value)
    assert out.resolved_start == (d(start) if start else None)
    assert out.resolved_end == (d(end or start) if start else None)
    assert out.flags == flags


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_no_deadline(raw):
    out = resolve(raw, d("2026-10-13"))
    assert (out.raw, out.type, out.flags) == (None, "none", ["no_deadline"])
    assert out.resolved_start is None and out.resolved_end is None


# Разложенный срок от модели: (raw, type, value) → (тип, начало, флаги)
STRUCTURED = [
    ("как договаривались", "weekday", "среда", "weekday", "2026-10-14", []),
    ("как договаривались", "weekday", "2", "weekday", "2026-10-14", []),
    (None, "tomorrow", None, "tomorrow", "2026-10-14", []),
    ("как обычно", "in_days", "три", "in_days", "2026-10-16", []),
    ("как обычно", "in_weeks", "2", "in_weeks", "2026-10-27", []),
    ("как обычно", "day_of_month", "20", "day_of_month", "2026-10-20", []),
    ("как обычно", "day_of_month", "тринадцатого", "day_of_month", "2026-11-13",
     ["same_day_of_month"]),
    ("как обычно", "explicit_date", "2026-10-30", "explicit_date", "2026-10-30", []),
    ("как обычно", "explicit_date", "30 октября", "explicit_date", "2026-10-30", []),
    ("как обычно", "end_of_month", None, "end_of_month", "2026-10-31", []),
    ("как обычно", "vague", None, "vague", None, ["vague"]),
    # модель ошиблась или прислала мусор → срок помечается, а не выдумывается
    ("как обычно", "weekday", "ерунда", "none", None, ["unparsed"]),
    ("как обычно", "in_days", "много", "none", None, ["unparsed"]),
    ("как обычно", "explicit_date", "31.02", "none", None, ["unparsed"]),
    ("как обычно", "none", None, "none", None, ["unparsed"]),
    ("как обычно", "выдуманный_тип", "1", "none", None, ["unparsed"]),
    # сказанное важнее типа от модели
    ("до пятницы", "tomorrow", None, "weekday", "2026-10-16", []),
    ("через три дня", "in_days", "5", "in_days", "2026-10-16", []),
]


@pytest.mark.parametrize("raw,type_,value,exp_type,start,flags", STRUCTURED)
def test_resolve_structured(raw, type_, value, exp_type, start, flags):
    out = resolve(raw, d("2026-10-13"), type_, value)
    assert out.raw == raw and out.type == exp_type and out.flags == flags
    assert out.resolved_start == (d(start) if start else None)


def _gold_cases():
    for path in sorted((DATA_DIR / "gold").glob("*.json")):
        gold = GoldCall.model_validate(load_json(path))
        call = Transcript.model_validate(
            load_json(DATA_DIR / "transcripts" / path.name)).call_date
        for c in gold.commitments:
            yield pytest.param(c.deadline, call, id=c.id)


@pytest.mark.parametrize("expected,call", list(_gold_cases()))
def test_matches_handwritten_gold(expected, call):
    """Эталон датасета размечен руками; код должен дать то же и по тексту, и по type/value."""
    assert resolve(expected.raw, call) == expected
    assert resolve_deadline(expected, call) == expected
    stripped = Deadline(raw=expected.raw, type=expected.type, value=expected.value)
    assert resolve_deadline(stripped, call) == expected


def test_format_deadline():
    call = d("2026-10-13")
    assert format_deadline(resolve("к среде", call)) == "к среде → ср, 14.10"
    assert (format_deadline(resolve("на следующей неделе", call))
            == "на следующей неделе → пн, 19.10 – вс, 25.10")
    assert format_deadline(resolve("на днях", call)) == "на днях"
    assert format_deadline(resolve(None, call)) == "без срока"
