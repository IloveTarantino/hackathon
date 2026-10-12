"""Проверка цитат: дословная цитата от модели ищется в расшифровке (rapidfuzz).

Найдена → автор и время берутся ИЗ РАСШИФРОВКИ (не от модели), пишется match_score.
Не найдена → verified=false, обязательство уходит в блок «Не подтверждено цитатой».
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from rapidfuzz import fuzz

from .contracts import Evidence, Segment, Transcript

THRESHOLD = 85.0
MIN_QUOTE_CHARS = 5   # «да», «ок» найдутся где угодно — такие цитаты ничего не подтверждают
MAX_WINDOW = 3        # сколько соседних фраз склеивать
JOIN_BONUS = 3.0      # склейка должна быть заметно лучше одной фразы, иначе берём одну

_ONES = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
         "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
         "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]


def _below_1000(n: int) -> list[str]:
    words = [_HUNDREDS[n // 100]] if n >= 100 else []
    n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
        if n:
            words.append(_ONES[n])
    elif n or not words:
        words.append(_ONES[n])
    return words


def number_to_words(n: int) -> str:
    """Число словами (до 999 999), как его произнесут: 5000 → «пять тысяч», 120 → «сто двадцать»."""
    if n >= 1_000_000:
        return str(n)
    words = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        head = _below_1000(thousands)
        last, tail = thousands % 10, thousands % 100
        if last == 1 and tail != 11:
            head[-1], unit = "одна", "тысяча"
        elif last in (2, 3, 4) and tail not in (12, 13, 14):
            head[-1], unit = ("две" if last == 2 else head[-1]), "тысячи"
        else:
            unit = "тысяч"
        words += head + [unit]
    if rest or not words:
        words += _below_1000(rest)
    return " ".join(words)


def normalize(text: str) -> str:
    """Нижний регистр, ё→е, без пунктуации, цифры — словами («5 000» → «пять тысяч»)."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", text)        # 5 000 → 5000
    text = text.replace("%", " процентов ")
    text = re.sub(r"(\d)-?(?:го|му|е|ое|й|ый|ого|ому|ти|х)\b", r"\1", text)  # 20-го → 20
    text = re.sub(r"\d+", lambda m: f" {number_to_words(int(m.group()))} ", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class QuoteMatch:
    verified: bool
    score: float
    speaker: Optional[str] = None
    t: Optional[float] = None
    segment_index: Optional[int] = None   # фраза, в которой начинается цитата
    span: int = 0                         # сколько фраз покрывает совпадение


def _score(quote: str, text: str) -> float:
    """partial_ratio со штрафом, если текст короче цитаты: иначе фраза «да» «находит»
    любую цитату, где есть «да»."""
    if not text:
        return 0.0
    return fuzz.partial_ratio(quote, text) * min(1.0, len(text) / len(quote))


def _windows(segments: Sequence[Segment]):
    """Склейки соседних фраз: подряд по времени и подряд у одного говорящего (ASR мог
    разрезать фразу паузой, а собеседник — вставить «угу» посередине)."""
    n = len(segments)
    seen = set()
    for size in range(2, MAX_WINDOW + 1):
        for i in range(n - size + 1):
            seen.add(tuple(range(i, i + size)))
    for speaker in {s.speaker for s in segments}:
        own = [i for i, s in enumerate(segments) if s.speaker == speaker]
        for size in range(2, MAX_WINDOW + 1):
            for k in range(len(own) - size + 1):
                idx = tuple(own[k:k + size])
                if idx[-1] - idx[0] <= size + 1:  # не дальше пары чужих реплик между ними
                    seen.add(idx)
    return sorted(seen, key=lambda idx: (len(idx), idx))  # короткие склейки — первыми


def find_quote(quote: str, segments: Sequence[Segment], threshold: float = THRESHOLD
               ) -> QuoteMatch:
    """Ищет цитату в расшифровке. Автор и время — от фразы, где цитата начинается."""
    q = normalize(quote)
    if len(q) < MIN_QUOTE_CHARS or not segments:
        return QuoteMatch(False, 0.0)
    texts = [normalize(s.text) for s in segments]

    best_i, best = max(enumerate(_score(q, t) for t in texts), key=lambda x: (x[1], -x[0]))
    best_idx: tuple[int, ...] = (best_i,)
    if best < 100:
        for idx in _windows(segments):
            score = _score(q, " ".join(texts[i] for i in idx))
            if score > best + JOIN_BONUS or (len(best_idx) > 1 and score > best):
                best, best_idx = score, idx

    first = best_idx[0]
    if len(best_idx) > 1:  # в какой из склеенных фраз цитата начинается
        joined = " ".join(texts[i] for i in best_idx)
        start = fuzz.partial_ratio_alignment(q, joined).dest_start
        pos = 0
        for i in best_idx:
            pos += len(texts[i]) + 1
            if start < pos:
                first = i
                break
    score = round(best, 1)
    if score < threshold:
        return QuoteMatch(False, score)
    seg = segments[first]
    return QuoteMatch(True, score, seg.speaker, seg.start, first, len(best_idx))


def verify_evidence(quotes: Sequence[str], transcript: Transcript,
                    threshold: float = THRESHOLD) -> tuple[list[Evidence], bool]:
    """Цитаты обязательства → evidence с автором и временем из расшифровки.

    verified = найдены ВСЕ цитаты (и есть хотя бы одна). У ненайденной speaker и t пустые.
    """
    evidence = []
    for quote in quotes:
        m = find_quote(quote, transcript.segments, threshold)
        evidence.append(Evidence(speaker=m.speaker, t=m.t, quote=quote, match_score=m.score))
    verified = bool(evidence) and all(e.t is not None for e in evidence)
    return evidence, verified
