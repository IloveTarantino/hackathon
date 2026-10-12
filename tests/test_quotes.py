"""Проверка цитат (T05)."""
import pytest

from callagree.config import DATA_DIR
from callagree.contracts import GoldCall, Segment, Transcript
from callagree.quotes import find_quote, normalize, number_to_words, verify_evidence
from callagree.store import load_json

LINES = [
    ("employee", "алло иван петрович добрый день это андрей компания техноснаб"),
    ("client", "да"),
    ("employee", "коммерческое предложение я вам пришлю до пятницы там будет два варианта"),
    ("client", "по оплате первую часть тысяч пять переведем до двадцатого"),
    ("employee", "если закажете от пятисот штук сделаем скидку десять процентов"),
    ("client", "счет на сто двадцать рублей я оплачу"),
    ("client", "до конца недели как зарплата придет"),
    ("employee", "договор я вам отправлю"),
    ("client", "угу"),
    ("employee", "курьером завтра до обеда"),
    ("client", "хорошо спасибо до свидания"),
]
SEGMENTS = [Segment(speaker=s, start=i * 5.0, end=i * 5.0 + 4, text=t)
            for i, (s, t) in enumerate(LINES)]


def transcript(call_id):
    return Transcript.model_validate(load_json(DATA_DIR / "transcripts" / f"{call_id}.json"))


@pytest.mark.parametrize("n,words", [
    (0, "ноль"), (5, "пять"), (15, "пятнадцать"), (20, "двадцать"), (42, "сорок два"),
    (120, "сто двадцать"), (500, "пятьсот"), (1000, "одна тысяча"), (2000, "две тысячи"),
    (5000, "пять тысяч"), (12400, "двенадцать тысяч четыреста"),
    (21000, "двадцать одна тысяча"), (111000, "сто одиннадцать тысяч"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


@pytest.mark.parametrize("text,expected", [
    ("Да, КП получили, спасибо!", "да кп получили спасибо"),
    ("Пришлём счёт — ещё сегодня.", "пришлем счет еще сегодня"),
    ("переведём 5 000 до 20-го", "переведем пять тысяч до двадцать"),
    ("скидку 10%", "скидку десять процентов"),
    ("  э-э...  ну  ", "э э ну"),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_exact_match_takes_speaker_and_time_from_transcript():
    m = find_quote("коммерческое предложение я вам пришлю до пятницы", SEGMENTS)
    assert m.verified and m.score == 100
    assert (m.speaker, m.t, m.segment_index, m.span) == ("employee", 10.0, 2, 1)


@pytest.mark.parametrize("quote,index", [
    ("Коммерческое предложение я вам пришлю до пятницы.", 2),   # регистр, пунктуация
    ("коммерческое предложение я вам вышлю до пятницы", 2),      # одно слово другое
    ("По оплате: первую часть, тысяч пять, переведём до двадцатого!", 3),  # ё
    ("если закажите от пятисот штук зделаем скидку десять процентов", 4),  # опечатки
])
def test_minor_differences(quote, index):
    m = find_quote(quote, SEGMENTS)
    assert m.verified and m.score >= 85
    assert m.segment_index == index and m.speaker == SEGMENTS[index].speaker


@pytest.mark.parametrize("quote,index", [
    ("первую часть тысяч 5 переведем до 20-го", 3),
    ("сделаем скидку 10%", 4),
    ("сделаем скидку 10 процентов", 4),
    ("счет на 120 рублей я оплачу", 5),
])
def test_digits_vs_words(quote, index):
    m = find_quote(quote, SEGMENTS)
    assert m.verified, m
    assert m.segment_index == index


@pytest.mark.parametrize("quote", [
    "мы гарантируем возврат денег в течение месяца",   # выдумано
    "коммерческое предложение пришлем после оплаты",    # похоже по теме, но так не говорили
    "договор подпишем на следующей неделе",
    "да",                                               # слишком короткая
    "",
])
def test_absent_quote_not_verified(quote):
    m = find_quote(quote, SEGMENTS)
    assert not m.verified
    assert m.speaker is None and m.t is None and m.segment_index is None


def test_short_segment_does_not_capture_long_quote():
    """Фраза «да» не должна «подтверждать» цитату, в которой есть слово «да»."""
    segs = [Segment(speaker="client", start=0, end=1, text="да"),
            Segment(speaker="client", start=2, end=3, text="да да конечно")]
    assert not find_quote("да мы оплатим счет до пятницы", segs).verified


def test_quote_split_across_adjacent_segments():
    m = find_quote("счет на сто двадцать рублей я оплачу до конца недели", SEGMENTS)
    assert m.verified and m.span == 2
    assert (m.segment_index, m.speaker, m.t) == (5, "client", 25.0)


def test_quote_split_by_interjection_of_other_speaker():
    m = find_quote("договор я вам отправлю курьером завтра до обеда", SEGMENTS)
    assert m.verified
    assert (m.segment_index, m.speaker, m.t) == (7, "employee", 35.0)


def test_quote_starting_in_second_segment_of_window():
    """Время берётся от фразы, где цитата начинается, а не от начала склейки."""
    m = find_quote("я оплачу до конца недели как зарплата придет", SEGMENTS)
    assert m.verified and m.segment_index in (5, 6)
    assert m.speaker == "client"


def test_single_segment_preferred_over_window():
    m = find_quote("до конца недели как зарплата придет", SEGMENTS)
    assert m.verified and (m.segment_index, m.span) == (6, 1)


def test_empty_transcript():
    assert not find_quote("коммерческое предложение пришлю", []).verified


def test_verify_evidence():
    t = Transcript(call_id="x", call_datetime="2026-10-13T14:05:00+03:00",
                   client_phone="+375291234567", segments=SEGMENTS)
    ev, ok = verify_evidence(["коммерческое предложение я вам пришлю до пятницы"], t)
    assert ok and (ev[0].speaker, ev[0].t, ev[0].match_score) == ("employee", 10.0, 100.0)

    ev, ok = verify_evidence(["коммерческое предложение я вам пришлю до пятницы",
                              "оплату гарантируем письмом"], t)
    assert not ok                      # одна из цитат выдумана
    assert ev[0].t == 10.0 and ev[1].t is None and ev[1].speaker is None
    assert ev[1].quote == "оплату гарантируем письмом" and ev[1].match_score < 85

    assert verify_evidence([], t) == ([], False)


def _gold():
    for path in sorted((DATA_DIR / "gold").glob("*.json")):
        yield GoldCall.model_validate(load_json(path))


def test_all_gold_quotes_found_with_right_speaker_and_time():
    n = 0
    for gold in _gold():
        t = transcript(gold.call_id)
        items = [(e.quote, e.speaker, e.t) for c in gold.commitments for e in c.evidence]
        items += [(x.quote, x.speaker, x.t) for x in gold.closures]
        for quote, speaker, start in items:
            m = find_quote(quote, t.segments)
            assert m.verified and m.score == 100, (gold.call_id, quote)
            assert (m.speaker, m.t) == (speaker, start), (gold.call_id, quote)
            n += 1
    assert n >= 30


def test_gold_quotes_not_found_in_other_calls():
    """Цитаты одного звонка не должны «находиться» в звонках других клиентов."""
    phones = {g.call_id: transcript(g.call_id).client_phone for g in _gold()}
    false_hits = []
    for gold in _gold():
        for other, phone in phones.items():
            if phone == phones[gold.call_id]:
                continue
            segs = transcript(other).segments
            for c in gold.commitments:
                for e in c.evidence:
                    if find_quote(e.quote, segs).verified:
                        false_hits.append((gold.call_id, other, e.quote))
    assert false_hits == []
