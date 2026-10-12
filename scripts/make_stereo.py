"""Склеивает две раздельные записи голосов в телефонное стерео: 8 кГц, A-law, левый канал —
сотрудник, правый — клиент. Записи выравниваются по хлопку в начале (самый громкий пик в
первые --search секунд каждой записи); сдвиги можно задать руками.

Запуск:
  .venv/bin/python scripts/make_stereo.py employee.m4a client.m4a data/audio/call_101.wav
  .venv/bin/python scripts/make_stereo.py a.wav b.wav out.wav --offsets 3.42 5.10

Нужны только ffmpeg и ffprobe в PATH. Инструкция записи — docs/RECORDING_BRIEFS.md.
"""
from __future__ import annotations

import argparse
import array
import subprocess
import sys

RATE = 8000
# Полоса телефонного канала: без неё запись с диктофона «слишком хорошая» для проверки ASR.
PHONE_BAND = "highpass=f=300,lowpass=f=3400"


def decode_head(path: str, seconds: float) -> array.array:
    """Первые seconds секунд файла: моно, 8 кГц, 16 бит."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-t", str(seconds), "-i", path, "-ac", "1", "-ar", str(RATE),
         "-f", "s16le", "-"], check=True, capture_output=True).stdout
    samples = array.array("h")
    samples.frombytes(raw[:len(raw) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def clap_offset(path: str, search: float = 30.0) -> float:
    """Время хлопка (с): самый громкий отсчёт в начале записи."""
    samples = decode_head(path, search)
    if not samples:
        raise SystemExit(f"{path}: не удалось прочитать звук")
    peak = max(range(len(samples)), key=lambda i: abs(samples[i]))
    return peak / RATE


def duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def build_command(employee: str, client: str, out: str, off_e: float, off_c: float,
                  length: float, skip: float = 0.5, phone_band: bool = True) -> list[str]:
    """Команда ffmpeg. skip — сколько отрезать после хлопка, чтобы он не попал в запись;
    length — длина результата: короткий канал дополняется тишиной, а не обрезает длинный."""
    chain = (f"aresample={RATE}" + (f",{PHONE_BAND}" if phone_band else "")
             + f",apad=whole_dur={length:.3f}")
    graph = (f"[0:a]pan=mono|c0=c0,{chain}[l];[1:a]pan=mono|c0=c0,{chain}[r];"
             f"[l][r]join=inputs=2:channel_layout=stereo[a]")
    return ["ffmpeg", "-v", "error", "-y",
            "-ss", f"{off_e + skip:.3f}", "-i", employee,
            "-ss", f"{off_c + skip:.3f}", "-i", client,
            "-filter_complex", graph, "-map", "[a]", "-t", f"{length:.3f}",
            "-ar", str(RATE), "-c:a", "pcm_alaw", out]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("employee", help="запись голоса сотрудника (любой формат, понятный ffmpeg)")
    p.add_argument("client", help="запись голоса клиента")
    p.add_argument("out", help="куда писать стерео wav")
    p.add_argument("--offsets", nargs=2, type=float, metavar=("СОТРУДНИК", "КЛИЕНТ"),
                   help="время хлопка в каждой записи, с (если автопоиск ошибся)")
    p.add_argument("--search", type=float, default=30.0, help="где искать хлопок, с от начала")
    p.add_argument("--skip", type=float, default=0.5, help="отрезать после хлопка, с")
    p.add_argument("--no-phone-band", action="store_true", help="не резать полосу 300–3400 Гц")
    a = p.parse_args(argv)

    off_e, off_c = a.offsets or (clap_offset(a.employee, a.search), clap_offset(a.client, a.search))
    print(f"хлопок: сотрудник {off_e:.2f} с, клиент {off_c:.2f} с")
    length = max(duration(a.employee) - off_e, duration(a.client) - off_c) - a.skip
    if length <= 0:
        raise SystemExit("после хлопка в записях ничего не осталось — проверьте --offsets")
    subprocess.run(build_command(a.employee, a.client, a.out, off_e, off_c, length, a.skip,
                                 not a.no_phone_band), check=True)
    print(f"готово: {a.out} (стерео, {RATE} Гц, A-law; левый — сотрудник, правый — клиент)")


if __name__ == "__main__":
    main()
