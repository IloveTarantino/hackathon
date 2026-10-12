"""CLI конвейера.

    python -m callagree.run --transcript data/transcripts/call_001.json
    python -m callagree.run --audio file.wav --phone +375291234567 --datetime 2026-10-13T14:05
    python -m callagree.run --reset --all        # очистить store и прогнать всю синтетику
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime

from . import pipeline
from .config import STORE_PATH, load_settings
from .dates import format_deadline
from .llm import LLMError, get_backend
from .store import load_store

WHO = {"employee": "мы", "client": "клиент", "third_party": "третье лицо"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m callagree.run",
                                description="Звонок → обязательства → data/store.json")
    p.add_argument("--transcript", nargs="+", metavar="JSON", help="расшифровки (контракт 1)")
    p.add_argument("--all", action="store_true", help="все синтетические звонки по порядку")
    p.add_argument("--audio", metavar="WAV", help="запись звонка (стерео: слева сотрудник)")
    p.add_argument("--phone", help="телефон клиента (для --audio)")
    p.add_argument("--datetime", help="дата и время звонка, Минск: 2026-10-13T14:05 (для --audio)")
    p.add_argument("--name", help="имя клиента (для --audio)")
    p.add_argument("--direction", choices=["outbound", "inbound"], default="outbound")
    p.add_argument("--call-id", help="id звонка (по умолчанию — имя файла)")
    p.add_argument("--reset", action="store_true", help="очистить хранилище перед прогоном")
    p.add_argument("--store", default=str(STORE_PATH), help="путь к store.json")
    p.add_argument("--backend", choices=["mock", "openai"], help="переопределить LLM_BACKEND")
    return p


def report(result: pipeline.CallResult, store) -> None:
    print(f"{result.call_id}: обязательств {len(result.commitments)}, "
          f"предложений закрыть {len(result.suggestions)}")
    for c in result.commitments:
        mark = "" if c.verified else "  [не подтверждено цитатой]"
        print(f"  {c.id} {WHO[c.debtor]:<11} {c.status:<11} {c.action} — "
              f"{format_deadline(c.deadline)}{mark}")
    by_id = {c.id: c for c in store.commitments}
    for s in result.suggestions:
        print(f"  {s.id} закрыть? {s.commitment_id} «{by_id[s.commitment_id].action}» ← «{s.quote}»")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.transcript or args.all or args.audio or args.reset):
        parser.error("нужно --transcript, --all, --audio или --reset")
    if args.audio and not (args.phone and args.datetime):
        parser.error("для --audio нужны --phone и --datetime")

    if args.reset:
        pipeline.reset_store(args.store)
        print(f"хранилище очищено: {args.store}")

    settings = load_settings()
    if args.backend:
        settings = replace(settings, llm_backend=args.backend)
    try:
        backend = get_backend(settings)
        results = []
        paths = list(args.transcript or []) + (pipeline.synthetic_transcripts() if args.all else [])
        if paths:
            results += pipeline.process_transcript_files(paths, args.store, backend)
        if args.audio:
            results.append(pipeline.process_audio(
                args.audio, args.phone, datetime.fromisoformat(args.datetime), args.store, backend,
                client_name=args.name, direction=args.direction, call_id=args.call_id))
    except (LLMError, pipeline.PipelineError, FileNotFoundError, ValueError) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1

    store = load_store(args.store)
    for r in results:
        report(r, store)
    if results:
        print(f"итого в {args.store}: клиентов {len(store.clients)}, звонков {len(store.calls)}, "
              f"обязательств {len(store.commitments)}, предложений закрыть "
              f"{len(store.close_suggestions)} (LLM: {backend.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
