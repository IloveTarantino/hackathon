"""Настройки из .env / переменных окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORE_PATH = DATA_DIR / "store.json"
TIMEZONE = "Europe/Minsk"


@dataclass(frozen=True)
class Settings:
    llm_backend: str = "mock"  # mock | openai
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = "not-needed"
    llm_timeout: float = 300.0  # секунд на один запрос (локальная модель небыстрая)
    asr_engine: str = "tone"  # tone | gigaam


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    d = Settings()
    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", d.llm_backend).strip().lower(),
        llm_base_url=os.getenv("LLM_BASE_URL", d.llm_base_url),
        llm_model=os.getenv("LLM_MODEL", d.llm_model),
        llm_api_key=os.getenv("LLM_API_KEY", d.llm_api_key),
        llm_timeout=float(os.getenv("LLM_TIMEOUT") or d.llm_timeout),
        asr_engine=os.getenv("ASR_ENGINE", d.asr_engine).strip().lower(),
    )
