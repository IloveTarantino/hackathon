import importlib

import pytest

MODULES = ["config", "contracts", "dates", "quotes", "llm", "extract", "closing", "asr",
           "pipeline", "store"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(f"callagree.{name}")


def test_settings_default_mock(monkeypatch):
    from callagree import config

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for k in ("LLM_BACKEND", "ASR_ENGINE"):
        monkeypatch.delenv(k, raising=False)
    s = config.load_settings()
    assert s.llm_backend == "mock"
    assert s.asr_engine == "tone"
