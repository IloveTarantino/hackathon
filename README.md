# Реестр обещаний — ИИ-агент фиксации договорённостей в телефонных разговорах

Не саммари звонка, а **реестр обещаний, который сам проверяет выполнение**: после звонка
система извлекает обязательства обеих сторон (кто, кому, что, к какому сроку, сумма, условие),
у каждого — дословная цитата и таймкод; во втором звонке предлагает закрыть выполненное.

## Быстрый старт (без LLM и без аудио — mock-режим)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # необязательно: без .env работает mock
.venv/bin/pytest              # тесты
.venv/bin/python -m callagree.run --reset --all   # вся синтетика → data/store.json
.venv/bin/streamlit run app.py
```

## Конвейер (CLI)

```bash
.venv/bin/python -m callagree.run --transcript data/transcripts/call_001.json
.venv/bin/python -m callagree.run --audio file.wav --phone +375291234567 --datetime 2026-10-13T14:05
.venv/bin/python -m callagree.run --reset          # очистить хранилище
```

Второй звонок тому же клиенту даёт «предложения закрыть» по обязательствам из прежних звонков.
`--backend mock|openai` переопределяет `LLM_BACKEND`; `--audio` заработает после подключения ASR.

## Структура

| Путь | Что |
|---|---|
| `callagree/contracts.py` | контракты данных (pydantic) |
| `callagree/dates.py` | нормализация сроков («до пятницы» → дата) |
| `callagree/quotes.py` | проверка цитат по расшифровке |
| `callagree/llm.py`, `extract.py`, `prompts/` | извлечение обязательств (mock / OpenAI-совместимый API) |
| `callagree/closing.py` | предложения закрыть по следующему звонку |
| `callagree/asr.py` | стерео wav → расшифровка (T-one / GigaAM) |
| `callagree/pipeline.py`, `run.py`, `store.py` | конвейер, CLI и хранилище `data/store.json` |
| `app.py` | интерфейс Streamlit |
| `data/transcripts`, `data/gold` | синтетические звонки и эталон |
| `scripts/build_synthetic.py` | сборка синтетики (тексты и разметка — в скрипте) |
| `scripts/make_stereo.py` | две записи голосов → стерео 8 кГц A-law |
| `docs/` | вводные для записи (`RECORDING_BRIEFS.md`), питч, демо, заметки по установке |

## Настройка LLM

`.env`: `LLM_BACKEND=mock|openai`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. Любой
OpenAI-совместимый сервер (Ollama, llama.cpp, vLLM) подключается сменой строки конфига.
Данные не покидают контур заказчика.

Состояние работ — в `MORNING.md`, журнал — в `PROGRESS.md`.
