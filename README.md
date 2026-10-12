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
.venv/bin/streamlit run app.py
```

## Структура

| Путь | Что |
|---|---|
| `callagree/contracts.py` | контракты данных (pydantic) |
| `callagree/dates.py` | нормализация сроков («до пятницы» → дата) |
| `callagree/quotes.py` | проверка цитат по расшифровке |
| `callagree/llm.py`, `extract.py`, `prompts/` | извлечение обязательств (mock / OpenAI-совместимый API) |
| `callagree/closing.py` | предложения закрыть по следующему звонку |
| `callagree/asr.py` | стерео wav → расшифровка (T-one / GigaAM) |
| `callagree/pipeline.py`, `store.py` | конвейер и хранилище `data/store.json` |
| `app.py` | интерфейс Streamlit |
| `data/transcripts`, `data/gold` | синтетические звонки и эталон |
| `docs/` | питч, демо, заметки по установке |

## Настройка LLM

`.env`: `LLM_BACKEND=mock|openai`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. Любой
OpenAI-совместимый сервер (Ollama, llama.cpp, vLLM) подключается сменой строки конфига.
Данные не покидают контур заказчика.

Состояние работ — в `MORNING.md`, журнал — в `PROGRESS.md`.
