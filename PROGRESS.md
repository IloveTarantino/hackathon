# PROGRESS.md — журнал работы (дописывай в конец)

## Решения
- (T01) Добавлен модуль callagree/config.py (настройки из .env, пути) — в списке модулей его не было, но он нужен всем.
- (T01) data/store.json в .gitignore: это рабочее состояние, пересобирается конвейером; примеры — в data/samples/.
- (T01) Зависимости ASR (torch, T-one) НЕ входят в requirements.txt — ставятся отдельно в T09 (тяжёлые, на диске мало места).
- (T01) На машине есть GPU: RTX 3050 6 ГБ → torch можно ставить с CUDA, но см. проблему с диском.

## Журнал

### T01 — каркас (13.10, ~00:50)
Сделано: .venv (Python 3.12), requirements.txt, пакет callagree/ (config + 9 модулей-заглушек), app.py-заглушка,
tests/test_smoke.py, .env.example, .gitignore, pytest.ini, README.md.
Проверено: `.venv/bin/pytest` → 11 passed; streamlit headless на порту 8599 → HTTP 200.
Проблема: на разделе /home свободно всего ~1,6 ГБ (93% занято). ~/.cache/pip занимает 2,0 ГБ, но он вне папки
проекта — не трогал. Для T09 (torch + модели) места почти наверняка не хватит → нужно решение людей (см. MORNING.md).
