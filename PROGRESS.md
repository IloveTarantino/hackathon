# PROGRESS.md — журнал работы (дописывай в конец)

## Решения
- (T01) Добавлен модуль callagree/config.py (настройки из .env, пути) — в списке модулей его не было, но он нужен всем.
- (T01) data/store.json в .gitignore: это рабочее состояние, пересобирается конвейером; примеры — в data/samples/.
- (T01) Зависимости ASR (torch, T-one) НЕ входят в requirements.txt — ставятся отдельно в T09 (тяжёлые, на диске мало места).
- (T01) На машине есть GPU: RTX 3050 6 ГБ → torch можно ставить с CUDA, но см. проблему с диском.
- (T02) Модели pydantic с extra="forbid": опечатка в поле (в т.ч. от LLM) падает на валидации, а не в интерфейсе.
- (T02) Дата/время без пояса трактуется как Europe/Minsk; опорная дата звонка = Transcript.call_date (по Минску).
- (T02) Сверх контракта: CloseSuggestion.match_score (необязательное), статусы предложения pending|accepted|rejected;
  Evidence.speaker/t могут быть null, если цитата не найдена (verified=false).
- (T02) lifecycle=done ставит только store.accept_suggestion() — т.е. только по кнопке человека.
- (T02) Повторная обработка того же call_id заменяет запись звонка (add_call) и его результаты (remove_call_results).
- (T02) Commitment.is_active: open, не rejected, статус не replaced/declined — это кандидаты на закрытие вторым звонком.

## Журнал

### T01 — каркас (13.10, ~00:50)
Сделано: .venv (Python 3.12), requirements.txt, пакет callagree/ (config + 9 модулей-заглушек), app.py-заглушка,
tests/test_smoke.py, .env.example, .gitignore, pytest.ini, README.md.
Проверено: `.venv/bin/pytest` → 11 passed; streamlit headless на порту 8599 → HTTP 200.
Проблема: на разделе /home свободно всего ~1,6 ГБ (93% занято). ~/.cache/pip занимает 2,0 ГБ, но он вне папки
проекта — не трогал. Для T09 (torch + модели) места почти наверняка не хватит → нужно решение людей (см. MORNING.md).

### T02 — контракты и хранилище (13.10, ~01:05)
Сделано: callagree/contracts.py (Segment, Transcript, Deadline, Amount, Evidence, Commitment, CloseSuggestion,
Client, CallRecord, Store), callagree/store.py (load/save с атомарной записью, выдача id, add_call, решения
человека: set_review, edit_commitment, accept/reject_suggestion), data/samples/ (3 примера + README).
Проверено: tests/test_contracts.py — валидация (плохие значения отклоняются), round-trip JSON для трёх
контрактов, сохранение/загрузка store, операции. `.venv/bin/pytest` → 49 passed.
