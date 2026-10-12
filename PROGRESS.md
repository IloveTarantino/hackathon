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
- (T03) Эталон data/gold/<call_id>.json = {call_id, commitments[], closures[]} (модели GoldCall/GoldClosure в contracts.py).
  id в эталоне — «call_001_c2» (стабильные, не зависят от порядка прогона); closures ссылаются на них.
- (T03) Датасет собирается скриптом scripts/build_synthetic.py (тексты и разметка — в нём, руками); время фраз
  расставляется детерминированно, speaker/t цитат берутся из расшифровки. Правка датасета = правка скрипта + перезапуск.
- (T03) Эталонные даты написаны руками по календарю, НЕ через dates.py — чтобы ими проверять dates.py.
- (T03) «Мастер приедет с 10 до 14»: окно времени остаётся в deadline.raw и в action, тип срока — tomorrow (в контракте
  нет времени суток). «29-го», сказанное 29-го → следующий месяц + same_day_of_month.
- (T03) Отказ (declined) оформлен как обязательство клиента со статусом declined и цитатой отказа.
- (T03) Синхронизация живых записей: хлопок при лежащих РЯДОМ диктофонах до того, как разойтись по комнатам
  (в наушниках чужой хлопок в свой микрофон не попадёт). Сверх задачи — scripts/make_stereo.py (автопоиск хлопка,
  склейка в стерео 8 кГц A-law, короткий канал дополняется тишиной).

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

## Решение команды (01:07, Patek)
- Проект перенесён на /mnt/LinuxData/hackathon (ext4, места много). Блокер «мало места на /home» СНЯТ.
- Кэши и временные файлы направлены в models/ переменными окружения (PIP_CACHE_DIR, HF_HOME, TORCH_HOME, XDG_CACHE_HOME, TMPDIR, OLLAMA_MODELS). Ставь torch и модели в .venv и models/ как обычно, T09 и T13 выполнять.

### T03 — синтетический датасет и вводные для записи (13.10, ~01:35)
Начато в прошлом запуске (оборвался по лимиту): scripts/build_synthetic.py и файлы уже были. В этом запуске:
вычитал все 8 диалогов и эталонные даты по календарю, пересобрал (файлы совпали), добавил тесты и документы.
Сделано: data/transcripts/call_001..008.json, data/gold/call_001..008.json (26 обязательств, 4 закрытия в двух
парах: 001→002 продажи, 003→004 сервис; 005–007 — B2B, 008 — без обязательств), GoldCall/GoldClosure в
contracts.py, docs/RECORDING_BRIEFS.md (4 вводные + запасная, инструкция записи), scripts/make_stereo.py.
Проверено: tests/test_synthetic.py (10 тестов: соответствие контрактам и скрипту, дословность цитат, покрытие
всех статусов/типов сроков/флагов, пары закрытий, склейка стерео на сгенерированном звуке через ffmpeg).
`.venv/bin/pytest` → 59 passed. Команды ffmpeg из инструкции прогнаны на сгенерированных m4a (результат:
pcm_alaw, 8000 Гц, 2 канала). Команда silencedetect на живом хлопке НЕ проверялась (помечено в документе).
