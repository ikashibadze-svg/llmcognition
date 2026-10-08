# Патч CMT-2 v0.2

**Важно:** это исправление ПОСЛЕ первого пилота, а не улучшение результата задним числом.
Первый пилот (`runs/pilot.jsonl`) не изменяется. В публичный GitHub не
выкладывать `.env`, `data/private`, `runs`, `reports`.

## Как перенести ZIP в Codespaces

Скачайте ZIP из сообщения ChatGPT на компьютер и **загрузите файл в Codespaces**
(перетащите его в VS Code Explorer на верхний уровень репозитория). Sandbox-файл
ChatGPT НЕ находится автоматически в удалённом `/workspaces/llmcognition`.

В Codespaces:

```bash
git pull origin main
unzip -o llmcognition_cmt2_v02_patch.zip -d .
python -m pytest -q
python -m llmcognition audit
python -m llmcognition run --backend openai --model gpt-4.1-mini \
 --limit 8 --selection stratified --out runs/pilot_v02.jsonl
python -m llmcognition report --results runs/pilot_v02.jsonl \
 --output-dir reports/pilot_v02
python scripts/diagnose_pilot.py --results runs/pilot_v02.jsonl \
 --output reports/pilot_v02/DIAGNOSTIC.md
cat reports/pilot_v02/DIAGNOSTIC.md

git status --short
git ls-files .env data/private runs reports
```

Не запускайте 100 API задач, пока не разберёмся с новым 8-case pilot.
Подробнее: `docs/PROTOCOL_V02.md`.
