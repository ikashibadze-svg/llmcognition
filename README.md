# LLMCognition — CMT-2 Compact (OpenAI API)

**Компактная версия v0.2.1** проверяемого эксперимента по функциональной метакогниции. Всего 10 исходных файлов; без `.venv`, `__pycache__`, результатов, API-ключей и скрытых ответов.

Объединён исходный код CMT-2 v0.2 без изменения экспериментальных правил. Старые `runs/pilot_v02.jsonl` и `data/private/tasks.jsonl` остаются пригодны; результаты в ZIP не входят.

## Setup (GitHub Codespaces / macOS / Linux)

```bash
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[api,dev]'
cp .env.example .env
# В .env укажи собственный OPENAI_API_KEY (не коммить).
python -m pytest -q
```

## Reproduce the 100-case benchmark

Generate 100 **cases**, NOT 100 files; one sealed `.jsonl` dataset is created (25 cases in each A/B/C/D condition):

```bash
python -m llmcognition generate --n 100
python -m llmcognition audit
python -m llmcognition run --backend openai --model gpt-4.1-mini --limit 8 --selection stratified --out runs/pilot_compact.jsonl
python -m llmcognition report --results runs/pilot_compact.jsonl --output-dir reports/pilot_compact
python scripts/diagnose_pilot.py --results runs/pilot_compact.jsonl --output reports/pilot_compact/DIAGNOSTIC.md
cat reports/pilot_compact/DIAGNOSTIC.md
```

Full study after the pilot (`--limit 100`, new output path):

```bash
python -m llmcognition run --backend openai --model gpt-4.1-mini --limit 100 --selection stratified --out runs/full_compact.jsonl
python -m llmcognition report --results runs/full_compact.jsonl --output-dir reports/full_compact
```

Offline dry run (no API call):

```bash
python -m llmcognition smoke
```

## Conditions & controls

- **A:** complete information, short inference path.
- **B:** one necessary relationship missing.
- **C:** one contradictory duplicate record requiring verification.
- **D:** complete information, longer inference path.
- Six paired policies: `adaptive`, `direct`, `random`, `count`, `always`, `symbolic`.
- Query IDs opaque; one informative query per B/C case; zero informative queries in A/D.
- The model sees `public_projection` only; ground truth remains sealed locally.
- Reports include accuracy, utility (correct − 0.1/query), selective-query discrimination, confidence Brier/AUROC, paired bootstrap/permutation controls.

## Interpretation & privacy

This is an exploratory synthetic benchmark, not proof of consciousness, and a positive result alone does not establish general metacognition. Correct inspection decisions and correct post-inspection inference are evaluated separately. The symbolic baseline is a task-specific expert, not a like-for-like LLM.

Never commit `.env`, `data/private`, `runs` or `reports`. The OpenAI API is used only if `--backend openai` is specified; no browsing or external target system is accessible to the model.

## Existing repository migration

The ZIP is **a replacement project**, not an overlay: extracting it on top of old modules leaves unused legacy files behind. For a truly compact repository, work in a new Git branch and remove obsolete tracked paths (`llmcognition/agent.py`, `cli.py`, `data.py`, `evaluation.py`, `parity.py`, `runner.py`, `validation.py`, old tests/docs) after confirming local changes are backed up. Do **not** delete `data/private`, `runs` or `reports`; these contain local research data. New imports are from `llmcognition.core` and `llmcognition.study`. All CLI examples above work after migration.
