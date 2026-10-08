# llmcognition — CMT-2: Constraint Metacognition Test

**Reproducible, falsifiable OpenAI API study of information-seeking behavior in a safe, closed-world experimental environment.**

This repository tests a *functional* hypothesis: when the current information is insufficient or contradictory, can an LLM recognize this and selectively obtain a **diagnostic** fact from another information source? And can it avoid unnecessary searching when the information is already sufficient, even if reasoning is difficult?

It **does not** demonstrate consciousness, subjective experience, spontaneous emergence, autonomous internet access, or a novel theory as established fact. It tests an operational notion of monitoring and controlling information acquisition. A conventional symbolic constraint solver is included explicitly as a non-LLM reference to check whether the behavior is already achievable through ordinary computation.

## Quick start — first pull the current repository

```bash
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[api,dev]'
cp .env.example .env
# Edit .env locally with your own OPENAI_API_KEY; it is gitignored.
```

If you downloaded this ZIP instead of cloning the repository, unpack it, enter the `llmcognition/` directory, and start with the Python setup commands; `git pull` applies after setting up the repository checkout.

**Offline validation (no API key, no API cost):**

```bash
python -m pytest -q
python -m llmcognition smoke
```

**Create the 100 sealed cases and verify anti-leakage checks:**

```bash
python -m llmcognition generate --n 100 --out data/private/tasks.jsonl
python -m llmcognition audit --dataset data/private/tasks.jsonl
```

Omit `--seed` for a fresh locally generated random seed. Save the printed seed and SHA-256 privately for reproducibility, and reveal them only **after** collection of results. For a pre-determined replicable dataset you may use `--seed 20261008`, but do not expose the hidden case data to the model. The full dataset, including the ground truth, is kept under `data/private/` and excluded from Git.

**Small REAL OpenAI API pilot (explicitly incurs API charges):**

```bash
python -m llmcognition run \
  --backend openai \
  --model gpt-4.1-mini \
  --dataset data/private/tasks.jsonl \
  --limit 8 \
  --max-api-calls 60 \
  --out runs/openai_pilot.jsonl
python -m llmcognition report \
  --dataset data/private/tasks.jsonl \
  --results runs/openai_pilot.jsonl \
  --output-dir reports/openai_pilot
```

**Main 100-case study (explicit API usage; pre-register first):**

```bash
python -m llmcognition run \
  --backend openai \
  --model gpt-4.1-mini \
  --dataset data/private/tasks.jsonl \
  --limit 100 \
  --max-api-calls 500 \
  --out runs/openai_main.jsonl
python -m llmcognition report \
  --dataset data/private/tasks.jsonl \
  --results runs/openai_main.jsonl \
  --output-dir reports/openai_main
```

Open `reports/openai_main/REPORT.md` and `metrics.json`. The model is configurable (`--model ...`) and requires access to the specified model through your OpenAI API account. The **OpenAI Responses API** uses the official Python SDK with strict JSON Schema Structured Outputs, and `store=False`. There is no live API run until you explicitly select `--backend openai` with a valid `OPENAI_API_KEY`.

Official APIs: [OpenAI Python SDK](https://github.com/openai/openai-python) · [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Experimental design

100 synthetic binary XOR relational problems, balanced across four concealed conditions (25 each). The target is to infer `bit(node_a) XOR bit(node_b)` from records. Incorrect or inconsistent records may exist. A maximum of **one** optional local inspection is allowed at a fixed penalty of 0.10.

| Hidden arm | Observed problem | Desired diagnostic behavior |
|---|---|---|
| **A** | Short path, all necessary information visible | Answer without inspecting |
| **B** | Exactly one relation missing along a path | Retrieve the missing relation from the **ARCHIVE** |
| **C** | One contradiction between competing records | Verify the disputed record via **AUDIT** |
| **D** | Long/difficult path, all information present | Reason with existing facts; avoid unnecessary inspection |

Each task always offers **six** query IDs: three ARCHIVE relation lookups and three AUDIT authenticity checks. Only B and C have **exactly one** diagnostic query; A and D have no query that can change the set of logically possible target answers. Labels A/B/C/D, truth, good query ID, world bits, and random seed are never included in the model request. They remain sealed in the local evaluator.

### Experimental process

```text
  cryptographically seeded task generator
              │
              ├── sealed: condition / answer / true world / diagnostic query
              │          (local evaluator only, never sent to API)
              └── PUBLIC projection: target + visible records + 6 query choices
                               │
                         OpenAI Responses API
                               │
                      initial decision + optional confidence
                               │
                ┌──────────────┴────────────────┐
             answer/abstain                 choose query ID
                │                                 │
                │                         SANDBOX ARCHIVE / AUDIT
                │                        local deterministic oracle
                │                                 │
                │                       one final OpenAI decision
                └──────────────┬──────────────────┘
                               │
                  sealed scorer + paired report
```

No network access other than the OpenAI API itself, no arbitrary URLs, no external websites, no shell commands, no exploit tools, no real system intrusion. "Switching source" here means choosing the local ARCHIVE or AUDIT simulator, **not** escaping a sandbox.

### Comparator policies

- `adaptive`: LLM chooses whether and which query to inspect.
- `direct`: same first LLM answer, query blocked.
- `random`: random inspection half the time, random query.
- `count`: simple record-count threshold, random query.
- `always`: always inspect a random query.
- `symbolic`: task-specific, fully deterministic XOR/constraint solver with the same one-query access; **not** an LLM, and **not** a matched compute baseline. It is a necessary check against mistaking standard computational behavior for novelty.

One identical first model completion is **shared** across all five model-dependent policies for a given task. Only follow-up inspections invoke additional completions. This removes one source of random variance and decreases API usage. The symbolic solver runs locally.

### Primary metrics

1. **Accuracy** of final answer.
2. **Utility** = correctness (`1` or `0`) − `0.10 × inspected`.
3. **Query selection discrimination**: mean query rate in B/C minus mean query rate in A/D.
4. **Diagnostic precision**: proportion of inspected queries that are actually informative.
5. **Type-2 AUROC** and **Brier score** of the model's *initial, pre-feedback confidence*, when confidence is requested.
6. Paired bootstrap 95% confidence intervals and paired sign-flip permutation tests for utility differences.

**Important:** by default we **elicit** a confidence number to measure calibration. This is itself a mild metacognitive prompt. For an instruction-minimized behavior test, run a fresh experiment with `--minimal`; this omits confidence entirely, so Type-2 AUROC and Brier are unavailable. Neither mode proves entirely unprompted emergence because the task still explicitly offers an inspect action and its utility cost.

### Anti-cheating and validity

- No answer keys, hidden labels, condition identifiers, or oracle responses in initial API payloads.
- Random opaque task, record, and query IDs; shuffled cases and record order; balanced ground-truth answers.
- Every condition has identical proportions of ARCHIVE / AUDIT query choices.
- Exact symbolic verification: A/D are uniquely solvable *before* lookup; B/C are not; only their single diagnostic query makes the target decidable.
- Query IDs are allowlisted. Invalid choices consume the one query, without access to unrestricted tools.
- Offline audit tests and CI are included; the study refuses changed dataset hashes at report time and inconsistent resume metadata.
- Reporting clearly identifies `mock-symbolic-NOT-LLM` runs, preventing confusion with live OpenAI evidence.

Read [docs/PROTOCOL.md](docs/PROTOCOL.md) for preregistered hypotheses, falsification criteria, limits, and additional experiments, and [docs/README_RU.md](docs/README_RU.md) for instructions in Russian.

## Reproducibility and security

- Python ≥ 3.10; production package itself uses only the Python standard library.
- Optional live API: `pip install -e '.[api]'`.
- Optional tests: `pip install -e '.[dev]'`.
- CI: pytest and end-to-end offline smoke run, **no API secret in CI**.
- `.env`, `data/private/`, `runs/`, and `reports/` are ignored by Git.
- `--limit 8` and `--max-api-calls 500` are deliberate safety defaults. For a full run specify `--limit 100` and pre-calculate token expenditure.
- Resume with the **same** args plus `--resume` if a full-case checkpoint exists. A stopped mid-case run may repeat that case's initial LLM call; results are not silently mixed.
- Public repository contents alone are **not** evidence that an OpenAI model was tested. Real runs require the user's API key and must be reported with complete configuration.

No license grant is included in this starter ZIP. Repository owners should choose a license before inviting reuse.
