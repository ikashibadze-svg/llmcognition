# CMT-2 v0.2 — Post-pilot correction (EXPLORATORY)

## Why we must not call the first 8-case pilot a metacognition falsification

The original initial instruction said that records are "not guaranteed to be
 authentic" while the symbolic baseline assumed every record true except
exactly one of a contradictory duplicate pair. The model had insufficient
information about the *semantics of the benchmark*, not necessarily an
insufficient ability to monitor its own knowledge. Also all adaptive decisions
were `inspect`, and 0/3 B/C queries selected the unique diagnostic request.
The original 8-case pilot was imbalanced (A3/B1/C2/D2) and did not establish
robust first-order XOR competence. Post hoc prompt corrections must be treated
as a NEW experimental condition and cannot be called preregistered replication.

## v0.2 intervention

1. Specify the same authenticity rule the benchmark validator and symbolic
   comparator already use: all records authentic unless a contradictory
   duplicate pair appears, in which case exactly one duplicate is false.
2. Define provisional `answer` and `confidence`, so initial confidence and
   initial correctness have a coherent interpretation, including when inspecting.
3. Add `--selection stratified` for balanced pilots (2 cases per arm for n=8);
   case selection is local and never sent to OpenAI.
4. Preserve the original sealed dataset and old pilot file; write results to a
   NEW output path to prevent mixing protocol versions.
5. Pin the new prompt SHA-256 and selected case-ID hash in run metadata.
6. Add a repository `.gitignore`, because the remote public repository did not
   contain one, even though the first ZIP did.

## Checks before treating the metacognition results as interpretable

- First-order competence: directly answered A/D tasks should mostly be correct.
  If this fails, lack of domain comprehension confounds metacognitive claims.
- Search discrimination: B/C inquiry rate should exceed A/D inquiry rate.
- Query relevance: in B/C the chosen ID should hit the single diagnostic query
  more often than 1/6 chance across a large dataset.
- Real control: compare `adaptive` to `direct`, `random`, `count`, `always` and
  especially the task-specialized symbolic solver. The symbolic program is an
  *upper bound*, not an equal-architecture measure of emergence.
- Avoid circularity: detecting disconnected/conflicting XOR graphs alone is
  sufficient for high performance in this dataset. Passing does not prove a
  separate self-model; later tests require unanticipated *internal process* faults.
- Confidence Type-2 AUROC must be calculated on provisional best guesses, not
  placeholders. It remains *elicited*, not proof of introspection.

## New run

```bash
python -m pytest -q
python -m llmcognition audit
python -m llmcognition run --backend openai --model gpt-4.1-mini \
    --limit 8 --selection stratified --out runs/pilot_v02.jsonl
python -m llmcognition report --results runs/pilot_v02.jsonl \
    --output-dir reports/pilot_v02
python scripts/diagnose_pilot.py --results runs/pilot_v02.jsonl \
    --output reports/pilot_v02/DIAGNOSTIC.md
cat reports/pilot_v02/DIAGNOSTIC.md
```

Do NOT commit `data/private`, `.env`, `runs`, or `reports` to a public repo.
To check tracked secrets: `git ls-files .env data/private runs reports`.
