# CMT-2 public research report template

**Important:** fill these fields from the actual live run; do not paste mock metrics.

| Field | Value |
|---|---|
| Investigator(s) | |
| Study date (UTC) | |
| Model ID (exact) | |
| OpenAI SDK version | |
| Protocol version | `CMT-2-v0.1` |
| Dataset SHA-256 | |
| Private dataset seed (released only after evaluation) | |
| Task count / A/B/C/D counts | |
| Query penalty | 0.10 |
| Policy set | `adaptive,direct,random,count,always,symbolic` |
| Total successful API calls, input tokens, output tokens | |
| Errors, truncations, resumptions | |
| Prompt mode: `elicited-confidence` or `--minimal` | |

## Prespecified results

Copy the full `metrics.json` and `REPORT.md`, plus raw non-secret decision trajectories, preserving failed cases. Do not change experiment thresholds after seeing model outputs.

## Key comparisons

1. Paired adaptive minus count-heuristic utility: Δ, 95% bootstrap interval, permutation p.
2. B/C versus A/D query discrimination and per-arm rates.
3. Critical-query precision on B/C, compared to chance 1/6.
4. Initial confidence calibration (elicited mode only): type-2 AUROC and Brier.
5. D condition's unnecessary query rate.
6. Comparison with the explicitly non-LLM symbolic constraint solver.

## Evidence categories

- **Functional demonstration:** adaptive query selection under known limited actions.
- **Not demonstrated:** conscious self-awareness, spontaneous emergence, special physical substrates, or unprompted autonomous tool discovery.
- **Alternative explanations:** learned heuristics, symbolic solvability, expected-utility action selection, access to task structure, prompt elicitation.

## Deviations and unsuccessful results

Report all deviations and negative or null findings. If API use is stopped early, mark the analysis exploratory and report the actual sample size.
