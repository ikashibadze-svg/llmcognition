# CMT-2 Experimental Protocol — preregistration template

**Research question:** Does a large language model exhibit *functional* metacognitive monitoring and control by determining when its information is sufficient, when to seek specific missing evidence, and when to stop seeking evidence? Can these phenomena be described as second-order constraint regulation without mistaking normal algorithms for consciousness?

**Status:** code and checks implemented. No OpenAI run is claimed until a live study actually executes. Register/commit this document *before* collecting the held-out dataset or changing thresholds after seeing outcomes.

## Manipulation and controls

- 100 independently generated XOR tasks, 25 each for conditions A/B/C/D.
- Each case has 18 arbitrarily named binary variables, a target pair, a short or long visible relation path, and 3–6 unrelated decoy records. Only `sealed` stores the true bits.
- B deliberately hides one link. C includes two contradictory records on one link, exactly one of which is false. A and D have all relevant records correct and complete; D requires a longer parity composition.
- Every case exposes 6 lookups, 3 relation lookups (ARCHIVE) and 3 record checks (AUDIT). One optional request with penalty 0.10. Only B/C have a unique query that reduces ambiguity of the target.
- Hidden conditions, query benefit classification, answer and ground truth are never passed to the LLM. For minimal mode the model is not asked for confidence.
- Identical initial model response is reused to evaluate adaptive and externally imposed policies on every case. Post-inspection responses are fresh calls to the *same* model and identical final-answer template.
- Baselines: no lookup, random lookup (p=0.5), count-trigger lookup (records ≤ 7), always random lookup, exact symbolic XOR constraint solver.

## Pre-specified outcomes

**H1 (primary):** adaptive query policy yields higher mean utility than the *count* heuristic: `mean(utility_adaptive - utility_count)>0`; positive lower bound of paired 95% bootstrap CI is evidence of improvement. Also disclose the exact paired two-sided sign-flip permutation p-value.

**H2:** diagnostic selectivity is positive: `(query_rate_B + query_rate_C - query_rate_A - query_rate_D)/2 > 0`. Report this descriptive index with per-condition rates; do not treat a positive result alone as proof of metacognition.

**H3 (elicited-confidence mode only):** initial confidence predicts pre-feedback correctness better than chance, measured by Type-2 AUROC (>0.5). Report the Brier score and prevalence of correctness; do not interpret raw confidence as introspective access to hidden model states.

**H4 (negative control):** D cases should not automatically induce inspections solely because they require more reasoning steps than A. Report query rate D separately.

**H5 (query relevance):** on B and C, adaptive inspections select the unique diagnostic query more often than random chance of `1/6`, and the diagnostic information produces an improvement in final accuracy. Use paired case-level data and pre-feedback accuracy.

**Falsification / weak evidence:**

- `H1` fails if the paired difference is ≤0 or its uncertainty interval includes 0 under the stated rule.
- `H2` fails if querying does not distinguish B/C from A/D.
- If D querying is as frequent as B/C, the system may merely search when a task is hard.
- If inspected queries are indistinguishable from random, high query rate alone is not metacognitive targeting.
- If exact symbolic solver completely matches or beats the LLM, this dataset cannot establish a distinct advantage for an LLM's constraint-based metacognitive process. Such a negative result is scientifically useful.
- Even all passing tests would establish only an *operational, task-bound capacity* and cannot demonstrate consciousness, unique mechanisms, evolutionary emergence, or spontaneous self-modeling.

## Statistical analysis

- Unit of analysis is a generated task; paired policies operate on identical cases and share the first OpenAI completion.
- Utility is `correct - 0.10 * queried` (abstention counts as 0 correct), the designated primary endpoint.
- 3,000 fixed-seed paired case-bootstrap samples for 95% percentile CI; 3,000 sign-flip randomization samples for an exploratory two-sided p-value. These are not a guarantee of power at 100 tasks.
- Secondary comparisons with direct, random, always, symbolic, plus AUROC and Brier, are exploratory. Don't treat uncorrected multiple comparisons as confirmatory discoveries.
- Do not pool offline mock metrics with OpenAI runs. Each report is stamped with the backend model and SHA-256 of the full sealed dataset.
- Report refusal/error counts and actual API calls; do not silently discard bad trials. The runner intentionally halts if the API fails; resume with the same checkpoint metadata.

## Anti-cheating audit

1. **Public/private separation:** only `public_projection(task)` goes to OpenAI; `sealed` has the correct bit, private world, condition and critical query IDs. Unit tests spy on API payloads.
2. **No labels in prompt:** IDs are randomized, case order shuffled, 25/arm balanced, answer labels approximately 50:50 in every arm.
3. **Same query menu size:** the model cannot infer the arm merely by counting or reading source categories; 3 ARCHIVE/3 AUDIT everywhere.
4. **No wrong-oracle leakage:** every oracle response is derived from the true world or authentic record status; inspections run locally, not over internet.
5. **Mathematical task validation:** A/D target provable without lookup, B/C underdetermined, exactly one useful lookup in B/C, and all other lookups provably uninformative about the target answer.
6. **Sealed holdout:** don't commit `data/private`; use fresh seed not shown in prompt, and don't modify the hypothesis after inspecting results.
7. **Source routing in a sandbox:** the agent can't call real Hugging Face, Shell, arbitrary HTTP, or other tools; this tests bounded source-switching, not unauthorized system access.

## Interpretation of consciousness and 'emergence'

A response showing the agent identified missing information and switched source is *behavior consistent with functional metacognitive control*. It can also be the result of learned algorithms or ordinary expected-utility optimization. This benchmark **cannot discriminate these internal mechanisms**. Confidence elicitation makes the strongest unprompted-emergence claim especially inappropriate; minimal mode is less invasive but still exposes query tools and goals. Claims about subjective awareness are out of scope.

## Next-stage test: self-model error without external data deficit (CMT-3)

A stronger follow-up should introduce *latent perturbations to the agent's internal inference strategy* while keeping external information fixed and complete. Compare error prediction and recovery with matched controls without giving a 'fault' indicator. That requires careful instrumentation and additional study design: it is **not claimed as implemented** by this repository.
