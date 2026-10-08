from __future__ import annotations


# === Consolidated from llmcognition/agent.py (v0.2 experiment; behavior preserved) ===

"""Model interface: one initial decision, optionally one bounded query, final decision.

The model receives only public_projection(task). No shell/web/GitHub tools, no
access to the generator seed or answer key. JSON schema requests decision and
optionally an elicited confidence estimate, not a chain of thought.
"""
import json
import os
from typing import Any

from .core import best_query, possibilities


INITIAL_PROMPT = """Solve the supplied binary-relation problem under these
explicit rules (the same rules apply to every task):
- Each record (u, v, value) claims bit[u] XOR bit[v] == value, with bits 0 or 1.
- All records are authentic unless TWO records for the SAME UNORDERED node pair
  give OPPOSITE values. If this occurs, exactly ONE of those two is false.
  All other records are authentic. There are no other hidden false records.
- If no contradictory pair exists, all supplied records are authentic.
- ARCHIVE returns an authentic relation for a selected pair; AUDIT verifies
  whether a selected record is authentic. Either source can be consulted via
  one of the six query IDs, at most ONCE in total.

Your goal is to determine bit[target[0]] XOR bit[target[1]]. You may either:
answer directly, abstain, or inspect one listed query ID. A correct FINAL bit
is worth 1 utility point; an incorrect final bit or abstention earns 0.
Inspection costs 0.10 points, even if the information is irrelevant.
Choose the action offering the highest expected utility.

Regardless of action, provide your best PROVISIONAL bit in `answer` (0 or 1),
except `abstain` may use -1. If the output schema has `confidence`, it is the probability that this
provisional bit is correct, not the probability your action is appropriate.
When not inspecting, use an empty `query_id`. Do not invent query IDs.
Do not provide an explanation. Return only the specified response object."""

FINAL_PROMPT = """Solve the same binary-relation problem with ONE inspection
result. The universal record-authenticity rules still apply: all records are
true except when a pair of duplicate records for the same unordered pair
conflict; then exactly one of those two is false, and every other record is
true. ARCHIVE supplies an authentic relation; AUDIT verifies a record.
There are no more inspections. Return the most supported final target XOR bit
(0 or 1), or -1 to abstain. If the output schema has `confidence`, it is the probability this final bit is
correct (if abstaining, use 0). Do not provide explanations."""


def schema(stage: str, elicit_confidence: bool) -> dict:
    if stage == "initial":
        properties: dict[str, Any] = {
            "action": {"type": "string", "enum": ["answer", "inspect", "abstain"]},
            "answer": {"type": "integer", "enum": [-1, 0, 1]},
            "query_id": {"type": "string"},
        }
    elif stage == "final":
        properties = {"answer": {"type": "integer", "enum": [-1, 0, 1]}}
    else:
        raise ValueError(stage)
    if elicit_confidence:
        properties["confidence"] = {"type": "number"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def validate_decision(decision: dict, *, initial: bool, elicit_confidence: bool) -> dict:
    if not isinstance(decision, dict) or type(decision.get("answer")) is not int:
        raise ValueError("Model answer is not an integer")
    if decision["answer"] not in (-1, 0, 1):
        raise ValueError("Model answer is not -1, 0 or 1")
    if initial:
        if decision.get("action") not in ("answer", "inspect", "abstain"):
            raise ValueError("Unrecognized action")
        if not isinstance(decision.get("query_id"), str):
            raise ValueError("query_id must be a string")
    if elicit_confidence:
        value = decision.get("confidence")
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError("Confidence must be in [0,1]")
    return decision


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4.1-mini", *, elicit_confidence: bool = True):
        # Optional .env load; deliberately no code path that prints the API key.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env (never commit it).")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the live API extra: pip install -e '.[api]'") from exc
        self.client = OpenAI(timeout=50.0, max_retries=2)
        self.model = model
        self.elicit_confidence = elicit_confidence
        self.api_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _call(self, stage: str, payload: dict) -> dict:
        response = self.client.responses.create(
            model=self.model,
            instructions=INITIAL_PROMPT if stage == "initial" else FINAL_PROMPT,
            input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
            text={"format": {
                "type": "json_schema",
                "name": "llmcognition_" + stage,
                "schema": schema(stage, self.elicit_confidence),
                "strict": True,
            }},
            max_output_tokens=500,
            store=False,
        )
        self.api_calls += 1
        usage = getattr(response, "usage", None)
        self.tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
        if not response.output_text:
            raise RuntimeError("OpenAI response contained no output_text (refusal or truncation)")
        return validate_decision(
            json.loads(response.output_text),
            initial=(stage == "initial"),
            elicit_confidence=self.elicit_confidence,
        )

    def decide(self, public: dict) -> dict:
        return self._call("initial", {"task": public})

    def finalize(self, public: dict, inspection: dict) -> dict:
        return self._call("final", {"task": public, "inspection": inspection})


class MockBackend:
    """Local symbolic reference behavior. This is NOT an OpenAI experiment."""

    model = "mock-symbolic-NOT-LLM"

    def __init__(self, *, elicit_confidence: bool = True):
        self.elicit_confidence = elicit_confidence
        self.api_calls = self.tokens_in = self.tokens_out = 0

    def decide(self, public: dict) -> dict:
        possible = possibilities(public)
        answer = min(possible) if len(possible) == 1 else 0
        q = best_query(public)
        out: dict[str, Any] = {
            "action": "inspect" if len(possible) > 1 and q else "answer",
            "answer": answer,
            "query_id": q["id"] if q and len(possible) > 1 else "",
        }
        if self.elicit_confidence:
            out["confidence"] = 0.99 if len(possible) == 1 else 0.5
        return out

    def finalize(self, public: dict, inspection: dict) -> dict:
        q = next((q for q in public["queries"] if q["id"] == inspection.get("query_id")), None)
        if q is None:
            poss = possibilities(public)
        else:
            result = inspection.get("value") if q["kind"] == "REL" else inspection.get("valid")
            poss = possibilities(public, {"query": q, "result": result})
        out: dict[str, Any] = {"answer": min(poss) if len(poss) == 1 else 0}
        if self.elicit_confidence:
            out["confidence"] = 0.99 if len(poss) == 1 else 0.5
        return out


# === Consolidated from llmcognition/runner.py (v0.2 experiment; behavior preserved) ===

"""Paired-policy study; all model-based policies share a *single initial* LLM call.

Results are checkpointed by complete case for safe resume. There is no network
access except the user's explicit OpenAI request in OpenAIBackend. Oracle calls
never reach a real website.
"""
import json
import hashlib
import random
from pathlib import Path

from .core import public_projection, query_oracle, sha256

POLICIES = ("adaptive", "direct", "random", "count", "always", "symbolic")


def _symbolic_result(task: dict) -> tuple[int, str]:
    public = public_projection(task)
    poss = possibilities(public)
    if len(poss) == 1:
        return next(iter(poss)), ""
    query = best_query(public)
    if not query:
        return -1, ""
    inspection = query_oracle(task, query["id"])
    evidence = {"query": query, "result": (
        inspection["value"] if query["kind"] == "REL" else inspection["valid"]
    )}
    after = possibilities(public, evidence)
    return (next(iter(after)) if len(after) == 1 else -1), query["id"]


def _choice(policy: str, public: dict, initial: dict, rng: random.Random) -> str:
    queries = public["queries"]
    if policy == "adaptive":
        return initial["query_id"] if initial["action"] == "inspect" else ""
    if policy == "direct":
        return ""
    if policy == "always":
        return rng.choice(queries)["id"]
    if policy == "random":
        return rng.choice(queries)["id"] if rng.random() < 0.5 else ""
    if policy == "count":
        return rng.choice(queries)["id"] if len(public["records"]) <= 7 else ""
    raise ValueError(policy)


def select_tasks(dataset: list[dict], limit: int, selection: str, seed: int) -> list[dict]:
    """Case selection is performed locally on sealed labels, never shown to model.

    Balanced selection is for small pilots; `prefix` retains the v0.1 behavior.
    The selected cases are fixed by seed and recorded by hash for checkpointing.
    """
    if selection == "prefix":
        return dataset[:limit]
    if selection != "stratified":
        raise ValueError("selection must be 'prefix' or 'stratified'")
    if limit % 4:
        raise ValueError("stratified --limit must be divisible by 4 (minimum 4)")
    quota = limit // 4
    buckets = {condition: [] for condition in "ABCD"}
    for task in dataset:
        buckets[task["sealed"]["condition"]].append(task)
    if any(len(group) < quota for group in buckets.values()):
        raise ValueError("insufficient cases for balanced selection")
    rng = random.Random(seed)
    chosen = []
    for condition in "ABCD":
        chosen.extend(rng.sample(buckets[condition], quota))
    rng.shuffle(chosen)
    return chosen


def run_study(
    dataset: list[dict],
    backend,
    *,
    output: str | Path,
    dataset_path: str | Path,
    policies: list[str] | None = None,
    limit: int = 8,
    seed: int = 20261008,
    query_cost: float = 0.10,
    max_api_calls: int = 1000,
    resume: bool = False,
    selection: str = "prefix",
) -> list[dict]:
    policies = policies or list(POLICIES)
    if not policies or len(set(policies)) != len(policies) or set(policies) - set(POLICIES):
        raise ValueError(f"policies must be unique members of {POLICIES}")
    if not 0 < limit <= len(dataset):
        raise ValueError("limit must be between 1 and dataset length")
    if not 0 < query_cost < 1:
        raise ValueError("query_cost must be between 0 and 1")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(dataset_path)
    selected = select_tasks(dataset, limit, selection, seed)
    selected_sha256 = hashlib.sha256("\n".join(t["id"] for t in selected).encode()).hexdigest()
    prompt_sha256 = hashlib.sha256((INITIAL_PROMPT + "\n" + FINAL_PROMPT).encode()).hexdigest()
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = {
        "dataset_sha256": digest,
        "backend_model": backend.model,
        "elicit_confidence": backend.elicit_confidence,
        "policies": policies,
        "limit": limit,
        "seed": seed,
        "query_cost": query_cost,
        "protocol": "CMT-2-v0.2-exploratory-post-pilot",
        "selection": selection,
        "selected_ids_sha256": selected_sha256,
        "prompt_sha256": prompt_sha256,
        "shared_initial_response": True,
    }
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists. Choose a new output or pass --resume")
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != meta:
            raise ValueError("Resume refused: study metadata changed")
        from .core import read_jsonl
        existing = read_jsonl(path)
    else:
        if resume and meta_path.exists():
            raise ValueError("Resume metadata exists without results")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        existing = []
    seen = {(r["case_id"], r["policy"]) for r in existing}
    # Each case is written as an indivisible batch; any partially written case
    # due to OS crash is refused rather than silently mixing model responses.
    for task in selected:
        case = task["id"]
        present = {p for k, p in seen if k == case}
        if present == set(policies):
            continue
        if present:
            raise ValueError(f"Incomplete checkpoint for case {case}: remove partial rows")
        public = public_projection(task)
        if backend.model.startswith("mock"):
            initial = backend.decide(public)
        else:
            if backend.api_calls >= max_api_calls:
                raise RuntimeError("API call cap reached; checkpoint is safe for --resume")
            initial = backend.decide(public)
        initial_answer = initial["answer"]
        rng = random.Random(seed ^ int.from_bytes(case.encode("ascii"), "little"))
        local_rows = []
        for policy in policies:
            chosen = ""
            if policy == "symbolic":
                final_answer, chosen = _symbolic_result(task)
                final_confidence = None
            else:
                chosen = _choice(policy, public, initial, rng)
                if chosen:
                    inspection = query_oracle(task, chosen)
                    if not backend.model.startswith("mock") and backend.api_calls >= max_api_calls:
                        raise RuntimeError("API call cap reached; checkpoint is safe for --resume")
                    final = backend.finalize(public, inspection)
                    final_answer = final["answer"]
                    final_confidence = final.get("confidence")
                elif policy == "adaptive" and initial["action"] == "abstain":
                    final_answer, final_confidence = -1, initial.get("confidence")
                else:
                    final_answer, final_confidence = initial_answer, initial.get("confidence")
            truth = task["sealed"]["truth"]
            local_rows.append({
                "case_id": case,
                "condition": task["sealed"]["condition"],
                "policy": policy,
                "model": backend.model if policy != "symbolic" else "deterministic_constraint_baseline",
                "initial_answer": initial_answer if policy != "symbolic" else None,
                "initial_confidence": initial.get("confidence") if policy != "symbolic" else None,
                "initial_action": initial.get("action") if policy != "symbolic" else None,
                "initial_query_id": initial.get("query_id") if policy != "symbolic" else None,
                "initial_correct": int(initial_answer == truth) if policy != "symbolic" else None,
                "selected_query_id": chosen or None,
                "queried": bool(chosen),
                "critical_hit": bool(chosen and chosen in task["sealed"]["critical_query_ids"]),
                "final_answer": final_answer,
                "final_confidence": final_confidence,
                "abstained": final_answer == -1,
                "correct": int(final_answer == truth),
                "utility": round(int(final_answer == truth) - query_cost * bool(chosen), 5),
            })
        with path.open("a", encoding="utf-8") as fp:
            for row in local_rows:
                fp.write(json.dumps(row, sort_keys=True) + "\n")
            fp.flush()
        existing.extend(local_rows)
        seen.update((r["case_id"], r["policy"]) for r in local_rows)
    return existing


# === Consolidated from llmcognition/evaluation.py (v0.2 experiment; behavior preserved) ===

"""Pre-registered endpoints and paired comparisons, standard-library only."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path



def mean(items: list[float]) -> float | None:
    return sum(items) / len(items) if items else None


def auc(pairs: list[tuple[float, int]]) -> float | None:
    positives = [c for c, y in pairs if y == 1]
    negatives = [c for c, y in pairs if y == 0]
    if not positives or not negatives:
        return None
    wins = sum(1 if pos > neg else .5 if pos == neg else 0 for pos in positives for neg in negatives)
    return wins / (len(positives) * len(negatives))


def _summarize(rows: list[dict]) -> dict:
    inspected = [r for r in rows if r["queried"]]
    conf = [(r["initial_confidence"], r["initial_correct"])
            for r in rows if r.get("initial_confidence") is not None and r.get("initial_correct") is not None]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    out = {
        "n": len(rows),
        "accuracy": mean([r["correct"] for r in rows]),
        "utility": mean([r["utility"] for r in rows]),
        "query_rate": mean([int(r["queried"]) for r in rows]),
        "abstain_rate": mean([int(r["abstained"]) for r in rows]),
        "critical_query_precision": mean([int(r["critical_hit"]) for r in inspected]),
        "initial_accuracy": mean([r["initial_correct"] for r in rows if r["initial_correct"] is not None]),
        "pre_feedback_brier": mean([(c - y) ** 2 for c, y in conf]),
        "type2_auroc": auc(conf),
    }
    out["by_condition"] = {}
    for condition, group in sorted(grouped.items()):
        out["by_condition"][condition] = {
            "n": len(group),
            "accuracy": mean([r["correct"] for r in group]),
            "utility": mean([r["utility"] for r in group]),
            "query_rate": mean([int(r["queried"]) for r in group]),
            "critical_query_rate": mean([int(r["critical_hit"]) for r in group]),
        }
    return out


def paired_test(a: dict[str, float], b: dict[str, float], *,
                seed: int = 1337, iterations: int = 3000) -> dict:
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return {"n_paired": 0, "error": "no overlapping cases"}
    diffs = [a[k] - b[k] for k in keys]
    obs = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs)
                   for _ in range(iterations))
    # Two-sided paired sign-flip randomization against a zero mean difference.
    extreme = sum(
        abs(sum(x * (1 if rng.randrange(2) else -1) for x in diffs) / len(diffs)) >= abs(obs) - 1e-12
        for _ in range(iterations)
    )
    return {
        "n_paired": len(keys),
        "difference_mean": obs,
        "bootstrap_95_ci": [boots[int(.025 * iterations)], boots[min(iterations-1, int(.975 * iterations))]],
        "paired_permutation_p_two_sided": (extreme + 1) / (iterations + 1),
        "iterations": iterations,
    }


def report(results: list[dict], *, dataset_path: str | Path,
           results_path: str | Path, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(str(results_path) + ".meta.json").read_text())
    if sha256(dataset_path) != meta["dataset_sha256"]:
        raise ValueError("Private dataset hash mismatch! Reporting aborted")
    key_list = [(r["case_id"], r["policy"]) for r in results]
    if len(key_list) != len(set(key_list)):
        raise ValueError("Duplicate study rows")
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_policy[row["policy"]].append(row)
    details = {p: _summarize(rs) for p, rs in sorted(by_policy.items())}
    comparisons = {}
    if "adaptive" in by_policy:
        a = {r["case_id"]: r["utility"] for r in by_policy["adaptive"]}
        for p, rows in by_policy.items():
            if p != "adaptive":
                comparisons[f"adaptive_minus_{p}"] = paired_test(
                    a, {r["case_id"]: r["utility"] for r in rows})
    discrimination = None
    if "adaptive" in details:
        group = details["adaptive"]["by_condition"]
        if set(group) == {"A", "B", "C", "D"}:
            discrimination = (group["B"]["query_rate"] + group["C"]["query_rate"]
                              - group["A"]["query_rate"] - group["D"]["query_rate"]) / 2
    data = {
        "protocol": meta.get("protocol", "CMT-2-v0.1"),
        "model": meta["backend_model"],
        "dataset_sha256": meta["dataset_sha256"],
        "query_cost": meta["query_cost"],
        "study_limit": meta["limit"],
        "n_complete_cases": len(set(r["case_id"] for r in results)),
        "policy_metrics": details,
        "paired_comparisons": comparisons,
        "epistemic_query_discrimination_BC_minus_AD": discrimination,
        "caveats": [
            "This establishes observable functional behavior, not subjective awareness.",
            "Confidence is elicited in default mode and is not unprompted metacognition.",
            "The symbolic baseline is task-specific, not an LLM with an equal architecture.",
            "If model is mock-symbolic-NOT-LLM, results do not test an OpenAI model.",
            "A single synthetic XOR task family does not establish general metacognition.",
            "Initial model calls are shared across policies, and policy decisions are paired.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    def fmt(x):
        return "NA" if x is None else f"{x:.3f}"
    table = "| Policy | N | Accuracy | Utility | Query rate | Critical precision | Type-2 AUROC |\n"
    table += "|---|---:|---:|---:|---:|---:|---:|\n"
    for name, item in details.items():
        table += f"| {name} | {item['n']} | {fmt(item['accuracy'])} | {fmt(item['utility'])} | {fmt(item['query_rate'])} | {fmt(item['critical_query_precision'])} | {fmt(item['type2_auroc'])} |\n"
    comparisons_md = "\n".join(
        f"- **{name}**: Δutility={fmt(result['difference_mean'])}, "
        f"95% paired bootstrap CI=[{fmt(result['bootstrap_95_ci'][0])}, "
        f"{fmt(result['bootstrap_95_ci'][1])}], "
        f"permutation p={fmt(result['paired_permutation_p_two_sided'])} (N={result['n_paired']})."
        for name, result in comparisons.items() if "difference_mean" in result
    )
    text = f"""# CMT-2 experiment report

Model/backend: `{meta['backend_model']}`  
Dataset SHA-256: `{meta['dataset_sha256']}`  
Completed cases: {data['n_complete_cases']} / {meta['limit']}  
Query penalty: {meta['query_cost']} per lookup

## Primary metrics

{table}

## Paired tests of utility (adaptive vs controls)

{comparisons_md or 'Not available.'}

## Epistemic discrimination

(B+C query rate)/2 − (A+D query rate)/2 = **{fmt(discrimination)}**.  
Positive is directionally consistent with selective information seeking,
but not sufficient to establish metacognition.

## Interpretation limits

""" + "\n".join(f"- {note}" for note in data["caveats"]) + "\n"
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    return data
