"""Pre-registered endpoints and paired comparisons, standard-library only."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .data import sha256


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
        "protocol": "CMT-2-v0.1",
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
