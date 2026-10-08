"""Paired-policy study; all model-based policies share a *single initial* LLM call.

Results are checkpointed by complete case for safe resume. There is no network
access except the user's explicit OpenAI request in OpenAIBackend. Oracle calls
never reach a real website.
"""
from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

from .data import public_projection, query_oracle, sha256
from .agent import INITIAL_PROMPT, FINAL_PROMPT
from .parity import best_query, possibilities

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
        from .data import read_jsonl
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
