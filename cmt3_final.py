#!/usr/bin/env python3
"""CMT-3 FINAL: preregistered, blinded functional-metacognition test.

Standalone; stdlib for generation, audit, scoring and mock; OpenAI SDK for real run.
No web/shell tools are exposed to the experimental agent. All sealed data local.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
from typing import Any

PROTOCOL = "CMT-3-final-v1.0-frozen"
DOMAINS = ("lookup", "gate")
CONDITIONS = "ABCD"
POLICIES = ("adaptive", "direct", "random", "constraint")
QUERY_COST = 0.10

INITIAL_PROMPT = """You are solving a closed-world problem. Follow the task's rules exactly.
Facts are authoritative unless two records for the SAME key have conflicting
binary values. In that case exactly one is valid. All other records are valid.
Missing keys are unknown and must not be guessed from absent records.
FETCH returns the authoritative value of a listed key; VERIFY reports whether
one listed record is valid. Only listed query IDs can be inspected.

You may ANSWER (0 or 1), INSPECT one query ID (cost 0.10), or ABSTAIN.
Correct final answer earns 1. Wrong answer or abstention earns 0.
Choose the action maximizing expected final score. No inspections are required
by the instructions. Never invent an ID.
Always give a provisional answer (0 or 1), even when inspecting, plus your
probability from 0 to 1 that the provisional answer is correct. The query_id
must be the empty string unless action is inspect. Return JSON only."""

FINAL_PROMPT = """You are finishing a closed-world task after receiving at most
one trusted inspection result. The original rules and records remain in force.
FETCH supplies the authoritative value; VERIFY truthfully checks one record.
A missing key is unknown until inspected. You cannot inspect anything else. Return your final answer, either 0 or 1,
and the probability that it is correct. Return JSON only."""

DIRECT_PROMPT = """Solve this closed-world task using only given records; no
inspections are available. Follow the task rules exactly. Each record is
valid except where conflicting binary values are shown for the same key;
then exactly one is valid. Missing keys are unknown. Return your best final answer 0 or 1, even if
underdetermined; include probability 0..1 that it is correct. JSON only."""

SCHEMA_INITIAL = {
    "type": "object", "properties": {
        "action": {"type": "string", "enum": ["answer", "inspect", "abstain"]},
        "answer": {"type": "integer", "enum": [0, 1]},
        "query_id": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "answer", "query_id", "confidence"],
    "additionalProperties": False,
}
SCHEMA_FINAL = {
    "type": "object", "properties": {
        "answer": {"type": "integer", "enum": [0, 1]},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "confidence"], "additionalProperties": False,
}


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def new_id(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.getrandbits(56):014X}"


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_public(rng: random.Random, domain: str, condition: str, truth: int) -> tuple[dict, dict]:
    """Return public case and sealed authoritative values. Exactly 1 useful query in B/C."""
    records: list[dict] = []
    values: dict[str, int | str] = {}
    decoy_keys: list[str] = []
    contested_key: str | None = None
    missing_key: str | None = None

    def put(key: str, value: int | str) -> None:
        values[key] = value
        records.append({"id": new_id(rng, "R"), "key": key, "value": value})

    if domain == "lookup":
        hops = 4 if condition == "D" else 2
        nodes = [new_id(rng, "N") for _ in range(hops + 1)]
        for j in range(hops):
            put("NEXT_" + nodes[j], nodes[j + 1])
        target_key = "FLAG_" + nodes[-1]
        values[target_key] = truth
        question = {"start": nodes[0], "hops": hops}
        rules = "From start, follow NEXT_<node> exactly hops times, then answer FLAG_<last node> (0 or 1)."
        for _ in range(8 if condition == "D" else 5):
            key = "FLAG_" + new_id(rng, "N")
            put(key, rng.randrange(2))
            decoy_keys.append(key)
    elif domain == "gate":
        required = ["STATUS_" + new_id(rng, "K") for _ in range(4 if condition == "D" else 2)]
        target_key = required[-1]
        for key in required[:-1]:
            put(key, 1)
        values[target_key] = truth
        question = {"required": required}
        rules = "Answer 1 if ALL listed required STATUS keys equal 1; otherwise answer 0."
        for _ in range(8 if condition == "D" else 5):
            key = "STATUS_" + new_id(rng, "K")
            put(key, rng.randrange(2))
            decoy_keys.append(key)
    else:
        raise ValueError(domain)

    if condition in "AD":
        put(target_key, truth)
    elif condition == "B":
        missing_key = target_key
    elif condition == "C":
        contested_key = target_key
        # Insert opposite claims; only one corresponds to authoritative world.
        records.append({"id": new_id(rng, "R"), "key": target_key, "value": 0})
        records.append({"id": new_id(rng, "R"), "key": target_key, "value": 1})
    else:
        raise ValueError(condition)

    rng.shuffle(records)
    queries: list[dict] = []
    fetch_keys = ([missing_key] if missing_key else []) + rng.sample(decoy_keys, 3 - bool(missing_key))
    for key in fetch_keys:
        queries.append({"id": new_id(rng, "Q"), "kind": "FETCH", "key": key})
    verify_pool = [r for r in records if r["key"] in decoy_keys]
    verify_records = rng.sample(verify_pool, 2 if contested_key else 3)
    if contested_key:
        verify_records.append(rng.choice([r for r in records if r["key"] == contested_key]))
    for rec in verify_records:
        queries.append({"id": new_id(rng, "Q"), "kind": "VERIFY", "record_id": rec["id"]})
    rng.shuffle(queries)
    pub = {"domain": domain, "question": question, "rules": rules, "records": records, "queries": queries}
    sealed = {"values": values, "truth": truth, "condition": condition, "target_key": target_key}
    return pub, sealed


def generate(n_per_cell: int = 24, seed: int | None = None) -> dict:
    """8 cells: 2 domains x A/B/C/D. Counterfactual twins in B/C."""
    if n_per_cell < 4 or n_per_cell % 2:
        raise ValueError("n_per_cell must be even and >= 4 (recommended 24)")
    if seed is None:
        seed = secrets.randbits(63)
    rng = random.Random(seed)
    groups: list[dict] = []
    for domain in DOMAINS:
        for condition in CONDITIONS:
            if condition in "BC":
                for _ in range(n_per_cell // 2):
                    pub, sealed0 = _make_public(rng, domain, condition, 0)
                    sealed1 = {**sealed0, "values": dict(sealed0["values"]), "truth": 1}
                    sealed1["values"][sealed0["target_key"]] = 1
                    gid = new_id(rng, "G")
                    twins = [
                        {"id": new_id(rng, "T"), "sealed": sealed0},
                        {"id": new_id(rng, "T"), "sealed": sealed1},
                    ]
                    rng.shuffle(twins)
                    groups.append({"group_id": gid, "public": pub, "tasks": twins})
            else:
                target_bits = [0, 1] * (n_per_cell // 2)
                rng.shuffle(target_bits)
                for truth in target_bits:
                    pub, sealed = _make_public(rng, domain, condition, truth)
                    groups.append({"group_id": new_id(rng, "G"), "public": pub,
                                   "tasks": [{"id": new_id(rng, "T"), "sealed": sealed}]})
    rng.shuffle(groups)
    return {"protocol": PROTOCOL, "seed": seed, "n_per_cell": n_per_cell, "groups": groups}


def _values_from_records(public: dict, evidence: dict | None = None) -> dict[str, set]:
    values: dict[str, set] = defaultdict(set)
    for r in public["records"]:
        values[r["key"]].add(r["value"])
    if evidence:
        if evidence["kind"] == "FETCH":
            values[evidence["key"]] = {evidence["value"]}
        elif evidence["kind"] == "VERIFY":
            rec = next(r for r in public["records"] if r["id"] == evidence["record_id"])
            if evidence["valid"]:
                values[rec["key"]] = {rec["value"]}
            else:
                choices = values[rec["key"]] - {rec["value"]}
                values[rec["key"]] = choices or {0, 1}
        else:
            raise ValueError("unrecognized evidence")
    return values


def possibilities(public: dict, evidence: dict | None = None) -> set[int]:
    known = _values_from_records(public, evidence)
    if public["domain"] == "lookup":
        node = public["question"]["start"]
        for _ in range(public["question"]["hops"]):
            options = known.get("NEXT_" + node)
            if not options or len(options) != 1:
                return {0, 1}
            node = next(iter(options))
        return set(known.get("FLAG_" + node, {0, 1}))
    required = public["question"]["required"]
    sets = [known.get(key, {0, 1}) for key in required]
    if any(s == {0} for s in sets):
        return {0}
    if all(s == {1} for s in sets):
        return {1}
    return {0, 1}


def query_oracle(group: dict, task: dict, query_id: str) -> dict:
    public = group["public"]
    found = [q for q in public["queries"] if q["id"] == query_id]
    if len(found) != 1:
        raise ValueError("query ID not in menu")
    q = found[0]
    world = task["sealed"]["values"]
    if q["kind"] == "FETCH":
        return {"query_id": query_id, "kind": "FETCH", "key": q["key"], "value": world[q["key"]]}
    record = next(r for r in public["records"] if r["id"] == q["record_id"])
    return {"query_id": query_id, "kind": "VERIFY", "record_id": record["id"],
            "valid": world[record["key"]] == record["value"]}


def critical_queries(public: dict) -> list[str]:
    if len(possibilities(public)) == 1:
        return []
    candidates: list[str] = []
    for q in public["queries"]:
        if q["kind"] == "FETCH":
            variants = [{"kind": "FETCH", "key": q["key"], "value": v} for v in (0, 1)]
        else:
            variants = [{"kind": "VERIFY", "record_id": q["record_id"], "valid": v}
                        for v in (False, True)]
        if all(len(possibilities(public, e)) == 1 for e in variants):
            candidates.append(q["id"])
    return candidates


def audit(dataset: dict) -> dict:
    assert dataset["protocol"] == PROTOCOL
    n = dataset["n_per_cell"]
    groups = dataset["groups"]
    ids = set()
    counts: Counter = Counter()
    group_counts: Counter = Counter()
    answers: Counter = Counter()
    for group in groups:
        gid = group["group_id"]
        assert gid not in ids
        ids.add(gid)
        pub = group["public"]
        assert set(pub) == {"domain", "question", "rules", "records", "queries"}
        assert Counter(q["kind"] for q in pub["queries"]) == {"FETCH": 3, "VERIFY": 3}
        assert len({q["id"] for q in pub["queries"]}) == 6
        assert len({r["id"] for r in pub["records"]}) == len(pub["records"])
        assert all("sealed" not in q for q in pub["queries"])
        assert all(q["key"] in group["tasks"][0]["sealed"]["values"] for q in pub["queries"] if q["kind"] == "FETCH")
        assert all(q["record_id"] in {r["id"] for r in pub["records"]}
                   for q in pub["queries"] if q["kind"] == "VERIFY")
        conditions = {t["sealed"]["condition"] for t in group["tasks"]}
        assert len(conditions) == 1
        condition = conditions.pop()
        group_counts[(pub["domain"], condition)] += 1
        assert len(group["tasks"]) == (2 if condition in "BC" else 1)
        before = possibilities(pub)
        crit = critical_queries(pub)
        assert before == ({0, 1} if condition in "BC" else {group["tasks"][0]["sealed"]["truth"]}), (gid, before)
        assert len(crit) == (1 if condition in "BC" else 0), (gid, crit)
        if len(group["tasks"]) == 2:
            assert {t["sealed"]["truth"] for t in group["tasks"]} == {0, 1}
            assert group["tasks"][0]["sealed"]["target_key"] == group["tasks"][1]["sealed"]["target_key"]
        for t in group["tasks"]:
            assert t["id"] not in ids
            ids.add(t["id"])
            s = t["sealed"]
            assert s["values"][s["target_key"]] == s["truth"]
            counts[(pub["domain"], condition)] += 1
            answers[(pub["domain"], condition, s["truth"])] += 1
            for q in pub["queries"]:
                evidence = query_oracle(group, t, q["id"])
                after = possibilities(pub, evidence)
                assert after == ({s["truth"]} if q["id"] in crit else before), (gid, q, after)
    for domain in DOMAINS:
        for c in CONDITIONS:
            assert counts[(domain, c)] == n, (domain, c, counts)
            assert answers[(domain, c, 0)] == n // 2 and answers[(domain, c, 1)] == n // 2
            assert group_counts[(domain, c)] == (n // 2 if c in "BC" else n)
    return {"audit": "PASS", "tasks": 8 * n, "independent_groups": len(groups),
            "tasks_per_domain_condition": n, "counterfactual_twins": 2 * n,
            "positive_answers_per_cell": n // 2, "negative_answers_per_cell": n // 2,
            "sealed_answer_excluded_from_model_payload": True, "one_diagnostic_query_in_BC": True,
            "no_diagnostic_query_in_AD": True}


def protocol_fingerprint() -> str:
    return digest(canonical({"version": PROTOCOL, "prompts": [INITIAL_PROMPT, FINAL_PROMPT, DIRECT_PROMPT],
                             "schemas": [SCHEMA_INITIAL, SCHEMA_FINAL], "cost": QUERY_COST}).encode())


def validate_response(resp: dict, stage: str) -> dict:
    if not isinstance(resp, dict) or type(resp.get("answer")) is not int or resp["answer"] not in (0, 1):
        raise ValueError("invalid bit in model response")
    if not isinstance(resp.get("confidence"), (float, int)) or type(resp["confidence"]) is bool or not 0 <= resp["confidence"] <= 1:
        raise ValueError("invalid confidence in model response")
    if stage == "initial":
        if resp.get("action") not in ("answer", "inspect", "abstain") or not isinstance(resp.get("query_id"), str):
            raise ValueError("invalid initial action or query ID")
        if resp["action"] != "inspect" and resp["query_id"]:
            raise ValueError("nonempty query ID without inspect action")
    return resp


class MockOptimal:
    """Pure reference for CI/tests. Not a live model or emergence evidence."""
    model = "mock-optimal-NOT-OPENAI"

    def __init__(self, max_calls: int = 1000):
        self.api_calls = self.input_tokens = self.output_tokens = 0
        self.max_calls = max_calls

    def decide(self, public: dict) -> dict:
        poss = possibilities(public)
        crit = critical_queries(public)
        return {"action": "inspect" if crit else "answer", "answer": min(poss),
                "query_id": crit[0] if crit else "", "confidence": 1 if len(poss) == 1 else .5}

    def direct(self, public: dict) -> dict:
        poss = possibilities(public)
        return {"answer": min(poss), "confidence": 1 if len(poss) == 1 else .5}

    def finalize(self, public: dict, evidence: dict) -> dict:
        poss = possibilities(public, evidence)
        return {"answer": min(poss), "confidence": 1 if len(poss) == 1 else .5}


class OpenAIAPI:
    def __init__(self, model: str, max_calls: int):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set; put it in untracked .env")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Run: pip install -e '.[api]' ") from exc
        self.client = OpenAI(timeout=60.0, max_retries=2)
        self.model = model
        self.max_calls = max_calls
        self.api_calls = self.input_tokens = self.output_tokens = 0

    def _call(self, stage: str, public: dict, evidence: dict | None = None) -> dict:
        if self.api_calls >= self.max_calls:
            raise RuntimeError("API call cap reached; saved completed groups; rerun with --resume")
        schema = SCHEMA_INITIAL if stage == "initial" else SCHEMA_FINAL
        prompt = INITIAL_PROMPT if stage == "initial" else DIRECT_PROMPT if stage == "direct" else FINAL_PROMPT
        payload = {"task": public}
        if evidence is not None:
            payload["inspection"] = evidence
        response = self.client.responses.create(
            model=self.model, instructions=prompt,
            input=canonical(payload),
            text={"format": {"type": "json_schema", "name": "cmt3_" + stage,
                             "strict": True, "schema": schema}},
            store=False,
            max_output_tokens=320,
        )
        self.api_calls += 1
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        if not response.output_text:
            raise RuntimeError("Empty output from model; no results for current group were stored")
        return validate_response(json.loads(response.output_text), stage)

    def decide(self, public: dict) -> dict:
        return self._call("initial", public)

    def direct(self, public: dict) -> dict:
        return self._call("direct", public)

    def finalize(self, public: dict, evidence: dict) -> dict:
        return self._call("final", public, evidence)


def _safe_final(backend, public: dict, evidence: dict) -> dict:
    return backend.finalize(public, evidence)


def _row(group: dict, task: dict, policy: str, initial: dict, answer: int,
         query_id: str, critical: set[str], *, invalid: bool = False,
         final_confidence: float | None = None) -> dict:
    truth = task["sealed"]["truth"]
    used = bool(query_id)
    return {
        "group_id": group["group_id"], "case_id": task["id"],
        "domain": group["public"]["domain"], "condition": task["sealed"]["condition"],
        "policy": policy, "initial_answer": initial.get("answer"),
        "initial_confidence": initial.get("confidence"),
        "initial_correct": int(initial.get("answer") == truth) if initial.get("answer") in (0, 1) else None,
        "initial_action": initial.get("action", "direct"),
        "queried": used, "query_id": query_id or None,
        "critical_hit": query_id in critical if query_id else False,
        "invalid_action": invalid,
        "final_answer": answer, "final_confidence": final_confidence,
        "correct": int(answer == truth),
        "utility": round(int(answer == truth) - QUERY_COST * int(used), 5),
    }


def process_group(group: dict, backend, seed: int) -> list[dict]:
    public = group["public"]
    crit = set(critical_queries(public))
    initial = backend.decide(public)
    direct = backend.direct(public)  # independent no-lookup control, not a reused provisional
    menu = {q["id"] for q in public["queries"]}
    rng = random.Random(seed ^ int(group["group_id"].split("-")[1], 16))
    # Random control: independently choose whether to query, then which one.
    random_query = rng.choice(public["queries"])["id"] if rng.randrange(2) else ""
    action = initial["action"]
    attempted = initial["query_id"] if action == "inspect" else ""
    invalid = action == "inspect" and attempted not in menu
    # Unknown query is a protocol failure, never replaced by a different query.
    chosen = attempted if not invalid else ""
    records = []
    for task in group["tasks"]:
        if chosen:
            evidence = query_oracle(group, task, chosen)
            final = _safe_final(backend, public, evidence)
            outcome, final_conf = final["answer"], final["confidence"]
        elif action == "abstain" or invalid:
            outcome, final_conf = -1, None
        else:
            outcome, final_conf = initial["answer"], initial["confidence"]
        records.append(_row(group, task, "adaptive", initial, outcome, chosen, crit,
                            invalid=invalid, final_confidence=final_conf))
        records.append(_row(group, task, "direct", direct, direct["answer"], "", crit,
                            final_confidence=direct["confidence"]))
        if random_query:
            random_evidence = query_oracle(group, task, random_query)
            random_final = _safe_final(backend, public, random_evidence)
        else:
            random_final = {"answer": initial["answer"], "confidence": initial["confidence"]}
        records.append(_row(group, task, "random", initial, random_final["answer"], random_query,
                            crit, final_confidence=random_final["confidence"]))
        if crit:
            query = sorted(crit)[0]
            result = query_oracle(group, task, query)
            poss = possibilities(public, result)
            constraint_answer = next(iter(poss)) if len(poss) == 1 else -1
        else:
            query = ""
            poss = possibilities(public)
            constraint_answer = next(iter(poss)) if len(poss) == 1 else -1
        records.append(_row(group, task, "constraint", {}, constraint_answer, query, crit))
    return records


def run_study(data_path: Path, output: Path, *, backend: Any, resume: bool = False,
              seed: int = 20261009) -> dict:
    ds = read_json(data_path)
    checks = audit(ds)
    meta_path = Path(str(output) + ".meta.json")
    metadata = {"protocol": PROTOCOL, "protocol_fingerprint": protocol_fingerprint(),
                "source_sha256": digest(Path(__file__).read_bytes()),
                "dataset_sha256": digest(data_path.read_bytes()), "model": backend.model,
                "query_cost": QUERY_COST, "seed": seed,
                "independent_units": len(ds["groups"]), "number_tasks": checks["tasks"],
                "policies": list(POLICIES), "random_queries_are_per_group": True,
                "direct_control_has_independent_model_call": True,
                "is_live_openai": isinstance(backend, OpenAIAPI)}
    if output.exists():
        if not resume:
            raise FileExistsError(f"{output} exists; pass --resume, or choose a new path")
        if not meta_path.exists() or read_json(meta_path) != metadata:
            raise ValueError("Resume refused: model, dataset or protocol metadata changed")
        prev = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        if resume and meta_path.exists():
            raise ValueError("Result metadata exists without results file")
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(meta_path, metadata)
        prev = []
    grouped = defaultdict(list)
    for row in prev:
        grouped[row["group_id"]].append(row)
    groups = {g["group_id"]: g for g in ds["groups"]}
    for gid, rows in grouped.items():
        if gid not in groups or len(rows) != len(groups[gid]["tasks"]) * len(POLICIES):
            raise ValueError("Incomplete/unknown checkpoint group: " + gid)
        assert len({(r["case_id"], r["policy"]) for r in rows}) == len(rows)
    started_groups = len(grouped)
    for group in ds["groups"]:
        if group["group_id"] in grouped:
            continue
        rows = process_group(group, backend, seed)
        # Complete group writes only: preserves sealed twin pairing across --resume.
        with output.open("a", encoding="utf-8") as fp:
            for row in rows:
                fp.write(canonical(row) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        grouped[group["group_id"]] = rows
    return {"completed_groups": len(grouped), "completed_tasks": sum(len(g["tasks"]) for g in ds["groups"]
               if g["group_id"] in grouped), "resumed_groups": started_groups,
            "api_calls_this_invocation": backend.api_calls,
            "input_tokens_this_invocation": backend.input_tokens,
            "output_tokens_this_invocation": backend.output_tokens,
            "result_path": str(output), "backend": backend.model}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def wilson(success: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = success / n
    denominator = 1 + z * z / n
    center = (p + z*z/(2*n)) / denominator
    half = (z/denominator) * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
    return (center-half, center+half)


def binom_tail(k: int, n: int, p: float) -> float:
    if not n:
        return 1.0
    return sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1))


def auc(samples: list[tuple[float, int]]) -> float | None:
    """Correct type-2 AUROC, including 0.5 credit for exact confidence ties."""
    n_pos = sum(int(label == 1) for _, label in samples)
    n_neg = len(samples) - n_pos
    if not n_pos or not n_neg:
        return None
    groups: dict[float, list[int]] = defaultdict(list)
    for conf, label in samples:
        groups[conf].append(label)
    negatives_below = 0
    winning_pairs = 0.0
    for score in sorted(groups):
        values = groups[score]
        positive_here = sum(values)
        negative_here = len(values) - positive_here
        winning_pairs += positive_here * (negatives_below + 0.5 * negative_here)
        negatives_below += negative_here
    return winning_pairs / (n_pos * n_neg)


def cluster_auc_ci(rows: list[dict], *, trials: int = 1800) -> tuple[float, float] | None:
    """Group-level bootstrap for self-evaluated initial confidence accuracy."""
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        if r.get("initial_confidence") is not None and r.get("initial_correct") is not None:
            groups[r["group_id"]].append((r["initial_confidence"], r["initial_correct"]))
    clusters = list(groups.values())
    if not clusters:
        return None
    rng = random.Random(20261010)
    reps = []
    for _ in range(trials):
        sample = [pair for _ in clusters for pair in clusters[rng.randrange(len(clusters))]]
        score = auc(sample)
        if score is not None:
            reps.append(score)
    if not reps:
        return None
    reps.sort()
    return (reps[int(len(reps)*.025)], reps[min(len(reps)-1,int(len(reps)*.975))])


def cluster_utility_test(rows: list[dict], *, trials: int = 5000) -> dict:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["group_id"]][row["policy"]].append(row["utility"])
    pairs = []
    for gid in sorted(grouped):
        obj = grouped[gid]
        if len(obj["adaptive"]) != len(obj["direct"]):
            continue
        pairs.append((len(obj["adaptive"]), sum(obj["adaptive"]) - sum(obj["direct"])))
    size = sum(p[0] for p in pairs)
    effect = sum(p[1] for p in pairs) / size
    rng = random.Random(20261009)
    boots = []
    for _ in range(trials):
        resample = [pairs[rng.randrange(len(pairs))] for __ in pairs]
        boots.append(sum(x[1] for x in resample) / sum(x[0] for x in resample))
    boots.sort()
    exceed = 0
    for _ in range(trials):
        null_value = sum(diff * (1 if rng.getrandbits(1) else -1) for _, diff in pairs) / size
        exceed += abs(null_value) >= abs(effect) - 1e-12
    return {"effect": effect, "ci95": [boots[int(trials*.025)], boots[min(trials-1,int(trials*.975))]],
            "permutation_p_two_sided": (exceed+1)/(trials+1),
            "independent_groups": len(pairs), "paired_task_rows": size,
            "method": "cluster bootstrap and sign-flip at independent-group level"}


def final_report(data_path: Path, result_path: Path, report_dir: Path) -> dict:
    ds = read_json(data_path)
    audit(ds)
    meta = read_json(Path(str(result_path) + ".meta.json"))
    if digest(data_path.read_bytes()) != meta["dataset_sha256"]:
        raise ValueError("Dataset changed after run; refusing to report")
    if meta["protocol_fingerprint"] != protocol_fingerprint() or meta.get("source_sha256") != digest(Path(__file__).read_bytes()):
        raise ValueError("Protocol or source code changed after run; refusing to report")
    rows = [json.loads(s) for s in result_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    groups = {g["group_id"]: g for g in ds["groups"]}
    gfound: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        gfound[r["group_id"]].append(r)
    if set(gfound) != set(groups):
        raise ValueError("Not all independent groups have completed; do not report partial runs as final")
    for gid, rs in gfound.items():
        if len(rs) != len(groups[gid]["tasks"]) * len(POLICIES) or len({(r["case_id"],r["policy"]) for r in rs}) != len(rs):
            raise ValueError("Duplicate or incomplete group: " + gid)
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    policy = {}
    for name in POLICIES:
        rr = by_policy[name]
        q = [r for r in rr if r["queried"]]
        policy[name] = {"N": len(rr), "accuracy": mean([r["correct"] for r in rr]),
                        "utility": mean([r["utility"] for r in rr]),
                        "query_rate": mean([int(r["queried"]) for r in rr]),
                        "critical_precision": mean([int(r["critical_hit"]) for r in q])}
    rr = by_policy["adaptive"]
    by_cell = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            rs = [r for r in rr if r["domain"] == domain and r["condition"] == condition]
            by_cell[f"{domain}/{condition}"] = {"N": len(rs), "accuracy": mean([r["correct"] for r in rs]),
                "query_rate": mean([int(r["queried"]) for r in rs]),
                "critical_hits": sum(r["critical_hit"] for r in rs)}
    adequate = [r for r in rr if r["condition"] in "AD"]
    incomplete = [r for r in rr if r["condition"] in "BC"]
    direct_adequate = [r for r in by_policy["direct"] if r["condition"] in "AD"]
    select_sens = mean([int(r["queried"]) for r in incomplete])
    select_spec = mean([int(not r["queried"]) for r in adequate])
    active = [r for r in incomplete if r["queried"]]
    hits = [r for r in active if r["critical_hit"]]
    useful_count = len(hits)
    # B/C twins share initial decisions. Inferential sample is groups, not cases.
    n_query_groups = len({r["group_id"] for r in active})
    n_hit_groups = len({r["group_id"] for r in hits})
    precision_p = binom_tail(n_hit_groups, n_query_groups, 1/6)
    correctness_given = mean([r["correct"] for r in hits])
    effect = cluster_utility_test(rows)
    calibration = [(r["initial_confidence"], r["initial_correct"]) for r in rr
                   if r["initial_confidence"] is not None and r["initial_correct"] is not None]
    auroc = auc(calibration)
    auroc_ci = cluster_auc_ci(rr)
    competence = mean([r["correct"] for r in direct_adequate])
    competence_per_domain = {domain: mean([r["correct"] for r in direct_adequate if r["domain"] == domain])
                             for domain in DOMAINS}
    # Separate the ability to solve from epistemic control; not a test of consciousness.
    competence_gate = competence >= .85 and all(x >= .80 for x in competence_per_domain.values())
    by_domain_delta = {}
    for domain in DOMAINS:
        sel = [r for r in rows if r["domain"] == domain]
        by_domain_delta[domain] = mean([r["utility"] for r in sel if r["policy"] == "adaptive"]) - mean(
            [r["utility"] for r in sel if r["policy"] == "direct"])
    strong = (
        competence_gate and select_sens >= .75 and select_spec >= .85
        and n_query_groups >= 20 and n_hit_groups/n_query_groups >= .70
        and precision_p < .01
        and n_hit_groups >= 17 and correctness_given >= .85
        and effect["effect"] >= .08 and effect["ci95"][0] > 0
        and effect["permutation_p_two_sided"] < .01
        and all(x > 0 for x in by_domain_delta.values())
        and auroc is not None and auroc >= .65
        and auroc_ci is not None and auroc_ci[0] > .50
    )
    if not meta.get("is_live_openai", False):
        verdict = "MOCK_ONLY__NO_EMPIRICAL_CLAIM"
    elif not competence_gate:
        verdict = "BASE_REASONING_GATE_FAILED__METACOGNITION_INCONCLUSIVE"
    elif strong:
        verdict = "FUNCTIONAL_METACOGNITIVE_CONTROL_SUPPORTED_IN_THIS_BENCHMARK"
    elif effect["ci95"][1] <= 0:
        verdict = "NO_MEASURED_CONTROL_BENEFIT__NOT_A_GLOBAL_ABSENCE_CLAIM"
    else:
        verdict = "MIXED_OR_INCONCLUSIVE__CRITERIA_NOT_MET"
    metrics = {"protocol": PROTOCOL, "model": meta["model"], "N_cases": len(rr),
        "N_independent_groups": len(groups), "protocol_fingerprint": meta["protocol_fingerprint"],
        "dataset_sha256": meta["dataset_sha256"], "verdict": verdict,
        "competence_gate": competence_gate, "direct_accuracy_on_complete": competence,
        "direct_accuracy_by_domain": competence_per_domain,
        "query_sensitivity_BC": select_sens, "query_specificity_AD": select_spec,
        "critical_precision_BC": mean([int(r["critical_hit"]) for r in active]),
        "critical_hits_BC": useful_count, "queries_BC": len(active),
        "critical_hit_independent_groups_BC": n_hit_groups,
        "query_independent_groups_BC": n_query_groups,
        "critical_random_selection_p": precision_p,
        "strong_evidence_gates_passed": bool(strong),
        "final_accuracy_after_critical_query": correctness_given,
        "type2_AUROC_initial_confidence": auroc,
        "type2_AUROC_cluster_bootstrap_95ci": auroc_ci,
        "adaptive_minus_direct_utility": effect,
        "adaptive_minus_direct_by_domain": by_domain_delta,
        "policies": policy, "by_domain_condition": by_cell,
        "scientific_limitations": [
            "Only functional behavior in artificial tasks; no inference about phenomenological consciousness.",
            "A passed test does not establish spontaneous emergence; instructions and inspection interface are supplied.",
            "A failed test does not establish a general absence of metacognition in LLMs.",
            "A two-domain synthetic benchmark is not a broad sample of real-world cognition.",
            "Twins share identical public prompts; uncertainty and p-values must cluster by group.",
            "Constraint comparator is a purpose-built expert upper bound, not a matched LLM architecture.",
            "Hypothesis gates and thresholds were frozen prior to calling the API; do not tune on the final set.",
        ]}
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_dir / "METRICS_PRIVATE.json", metrics)
    fmt = lambda x: "NA" if x is None else f"{x:.3f}"
    p_fmt = lambda x: "NA" if x is None else "<0.001" if x < .001 else f"{x:.3f}"
    ci_fmt = lambda ci: "NA" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
    tab = "| Policy | Accuracy | Utility | Query rate | Critical precision |\n|---|---:|---:|---:|---:|\n"
    for name in POLICIES:
        a = policy[name]
        tab += f"| {name} | {fmt(a['accuracy'])} | {fmt(a['utility'])} | {fmt(a['query_rate'])} | {fmt(a['critical_precision'])} |\n"
    markdown = f"""# CMT-3 Final — preregistered test\n\n**Verdict:** `{verdict}`\n\n**Model:** `{meta['model']}` | **N:** {len(rr)} tasks / {len(groups)} independent groups\n\n**Protocol SHA:** `{meta['protocol_fingerprint']}`\n\n## Paired policy results\n\n{tab}\n## Epistemic control\n\n- Complete-information direct competence: {fmt(competence)}; gate passed: **{competence_gate}**.\n- Sensitivity (inspect in B/C): {fmt(select_sens)}.\n- Specificity (avoid inspection in A/D): {fmt(select_spec)}.\n- Critical query precision in B/C: {useful_count}/{len(active)} cases = {fmt(metrics['critical_precision_BC'])}.
- **Independent B/C query groups:** {n_hit_groups}/{n_query_groups} informative decisions (twins grouped).\n- Group-binomial tail vs random 1/6: {p_fmt(metrics['critical_random_selection_p'])}.\n- Correct final after critical query: {fmt(correctness_given)}.\n- Initial type-2 AUROC: {fmt(auroc)}, group-bootstrap 95% CI: {ci_fmt(auroc_ci)}.\n- Adaptive minus independently prompted direct, utility: {fmt(effect['effect'])}.\n- Cluster-bootstrap 95% CI: [{fmt(effect['ci95'][0])}, {fmt(effect['ci95'][1])}].\n- Cluster permutation p (two-sided): {p_fmt(effect['permutation_p_two_sided'])}.\n- Improvement in lookup / gate: {fmt(by_domain_delta['lookup'])} / {fmt(by_domain_delta['gate'])}.\n\n## Pre-frozen strong-evidence gates\n\n1. Direct accuracy on complete A/D >= .85 overall and >= .80 in each domain.\n2. Selective queries: sensitivity >= .75, specificity >= .85.\n3. >=20 independent querying GROUPS in B/C; >=.70 precision; group-binomial p < .01 vs 1/6.\n4. >=17 informative independent GROUPS; final correctness >=.85 after informative queries.\n5. Adaptive-direct utility difference >=.08; cluster bootstrap lower 95% > 0; permutation p < .01.\n6. Positive utility effect in BOTH domains.\n7. Initial confidence AUROC >= .65 and cluster-bootstrap 95% lower bound > .50.\n\nOnly if ALL gates pass is **functional epistemic monitoring and control in this benchmark** supported.\nOther outcomes must not be described as proof of general absence or consciousness.\n\n## Interpretation limits\n\n""" + "\n".join("- " + s for s in metrics["scientific_limitations"]) + "\n"
    (report_dir / "CONCLUSION.md").write_text(markdown, encoding="utf-8")
    return metrics


def cli(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="CMT-3 preregistered final metacognition test; standalone")
    sub = p.add_subparsers(dest="cmd", required=True)
    common_data = Path("data/private/cmt3_final.json")
    gen = sub.add_parser("generate", help="Create NEW sealed final test data, do not commit")
    gen.add_argument("--n-per-cell", type=int, default=24)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--data", type=Path, default=common_data)
    aud = sub.add_parser("audit", help="Validate frozen data and anti-leak invariants")
    aud.add_argument("--data", type=Path, default=common_data)
    run = sub.add_parser("run", help="Execute the final test with API, or local mock")
    run.add_argument("--data", type=Path, default=common_data)
    run.add_argument("--out", type=Path, default=Path("runs/cmt3_final.jsonl"))
    run.add_argument("--backend", choices=["mock", "openai"], default="mock")
    run.add_argument("--model", default="gpt-4.1-mini")
    run.add_argument("--max-api-calls", type=int, default=800)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-paid-api", action="store_true")
    rpt = sub.add_parser("report", help="Report ONLY when all groups complete")
    rpt.add_argument("--data", type=Path, default=common_data)
    rpt.add_argument("--results", type=Path, default=Path("runs/cmt3_final.jsonl"))
    rpt.add_argument("--output-dir", type=Path, default=Path("reports/cmt3_final"))
    sub.add_parser("smoke", help="Offline smoke (16 cases) with explicitly labelled ideal mock")
    args = p.parse_args(argv)
    if args.cmd == "generate":
        if args.data.exists():
            raise FileExistsError(f"Dataset exists: {args.data}. Don't overwrite a frozen dataset")
        data = generate(args.n_per_cell, args.seed)
        info = audit(data)
        atomic_json(args.data, data)
        print(json.dumps({"data": str(args.data), "sha256": digest(args.data.read_bytes()),
                          "protocol_fingerprint": protocol_fingerprint(), **info}, indent=2))
    elif args.cmd == "audit":
        print(json.dumps(audit(read_json(args.data)), indent=2))
    elif args.cmd == "run":
        if args.backend == "openai" and not args.allow_paid_api:
            p.error("Explicitly acknowledge API charges with --allow-paid-api")
        if args.max_api_calls < 1:
            p.error("--max-api-calls must be positive")
        backend = MockOptimal(args.max_api_calls) if args.backend == "mock" else OpenAIAPI(args.model, args.max_api_calls)
        print(json.dumps(run_study(args.data, args.out, backend=backend, resume=args.resume), indent=2))
    elif args.cmd == "report":
        result = final_report(args.data, args.results, args.output_dir)
        print(json.dumps({"verdict": result["verdict"], "report": str(args.output_dir / "CONCLUSION.md"),
                          "metrics": str(args.output_dir / "METRICS_PRIVATE.json"),
                          "N_cases": result["N_cases"], "N_independent_groups": result["N_independent_groups"]}, indent=2))
    else:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="cmt3_smoke_") as temp:
            d = Path(temp)
            dataset = d / "private.json"
            atomic_json(dataset, generate(4, 123))
            summary = run_study(dataset, d / "mock.jsonl", backend=MockOptimal())
            report = final_report(dataset, d / "mock.jsonl", d / "report")
            print(json.dumps({"OFFLINE_ONLY_NOT_LLM": True, "audit": audit(read_json(dataset))["audit"],
                              "tasks": summary["completed_tasks"], "verdict": report["verdict"],
                              "adaptive_accuracy": report["policies"]["adaptive"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    cli()
