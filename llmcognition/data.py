"""Sealed XOR relational tasks: A sufficient, B missing, C conflicting, D hard.

The model never sees the private keys in a case. Truth and query results are
held locally and published only after scoring. Query IDs and task IDs carry
no condition labels; all four conditions have the same numbers of query kinds.
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
from pathlib import Path
from typing import Any

from .parity import possibilities, query_diagnostic


CONDITIONS = ("A", "B", "C", "D")


def _token(rng: random.Random, prefix: str, used: set[str], length: int = 7) -> str:
    while True:
        value = f"{prefix}-{rng.getrandbits(4*length):0{length}x}".upper()
        if value not in used:
            used.add(value)
            return value


def _task(rng: random.Random, condition: str, target_value: int, case_id: str) -> dict:
    used: set[str] = set()
    nodes = [_token(rng, "N", used, 5) for _ in range(18)]
    world = {n: rng.randrange(2) for n in nodes}
    # Balance hidden answer bits independently of the case order and task label.
    world[nodes[1]] = world[nodes[0]] ^ target_value
    path_length = {"A": 3, "B": 4, "C": 3, "D": 8}[condition]
    path = [nodes[0], *nodes[2 : path_length + 1], nodes[1]]
    noise = nodes[path_length + 1 :]
    used_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    invalid: list[str] = []
    gap = rng.randrange(1, path_length - 1) if condition == "B" else None
    contested = rng.randrange(path_length) if condition == "C" else None

    def record(u: str, v: str, value: int) -> dict:
        item = {"id": _token(rng, "R", used_ids), "u": u, "v": v, "value": value}
        records.append(item)
        return item

    conflict_ids: tuple[str, str] | None = None
    for i in range(path_length):
        u, v = path[i], path[i + 1]
        if i == gap:
            continue
        actual = world[u] ^ world[v]
        correct = record(u, v, actual)
        if i == contested:
            false = record(u, v, actual ^ 1)
            invalid.append(false["id"])
            conflict_ids = (correct["id"], false["id"])

    # Decoy records live entirely outside the target's connected path.
    # Random count overlaps across A/B/C; long-chain D is *harder*, not missing.
    for i in range(rng.randint(3, 6)):
        u, v = noise[i], noise[i + 1]
        record(u, v, world[u] ^ world[v])

    # Three REL queries in all conditions; only in B is one a missing bridge.
    rel_pairs: list[tuple[str, str]] = []
    if gap is not None:
        rel_pairs.append((path[gap], path[gap + 1]))
    elif contested is not None:
        # Redundant relations on either side of the contested edge.
        rel_pairs.append((noise[0], noise[1]))
    else:
        rel_pairs.append((path[0], path[1]))
    rel_pairs.extend(((noise[0], noise[1]), (noise[1], noise[2])))
    if condition == "C":
        # All REL candidates avoid the uncertain bridge.
        assert contested is not None
    rel_queries = [
        {"id": _token(rng, "Q", used_ids), "kind": "REL", "source": "ARCHIVE", "u": u, "v": v}
        for u, v in rel_pairs
    ]

    # Three VERIFY queries. C has exactly one verification of contested pair.
    check_ids: list[str]
    if conflict_ids:
        check_ids = [rng.choice(conflict_ids)]
        pool = [r["id"] for r in records if r["id"] not in conflict_ids]
    else:
        pool = [r["id"] for r in records]
        check_ids = []
    check_ids += rng.sample(pool, 3 - len(check_ids))
    verify_queries = [
        {"id": _token(rng, "Q", used_ids), "kind": "VERIFY", "source": "AUDIT", "record_id": rec_id}
        for rec_id in check_ids
    ]
    queries = rel_queries + verify_queries
    rng.shuffle(records)
    rng.shuffle(queries)
    public = {"target": [nodes[0], nodes[1]], "records": records, "queries": queries}
    before = possibilities(public)
    critical = [q["id"] for q in queries if query_diagnostic(public, q)[0] > 0]
    if condition in ("A", "D"):
        assert before == {target_value} and not critical
    else:
        assert before == {0, 1} and len(critical) == 1

    return {
        "id": case_id,
        "public": public,
        "sealed": {
            "condition": condition,
            "truth": target_value,
            "world": world,
            "invalid_record_ids": invalid,
            "critical_query_ids": critical,
        },
    }


def generate(n: int = 100, seed: int | None = None) -> tuple[list[dict], int]:
    if n <= 0 or n % len(CONDITIONS):
        raise ValueError("n must be a positive multiple of 4; use 100 for the main study")
    seed = seed if seed is not None else secrets.randbits(63)
    rng = random.Random(seed)
    arms: list[tuple[str, int]] = []
    for condition in CONDITIONS:
        values = [0] * (n // 8) + [1] * (n // 8)
        if len(values) < n // 4:
            values.append(rng.randrange(2))
        rng.shuffle(values)
        arms.extend((condition, v) for v in values)
    rng.shuffle(arms)
    used_case_ids: set[str] = set()
    result = [_task(rng, c, v, _token(rng, "T", used_case_ids, 9)) for c, v in arms]
    return result, seed


def public_projection(task: dict) -> dict:
    """Allowlist: MUST be the only representation sent to OpenAI."""
    p = task["public"]
    return {
        "target": list(p["target"]),
        "records": [
            {k: r[k] for k in ("id", "u", "v", "value")} for r in p["records"]
        ],
        "queries": [
            ({k: q[k] for k in ("id", "kind", "source", "u", "v")}
             if q["kind"] == "REL" else
             {k: q[k] for k in ("id", "kind", "source", "record_id")})
            for q in p["queries"]
        ],
    }


def query_oracle(task: dict, query_id: str) -> dict:
    """Local closed-world oracle; no websites, filesystem tools, or code execution."""
    matches = [q for q in task["public"]["queries"] if q["id"] == query_id]
    if len(matches) != 1:
        return {"error": "UNKNOWN_QUERY_ID"}
    q = matches[0]
    sealed = task["sealed"]
    if q["kind"] == "REL":
        value = sealed["world"][q["u"]] ^ sealed["world"][q["v"]]
        return {"query_id": query_id, "kind": "REL", "u": q["u"], "v": q["v"], "value": value}
    return {
        "query_id": query_id,
        "kind": "VERIFY",
        "record_id": q["record_id"],
        "valid": q["record_id"] not in sealed["invalid_record_ids"],
    }


def atomic_write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temp.replace(p)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()
