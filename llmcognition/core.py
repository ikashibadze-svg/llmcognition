from __future__ import annotations


# === Consolidated from llmcognition/parity.py (v0.2 experiment; behavior preserved) ===

"""Small symbolic solver used for *validation* and an explicitly labeled baseline.

There is no call to this module from the OpenAI decision interface. The model
sees only the public task projection, never this solver's private judgments.
"""
from collections import defaultdict


class ParityDSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.xor_parent: dict[str, int] = {}

    def find(self, x: str) -> tuple[str, int]:
        if x not in self.parent:
            self.parent[x] = x
            self.xor_parent[x] = 0
        if self.parent[x] == x:
            return x, 0
        p = self.parent[x]
        root, value = self.find(p)
        self.xor_parent[x] ^= value
        self.parent[x] = root
        return root, self.xor_parent[x]

    def add(self, a: str, b: str, value: int) -> bool:
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == value
        self.parent[ra] = rb
        self.xor_parent[ra] = pa ^ pb ^ value
        return True

    def query(self, a: str, b: str) -> int | None:
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        return pa ^ pb if ra == rb else None


def _conflicting_pairs(records: list[dict]) -> list[tuple[str, str]]:
    """Known benchmark rule: conflicting duplicate records have uncertain status."""
    by_pair: dict[frozenset[str], list[dict]] = defaultdict(list)
    for record in records:
        by_pair[frozenset((record["u"], record["v"]))].append(record)
    result: list[tuple[str, str]] = []
    for group in by_pair.values():
        for i, first in enumerate(group):
            for other in group[i + 1 :]:
                if first["value"] != other["value"]:
                    result.append((first["id"], other["id"]))
    if len(result) > 1:
        raise ValueError("Benchmark supports at most one conflicting duplicate pair")
    return result


def possibilities(public: dict, evidence: dict | None = None) -> set[int]:
    """Answers consistent with public records + *observed* oracle evidence.

    This never reads a private world or ground truth. Contradictory duplicates
    are modeled as two possible repairs. Evidence filters the repairs.
    """
    evidence = evidence or {}
    records = [dict(rec) for rec in public["records"]]
    conflict = _conflicting_pairs(records)
    dropped_options: list[set[str]] = [set()]
    if conflict:
        a, b = conflict[0]
        dropped_options = [{a}, {b}]
    output: set[int] = set()
    for dropped in dropped_options:
        result = evidence.get("result")
        query = evidence.get("query")
        if query and query["kind"] == "VERIFY":
            rec = query["record_id"]
            # Trust a verified record; reject a proven-inaccurate one.
            if (rec in dropped) == bool(result):
                continue
        dsu = ParityDSU()
        consistent = True
        for record in records:
            if record["id"] not in dropped:
                if not dsu.add(record["u"], record["v"], int(record["value"])):
                    consistent = False
                    break
        if not consistent:
            continue
        if query and query["kind"] == "REL":
            if not dsu.add(query["u"], query["v"], int(result)):
                continue
        target = dsu.query(*public["target"])
        if target is None:
            output.update((0, 1))
        else:
            output.add(target)
    return output


def query_diagnostic(public: dict, query: dict) -> tuple[int, float]:
    """Guaranteed reduction in answer ambiguity across feasible query replies.

    Used only by the symbolic comparator, never as secret guidance to the LLM.
    """
    before = possibilities(public)
    replies = [0, 1] if query["kind"] == "REL" else [True, False]
    sizes = [
        len(after)
        for reply in replies
        if (after := possibilities(public, {"query": query, "result": reply}))
    ]
    if not sizes:
        return (0, 0.0)
    return (len(before) - max(sizes), len(before) - sum(sizes) / len(sizes))


def best_query(public: dict) -> dict | None:
    options = [(query_diagnostic(public, q), q) for q in public["queries"]]
    candidates = [(score, q) for score, q in options if score[0] > 0]
    return max(candidates, key=lambda x: (x[0][0], x[0][1], x[1]["id"]))[1] if candidates else None


# === Consolidated from llmcognition/data.py (v0.2 experiment; behavior preserved) ===

"""Sealed XOR relational tasks: A sufficient, B missing, C conflicting, D hard.

The model never sees the private keys in a case. Truth and query results are
held locally and published only after scoring. Query IDs and task IDs carry
no condition labels; all four conditions have the same numbers of query kinds.
"""
import hashlib
import json
import random
import secrets
from pathlib import Path
from typing import Any



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


# === Consolidated from llmcognition/validation.py (v0.2 experiment; behavior preserved) ===

"""Pre-run anti-cheating and benchmark integrity checks."""
from collections import Counter



def audit_tasks(tasks: list[dict]) -> dict:
    ids: set[str] = set()
    counts: Counter = Counter()
    bit_counts: Counter = Counter()
    for task in tasks:
        case_id = task["id"]
        assert case_id not in ids, f"Duplicate case ID: {case_id}"
        ids.add(case_id)
        pub = public_projection(task)
        sealed = task["sealed"]
        condition = sealed["condition"]
        assert condition in ("A", "B", "C", "D")
        assert task["public"] == pub, "Unrecognized public fields: potential leak"
        assert not any(k in pub for k in ("truth", "condition", "sealed", "world"))
        assert len(pub["queries"]) == 6
        assert all((q["source"] == "ARCHIVE") == (q["kind"] == "REL")
                   for q in pub["queries"])
        assert Counter(q["kind"] for q in pub["queries"]) == {"REL": 3, "VERIFY": 3}
        assert len({q["id"] for q in pub["queries"]}) == 6
        assert len({r["id"] for r in pub["records"]}) == len(pub["records"])
        assert all(q["record_id"] in {r["id"] for r in pub["records"]}
                   for q in pub["queries"] if q["kind"] == "VERIFY")
        assert all(r["value"] in (0, 1) for r in pub["records"])
        assert sealed["world"][pub["target"][0]] ^ sealed["world"][pub["target"][1]] == sealed["truth"]
        initial = possibilities(pub)
        expected = {sealed["truth"]} if condition in ("A", "D") else {0, 1}
        assert initial == expected, f"Bad initial identifiability: {case_id}"
        crit = [q["id"] for q in pub["queries"] if query_diagnostic(pub, q)[0] > 0]
        assert crit == sealed["critical_query_ids"] or set(crit) == set(sealed["critical_query_ids"])
        assert len(crit) == (1 if condition in ("B", "C") else 0)
        for q in pub["queries"]:
            res = query_oracle(task, q["id"])
            val = res.get("value") if q["kind"] == "REL" else res.get("valid")
            after = possibilities(pub, {"query": q, "result": val})
            if q["id"] in crit:
                assert after == {sealed["truth"]}, (case_id, q, after)
            else:
                assert after == initial, (case_id, q, after, initial)
        counts[condition] += 1
        bit_counts[(condition, sealed["truth"])] += 1
    if len(tasks) % 4 == 0:
        assert len(set(counts.values())) == 1, f"Arms are not balanced: {counts}"
        for condition in counts:
            assert abs(bit_counts[(condition, 0)] - bit_counts[(condition, 1)]) <= 1
    return {
        "number_tasks": len(tasks),
        "conditions": dict(counts),
        "answers_by_condition": {c: {"0": bit_counts[(c, 0)], "1": bit_counts[(c, 1)]}
                                 for c in sorted(counts)},
        "checks": "PASSED",
        "blind_projection": "PASSED",
        "diagnostic_queries": "EXACTLY_ONE_IN_B_AND_C",
        "noncritical_queries": "NO_INFORMATION_GAIN",
    }
