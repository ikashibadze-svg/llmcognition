"""Small symbolic solver used for *validation* and an explicitly labeled baseline.

There is no call to this module from the OpenAI decision interface. The model
sees only the public task projection, never this solver's private judgments.
"""
from __future__ import annotations

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
