"""Pre-run anti-cheating and benchmark integrity checks."""
from collections import Counter

from .data import public_projection, query_oracle
from .parity import possibilities, query_diagnostic


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
