import json
from collections import Counter

import pytest

from llmcognition.data import generate, public_projection, query_oracle
from llmcognition.parity import best_query, possibilities, query_diagnostic
from llmcognition.validation import audit_tasks


def test_generate_hundred_sealed_audited():
    tasks, seed = generate(100, 20261008)
    result = audit_tasks(tasks)
    assert seed == 20261008
    assert result["checks"] == "PASSED"
    assert result["conditions"] == dict.fromkeys("ABCD", 25)


def test_deterministic_seed_and_unique_ids():
    a, _ = generate(100, 99)
    b, _ = generate(100, 99)
    assert a == b
    assert len({t["id"] for t in a}) == 100
    assert len({r["id"] for t in a for r in t["public"]["records"]}) > 5


def test_invalid_task_counts():
    with pytest.raises(ValueError):
        generate(7, 42)
    with pytest.raises(ValueError):
        generate(0, 42)


def test_public_projection_cannot_leak_sealed_truth():
    cases, _ = generate(16, 2)
    for task in cases:
        shown = public_projection(task)
        assert set(shown) == {"target", "records", "queries"}
        raw = json.dumps(shown)
        assert '"truth"' not in raw
        assert '"world"' not in raw
        assert '"condition"' not in raw
        assert '"critical_query_ids"' not in raw
        assert "sealed" not in shown


def test_ground_truth_not_derived_from_query_id():
    tasks, _ = generate(100, 12)
    # IDs remain opaque random hex strings, with no arm markers or sequential IDs.
    assert all(task["id"].startswith("T-") for task in tasks)
    assert set(c for c in Counter(task["sealed"]["condition"] for task in tasks)) == set("ABCD")


def test_exactly_one_informative_query_for_missing_and_conflicting():
    tasks, _ = generate(100, 72)
    for task in tasks:
        public = task["public"]
        truth = task["sealed"]["truth"]
        need = task["sealed"]["condition"] in ("B", "C")
        assert len(task["sealed"]["critical_query_ids"]) == int(need)
        if need:
            assert possibilities(public) == {0, 1}
            chosen = best_query(public)
            assert chosen and query_diagnostic(public, chosen)[0] == 1
            ans = query_oracle(task, chosen["id"])
            observed = ans.get("value") if chosen["kind"] == "REL" else ans.get("valid")
            assert possibilities(public, {"query": chosen, "result": observed}) == {truth}
        else:
            assert possibilities(public) == {truth}
            assert best_query(public) is None


def test_invalid_query_does_not_reveal_anything():
    task = generate(4, 10)[0][0]
    assert query_oracle(task, "Q-NOT-IN-CATALOG") == {"error": "UNKNOWN_QUERY_ID"}
