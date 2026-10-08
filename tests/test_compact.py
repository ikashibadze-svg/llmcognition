

# === test_generation.py ===
import json
from collections import Counter

import pytest

from llmcognition.core import generate, public_projection, query_oracle
from llmcognition.core import best_query, possibilities, query_diagnostic
from llmcognition.core import audit_tasks


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


# === test_protocol_v02.py ===
from llmcognition.study import INITIAL_PROMPT, FINAL_PROMPT, MockBackend
from llmcognition.core import generate, atomic_write_jsonl, public_projection
from llmcognition.core import possibilities
from llmcognition.study import select_tasks, run_study
from llmcognition.study import report


def test_explicit_authenticity_contract_in_both_phases():
    assert "exactly ONE" in INITIAL_PROMPT
    assert "exactly one" in FINAL_PROMPT
    assert "If no contradictory" in INITIAL_PROMPT
    assert "PROVISIONAL" in INITIAL_PROMPT
    assert "probability" in INITIAL_PROMPT
    assert "ground truth" not in INITIAL_PROMPT
    assert "critical_query_ids" not in INITIAL_PROMPT


def test_stratified_8_has_exactly_two_per_condition():
    data, _ = generate(100, 20261009)
    sample = select_tasks(data, 8, "stratified", 123)
    assert len(sample) == 8
    assert sorted([t["sealed"]["condition"] for t in sample]) == list("AABBCCDD")
    assert [t["id"] for t in sample] == [t["id"] for t in select_tasks(data, 8, "stratified", 123)]
    assert {t["id"] for t in sample} != {t["id"] for t in select_tasks(data, 8, "stratified", 124)}


def test_validation_matches_contract():
    data, _ = generate(100, 20261009)
    for task in data:
        p = public_projection(task)
        arm = task["sealed"]["condition"]
        if arm in "AD":
            assert possibilities(p) == {task["sealed"]["truth"]}
        else:
            assert possibilities(p) == {0, 1}
        assert "sealed" not in p and "condition" not in str(p)


def test_new_run_metadata_and_score(tmp_path):
    data, _ = generate(100, 42)
    path = tmp_path / "input.jsonl"
    atomic_write_jsonl(path, data)
    results_path = tmp_path / "results.jsonl"
    rows = run_study(data, MockBackend(), output=results_path, dataset_path=path,
                     limit=8, selection="stratified")
    assert len(rows) == 8 * 6
    by_condition = {arm: [r for r in rows if r["policy"] == "adaptive" and r["condition"] == arm]
                    for arm in "ABCD"}
    assert all(len(group) == 2 for group in by_condition.values())
    assert all(r["correct"] == 1 for r in rows if r["policy"] == "adaptive")
    assert all(r["queried"] for arm in "BC" for r in by_condition[arm])
    assert all(not r["queried"] for arm in "AD" for r in by_condition[arm])
    metrics = report(rows, dataset_path=path, results_path=results_path, output_dir=tmp_path / "report")
    assert metrics["protocol"] == "CMT-2-v0.2-exploratory-post-pilot"
    assert metrics["epistemic_query_discrimination_BC_minus_AD"] == 1.0
    import json
    m = json.loads((tmp_path / "results.jsonl.meta.json").read_text())
    assert m["selection"] == "stratified"
    assert len(m["selected_ids_sha256"]) == 64
    assert len(m["prompt_sha256"]) == 64


# === test_runner.py ===
import json

import pytest

from llmcognition.study import MockBackend, OpenAIBackend, schema, validate_decision
from llmcognition.core import atomic_write_jsonl, generate, read_jsonl
from llmcognition.study import report
from llmcognition.study import POLICIES, run_study


def _setup(tmp_path, n=12):
    tasks, _ = generate(n, 42)
    datafile = tmp_path / "private.jsonl"
    atomic_write_jsonl(datafile, tasks)
    return tasks, datafile


def test_pipeline_and_full_report(tmp_path):
    tasks, dataset = _setup(tmp_path)
    backend = MockBackend()
    result_path = tmp_path / "run.jsonl"
    rows = run_study(tasks, backend, output=result_path, dataset_path=dataset, limit=12)
    assert len(rows) == 12 * len(POLICIES)
    assert all(0 <= row["correct"] <= 1 for row in rows)
    assert all(row["correct"] == 1 for row in rows if row["policy"] in ("symbolic", "adaptive"))
    data = report(rows, dataset_path=dataset, results_path=result_path, output_dir=tmp_path / "report")
    assert data["n_complete_cases"] == 12
    assert data["policy_metrics"]["adaptive"]["accuracy"] == 1.0
    assert data["epistemic_query_discrimination_BC_minus_AD"] == 1.0
    assert (tmp_path / "report" / "REPORT.md").exists()


def test_resume_rejects_changed_dataset_or_policies(tmp_path):
    tasks, dataset = _setup(tmp_path)
    path = tmp_path / "r.jsonl"
    run_study(tasks, MockBackend(), output=path, dataset_path=dataset, limit=8)
    with pytest.raises(FileExistsError):
        run_study(tasks, MockBackend(), output=path, dataset_path=dataset, limit=8)
    resume_rows = run_study(tasks, MockBackend(), output=path, dataset_path=dataset, limit=8, resume=True)
    assert len(resume_rows) == 8 * len(POLICIES)
    with pytest.raises(ValueError, match="metadata changed"):
        run_study(tasks, MockBackend(), output=path, dataset_path=dataset,
                  limit=8, policies=["adaptive"], resume=True)


def test_report_rejects_dataset_tampering(tmp_path):
    tasks, dataset = _setup(tmp_path)
    path = tmp_path / "r.jsonl"
    run_study(tasks, MockBackend(), output=path, dataset_path=dataset, limit=4)
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        report(read_jsonl(path), dataset_path=dataset, results_path=path, output_dir=tmp_path / "out")


def test_confidence_toggle_json_schema():
    assert "confidence" in schema("initial", True)["properties"]
    assert "confidence" not in schema("initial", False)["properties"]
    assert schema("initial", True)["additionalProperties"] is False
    assert "action" not in schema("final", True)["properties"]
    with pytest.raises(ValueError):
        validate_decision({"answer": True, "action": "answer", "query_id": ""}, initial=True, elicit_confidence=False)
    with pytest.raises(ValueError):
        validate_decision({"answer": 0, "action": "answer", "query_id": "", "confidence": 1.7}, initial=True, elicit_confidence=True)


class FakeUsage:
    input_tokens = 12
    output_tokens = 10


class FakeResponse:
    output_text = '{"action":"answer","answer":0,"query_id":"","confidence":0.7}'
    usage = FakeUsage()


class FakeClient:
    def __init__(self):
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


def test_openai_api_schema_payload_and_store_false():
    tasks, _ = generate(4, 7)
    from llmcognition.core import public_projection
    # Use a fake client: no credentials and no actual API calls.
    backend = object.__new__(OpenAIBackend)
    backend.client = FakeClient()
    backend.model = "gpt-4.1-mini"
    backend.elicit_confidence = True
    backend.api_calls = backend.tokens_in = backend.tokens_out = 0
    result = backend.decide(public_projection(tasks[0]))
    assert result["answer"] == 0
    payload = backend.client.calls[0]
    assert payload["store"] is False
    assert payload["model"] == "gpt-4.1-mini"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["type"] == "json_schema"
    assert '"truth"' not in payload["input"]
    assert '"world"' not in payload["input"]
    assert '"condition"' not in payload["input"]
    assert backend.api_calls == 1
    assert backend.tokens_in == 12


def test_no_oracle_sharing_to_initial_model(tmp_path):
    tasks, dataset = _setup(tmp_path, n=4)
    class Probe(MockBackend):
        def decide(self, public):
            assert "sealed" not in public
            assert "truth" not in json.dumps(public)
            assert "critical_query_ids" not in json.dumps(public)
            return super().decide(public)
    rows = run_study(tasks, Probe(), output=tmp_path / "o.jsonl",
                     dataset_path=dataset, limit=4, policies=["adaptive", "direct"])
    assert len(rows) == 8


def test_openai_post_inspection_payload_contains_only_local_result():
    from llmcognition.core import public_projection
    tasks, _ = generate(4, 33)
    backend = object.__new__(OpenAIBackend)
    fake = FakeClient()
    backend.client = fake
    backend.model = "gpt-4.1-mini"
    backend.elicit_confidence = True
    backend.api_calls = backend.tokens_in = backend.tokens_out = 0
    fake.create = lambda **kwargs: type("FinalResponse", (), {
        "output_text": '{"answer":1,"confidence":0.9}', "usage": FakeUsage()
    })()
    final = backend.finalize(public_projection(tasks[0]), {"query_id": "Q-X", "kind": "VERIFY", "valid": True})
    assert final["answer"] == 1
    assert backend.api_calls == 1


# === test_statistics.py ===
from llmcognition.study import auc, paired_test


def test_auc_ties_and_extremes():
    assert auc([(0.8, 1), (0.4, 0)]) == 1.0
    assert auc([(0.4, 1), (0.8, 0)]) == 0.0
    assert auc([(0.5, 1), (0.5, 0)]) == 0.5
    assert auc([(0.7, 1), (0.8, 1)]) is None


def test_paired_ci_contains_large_positive_effect():
    a = {str(i): 1.0 for i in range(30)}
    b = {str(i): 0.0 for i in range(30)}
    result = paired_test(a, b, iterations=1000)
    assert result["n_paired"] == 30
    assert result["difference_mean"] == 1.0
    assert result["bootstrap_95_ci"] == [1.0, 1.0]
    assert result["paired_permutation_p_two_sided"] < .01
