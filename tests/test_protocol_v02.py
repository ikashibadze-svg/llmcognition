from llmcognition.agent import INITIAL_PROMPT, FINAL_PROMPT, MockBackend
from llmcognition.data import generate, atomic_write_jsonl, public_projection
from llmcognition.parity import possibilities
from llmcognition.runner import select_tasks, run_study
from llmcognition.evaluation import report


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
