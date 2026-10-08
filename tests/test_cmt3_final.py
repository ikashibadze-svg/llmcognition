"""CMT-3 tests. No internet and no OpenAI API calls."""
import json
from pathlib import Path

import pytest

import cmt3_final as c


def test_deterministic_generation_and_sealed_audit():
    ds = c.generate(8, seed=20261009)
    assert c.canonical(ds) == c.canonical(c.generate(8, seed=20261009))
    result = c.audit(ds)
    assert result["audit"] == "PASS"
    assert result["tasks"] == 64
    assert result["independent_groups"] == 48
    assert result["counterfactual_twins"] == 16


def test_default_final_design():
    ds = c.generate(24, seed=20261010)
    result = c.audit(ds)
    assert result["tasks"] == 192
    assert result["independent_groups"] == 144
    assert result["positive_answers_per_cell"] == 12


@pytest.mark.parametrize("domain", ["lookup", "gate"])
@pytest.mark.parametrize("condition", list("ABCD"))
def test_each_domain_condition_has_precise_identifiability(domain, condition):
    ds = c.generate(4, 13203)
    relevant = [g for g in ds["groups"] if g["public"]["domain"] == domain and
                g["tasks"][0]["sealed"]["condition"] == condition]
    assert relevant
    for group in relevant:
        initial = c.possibilities(group["public"])
        critical = c.critical_queries(group["public"])
        assert len(critical) == (1 if condition in "BC" else 0)
        for task in group["tasks"]:
            truth = task["sealed"]["truth"]
            assert initial == ({0, 1} if condition in "BC" else {truth})
            for q in group["public"]["queries"]:
                result = c.query_oracle(group, task, q["id"])
                after = c.possibilities(group["public"], result)
                assert after == ({truth} if q["id"] in critical else initial)


def test_twin_pairs_are_indistinguishable_before_oracle():
    ds = c.generate(24, 14725)
    for group in ds["groups"]:
        public = group["public"]
        assert "truth" not in public and "sealed" not in public and "group_id" not in public
        if len(group["tasks"]) == 2:
            assert {t["sealed"]["truth"] for t in group["tasks"]} == {0, 1}
            qid = c.critical_queries(public)[0]
            assert {json.dumps(c.query_oracle(group, t, qid), sort_keys=True) for t in group["tasks"]}.__len__() == 2


def test_invalid_query_rejected():
    ds = c.generate(4, 77)
    with pytest.raises(ValueError, match="not in menu"):
        c.query_oracle(ds["groups"][0], ds["groups"][0]["tasks"][0], "Q-UNKNOWN")


def test_response_validation():
    x = {"action": "inspect", "answer": 0, "query_id": "Q-123", "confidence": .5}
    assert c.validate_response(x, "initial") == x
    for wrong in [dict(x, answer=True), dict(x, confidence=1.1), dict(x, query_id=32), dict(x, action="hack")]:
        with pytest.raises(ValueError):
            c.validate_response(wrong, "initial")


def test_offline_smoke_resume_and_final_report(tmp_path):
    src = tmp_path / "sealed.json"
    c.atomic_json(src, c.generate(4, 1111))
    dest = tmp_path / "results.jsonl"
    summary = c.run_study(src, dest, backend=c.MockOptimal())
    assert summary["completed_tasks"] == 32
    assert summary["completed_groups"] == 24
    before = dest.read_bytes()
    resumed = c.run_study(src, dest, backend=c.MockOptimal(), resume=True)
    assert resumed["resumed_groups"] == 24
    assert dest.read_bytes() == before
    metrics = c.final_report(src, dest, tmp_path / "report")
    assert metrics["N_cases"] == 32
    assert metrics["verdict"] == "MOCK_ONLY__NO_EMPIRICAL_CLAIM"
    assert metrics["policies"]["adaptive"]["accuracy"] == 1.0
    assert (tmp_path / "report" / "CONCLUSION.md").exists()


def test_resume_metadata_prevents_model_swap(tmp_path):
    path = tmp_path / "seeded.json"
    c.atomic_json(path, c.generate(4, 11))
    out = tmp_path / "study.jsonl"
    c.run_study(path, out, backend=c.MockOptimal())

    class ChangedMock(c.MockOptimal):
        model = "mock-changed"

    with pytest.raises(ValueError, match="metadata changed"):
        c.run_study(path, out, backend=ChangedMock(), resume=True)


def test_report_refuses_partial(tmp_path):
    path = tmp_path / "sealed.json"
    c.atomic_json(path, c.generate(4, 111))
    out = tmp_path / "study.jsonl"
    c.run_study(path, out, backend=c.MockOptimal())
    rows = out.read_text().splitlines()
    # Drop complete final independent group, no partial-case corruption.
    last = json.loads(rows[-1])["group_id"]
    out.write_text("\n".join(s for s in rows if json.loads(s)["group_id"] != last) + "\n")
    with pytest.raises(ValueError, match="Not all independent groups"):
        c.final_report(path, out, tmp_path / "report")


def test_mock_full_positive_control_has_zero_false_empirical_claim(tmp_path):
    path = tmp_path / "sealed.json"
    c.atomic_json(path, c.generate(24, 13579))
    out = tmp_path / "study.jsonl"
    c.run_study(path, out, backend=c.MockOptimal())
    metrics = c.final_report(path, out, tmp_path / "report")
    assert metrics["strong_evidence_gates_passed"] is True
    assert metrics["verdict"] == "MOCK_ONLY__NO_EMPIRICAL_CLAIM"
    assert metrics["N_cases"] == 192
    assert metrics["N_independent_groups"] == 144
    assert metrics["query_independent_groups_BC"] == 48
    assert metrics["critical_hit_independent_groups_BC"] == 48
    assert metrics["adaptive_minus_direct_utility"]["effect"] > .1


class AlwaysAnswer(c.MockOptimal):
    model = "blind-no-query-NOT-OPENAI"

    def decide(self, public):
        response = self.direct(public)
        return {**response, "action": "answer", "query_id": ""}


def test_offline_negative_control_is_not_labeled_metacognition(tmp_path):
    path = tmp_path / "sealed.json"
    c.atomic_json(path, c.generate(24, 98765))
    out = tmp_path / "study.jsonl"
    c.run_study(path, out, backend=AlwaysAnswer())
    metrics = c.final_report(path, out, tmp_path / "report")
    assert metrics["strong_evidence_gates_passed"] is False
    assert metrics["query_sensitivity_BC"] == 0
    assert metrics["verdict"] == "MOCK_ONLY__NO_EMPIRICAL_CLAIM"


def test_frozen_dataset_never_overwritten(tmp_path):
    path = tmp_path / "sealed.json"
    c.atomic_json(path, c.generate(4, 77))
    with pytest.raises(FileExistsError):
        c.cli(["generate", "--data", str(path)])


def test_cluster_precision_not_double_counted():
    ds = c.generate(24, 99999)
    pairs = [g for g in ds["groups"] if len(g["tasks"]) == 2]
    assert len(pairs) == 48
    # Each pair is a single independent query choice, never two samples.
    assert sum(len(g["tasks"]) for g in pairs) == 96


def test_type2_auc_exact_ties_and_cluster_bootstrap():
    assert c.auc([(0.5, 0), (0.5, 1)]) == 0.5
    assert c.auc([(0.1, 0), (0.9, 1)]) == 1.0
    assert c.auc([(0.9, 0), (0.1, 1)]) == 0.0
    rows = [
        {"group_id": "g1", "initial_confidence": .8, "initial_correct": 1},
        {"group_id": "g2", "initial_confidence": .2, "initial_correct": 0},
        {"group_id": "g3", "initial_confidence": .5, "initial_correct": 1},
        {"group_id": "g3", "initial_confidence": .5, "initial_correct": 0},
    ]
    ci = c.cluster_auc_ci(rows, trials=200)
    assert ci is not None and 0 <= ci[0] <= ci[1] <= 1
