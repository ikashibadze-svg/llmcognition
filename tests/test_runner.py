import json

import pytest

from llmcognition.agent import MockBackend, OpenAIBackend, schema, validate_decision
from llmcognition.data import atomic_write_jsonl, generate, read_jsonl
from llmcognition.evaluation import report
from llmcognition.runner import POLICIES, run_study


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
    from llmcognition.data import public_projection
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
    from llmcognition.data import public_projection
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
