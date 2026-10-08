"""Model interface: one initial decision, optionally one bounded query, final decision.

The model receives only public_projection(task). No shell/web/GitHub tools, no
access to the generator seed or answer key. JSON schema requests decision and
optionally an elicited confidence estimate, not a chain of thought.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .parity import best_query, possibilities


INITIAL_PROMPT = """Solve the supplied binary-relation problem.
Each record asserts that the XOR of two node bits equals 0 or 1. Records are
not guaranteed to be authentic. The ARCHIVE source supplies a relation, while
the AUDIT source verifies a record. You may choose one action: answer with your
current bit, abstain, or inspect ONE listed query ID to receive a local result. Each inspection costs
0.10 utility points. A correct final answer earns 1 point; a wrong answer or
abstention earns 0. There is no further tool access. Select the action offering
the best expected utility. Never invent an unavailable query.
Do not give explanations. Return only the specified response object."""

FINAL_PROMPT = """Solve the same binary-relation problem using the supplied
single inspection result. No more inspections are possible. Provide the most
supported final bit, or abstain. Do not give explanations."""


def schema(stage: str, elicit_confidence: bool) -> dict:
    if stage == "initial":
        properties: dict[str, Any] = {
            "action": {"type": "string", "enum": ["answer", "inspect", "abstain"]},
            "answer": {"type": "integer", "enum": [-1, 0, 1]},
            "query_id": {"type": "string"},
        }
    elif stage == "final":
        properties = {"answer": {"type": "integer", "enum": [-1, 0, 1]}}
    else:
        raise ValueError(stage)
    if elicit_confidence:
        properties["confidence"] = {"type": "number"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def validate_decision(decision: dict, *, initial: bool, elicit_confidence: bool) -> dict:
    if not isinstance(decision, dict) or type(decision.get("answer")) is not int:
        raise ValueError("Model answer is not an integer")
    if decision["answer"] not in (-1, 0, 1):
        raise ValueError("Model answer is not -1, 0 or 1")
    if initial:
        if decision.get("action") not in ("answer", "inspect", "abstain"):
            raise ValueError("Unrecognized action")
        if not isinstance(decision.get("query_id"), str):
            raise ValueError("query_id must be a string")
    if elicit_confidence:
        value = decision.get("confidence")
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError("Confidence must be in [0,1]")
    return decision


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4.1-mini", *, elicit_confidence: bool = True):
        # Optional .env load; deliberately no code path that prints the API key.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env (never commit it).")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the live API extra: pip install -e '.[api]'") from exc
        self.client = OpenAI(timeout=50.0, max_retries=2)
        self.model = model
        self.elicit_confidence = elicit_confidence
        self.api_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _call(self, stage: str, payload: dict) -> dict:
        response = self.client.responses.create(
            model=self.model,
            instructions=INITIAL_PROMPT if stage == "initial" else FINAL_PROMPT,
            input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
            text={"format": {
                "type": "json_schema",
                "name": "llmcognition_" + stage,
                "schema": schema(stage, self.elicit_confidence),
                "strict": True,
            }},
            max_output_tokens=500,
            store=False,
        )
        self.api_calls += 1
        usage = getattr(response, "usage", None)
        self.tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
        if not response.output_text:
            raise RuntimeError("OpenAI response contained no output_text (refusal or truncation)")
        return validate_decision(
            json.loads(response.output_text),
            initial=(stage == "initial"),
            elicit_confidence=self.elicit_confidence,
        )

    def decide(self, public: dict) -> dict:
        return self._call("initial", {"task": public})

    def finalize(self, public: dict, inspection: dict) -> dict:
        return self._call("final", {"task": public, "inspection": inspection})


class MockBackend:
    """Local symbolic reference behavior. This is NOT an OpenAI experiment."""

    model = "mock-symbolic-NOT-LLM"

    def __init__(self, *, elicit_confidence: bool = True):
        self.elicit_confidence = elicit_confidence
        self.api_calls = self.tokens_in = self.tokens_out = 0

    def decide(self, public: dict) -> dict:
        possible = possibilities(public)
        answer = min(possible) if len(possible) == 1 else 0
        q = best_query(public)
        out: dict[str, Any] = {
            "action": "inspect" if len(possible) > 1 and q else "answer",
            "answer": answer,
            "query_id": q["id"] if q and len(possible) > 1 else "",
        }
        if self.elicit_confidence:
            out["confidence"] = 0.99 if len(possible) == 1 else 0.5
        return out

    def finalize(self, public: dict, inspection: dict) -> dict:
        q = next((q for q in public["queries"] if q["id"] == inspection.get("query_id")), None)
        if q is None:
            poss = possibilities(public)
        else:
            result = inspection.get("value") if q["kind"] == "REL" else inspection.get("valid")
            poss = possibilities(public, {"query": q, "result": result})
        out: dict[str, Any] = {"answer": min(poss) if len(poss) == 1 else 0}
        if self.elicit_confidence:
            out["confidence"] = 0.99 if len(poss) == 1 else 0.5
        return out
