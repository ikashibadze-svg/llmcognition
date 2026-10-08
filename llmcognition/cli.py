"""python -m llmcognition --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import MockBackend, OpenAIBackend
from .data import atomic_write_jsonl, generate, public_projection, read_jsonl, sha256
from .evaluation import report
from .runner import POLICIES, run_study
from .validation import audit_tasks


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CMT-2: sealed, falsifiable metacognition experiments")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Generate private cases; do not commit the private file")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=None, help="Omit for a fresh cryptographic random seed")
    g.add_argument("--out", default="data/private/tasks.jsonl")
    g.add_argument("--public-preview", default=None, help="Optional public-only data export")
    a = sub.add_parser("audit", help="Validate task generation / hidden-answer isolation")
    a.add_argument("--dataset", default="data/private/tasks.jsonl")
    r = sub.add_parser("run", help="Run OpenAI API or explicitly labeled offline mock")
    r.add_argument("--dataset", default="data/private/tasks.jsonl")
    r.add_argument("--backend", choices=["openai", "mock"], default="mock")
    r.add_argument("--model", default="gpt-4.1-mini")
    r.add_argument("--policies", default=",".join(POLICIES))
    r.add_argument("--limit", type=int, default=8, help="Default 8 to prevent unintended API spending")
    r.add_argument("--seed", type=int, default=20261008)
    r.add_argument("--selection", choices=["prefix", "stratified"], default="prefix",
                   help="stratified: balanced A/B/C/D selection for small pilots")
    r.add_argument("--query-cost", type=float, default=0.10)
    r.add_argument("--max-api-calls", type=int, default=500)
    r.add_argument("--minimal", action="store_true", help="Do not solicit confidence estimates; type-2 AUROC unavailable")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--out", default="runs/study.jsonl")
    e = sub.add_parser("report", help="Calculate paired statistics and markdown report")
    e.add_argument("--dataset", default="data/private/tasks.jsonl")
    e.add_argument("--results", default="runs/study.jsonl")
    e.add_argument("--output-dir", default="reports/study")
    s = sub.add_parser("smoke", help="End-to-end offline 8-case demo; no API calls")
    s.add_argument("--output-dir", default="runs/smoke")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "generate":
        tasks, seed = generate(args.n, args.seed)
        checks = audit_tasks(tasks)
        atomic_write_jsonl(args.out, tasks)
        if args.public_preview:
            atomic_write_jsonl(args.public_preview,
                               [{"id": t["id"], "public": public_projection(t)} for t in tasks])
        print(json.dumps({"seed": seed, "private_dataset": args.out,
                          "sha256": sha256(args.out), "audit": checks}, indent=2))
    elif args.command == "audit":
        tasks = read_jsonl(args.dataset)
        print(json.dumps(audit_tasks(tasks), indent=2))
    elif args.command == "run":
        tasks = read_jsonl(args.dataset)
        audit_tasks(tasks)
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
        backend = (MockBackend(elicit_confidence=not args.minimal) if args.backend == "mock"
                   else OpenAIBackend(model=args.model, elicit_confidence=not args.minimal))
        results = run_study(tasks, backend, output=args.out, dataset_path=args.dataset,
                            policies=policies, limit=args.limit, seed=args.seed,
                            query_cost=args.query_cost, max_api_calls=args.max_api_calls,
                            resume=args.resume, selection=args.selection)
        print(json.dumps({"rows": len(results), "cases": len(set(r['case_id'] for r in results)),
                          "backend": backend.model, "actual_api_calls": backend.api_calls,
                          "input_tokens": backend.tokens_in, "output_tokens": backend.tokens_out,
                          "results": args.out}, indent=2))
    elif args.command == "report":
        rows = read_jsonl(args.results)
        data = report(rows, dataset_path=args.dataset,
                      results_path=args.results, output_dir=args.output_dir)
        print(json.dumps({"output_dir": args.output_dir, "completed_cases": data["n_complete_cases"],
                          "condition_discrimination": data["epistemic_query_discrimination_BC_minus_AD"]}, indent=2))
    elif args.command == "smoke":
        home = Path(args.output_dir)
        home.mkdir(parents=True, exist_ok=True)
        dataset = home / "tasks.jsonl"
        result = home / "mock.jsonl"
        tasks, seed = generate(8, 42)
        checks = audit_tasks(tasks)
        atomic_write_jsonl(dataset, tasks)
        if result.exists():
            result.unlink()
        meta = Path(str(result) + ".meta.json")
        if meta.exists():
            meta.unlink()
        backend = MockBackend()
        rows = run_study(tasks, backend, output=result, dataset_path=dataset, limit=8)
        summary = report(rows, dataset_path=dataset, results_path=result,
                         output_dir=home / "report")
        print(json.dumps({"status": "PASS", "type": "OFFLINE_MOCK_NOT_OPENAI",
                          "audited": checks["number_tasks"], "rows": len(rows),
                          "output": str(home / 'report' / 'REPORT.md'),
                          "query_discrimination": summary["epistemic_query_discrimination_BC_minus_AD"]}, indent=2))


if __name__ == "__main__":
    main()
