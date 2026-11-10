"""CLI:  python -m agent.main "Which dataset version was used for experiment 25?" """

from __future__ import annotations

import argparse
import sys

from agent.bootstrap import build_agent
from agent.config import Settings
from agent.logging_utils import configure_logging
from agent.report import render_report
from agent.schemas import ChatRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlops-agent")
    parser.add_argument("message", nargs="*", help="question; omit for an interactive session")
    parser.add_argument("--confirm", action="store_true", help="allow state-changing tools if you ask for them")
    parser.add_argument("--degradation-note", default=None, help="evidence of model degradation to consider")
    parser.add_argument("--json", action="store_true", help="print the full JSON response")
    parser.add_argument("--evidence", action="store_true", help="also print facts/uncertainties/actions")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    agent = build_agent(settings)

    def ask(text: str) -> None:
        req = ChatRequest(message=text, confirm_mutations=args.confirm, degradation_note=args.degradation_note)
        resp = agent.handle(req)
        if args.json:
            print(resp.model_dump_json(indent=2))
            return
        print(resp.answer)
        if args.evidence:
            print("\n--- evidence ---")
            print(render_report(resp.facts, resp.uncertainties, resp.recommendations))
        print(f"\n(request_id={resp.request_id}, tools={[r.tool for r in resp.tool_trace]})")

    if args.message:
        ask(" ".join(args.message))
        return 0
    print("MLOps agent - type a question, or 'exit'.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if line.lower() in {"exit", "quit"}:
            return 0
        if line:
            ask(line)


if __name__ == "__main__":
    sys.exit(main())
