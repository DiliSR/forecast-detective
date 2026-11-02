"""Ask the explanation agent a question in plain English. Needs OPENAI_API_KEY in .env.

    python scripts/ask.py "why was the forecast so off on Christmas Eve evening?"
    python scripts/ask.py "explain the LSTM forecast for 16 October at noon"
    python scripts/ask.py --chat                      # keep talking; follow-ups remember earlier turns

The agent works out the hour and model(s), runs the explainers (cached after the first time), retrieves background
and prints a cited, checked explanation.
"""

import argparse
import warnings

from forecast_explainer.agent import run_agent

warnings.filterwarnings("ignore")


def show(result: dict, verbose: bool) -> None:
    """Print the agent's note, each explanation with its check result and sources, and optionally the trace."""
    if verbose:
        for step in result["trace"]:
            print(f"  [{step['seconds']:>4}s] {step['tool']}({step['arguments']})")
    print(f"\n{result['message']}\n")
    for n in result["narratives"]:
        c = n["checks"]
        status = "checks passed" if c["passed"] else f"CHECKS FAILED (unsupported numbers: {[x['number'] for x in c['unsupported_numbers']]}, " \
                                                    f"unknown ids: {c['unknown_citations']}, flags cited: {not c['flags_not_cited']})"
        print(f"=== {n['timestamp'][:16]} | {'+'.join(n['models'])} | {n['model']} | {status} ===\n")
        print(n["text"], "\n")
        for s in n["sources"]:
            where = f", p.{s['page']}" if s["page"] else f" > {s['section']}"
            print(f"  [{s['sid']}] {s['citation']}{where}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="your question, in plain English")
    parser.add_argument("--chat", action="store_true", help="interactive conversation")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="show every tool call")
    args = parser.parse_args()

    history: list[dict] = []
    prompt = args.prompt
    while True:
        if not prompt:
            prompt = input("\nyou> ").strip()
            if prompt in ("", "exit", "quit"):
                return
        result = run_agent(prompt, history, model=args.model, use_rag=not args.no_rag)
        show(result, args.verbose or args.chat)
        history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": result["history_text"]}]
        if not args.chat:
            return
        prompt = None


if __name__ == "__main__":
    main()
