"""Explain one forecast hour in words, with citations. Needs OPENAI_API_KEY in .env.

    python scripts/narrate.py "2014-12-24 18:00"
    python scripts/narrate.py "2014-12-24 18:00" --question "why did both models over-forecast?"
    python scripts/narrate.py "2014-12-24 18:00" --no-rag        # ablation: leave out the knowledge base
    python scripts/narrate.py "2014-12-24 18:00" --refresh       # ignore the cached answer

Everything before the LLM call is cached (explanations) or deterministic (evidence, retrieval).
"""

import argparse
import warnings

from forecast_explainer.narrate import narrate

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timestamp", help='target hour, e.g. "2014-12-24 18:00"')
    parser.add_argument("--client", default="total")
    parser.add_argument("--question", default=None)
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out = narrate(args.timestamp, args.client, question=args.question, model=args.model,
                  use_rag=not args.no_rag, refresh=args.refresh)
    print(f"\n=== {out['timestamp']}  model {out['model']}  rag={out['use_rag']}  ({out['source']}) ===\n")
    print(out["text"])
    c = out["checks"]
    print(f"\n--- checks: {'PASSED' if c['passed'] else 'FAILED'} after {out['n_attempts']} attempt(s); "
          f"{c['n_citations']} citations; unknown {c['unknown_citations']}; "
          f"unsupported numbers {[n['number'] for n in c['unsupported_numbers']]}; uncited-number sentences {len(c['uncited_numbers'])}")
    if out["sources"]:
        print("\nsources:")
        for s in out["sources"]:
            where = f", p.{s['page']}" if s["page"] else f" > {s['section']}"
            print(f"  [{s['sid']}] {s['citation']}{where}")


if __name__ == "__main__":
    main()
