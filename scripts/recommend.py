"""Ask the planner which forecasting model to use, for the whole test period or for a period / kind of hour.

    python scripts/recommend.py                                   # whole test period, with the written reasoning
    python scripts/recommend.py --focus "evening 18-23"           # one kind of hour
    python scripts/recommend.py --start 2014-12-20 --end 2014-12-31 --question "which model for Christmas week?"
    python scripts/recommend.py --no-llm                          # only the computed decision and evidence; no API key needed

The decision (which model, how sure, where to switch) is computed from the data with a statistical test; the LLM only
writes the reasoning for it, and that text is checked against the same evidence.
"""

import argparse
import warnings

from forecast_explainer import planner

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: start of the test period)")
    parser.add_argument("--end", help="last day, YYYY-MM-DD (default: end of the test period)")
    parser.add_argument("--focus", choices=planner.FOCUS_CHOICES, help="compare only this kind of hour")
    parser.add_argument("--question", help="your question in words, so the answer addresses it")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--no-rag", action="store_true", help="write the reasoning without retrieved sources")
    parser.add_argument("--no-explanations", action="store_true", help="skip each model's worst-hour drivers (faster the first time)")
    parser.add_argument("--no-llm", action="store_true", help="print only the computed decision and evidence")
    args = parser.parse_args()

    if args.no_llm:
        analysis = planner.analyse(args.start, args.end, args.focus, with_explanations=not args.no_explanations)
        print(f"\n{analysis['decision']['headline']}")
        for line in analysis["decision"]["switches"]:
            print(f"  - {line}")
        print()
        for tag, text in planner.evidence_blocks(analysis).items():
            print(f"[{tag}]\n{text}\n")
        return

    result = planner.recommend(args.start, args.end, args.focus, question=args.question, model=args.model,
                               use_rag=not args.no_rag, with_explanations=not args.no_explanations)
    c = result["checks"]
    print(f"\n{result['decision']['headline']}")
    for line in result["decision"]["switches"]:
        print(f"  - {line}")
    status = "checks passed" if c["passed"] else f"CHECKS FAILED (unsupported numbers: {[x['number'] for x in c['unsupported_numbers']]})"
    print(f"\n=== {result['model']} | {status} | {c['words']} words | {result['source']} ===\n")
    print(result["text"], "\n")
    for s in result["sources"]:
        where = f", p.{s['page']}" if s["page"] else f" > {s['section']}"
        print(f"  [{s['sid']}] {s['citation']}{where}")


if __name__ == "__main__":
    main()
