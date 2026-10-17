"""inspect_failures.py - show questions that several tiers got wrong, with the raw answers.

    python src/inspect_failures.py public_questions          # wrong in 2+ tiers
    python src/inspect_failures.py public_questions --min-tiers 1 --limit 5

Why: when a question is missed by several different models, either it is genuinely
hard, or our scoring/reference is wrong. Reading the raw answers tells you which.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
TIERS = ("cheap", "mid", "strong")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="question file name without .jsonl, e.g. public_questions")
    ap.add_argument("--min-tiers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--chars", type=int, default=300)
    args = ap.parse_args()

    by_id = defaultdict(dict)
    for tier in TIERS:
        path = RESULTS / f"{args.stem}_{tier}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]][tier] = r

    shown = 0
    for qid, tiers in by_id.items():
        wrong = [t for t, r in tiers.items() if r.get("score") is not None and r["score"] <= 2]
        if len(wrong) < args.min_tiers:
            continue
        print(f"\\n=== {qid}  wrong in: {wrong}")
        for t, r in tiers.items():
            ans = (r.get("answer") or "").replace("\\n", " ")
            print(f"  [{t}] score={r.get('score')} stop={r.get('stop_reason')} | {r.get('judge_reason', '')}")
            print(f"        {ans[-args.chars:]}")
        shown += 1
        if shown >= args.limit:
            break
    print(f"\\nShown {shown} question(s).")


if __name__ == "__main__":
    main()
