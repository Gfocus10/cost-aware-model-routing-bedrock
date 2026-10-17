"""classify.py - the router's "brain": rate how hard each incoming request is (1-5).

    python src/classify.py public_questions
    python src/classify.py pilot_questions
    python src/classify.py public_test

The classifier sees exactly what the answering model would see (the question plus any
policy text). It never sees reference answers, labels, or any model's results.
Scores are cached in results/<set>_difficulty.jsonl, so thresholds can be tuned later
by src/sweep.py with NO further model calls. Re-running resumes where it stopped.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from models import CREDENTIALS_HELP, call_model, is_credentials_error
from run_eval import DATA, RESULTS, build_prompt, load_questions

SYSTEM = (
    "You triage requests for a model router that chooses between a small cheap model and a "
    "large expensive model. Rate how hard the request is for a SMALL language model:\n"
    "1 = trivial: a single fact lookup or one simple step\n"
    "2 = easy: a couple of facts or one simple calculation\n"
    "3 = moderate: several facts to combine, or light reasoning\n"
    "4 = hard: multi-step reasoning, several rules interacting, or tricky knowledge\n"
    "5 = very hard: long chains of reasoning, conflicting rules, or easy-to-miss details\n"
    'Reply with JSON only, like {"difficulty": 3}.'
)


def parse(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0)).get("difficulty")
    except json.JSONDecodeError:
        return None
    return int(round(d)) if isinstance(d, (int, float)) and 1 <= d <= 5 else None


def classify(request_text: str, tier: str) -> dict:
    prompt = f"Request to rate:\n<<<\n{request_text}\n>>>"
    cost = 0.0
    for _ in range(2):  # retry once if the reply isn't valid JSON
        r = call_model(tier, prompt, max_tokens=40, system=SYSTEM)
        cost += r["cost_usd"]
        d = parse(r["text"])
        if d is not None:
            return {"difficulty": d, "cost_usd": cost, "latency_ms": r["latency_ms"]}
    return {"difficulty": None, "cost_usd": cost, "latency_ms": r["latency_ms"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="question file name without .jsonl, e.g. public_questions")
    ap.add_argument("--tier", default="mid", choices=["cheap", "mid", "strong"],
                    help="which model does the rating (default: mid = Nova Lite)")
    args = ap.parse_args()

    policies_path = DATA / "policies.json"
    policies = json.loads(policies_path.read_text(encoding="utf-8")) if policies_path.exists() else {}
    questions = load_questions(DATA / f"{args.stem}.jsonl")

    out_path = RESULTS / f"{args.stem}_difficulty.jsonl"
    rows = []
    if out_path.exists():
        rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Only successful ratings count as done; failed rows (e.g. expired credentials) are dropped and retried.
    good = [r for r in rows if r.get("difficulty") is not None]
    if len(good) != len(rows):
        print(f"Dropping {len(rows) - len(good)} failed rows from an earlier run; they will be redone.")
        out_path.write_text("".join(json.dumps(r) + "\n" for r in good), encoding="utf-8")
        rows = good
    done = {r["id"] for r in rows}

    with open(out_path, "a", encoding="utf-8") as out:
        for q in questions:
            if q["id"] in done:
                continue
            try:
                res = classify(build_prompt(q, policies), args.tier)
            except Exception as e:
                if is_credentials_error(e):
                    print(CREDENTIALS_HELP)
                    sys.exit(1)  # stop now instead of burning through every question
                res = {"difficulty": None, "cost_usd": 0.0, "latency_ms": 0, "error": f"{type(e).__name__}: {e}"}
            row = {"id": q["id"], "classifier_tier": args.tier, **res}
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            print(f"{q['id']}: difficulty={row['difficulty']}" + (f"  ERROR {row['error']}" if "error" in row else ""))

    dist = Counter(r["difficulty"] for r in rows)
    print(f"\nDifficulty distribution: {dict(sorted(dist.items(), key=lambda kv: (kv[0] is None, kv[0])))}")
    print(f"Classifier cost: ${sum(r['cost_usd'] for r in rows):.6f}  ->  {out_path}")


if __name__ == "__main__":
    main()
