"""run_eval.py - run every question through ONE tier, score each answer, log results.

    python src/run_eval.py --tier cheap                          # BrightCart pilot set
    python src/run_eval.py --tier cheap --questions data/public_questions.jsonl

Scoring depends on the question's answer_type:
  short_fact / json / rubric -> LLM judge (judge.py), score 1-5
  mcq / number               -> exact match in code, score 5 if correct else 1 (no judge cost)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from judge import judge_answer
from models import CREDENTIALS_HELP, call_model, is_credentials_error

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DOMAIN_SYSTEM = (
    "You are a customer-support assistant for BrightCart, an online home-goods store. "
    "Answer accurately and concisely. When store policies are provided, use only those."
)
NEUTRAL_SYSTEM = "You are a helpful assistant. Follow the answer format instructions exactly."
EXACT_TYPES = {"mcq", "number"}


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(q: dict, policies: dict) -> str:
    """The model sees the question plus the policy text it needs, never labels or answers."""
    if not q["policy_ids"]:
        return q["question"]
    policy_text = "\n".join(policies[p] for p in q["policy_ids"])
    return f"Store policies:\n{policy_text}\n\nTask:\n{q['question']}"


# ---------- exact-match scoring (no LLM judge needed) ----------

def extract_letter(text: str):
    """Find the chosen option letter A-D in a model reply, or None.

    Models don't always obey "answer with only the letter" - some explain first and
    conclude at the end - so after the clean cases we look for the LAST stated choice.
    """
    t = text.strip()
    m = re.fullmatch(r"\(?([A-D])\)?[.\s]*", t)  # reply is just the letter
    if m:
        return m.group(1)
    last_line = t.splitlines()[-1] if t else ""
    m = re.fullmatch(r"\W*\(?([A-D])\)?\W*", last_line)  # reply ends with a bare letter
    if m:
        return m.group(1)
    for pattern in (
        r"(?i:answer)\W{0,10}(?:is\W{0,3})?\(?([A-D])\b",             # "The answer is B", "Answer: **B**"
        r"(?i:correct|option|choice)\W{0,12}(?:is\W{0,3})?\(?([A-D])\b",  # "the correct option is B"
        r"\*\*\(?([A-D])\b",                                           # bolded letter: **B) ...**
    ):
        found = re.findall(pattern, t)
        if found:
            return found[-1]
    m = re.match(r"\(?([A-D])\b", t)  # reply starts with the letter
    return m.group(1) if m else None


def extract_number(text: str):
    """Find the final numeric answer: prefer 'Final answer: N', else the last number."""
    m = re.search(r"(?i:final answer)[\s:*$]*(-?[\d,]*\.?\d+)", text)  # [\s:*$] so a minus sign is not swallowed
    raw = m.group(1) if m else None
    if raw is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        raw = nums[-1] if nums else None
    if raw is None:
        return None
    try:
        return float(raw.replace(",", "").rstrip("."))
    except ValueError:
        return None


def score_exact(q: dict, text: str) -> dict:
    if q["answer_type"] == "mcq":
        got = extract_letter(text)
        correct = got == q["reference"]
    else:
        got = extract_number(text)
        correct = got is not None and abs(got - float(q["reference"])) < 1e-6
    return {"score": 5 if correct else 1, "correct": correct, "unparsed": got is None,
            "reason": f"exact match: expected {q['reference']}, got {got}", "judge_cost_usd": 0.0}


# ---------- run + summarize ----------

def load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("score") is not None]
    if not ok:
        print("No scored rows to summarize.")
        return
    n = len(ok)
    print(f"\nScored {n} of {len(rows)} questions")
    print(f"  avg score        : {sum(r['score'] for r in ok) / n:.2f} / 5")

    # Rows the model never really answered must not be silently counted as "wrong".
    filtered = [r for r in ok if r.get("stop_reason") == "content_filtered"]
    truncated = [r for r in ok if r.get("stop_reason") == "max_tokens"]
    unparsed = [r for r in ok if r.get("unparsed") and r not in filtered]
    exact = [r for r in ok if "correct" in r and r not in filtered]
    if exact:
        print(f"  accuracy (exact) : {sum(r['correct'] for r in exact) / len(exact):.0%} of {len(exact)} exact-match questions")
    if filtered:
        print(f"  WARNING content-filtered by Bedrock (excluded from accuracy): {[r['id'] for r in filtered]}")
    if truncated:
        print(f"  WARNING cut off by max_tokens: {len(truncated)} rows {[r['id'] for r in truncated][:10]}")
    if unparsed:
        print(f"  WARNING answer could not be parsed: {len(unparsed)} rows {[r['id'] for r in unparsed][:10]}")

    print(f"  answer cost      : ${sum(r['cost_usd'] for r in ok):.6f}")
    print(f"  avg answer cost  : ${sum(r['cost_usd'] for r in ok) / n:.6f}")
    print(f"  avg latency      : {sum(r['latency_ms'] for r in ok) / n:.0f} ms")
    print(f"  judge cost       : ${sum(r['judge_cost_usd'] for r in ok):.4f} (evaluation overhead)")
    by_diff = defaultdict(list)
    for r in ok:
        by_diff[r["difficulty_guess"]].append(r["score"])
    for d in ("easy", "medium", "hard"):
        if d in by_diff:
            print(f"  {d:6} avg score : {sum(by_diff[d]) / len(by_diff[d]):.2f} ({len(by_diff[d])} q)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["cheap", "mid", "strong"])
    ap.add_argument("--questions", default=str(DATA / "pilot_questions.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--max-tokens", type=int, default=300,
                    help="default cap; a question's own max_tokens field overrides it")
    args = ap.parse_args()

    policies_path = DATA / "policies.json"
    policies = json.loads(policies_path.read_text(encoding="utf-8")) if policies_path.exists() else {}
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{Path(args.questions).stem}_{args.tier}.jsonl"
    rows = load_done(out_path)
    # Only scored rows count as done; failed rows (e.g. expired credentials) are dropped and retried.
    good = [r for r in rows if r.get("score") is not None]
    if len(good) != len(rows):
        print(f"Dropping {len(rows) - len(good)} failed rows from an earlier run; they will be redone.")
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good), encoding="utf-8")
        rows = good
    done_ids = {r["id"] for r in rows}

    with open(out_path, "a", encoding="utf-8") as out:
        for q in questions:
            if q["id"] in done_ids:
                continue
            exact = q["answer_type"] in EXACT_TYPES
            try:
                resp = call_model(args.tier, build_prompt(q, policies),
                                  max_tokens=q.get("max_tokens", args.max_tokens),
                                  system=NEUTRAL_SYSTEM if exact else DOMAIN_SYSTEM)
                verdict = score_exact(q, resp["text"]) if exact else judge_answer(q, resp["text"])
                row = {
                    "id": q["id"], "difficulty_guess": q["difficulty_guess"],
                    "category": q["category"], "tier": args.tier,
                    "answer": resp["text"], "input_tokens": resp["input_tokens"],
                    "output_tokens": resp["output_tokens"], "latency_ms": resp["latency_ms"],
                    "cost_usd": resp["cost_usd"], "stop_reason": resp["stop_reason"],
                    "score": verdict["score"], "judge_reason": verdict["reason"],
                    "judge_cost_usd": verdict["judge_cost_usd"],
                }
                if "correct" in verdict:
                    row["correct"] = verdict["correct"]
                    row["unparsed"] = verdict["unparsed"]
            except Exception as e:  # keep going; one bad call shouldn't lose the whole run
                if is_credentials_error(e):
                    print(CREDENTIALS_HELP)
                    sys.exit(1)  # every later call would fail too, so stop here
                row = {"id": q["id"], "difficulty_guess": q["difficulty_guess"],
                       "category": q["category"], "tier": args.tier,
                       "score": None, "error": f"{type(e).__name__}: {e}",
                       "cost_usd": 0.0, "latency_ms": 0, "judge_cost_usd": 0.0}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            print(f"{q['id']}: score={row.get('score')} cost=${row.get('cost_usd', 0):.6f}"
                  + (f"  ERROR {row['error']}" if "error" in row else ""))

    print(f"\nResults file: {out_path}")
    summarize(rows)


if __name__ == "__main__":
    main()
