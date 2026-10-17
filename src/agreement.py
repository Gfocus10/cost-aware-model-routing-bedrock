"""agreement.py - offline test of an "agreement check" router. No model calls, no cost.

Idea: run the two cheap models (cheap + mid) on every question. If they give the SAME
answer, accept it. If they DISAGREE, at least one is wrong, so escalate to the strong model.
Optionally, the difficulty classifier can send the hardest requests straight to the strong
model without running the cheap pair at all.

    python src/agreement.py public_questions

Only works for exact-match questions (multiple-choice / number), because "same answer" needs
a letter or a number to compare. Free-text BrightCart answers can't be compared this way.
Used on DEV sets only; the test set is evaluated once, at the end.
"""
import sys
from pathlib import Path

import sweep
from run_eval import DATA, extract_letter, extract_number, load_questions

TIERS = sweep.TIERS


def extract(q: dict, text: str):
    return extract_letter(text) if q["answer_type"] == "mcq" else extract_number(text)


def same(a, b) -> bool:
    if a is None or b is None:  # an unreadable answer counts as disagreement
        return False
    return a == b if isinstance(a, str) else abs(a - b) < 1e-6


def load(stem: str) -> list[dict]:
    questions = {q["id"]: q for q in load_questions(DATA / f"{stem}.jsonl")}
    tiers = {t: sweep.read(sweep.RESULTS / f"{stem}_{t}.jsonl") for t in TIERS}
    diff = sweep.read(sweep.RESULTS / f"{stem}_difficulty.jsonl")
    skip = sweep.load_excluded(stem)
    items = []
    for qid in sorted(set.intersection(*(set(d) for d in tiers.values()), set(diff)) - skip):
        q = questions[qid]
        if q["answer_type"] not in ("mcq", "number"):
            continue
        rows = {t: tiers[t][qid] for t in TIERS}
        if any(r.get("score") is None or r.get("stop_reason") == "content_filtered" for r in rows.values()):
            continue
        if diff[qid].get("difficulty") is None:
            continue
        items.append({
            "key": f"{stem}:{qid}", "d": diff[qid]["difficulty"], "cls_cost": diff[qid]["cost_usd"],
            "passed": {t: rows[t]["score"] >= sweep.PASS_SCORE for t in TIERS},
            "cost": {t: rows[t]["cost_usd"] for t in TIERS},
            "agree": same(extract(q, rows["cheap"]["answer"]), extract(q, rows["mid"]["answer"])),
        })
    return items


def cascade(items: list[dict], direct_at: int | None = None) -> dict:
    """Agreement router. If direct_at is set, difficulty >= direct_at goes straight to strong."""
    passed, cost, direct, escalated = 0, 0.0, 0, 0
    for it in items:
        if direct_at is not None:
            cost += it["cls_cost"]
        if direct_at is not None and it["d"] >= direct_at:
            cost += it["cost"]["strong"]
            passed += it["passed"]["strong"]
            direct += 1
            continue
        cost += it["cost"]["cheap"] + it["cost"]["mid"]
        if it["agree"]:
            passed += it["passed"]["cheap"]
        else:
            cost += it["cost"]["strong"]
            passed += it["passed"]["strong"]
            escalated += 1
    n = len(items)
    return {"pass": passed / n, "cost": cost, "direct": direct, "escalated": escalated}


def main(stem: str) -> None:
    items = load(stem)
    n = len(items)
    if n == 0:
        raise SystemExit("No exact-match questions found. Run the baselines and classify.py for this set first.")
    strong_cost = sum(it["cost"]["strong"] for it in items)
    print(f"Set: {stem} | {n} exact-match questions\n")
    sweep.baselines(items)

    agree = [it for it in items if it["agree"]]
    disagree = [it for it in items if not it["agree"]]
    print(f"\nCheap and mid agree on {len(agree)} of {n} questions; they are correct on "
          f"{sum(it['passed']['cheap'] for it in agree)} of those.")
    print(f"They disagree on {len(disagree)}: cheap right {sum(it['passed']['cheap'] for it in disagree)}, "
          f"mid right {sum(it['passed']['mid'] for it in disagree)}, "
          f"strong right {sum(it['passed']['strong'] for it in disagree)}.")
    both_wrong = [it["key"] for it in agree if not it["passed"]["cheap"]]
    print(f"Agreed but WRONG (the check cannot catch these): {both_wrong}")

    print(f"\n{'router':<44}{'pass rate':>10}{'total cost':>13}{'% of strong':>13}   notes")
    for label, (t1, t2) in (("difficulty router (3,4)", (3, 4)), ("difficulty router (3,3)", (3, 3))):
        r = sweep.evaluate(items, t1, t2)
        print(f"{label:<44}{r['pass']:>10.1%}{r['cost']:>13.6f}{r['cost'] / strong_cost:>13.1%}"
              f"   cheap/mid/strong {r['split']['cheap']}/{r['split']['mid']}/{r['split']['strong']}")
    for label, d in (("agreement check", None),
                     ("agreement + difficulty>=4 straight to strong", 4),
                     ("agreement + difficulty>=3 straight to strong", 3)):
        r = cascade(items, d)
        print(f"{label:<44}{r['pass']:>10.1%}{r['cost']:>13.6f}{r['cost'] / strong_cost:>13.1%}"
              f"   direct {r['direct']}, escalated {r['escalated']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "public_questions")
