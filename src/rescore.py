"""rescore.py - re-score saved exact-match answers with the current extraction code.

    python src/rescore.py public_questions

Use it after fixing answer extraction, so you don't have to pay to re-run the models.
Only mcq/number rows are touched; judge-scored rows and content-filtered rows are left alone.
"""
import json
import sys
from pathlib import Path

from run_eval import DATA, RESULTS, load_questions, score_exact

TIERS = ("cheap", "mid", "strong")


def main(stem: str) -> None:
    questions = {q["id"]: q for q in load_questions(DATA / f"{stem}.jsonl")}
    for tier in TIERS:
        path = RESULTS / f"{stem}_{tier}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        changed = []
        for r in rows:
            q = questions.get(r["id"])
            if (not q or q["answer_type"] not in ("mcq", "number") or r.get("score") is None
                    or r.get("stop_reason") == "content_filtered" or not r.get("answer")):
                continue
            v = score_exact(q, r["answer"])
            if v["correct"] != r.get("correct") or v["unparsed"] != r.get("unparsed"):
                changed.append(f"{r['id']} ({'now correct' if v['correct'] else 'now wrong'})")
            r.update(score=v["score"], correct=v["correct"], unparsed=v["unparsed"], judge_reason=v["reason"])
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{tier}: re-scored {len(rows)} rows; changed: {changed or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "public_questions")
