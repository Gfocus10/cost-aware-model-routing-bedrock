"""judge.py - LLM-as-judge: scores one answer from 1 to 5 against a reference or rubric.

Design choices:
- The judge is BLIND: it never sees which model wrote the answer, so it can't favor a tier.
- It scores against a reference answer or rubric, not against its own opinion.
- temperature is 0 so the same answer gets the same score.
- Judge cost is tracked separately: it is evaluation overhead, not part of the cost a
  router would incur in production.
- Known limitation: if the judge model is also one of the models being tested, it may
  prefer its own style.
"""
import json
import re

from models import call_model

JUDGE_TIER = "strong"

JUDGE_SYSTEM = (
    "You are a strict, fair grader for a customer-support assistant. "
    "You compare a candidate answer to a reference answer or a rubric. "
    "Reply with JSON only, no other text."
)

SCALE = (
    "Scoring scale:\n"
    "5 = fully correct and complete\n"
    "4 = correct, with a minor omission or imprecision\n"
    "3 = partly correct; an important detail is missing or wrong\n"
    "2 = mostly wrong or misleading\n"
    "1 = wrong, irrelevant, or empty"
)


def _expected_block(q: dict) -> str:
    """Format the reference answer or rubric for the judge prompt."""
    if q.get("answer_type") == "rubric":
        items = "\n".join(f"- {r}" for r in q["rubric"])
        return f"Rubric (the answer should satisfy these):\n{items}"
    ref = q["reference"]
    if not isinstance(ref, str):
        ref = json.dumps(ref)
    return f"Reference answer:\n{ref}"


def build_judge_prompt(q: dict, answer: str) -> str:
    return (
        f"Question given to the assistant:\n{q['question']}\n\n"
        f"{_expected_block(q)}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"{SCALE}\n\n"
        'Return JSON exactly like: {"score": <1-5>, "reason": "<one short sentence>"}'
    )


def _parse(text: str):
    """Pull {"score": n, "reason": "..."} out of the judge's reply. None if unusable."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    score = obj.get("score")
    if isinstance(score, (int, float)) and 1 <= score <= 5:
        return int(round(score)), str(obj.get("reason", ""))
    return None


def judge_answer(q: dict, answer: str) -> dict:
    """Score one answer. Retries once if the reply isn't valid JSON."""
    prompt = build_judge_prompt(q, answer)
    cost = 0.0
    for _ in range(2):
        r = call_model(JUDGE_TIER, prompt, max_tokens=200, system=JUDGE_SYSTEM)
        cost += r["cost_usd"]
        parsed = _parse(r["text"])
        if parsed:
            return {"score": parsed[0], "reason": parsed[1], "judge_cost_usd": cost}
    return {"score": None, "reason": "judge reply could not be parsed", "judge_cost_usd": cost}


if __name__ == "__main__":
    # Sanity check: a correct answer should score high, a wrong one low.
    sample = {
        "question": "A customer is about to place a $38 order and asks how much standard shipping will cost.",
        "answer_type": "short_fact",
        "reference": "$5.99 (the order is under the $50 free-shipping threshold).",
    }
    for label, ans in [("correct", "Standard shipping will cost $5.99."),
                       ("wrong", "Shipping is free on your order.")]:
        res = judge_answer(sample, ans)
        print(f"{label:8} -> score {res['score']} | {res['reason']}")
