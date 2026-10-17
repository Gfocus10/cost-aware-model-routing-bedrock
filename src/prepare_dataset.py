"""prepare_dataset.py - download public benchmarks and sample them into our JSONL format.

Run once:  python src/prepare_dataset.py
Output:    data/public_questions.jsonl

Sources (all test splits, sampled with a fixed seed so anyone can reproduce your set):
  - ARC-Easy        multiple-choice science   -> labelled "easy"
  - ARC-Challenge   multiple-choice science   -> labelled "hard"
  - GSM8K           grade-school math, numeric answer -> "medium" or "hard" by number of steps
These difficulty labels are only guesses. The real difficulty comes from how the
cheap model actually performs.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCD"


def sample_arc(config: str, n: int, rng: random.Random, prefix: str, difficulty: str, exclude: set) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    # Keep only 4-option questions so every answer is A-D.
    idxs = [i for i, row in enumerate(ds)
            if len(row["choices"]["text"]) == 4
            and f"allenai/ai2_arc/{config}/test#{i}" not in exclude]
    rows = []
    for k, i in enumerate(rng.sample(idxs, n), start=1):
        row = ds[i]
        labels = row["choices"]["label"]
        letter = LETTERS[labels.index(row["answerKey"])]
        options = "\n".join(f"{LETTERS[j]}) {t}" for j, t in enumerate(row["choices"]["text"]))
        rows.append({
            "id": f"{prefix}{k:02d}",
            "difficulty_guess": difficulty,
            "category": f"science_{difficulty}",
            "policy_ids": [],
            "question": f"{row['question']}\n\n{options}\n\nReply with only the single letter (A, B, C, or D) and nothing else. Do not explain.",
            "answer_type": "mcq",
            "reference": letter,
            "max_tokens": 400,  # headroom: some models explain first, and a cut-off reply scores as wrong
            "source": f"allenai/ai2_arc/{config}/test#{i}",
        })
    return rows


def sample_gsm8k(n: int, rng: random.Random, exclude: set) -> list[dict]:
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    pool = [i for i in range(len(ds)) if f"openai/gsm8k/main/test#{i}" not in exclude]
    for k, i in enumerate(rng.sample(pool, n), start=1):
        row = ds[i]
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        steps = row["answer"].count("<<")  # each calculation in the solution is marked with <<...>>
        rows.append({
            "id": f"GSM{k:02d}",
            "difficulty_guess": "hard" if steps >= 4 else "medium",
            "category": "math_word_problem",
            "policy_ids": [],
            "question": (f"{row['question']}\n\nSolve this step by step, keeping it brief. "
                         "End with a final line in exactly this form: Final answer: <number>"),
            "answer_type": "number",
            "reference": answer,
            "max_tokens": 500,
            "source": f"openai/gsm8k/main/test#{i}",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-easy", type=int, default=30, help="ARC-Easy questions")
    ap.add_argument("--n-challenge", type=int, default=30, help="ARC-Challenge questions")
    ap.add_argument("--n-gsm", type=int, default=30, help="GSM8K questions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "public_questions.jsonl"))
    ap.add_argument("--exclude", default=None,
                    help="an existing questions file; its source questions are never sampled again "
                         "(use this to build a separate TEST set that shares nothing with your DEV set)")
    args = ap.parse_args()

    exclude = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            exclude = {json.loads(l)["source"] for l in f if l.strip() and "source" in json.loads(l)}
        print(f"Excluding {len(exclude)} questions already used in {args.exclude}")

    rng = random.Random(args.seed)
    rows = (
        sample_arc("ARC-Easy", args.n_easy, rng, "ARCE", "easy", exclude)
        + sample_arc("ARC-Challenge", args.n_challenge, rng, "ARCC", "hard", exclude)
        + sample_gsm8k(args.n_gsm, rng, exclude)
    )
    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} questions to {args.out}")


if __name__ == "__main__":
    main()
