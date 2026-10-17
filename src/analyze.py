"""analyze.py - how much could a router save? Compares static tiers with an 'oracle' router.

    python src/analyze.py                      # both question sets
    python src/analyze.py public_questions     # one set

A question counts as PASSED when its score is >= 4 (exact-match questions score 5 or 1).
The oracle picks, for every question, the CHEAPEST tier that passed it. No real router can
beat it (it peeks at the answers), so it is the ceiling for what routing can achieve.
Questions that Bedrock's content filter blocked for ANY tier are excluded for ALL tiers,
so every tier is compared on the same questions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_excluded(stem: str) -> set:
    """Question IDs listed in data/excluded.json (bad keys / ambiguous wording), skipped for ALL tiers."""
    path = ROOT / "data" / "excluded.json"
    return set(json.loads(path.read_text(encoding="utf-8")).get(stem, {})) if path.exists() else set()
TIERS = ("cheap", "mid", "strong")  # cheapest first
PASS_SCORE = 4


def load(stem: str) -> dict:
    data = {}
    for tier in TIERS:
        path = RESULTS / f"{stem}_{tier}.jsonl"
        if not path.exists():
            print(f"  (missing {path.name}, skipping set)")
            return {}
        data[tier] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                data[tier][r["id"]] = r
    return data


def analyze(stem: str) -> None:
    print(f"\n===== {stem} =====")
    data = load(stem)
    if not data:
        return
    ids = set.intersection(*(set(d) for d in data.values()))
    manual = sorted(ids & load_excluded(stem))
    if manual:
        print(f"Excluded by hand (see data/excluded.json): {manual}")
    ids -= set(manual)
    excluded = sorted(
        i for i in ids
        if any(data[t][i].get("stop_reason") == "content_filtered" or data[t][i].get("score") is None
               for t in TIERS)
    )
    ids = sorted(ids - set(excluded))
    n = len(ids)
    if excluded:
        print(f"Excluded for all tiers (filtered or errored in some tier): {excluded}")
    print(f"Compared on {n} questions\n")

    def passed(t, i):
        return data[t][i]["score"] >= PASS_SCORE

    print(f"{'config':<14}{'pass rate':>10}{'total cost':>14}{'vs strong cost':>16}")
    strong_cost = sum(data["strong"][i]["cost_usd"] for i in ids)
    for t in TIERS:
        rate = sum(passed(t, i) for i in ids) / n
        cost = sum(data[t][i]["cost_usd"] for i in ids)
        print(f"always-{t:<7}{rate:>10.1%}{cost:>14.6f}{cost / strong_cost:>15.1%}")

    oracle_pass, oracle_cost, chosen = 0, 0.0, {t: 0 for t in TIERS}
    for i in ids:
        pick = next((t for t in TIERS if passed(t, i)), None)
        if pick is None:           # nobody passes: a router would just use the cheapest
            pick = "cheap"
        else:
            oracle_pass += 1
        chosen[pick] += 1
        oracle_cost += data[pick][i]["cost_usd"]
    print(f"{'oracle router':<14}{oracle_pass / n:>10.1%}{oracle_cost:>14.6f}{oracle_cost / strong_cost:>15.1%}")
    print(f"\nOracle routing split: " + ", ".join(f"{t} {chosen[t]}" for t in TIERS))

    cheap_fail_strong_ok = [i for i in ids if not passed("cheap", i) and passed("strong", i)]
    strong_fail_cheap_ok = [i for i in ids if passed("cheap", i) and not passed("strong", i)]
    nobody = [i for i in ids if not any(passed(t, i) for t in TIERS)]
    print(f"Cheap fails but strong passes ({len(cheap_fail_strong_ok)}): {cheap_fail_strong_ok}")
    print(f"Strong fails but cheap passes ({len(strong_fail_cheap_ok)}): {strong_fail_cheap_ok}")
    print(f"Nobody passes ({len(nobody)}): {nobody}")


if __name__ == "__main__":
    for stem in (sys.argv[1:] or ["pilot_questions", "public_questions"]):
        analyze(stem)
