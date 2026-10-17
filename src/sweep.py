"""sweep.py - evaluate a difficulty-threshold router using the results we already have.

Routing rule:  difficulty <= T1 -> cheap,  T1 < difficulty <= T2 -> mid,  difficulty > T2 -> strong.

    python src/sweep.py public_questions pilot_questions     # tune on the DEV sets
    python src/sweep.py public_test --apply 2,4              # ONE final run on the TEST set

A question counts as passed when its score is >= 4.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_excluded(stem: str) -> set:
    """Question IDs listed in data/excluded.json (bad keys / ambiguous wording), skipped for ALL tiers."""
    path = ROOT / "data" / "excluded.json"
    return set(json.loads(path.read_text(encoding="utf-8")).get(stem, {})) if path.exists() else set()
TIERS = ("cheap", "mid", "strong")
PASS_SCORE = 4


def read(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Run the baselines / classify.py for that set first.")
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def load_items(stems: list[str]) -> list[dict]:
    items = []
    for stem in stems:
        tiers = {t: read(RESULTS / f"{stem}_{t}.jsonl") for t in TIERS}
        diff = read(RESULTS / f"{stem}_difficulty.jsonl")
        skip = load_excluded(stem)
        for qid in set.intersection(*(set(d) for d in tiers.values()), set(diff)) - skip:
            rows = [tiers[t][qid] for t in TIERS]
            if any(r.get("score") is None or r.get("stop_reason") == "content_filtered" for r in rows):
                continue  # excluded for all tiers
            if diff[qid].get("difficulty") is None:
                continue
            items.append({
                "key": f"{stem}:{qid}", "d": diff[qid]["difficulty"], "cls_cost": diff[qid]["cost_usd"],
                "passed": {t: tiers[t][qid]["score"] >= PASS_SCORE for t in TIERS},
                "cost": {t: tiers[t][qid]["cost_usd"] for t in TIERS},
            })
    return items


def route(d: int, t1: int, t2: int) -> str:
    return "cheap" if d <= t1 else ("mid" if d <= t2 else "strong")


def evaluate(items: list[dict], t1: int, t2: int) -> dict:
    n = len(items)
    passed, cost, split = 0, 0.0, {t: 0 for t in TIERS}
    for it in items:
        tier = route(it["d"], t1, t2)
        split[tier] += 1
        passed += it["passed"][tier]
        cost += it["cost"][tier] + it["cls_cost"]
    return {"pass": passed / n, "cost": cost, "split": split}


def auc(items: list[dict]) -> float | None:
    """Chance that a question the cheap model FAILS gets a higher difficulty than one it passes (0.5 = useless)."""
    fails = [it["d"] for it in items if not it["passed"]["cheap"]]
    oks = [it["d"] for it in items if it["passed"]["cheap"]]
    if not fails or not oks:
        return None
    wins = sum((f > o) + 0.5 * (f == o) for f in fails for o in oks)
    return wins / (len(fails) * len(oks))


def baselines(items: list[dict]) -> None:
    n = len(items)
    strong_cost = sum(it["cost"]["strong"] for it in items)
    print(f"{'config':<22}{'pass rate':>10}{'total cost':>13}{'% of strong':>13}")
    for t in TIERS:
        rate = sum(it["passed"][t] for it in items) / n
        cost = sum(it["cost"][t] for it in items)
        print(f"{'always-' + t:<22}{rate:>10.1%}{cost:>13.6f}{cost / strong_cost:>13.1%}")
    o_pass, o_cost = 0, 0.0
    for it in items:
        pick = next((t for t in TIERS if it["passed"][t]), None)
        o_pass += pick is not None
        o_cost += it["cost"][pick or "cheap"]
    print(f"{'oracle (ceiling)':<22}{o_pass / n:>10.1%}{o_cost:>13.6f}{o_cost / strong_cost:>13.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--apply", default=None, help='evaluate ONE config, e.g. "2,4" (T1,T2)')
    args = ap.parse_args()

    items = load_items(args.stems)
    n = len(items)
    if n == 0:
        raise SystemExit("No usable questions found. Run classify.py for each set first (and make sure "
                         "no ratings failed), and check the three baseline result files exist.")
    strong_cost = sum(it["cost"]["strong"] for it in items)
    print(f"Sets: {args.stems} | {n} questions | cheap-model failures: {sum(not it['passed']['cheap'] for it in items)}\n")
    baselines(items)
    a = auc(items)
    print(f"\nDoes difficulty predict cheap-model failure? AUC = {a:.2f}  (0.5 = no signal, 1.0 = perfect)"
          if a is not None else "\nAUC unavailable (need both passes and failures)")

    if args.apply:
        t1, t2 = (int(x) for x in args.apply.split(","))
        configs = [(t1, t2)]
    else:
        configs = [(t1, t2) for t1 in range(1, 5) for t2 in range(t1, 6)]
    results = [(c, evaluate(items, *c)) for c in configs]

    # Pareto frontier: a config is kept only if no cheaper config has an equal or better pass rate.
    frontier = set()
    best = -1.0
    for c, r in sorted(results, key=lambda cr: (cr[1]["cost"], -cr[1]["pass"])):
        if r["pass"] > best:
            frontier.add(c)
            best = r["pass"]

    print(f"\n{'router (T1,T2)':<16}{'pass rate':>10}{'total cost':>13}{'% of strong':>13}   routed cheap/mid/strong")
    for c, r in sorted(results, key=lambda cr: cr[1]["cost"]):
        s = r["split"]
        mark = "  <- Pareto" if c in frontier and not args.apply else ""
        print(f"{str(c):<16}{r['pass']:>10.1%}{r['cost']:>13.6f}{r['cost'] / strong_cost:>13.1%}"
              f"   {s['cheap']}/{s['mid']}/{s['strong']}{mark}")


if __name__ == "__main__":
    main()
