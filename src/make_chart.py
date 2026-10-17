"""make_chart.py - cost-vs-quality chart and summary table from the saved results. No model calls.

    python src/make_chart.py

Reads results/ for the dev set (public_questions) and the held-out test set (public_test), computes
every setup (always-X, oracle, the two difficulty routers, the agreement check) and writes:
    results/cost_vs_quality.png   one panel per set; up-and-to-the-left is better
    results/summary.csv           the same numbers as a table.
"""
import sweep
import agreement
import matplotlib.pyplot as plt
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed


OUT_DIR = sweep.RESULTS
SETS = [("public_questions", "Dev set"), ("public_test", "Test set (unseen)")]
DIFFICULTY_ROUTERS = [("difficulty router (3,4)", (3, 4)),
                      ("difficulty router (3,3)", (3, 3))]


def compute_rows(items: list[dict]) -> list[dict]:
    """One dict per setup: name, kind, pass_rate (%), cost_usd, cost_pct (of always-strong)."""
    n = len(items)
    strong_cost = sum(it["cost"]["strong"] for it in items)
    rows = []

    def add(name, kind, passed, cost):
        rows.append({"name": name, "kind": kind, "pass_rate": 100 * passed / n,
                     "cost_usd": cost, "cost_pct": 100 * cost / strong_cost})

    for t in sweep.TIERS:
        add(f"always-{t}", "baseline", sum(it["passed"][t]
            for it in items), sum(it["cost"][t] for it in items))
    o_pass, o_cost = 0, 0.0
    for it in items:
        pick = next((t for t in sweep.TIERS if it["passed"][t]), None)
        o_pass += pick is not None
        o_cost += it["cost"][pick or "cheap"]
    add("oracle (ceiling)", "oracle", o_pass, o_cost)
    for name, (t1, t2) in DIFFICULTY_ROUTERS:
        r = sweep.evaluate(items, t1, t2)
        add(name, "difficulty", r["pass"] * n, r["cost"])
    r = agreement.cascade(items)
    add("agreement check", "agreement", r["pass"] * n, r["cost"])
    return rows


STYLE = {"baseline": ("#888888", "o", 70), "oracle": ("#000000", "D", 70),
         "difficulty": ("#1f77b4", "s", 80), "agreement": ("#2ca02c", "*", 260)}
# label offsets in points, so neighbouring labels don't collide
OFFSETS = {"always-cheap": (-8, -16), "always-mid": (8, -16), "always-strong": (-8, -16),
           "oracle (ceiling)": (8, 8), "difficulty router (3,4)": (-8, 8),
           "difficulty router (3,3)": (8, -4), "agreement check": (10, 6)}


def plot(rows_by_set: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(rows_by_set), figsize=(
        7 * len(rows_by_set), 5.8), sharey=True)
    axes = [axes] if len(rows_by_set) == 1 else list(axes)
    low = min(r["pass_rate"] for rows in rows_by_set.values() for r in rows)
    for ax, (title, rows) in zip(axes, rows_by_set.items()):
        for r in rows:
            color, marker, size = STYLE[r["kind"]]
            ax.scatter(r["cost_pct"], r["pass_rate"], c=color,
                       marker=marker, s=size, zorder=3)
            dx, dy = OFFSETS.get(r["name"], (6, 6))
            ax.annotate(f"{r['name']}\n{r['pass_rate']:.1f}% | {r['cost_pct']:.1f}%", (r["cost_pct"], r["pass_rate"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=8,
                        ha="right" if dx < 0 else "left")
        ax.set_xscale("log")
        ax.set_xlim(0.3, 400)
        ax.set_ylim(low - 3, 102.5)
        ax.set_xlabel("Cost as % of always-Sonnet (log scale)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Pass rate (%)")
    fig.suptitle(
        "Model routing: quality vs cost  (up and to the left is better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def main() -> None:
    rows_by_set = {}
    for stem, label in SETS:
        items = agreement.load(stem)
        if not items:
            print(f"(no usable results for {stem}, skipping)")
            continue
        rows_by_set[f"{label} - {len(items)} questions"] = compute_rows(items)
    if not rows_by_set:
        raise SystemExit(
            "No results found. Run the baselines and classify.py first.")

    OUT_DIR.mkdir(exist_ok=True)
    plot(rows_by_set, OUT_DIR / "cost_vs_quality.png")
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "setup", "pass_rate_pct",
                   "cost_usd", "cost_pct_of_strong"])
        for title, rows in rows_by_set.items():
            for r in rows:
                w.writerow(
                    [title, r["name"], f"{r['pass_rate']:.1f}", f"{r['cost_usd']:.6f}", f"{r['cost_pct']:.1f}"])
    for title, rows in rows_by_set.items():
        print(
            f"\n**{title}**\n\n| Setup | Pass rate | Cost vs always-Sonnet |\n|---|---|---|")
        for r in rows:
            print(
                f"| {r['name']} | {r['pass_rate']:.1f}% | {r['cost_pct']:.1f}% |")
    print(
        f"\nWrote {OUT_DIR / 'cost_vs_quality.png'} and {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
