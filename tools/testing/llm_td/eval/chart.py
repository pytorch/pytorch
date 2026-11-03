"""Plot changes caught vs test time run on a typical change, from score.py --json rows.

    python -m tools.testing.llm_td.eval.chart rows.json out.svg

Needs matplotlib, which no requirements file installs; the SVG's path data
depends on its version.

Each model and effort level is a line through its hedge budgets (0% to 20%);
the dashed line is the Pareto frontier over every point. A revert counts when
any of its failures is selected (its PR would have gone red); a failing PR
head only when every one of its non-flaky failures is.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from tools.testing.llm_td.eval.score import summarize


MODELS = {"v3_opus55": "Opus 5.5", "v3_sonnet5": "Sonnet 5", "v3_haiku45": "Haiku 4.5"}
COLOR = {
    "v3_opus55_low": "#9ecae1", "v3_opus55_medium": "#3182bd", "v3_opus55_high": "#08306b",
    "v3_sonnet5_medium": "#74c476", "v3_sonnet5_high": "#006d2c", "v3_haiku45_low": "#fd8d3c",
}  # fmt: skip
MARKER = {"low": "o", "medium": "s", "high": "^"}
HEDGES = {"final": 0, "final+td5": 5, "final+td10": 10, "final+td15": 15, "final+td20": 20}  # fmt: skip
PRODUCTION = ("v3_opus55_medium", "final+td20")


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The points no other point beats on both lower x and higher y, by x."""
    out: list[tuple[float, float]] = []
    for x, y in sorted(points, key=lambda p: (p[0], -p[1])):
        if not out or y > out[-1][1]:
            out.append((x, y))
    return out


def caught(summary: dict, kind: str) -> tuple[float, int]:
    """% of a score.summarize row's failing samples caught, and their count: a
    revert with any failure selected, a PR head with every failure selected."""
    n = summary["n_failing"]
    if not n:
        raise SystemExit(f"no failing {kind} rows for {summary['tag']}")
    return 100 * summary["caught_any" if kind == "revert" else "caught_all"], n


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("rows", type=Path, help="score.py --json rows")
    p.add_argument("out", type=Path, help="the SVG to write")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("svg")
    matplotlib.rcParams["svg.hashsalt"] = "llm_td"
    # Labels stay text, so they can be diffed against REPORT.md.
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    rows = summarize(json.loads(args.rows.read_text()))
    g = {(x["tag"], x["policy"], x["kind"]): x for x in rows}

    def summary(tag: str, pol: str, kind: str) -> dict:
        if (tag, pol, kind) not in g:
            raise SystemExit(f"no {kind} rows for {tag} {pol} in {args.rows}")
        return g[(tag, pol, kind)]

    def cost(tag: str, pol: str) -> float:
        return 100 * summary(tag, pol, "neutral")["cpu"]

    def rate(tag: str, pol: str, kind: str) -> tuple[float, int]:
        return caught(summary(tag, pol, kind), kind)

    # fmt: off
    tags = sorted({t for t, _, _ in g if t != "td" and not t.endswith(("_r2", "_agentic"))})
    if unknown := [t for t in tags if t not in COLOR or t.rsplit("_", 1)[0] not in MODELS]:
        raise SystemExit(f"no name or color for {unknown}; add them to MODELS and COLOR")
    titles = {"revert": "Reverted commits caught", "head": "Failing PR pushes with every real failure selected"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, kind in zip(axes, ["revert", "head"]):
        points = []
        for tag in tags:
            prefix, effort = tag.rsplit("_", 1)
            pols = [p for p in HEDGES if (tag, p, kind) in g]
            if not pols:
                raise SystemExit(f"no {kind} rows for {tag}")
            xy = [(cost(tag, p), rate(tag, p, kind)[0]) for p in pols]
            points += xy
            ax.plot(*zip(*xy), color=COLOR[tag], marker=MARKER[effort], markersize=5, linewidth=1, label=f"{MODELS[prefix]} {effort}")
            if tag == PRODUCTION[0] and kind == "head":
                for p, (x, y) in zip(pols, xy):
                    ax.annotate(f"+{HEDGES[p]}% hedge" if HEDGES[p] else "no hedge", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
        x, y = cost(*PRODUCTION), rate(*PRODUCTION, kind)[0]
        ax.scatter(x, y, s=220, marker="*", color="gold", edgecolors="black", zorder=5, label="proposed: Opus medium + 20% hedge")
        td_x, (td_y, n) = cost("td", "td25"), rate("td", "td25", kind)
        points.append((td_x, td_y))
        ax.axhline(td_y, color="#d62728", linestyle=":", linewidth=1, zorder=0, label="TD today's level")
        ax.scatter(td_x, td_y, s=70, marker="X", color="#d62728", zorder=5, label="TD today (top 25% of files)")
        fx, fy = zip(*frontier(points))
        ax.plot(fx, fy, color="gray", linestyle="--", linewidth=1, zorder=0, label="Pareto frontier")
        ax.set_title(f"{titles[kind]} (n={n})", fontsize=10)
        ax.set_xlabel("CPU test time run on a typical change (% of a full pass, in seconds)")
        ax.set_xlim(0, 55)
        ax.grid(alpha=0.3)
    # fmt: on
    axes[0].set_ylabel("%")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None})
    # matplotlib ends path lines with a space, which the SPACES linter rejects.
    lines = svg.getvalue().splitlines()
    args.out.write_text("".join(line.rstrip() + "\n" for line in lines))


if __name__ == "__main__":
    main()
