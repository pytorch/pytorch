"""Plot changes caught vs test time run on a typical change, from score.py --json rows.

python -m tools.testing.llm_td.eval.chart rows.json out.svg

Each model and effort level is a line through its hedge budgets (0% to 20%);
the dashed line is the Pareto frontier over every point. A revert counts when
any of its failures is selected (its PR would have gone red); a failing PR
head only when every one of its non-flaky failures is.
"""

from __future__ import annotations

import collections
import io
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib


matplotlib.use("svg")
matplotlib.rcParams["svg.hashsalt"] = "llm_td"
import matplotlib.pyplot as plt


MODELS = {"v2_opus55": "Opus 5.5", "v2_sonnet5": "Sonnet 5", "v2_haiku45": "Haiku 4.5"}
COLOR = {
    "v2_opus55_low": "#9ecae1", "v2_opus55_medium": "#3182bd", "v2_opus55_high": "#08306b",
    "v2_sonnet5_medium": "#74c476", "v2_sonnet5_high": "#006d2c", "v2_haiku45_low": "#fd8d3c",
}  # fmt: skip
MARKER = {"low": "o", "medium": "s", "high": "^"}
HEDGES = {"final": 0, "final+td5": 5, "final+td10": 10, "final+td15": 15, "final+td20": 20}  # fmt: skip
PRODUCTION = ("v2_opus55_medium", "final+td20")


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The points no other point beats on both lower x and higher y, by x."""
    out: list[tuple[float, float]] = []
    for x, y in sorted(points, key=lambda p: (p[0], -p[1])):
        if not out or y > out[-1][1]:
            out.append((x, y))
    return out


def main() -> None:
    rows = json.loads(Path(sys.argv[1]).read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["tag"], r["policy"], r["kind"])].append(r)

    def cost(tag: str, pol: str) -> float:
        return 100 * st.mean(x["cpu"] for x in g[(tag, pol, "neutral")])

    def caught(tag: str, pol: str, kind: str) -> tuple[float, int]:
        """Reverts with any failure selected; PR heads with every failure selected."""
        fail = [x for x in g[(tag, pol, kind)] if x["pairs"]]
        full = kind != "revert"
        ok = [x["hit"] == x["pairs"] if full else x["hit"] > 0 for x in fail]
        return 100 * sum(ok) / len(fail), len(fail)

    # fmt: off
    tags = sorted({t for t, _, _ in g if t != "td" and not t.endswith(("_r2", "_agentic"))})
    titles = {"revert": "Reverted commits caught", "head": "Failing PR pushes with every real failure selected"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, kind in zip(axes, ["revert", "head"]):
        points = []
        for tag in tags:
            prefix, effort = tag.rsplit("_", 1)
            name, color = MODELS[prefix], COLOR[tag]
            pols = [p for p in HEDGES if (tag, p, kind) in g]
            xy = [(cost(tag, p), caught(tag, p, kind)[0]) for p in pols]
            points += xy
            ax.plot(*zip(*xy), color=color, marker=MARKER[effort], markersize=5, linewidth=1, label=f"{name} {effort}")
            if tag == PRODUCTION[0]:
                for p, (x, y) in zip(pols, xy):
                    ax.annotate(f"{HEDGES[p]}%", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7)
        x, y = cost(*PRODUCTION), caught(*PRODUCTION, kind)[0]
        ax.scatter(x, y, s=220, marker="*", color="gold", edgecolors="black", zorder=5, label="proposed: Opus medium + 20% hedge")
        td_x, (td_y, n) = cost("td", "td25"), caught("td", "td25", kind)
        points.append((td_x, td_y))
        ax.axhline(td_y, color="#d62728", linestyle=":", linewidth=1, zorder=0, label="TD today's level")
        ax.scatter(td_x, td_y, s=70, marker="X", color="#d62728", zorder=5, label="TD today (top 25%)")
        fx, fy = zip(*frontier(points))
        ax.plot(fx, fy, color="gray", linestyle="--", linewidth=1, zorder=0, label="Pareto frontier")
        ax.set_title(f"{titles[kind]} (n={n})", fontsize=10)
        ax.set_xlabel("CPU test time run on a typical change (% of a full pass)")
        ax.set_xlim(0, 55)
        ax.grid(alpha=0.3)
    # fmt: on
    axes[0].set_ylabel("%")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    # matplotlib ends path lines with a space, which the SPACES linter rejects.
    lines = svg.getvalue().splitlines()
    Path(sys.argv[2]).write_text("".join(line.rstrip() + "\n" for line in lines))


if __name__ == "__main__":
    main()
