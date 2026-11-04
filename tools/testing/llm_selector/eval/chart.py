"""Plot changes caught vs test time run on a typical change, from score.py --json rows.

    python -m tools.testing.llm_selector.eval.chart rows.json out.svg

Needs matplotlib, which the Test tools CI job's image does not install and no
requirements file pins. The SVG's path data depends on its version; REPORT.md's
chart was rendered with matplotlib 3.11.1.

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

from tools.testing.llm_selector.eval.score import summarize


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


def panels(rows: list[dict], src: object = "the rows") -> dict[str, dict]:
    """What each panel ("revert", "head") plots, from score.py --json rows:
    per model and effort its label, color, marker and (cpu %, caught %) per
    hedge, the hedge labels (on the proposed tag's head panel), the proposed
    and TD points, the frontier and the failing-sample count."""
    g = {(x["tag"], x["policy"], x["kind"]): x for x in summarize(rows)}

    def summary(tag: str, pol: str, kind: str) -> dict:
        if (tag, pol, kind) not in g:
            raise SystemExit(f"no {kind} rows for {tag} {pol} in {src}")
        return g[(tag, pol, kind)]

    def point(tag: str, pol: str, kind: str) -> tuple[float, float]:
        return 100 * summary(tag, pol, "neutral")["cpu"], caught(summary(tag, pol, kind), kind)[0]  # fmt: skip

    # fmt: off
    tags = sorted({t for t, _, _ in g if t != "td" and not t.endswith(("_r2", "_agentic"))})
    if unknown := [t for t in tags if t not in COLOR or t.rsplit("_", 1)[0] not in MODELS or t.rsplit("_", 1)[1] not in MARKER]:
        raise SystemExit(f"no name, color or marker for {unknown}; add them to MODELS, COLOR and MARKER")
    out = {}
    for kind in ["revert", "head"]:
        lines, labels = [], []
        for tag in tags:
            prefix, effort = tag.rsplit("_", 1)
            pols = [p for p in HEDGES if (tag, p, kind) in g]
            if not pols:
                raise SystemExit(f"no {kind} rows for {tag}")
            xy = [point(tag, p, kind) for p in pols]
            lines.append({"label": f"{MODELS[prefix]} {effort}", "color": COLOR[tag], "marker": MARKER[effort], "xy": xy})
            if tag == PRODUCTION[0] and kind == "head":
                labels = [(f"+{HEDGES[p]}% hedge" if HEDGES[p] else "no hedge", q) for p, q in zip(pols, xy)]
        td = point("td", "td25", kind)
        points = [q for line in lines for q in line["xy"]] + [td]
        n = caught(summary("td", "td25", kind), kind)[1]
        out[kind] = {"lines": lines, "labels": labels, "star": point(*PRODUCTION, kind), "td": td, "frontier": frontier(points), "n": n, "xmax": max(x for x, _ in points)}
    # fmt: on
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("rows", type=Path, help="score.py --json rows")
    p.add_argument("out", type=Path, help="the SVG to write")
    args = p.parse_args()
    data = panels(json.loads(args.rows.read_text()), args.rows)

    import matplotlib

    # Labels stay text, so they can be diffed against REPORT.md.
    with matplotlib.rc_context({"svg.hashsalt": "llm_td", "svg.fonttype": "none"}):
        svg = render(data)
    # matplotlib ends path lines with a space, which the SPACES linter rejects.
    args.out.write_text("".join(line.rstrip() + "\n" for line in svg.splitlines()))


def render(data: dict[str, dict]) -> str:
    from matplotlib.figure import Figure

    # fmt: off
    titles = {"revert": "Reverted commits caught", "head": "Failing PR pushes with every real failure selected"}
    fig = Figure(figsize=(11, 4.8))
    axes = fig.subplots(1, 2)
    for ax, (kind, d) in zip(axes, data.items()):
        for line in d["lines"]:
            ax.plot(*zip(*line["xy"]), color=line["color"], marker=line["marker"], markersize=5, linewidth=1, label=line["label"])
        for text, q in d["labels"]:
            ax.annotate(text, q, textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.scatter(*d["star"], s=220, marker="*", color="gold", edgecolors="black", zorder=5, label="proposed: Opus medium + 20% hedge")
        ax.axhline(d["td"][1], color="#d62728", linestyle=":", linewidth=1, zorder=0, label="TD today's level")
        ax.scatter(*d["td"], s=70, marker="X", color="#d62728", zorder=5, label="TD today (top 25% of TestRuns)")
        ax.plot(*zip(*d["frontier"]), color="gray", linestyle="--", linewidth=1, zorder=0, label="Pareto frontier")
        ax.set_title(f"{titles[kind]} (n={d['n']})", fontsize=10)
        ax.set_xlabel("CPU test time run on a typical change (% of a full pass, in seconds)")
        ax.set_xlim(0, max(55, 5 + d["xmax"]))
        ax.grid(alpha=0.3)
    # fmt: on
    axes[0].set_ylabel("%")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": None})
    return svg.getvalue()


if __name__ == "__main__":
    main()
