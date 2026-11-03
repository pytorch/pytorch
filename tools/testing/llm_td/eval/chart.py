"""Plot changes caught vs GPU test time run, from score.py --json rows.

python -m tools.testing.llm_td.eval.chart rows.json out.svg
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib


matplotlib.use("svg")
import matplotlib.pyplot as plt


LABEL = {"v2_opus55": "Opus 5.5", "v2_sonnet5": "Sonnet 5", "v2_haiku45": "Haiku 4.5"}
COLOR = {
    "Opus 5.5": "#1f77b4",
    "Sonnet 5": "#2ca02c",
    "Haiku 4.5": "#ff7f0e",
    "TD (top 25%)": "#d62728",
}


def main() -> None:
    rows = json.loads(Path(sys.argv[1]).read_text())
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["tag"], r["policy"], r["kind"])].append(r)
    failing = {
        k: len({r["id"] for r in rows if r["kind"] == k and r["pairs"]})
        for k in ("pr", "revert")
    }
    titles = {
        "pr": f"PR-caused failures (n={failing['pr']} PRs)",
        "revert": f"Autoreverted commits (n={failing['revert']})",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, kind in zip(axes, ["pr", "revert"]):
        for (tag, pol, k), xs in sorted(g.items()):
            if (
                k != kind
                or pol not in ("final", "final+td5", "td25")
                or tag.endswith(("_r2", "_agentic"))
            ):
                continue
            fail = [x for x in xs if x["pairs"]]
            y = 100 * sum(x["hit"] > 0 for x in fail) / len(fail)
            x = 100 * st.mean(x["gpu"] for x in xs)
            if tag == "td":
                name, eff = "TD (top 25%)", ""
            else:
                prefix, eff = tag.rsplit("_", 1)
                name = LABEL[prefix]
            hedged = pol == "final+td5"
            ax.scatter(x, y, s=40, marker="s" if tag == "td" else "o", label=name + " + 5% TD hedge" if hedged else name,
                       facecolors="none" if hedged else COLOR[name], edgecolors=COLOR[name])  # fmt: skip
            ax.annotate(
                eff, (x, y), textcoords="offset points", xytext=(5, -3), fontsize=8
            )
        ax.set_title(titles[kind], fontsize=10)
        ax.set_xlabel("GPU test time run (% of a full pass)")
        ax.set_xlim(0, 70)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("changes with at least one failure caught (%)")
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(sys.argv[2], metadata={"Date": None})


if __name__ == "__main__":
    main()
