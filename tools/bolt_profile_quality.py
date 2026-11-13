#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path


_PROFILED_RE = re.compile(
    r"BOLT-INFO: (\d+) out of (\d+) functions in the binary .*"
    r"have non-empty execution profile"
)
_STALE_FUNCTIONS_RE = re.compile(
    r"BOLT-(?:WARNING|ERROR): (\d+) \([^)]*of all profiled\) functions? have invalid"
)
_STALE_SAMPLES_RE = re.compile(
    r"BOLT-(?:WARNING|ERROR): (\d+) out of (\d+) samples in the binary .*"
    r"belong to functions with invalid"
)
_INFERRED_RE = re.compile(r"BOLT-INFO: inferred profile for (\d+) ")
_MATCH_RE = re.compile(
    r"BOLT-INFO: inference found an? (exact match|call match|loose match) "
    r"for [0-9.]+% of basic blocks "
    r"\((\d+) out of (\d+) stale\) responsible for [0-9.]+% samples "
    r"\((\d+) out of (\d+) stale\)"
)


_TABLE_HEADER = """\
Library                      |  Functions |           Branches |     Blocks |        Block execs"""
_TABLE_TEMPLATE = """\
-----------------------------+------------+--------------------+------------+--------------------
{library:<28} | {binary:>10} |                    |            |
|_ With profiles             | {profiled:>10} | {total_edges:>18} |            |
   |_ Fresh profiles         | {valid_funcs:>10} | {valid_edges:>18} |            |
   |_ Stale profiles         | {stale_funcs:>10} | {stale_edges:>18} |            |
      |_ Inferred Functions  | {recovered:>10} |                    | {total_blocks:>10} | {total_execs:>18}
         |_ Exact match      |            |                    | {exact_blocks:>10} | {exact_execs:>18}
         |_ Call match       |            |                    | {call_blocks:>10} | {call_execs:>18}
         |_ Loose match      |            |                    | {loose_blocks:>10} | {loose_execs:>18}
"""

_REPORT_GUIDANCE = """\
Percentages are relative to the parent value in the same column.

"Inferred Functions" refers to the number of functions that received profiles after stale
profile matching (inference). The "Blocks" count in that row represent the total number of basic
blocks in inferred functions and NOT the number of successfully matched basic blocks. Block
matching numbers are shown in the children rows and are used to judge the quality of inference.

Exact and call matches are good indicators of correspondence with the current binary. Loose
matches use weaker heuristics and give less confidence in the recovered profile. A high share of
loose matches, especially in block execs, is a reason to consider refreshing the profiles.
"""


def _counts(
    pattern: re.Pattern[str],
    text: str,
    description: str,
    default: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    match = pattern.search(text)
    if match is None:
        if default is not None:
            return default
        raise ValueError(f"BOLT log does not contain {description}")
    return tuple(int(value) for value in match.groups())


def _percent(value: int, total: int) -> str:
    return f"{value / total:.1%}" if total else "-"


def parse_bolt_log(path: Path) -> tuple[int, str]:
    text = path.read_text(encoding="utf-8")
    profiled, binary = _counts(_PROFILED_RE, text, "profiled function counts")
    (stale,) = _counts(_STALE_FUNCTIONS_RE, text, "stale function counts", default=(0,))
    # LLVM omits raw edge totals without staleness; -1 sorts unknown totals last.
    stale_edges, total_edges = _counts(
        _STALE_SAMPLES_RE, text, "edge counts", default=(0, -1)
    )
    inferred = _INFERRED_RE.search(text)
    recovered = int(inferred.group(1)) if inferred else 0
    matches = {
        match.group(1): tuple(int(value) for value in match.groups()[1:])
        for match in _MATCH_RE.finditer(text)
    }
    total_blocks = max((counts[1] for counts in matches.values()), default=0)
    total_execs = max((counts[3] for counts in matches.values()), default=0)
    values = {
        "library": path.stem.replace("llvm-bolt-", "", 1) + ".so",
        "binary": f"{binary:,}",
        "profiled": f"{profiled:,}",
        "total_edges": f"{total_edges:,}" if total_edges >= 0 else "-",
        "valid_funcs": _percent(profiled - stale, profiled),
        "valid_edges": _percent(total_edges - stale_edges, total_edges)
        if total_edges >= 0
        else "-",
        "stale_funcs": _percent(stale, profiled),
        "stale_edges": _percent(stale_edges, total_edges) if total_edges >= 0 else "-",
        "recovered": _percent(recovered, stale),
        "total_blocks": f"{total_blocks:,}" if matches else "",
        "total_execs": f"{total_execs:,}" if matches else "",
    }
    for name in ("exact", "call", "loose"):
        blocks, _, block_execs, _ = matches.get(f"{name} match", (0, 0, 0, 0))
        values[f"{name}_blocks"] = _percent(blocks, total_blocks)
        values[f"{name}_execs"] = _percent(block_execs, total_execs)
    return total_edges, _TABLE_TEMPLATE.format(**values).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print BOLT profile quality by descending edge count"
    )
    parser.add_argument("log_dir", type=Path, help="BOLT log directory")
    args = parser.parse_args()
    logs = sorted(args.log_dir.glob("llvm-bolt-*.txt"))
    if not logs:
        raise ValueError("No llvm-bolt-*.txt logs found")
    libraries = [parse_bolt_log(path) for path in logs]
    libraries.sort(key=lambda item: (-item[0], item[1]))
    print(_TABLE_HEADER)
    print("\n".join(table for _, table in libraries))
    print("\n" + _REPORT_GUIDANCE.rstrip())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Warning: BOLT profile quality summary failed: {error}", file=sys.stderr)
