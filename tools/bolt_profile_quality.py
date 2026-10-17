#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import tabulate
import yaml


# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]

_PROFILED_RE = re.compile(
    r"BOLT-INFO: (?P<functions>\d+) out of (?P<binary_functions>\d+) "
    r"functions in the binary .* have non-empty execution profile"
)
_STALE_FUNCTIONS_RE = re.compile(
    r"BOLT-(?:WARNING|ERROR): (?P<functions>\d+) "
    r"\([^)]*of all profiled\) functions? have invalid"
)
_STALE_SAMPLES_RE = re.compile(
    r"BOLT-(?:WARNING|ERROR): (?P<stale>\d+) out of "
    r"(?P<matched>\d+) samples in the binary .*"
    r"belong to functions with invalid"
)
_NON_SIMPLE_RE = re.compile(
    r"BOLT-INFO: (?P<functions>\d+) functions? with profile could not be optimized"
)
_IGNORED_RE = re.compile(
    r"BOLT-INFO: profile for (?P<functions>\d+) objects was ignored"
)
_INFERRED_RE = re.compile(
    r"BOLT-INFO: inferred profile for (?P<functions>\d+) .* samples "
    r"\((?P<samples>\d+) out of (?P<matched_samples>\d+)\)"
)
_MATCH_RE = re.compile(
    r"BOLT-INFO: inference found an? (?P<kind>exact|call|loose) match "
    r"for [0-9.]+% of basic blocks "
    r"\((?P<blocks>\d+) out of (?P<total_blocks>\d+) stale\) "
    r"responsible for [0-9.]+% samples "
    r"\((?P<block_execs>\d+) out of (?P<total_block_execs>\d+) stale\)"
)


_HEADERS = (
    "Profile",
    "Functions",
    "%",
    "Samples",
    "%",
    "Blocks",
    "%",
    "Block execs",
    "%",
)
TableRow = list[str | int | float | None]

_REPORT_GUIDANCE = """\
"""


def _counts(
    pattern: re.Pattern[str],
    text: str,
) -> dict[str, int]:
    if match := pattern.search(text):
        return {name: int(value) for name, value in match.groupdict().items()}
    return {}


def _safediv(value: int, total: int) -> float | None:
    return value / total if total > 0 else None


def read_profile_yaml(profile_path: Path) -> tuple[int, int]:
    with profile_path.open(encoding="utf-8") as stream:
        profile = yaml.load(stream, Loader=Loader)
    functions = len(profile["functions"])
    samples = sum(
        edge.get("cnt", 0)
        for function in profile["functions"]
        for block in function.get("blocks", [])
        for edge in block.get("succ", [])
    )
    return functions, samples


def read_profile_quality(profile_path: Path, log_path: Path) -> list[TableRow]:
    # read the optimization log
    text = log_path.read_text(encoding="utf-8")
    profiled = _counts(_PROFILED_RE, text)
    non_simple = _counts(_NON_SIMPLE_RE, text)
    ignored = _counts(_IGNORED_RE, text)
    stale = _counts(_STALE_FUNCTIONS_RE, text)
    samples = _counts(_STALE_SAMPLES_RE, text)
    inferred = _counts(_INFERRED_RE, text)
    matches = {
        match["kind"]: {
            "blocks": int(match["blocks"]),
            "total_blocks": int(match["total_blocks"]),
            "block_execs": int(match["block_execs"]),
            "total_block_execs": int(match["total_block_execs"]),
        }
        for match in _MATCH_RE.finditer(text)
    }
    block_totals = {
        (counts["total_blocks"], counts["total_block_execs"])
        for counts in matches.values()
    }
    if len(block_totals) != 1:
        raise ValueError(f"{log_path}: inconsistent inference block totals")
    total_blocks, total_execs = block_totals.pop()

    # read the yaml profile and sanity check that bolt optimization log
    # accounts for the same number of functions as present in the profile
    yaml_functions, yaml_samples = read_profile_yaml(profile_path)
    profiled_functions = profiled["functions"]
    ignored_functions = ignored["functions"] + non_simple["functions"]
    logged_functions = profiled_functions + ignored_functions
    if yaml_functions != logged_functions:
        print(
            f"Warning: {profile_path}: {yaml_functions} profile entries do not match {log_path}: "
            f"{profiled_functions} profiled + {non_simple['functions']} non-simple + "
            f"{ignored['functions']} ignored",
            file=sys.stderr,
        )

    # construct the table
    rows: list[TableRow] = [[profile_path.name, yaml_functions, None, yaml_samples]]

    matched_samples = samples["matched"]
    ignored_samples = yaml_samples - matched_samples
    rows.append(
        [
            "|_ Ignored + Non-Simple",
            ignored_functions,
            _safediv(ignored_functions, yaml_functions),
            ignored_samples,
            _safediv(ignored_samples, yaml_samples),
        ]
    )

    rows.append(
        [
            "|_ Matched",
            profiled_functions,
            _safediv(profiled_functions, yaml_functions),
            matched_samples,
            _safediv(matched_samples, yaml_samples),
        ]
    )

    stale_functions = stale["functions"]
    stale_samples = samples["stale"]
    fresh_functions = profiled_functions - stale_functions
    fresh_samples = matched_samples - stale_samples
    rows.append(
        [
            "   |_ Fresh profiles",
            fresh_functions,
            _safediv(fresh_functions, profiled_functions),
            fresh_samples,
            _safediv(fresh_samples, matched_samples),
        ]
    )
    rows.append(
        [
            "   |_ Stale profiles",
            stale_functions,
            _safediv(stale_functions, profiled_functions),
            stale_samples,
            _safediv(stale_samples, matched_samples),
        ]
    )

    rows.append(
        [
            "      |_ Inferred Functions",
            inferred["functions"],
            _safediv(inferred["functions"], stale_functions),
            inferred["samples"],
            _safediv(inferred["samples"], stale_samples),
            total_blocks,
            None,
            total_execs,
        ]
    )

    for name, counts in matches.items():
        rows.append(
            [
                f"         |_ {name.capitalize()} match",
                None,
                None,
                None,
                None,
                counts["blocks"],
                _safediv(counts["blocks"], counts["total_blocks"]),
                counts["block_execs"],
                _safediv(counts["block_execs"], counts["total_block_execs"]),
            ]
        )
    return rows


_REPORT_GUIDANCE = """\
Print BOLT profile quality by looking at the yaml profiles used for optimization and the optimization logs.

=== EXAMPLE ===

Profile                        Functions       %          Samples       %    Blocks      %     Block execs      %
---------------------------  -----------  ------  ---------------  ------  --------  -----  --------------  -----
libtorch_python.yaml               6,811          321,563,664,624
|_ Ignored + Non-Simple            2,350   34.5%   20,789,057,795    6.5%
|_ Matched                         4,461   65.5%  300,774,606,829   93.5%
   |_ Fresh profiles                 814   18.2%   41,414,527,046   13.8%
   |_ Stale profiles               3,647   81.8%  259,360,079,783   86.2%
      |_ Inferred Functions        3,647  100.0%  259,360,079,783  100.0%    52,106         47,090,218,678
         |_ Exact match                                                      13,928  26.7%   6,877,780,011  14.6%
         |_ Call match                                                        3,568   6.8%   4,213,401,480   8.9%
         |_ Loose match                                                      19,490  37.4%  35,983,392,509  76.4%

=== GUIDANCE ===

Percentages are relative to the parent value in the same column.

"Non-Simple" functions cannot be optimized by bolt. Reasons include unsupported relocations, jump table handling,
failed disassembly or control flow reconstruction. "Ignored" functions are profile entries that were not found
in the binary, this happens when bolt's heuristics for function matching fails, eg function renames. "Matched"
function are the ones that are candidates for optimization by bolt.

"Inferred Functions" refers to the number of functions that received profiles after stale profile matching
(inference). The "Blocks" count in that row represent the total number of basic blocks in inferred functions and
NOT the number of successfully matched basic blocks. Block matching numbers are shown in the children rows and
are used to judge the quality of inference.

Exact and call matches are good indicators of correspondence with the current binary. Loose matches use weaker
heuristics and give less confidence in the recovered profile.

High values for the following can signal if the profiles need to be updated: "Ignored + Non-Simple" sample %,
"Stale profile" sample %, and "Loose match" block exec %.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=_REPORT_GUIDANCE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        nargs=2,
        action="append",
        required=True,
        type=Path,
        metavar=("YAML", "LOG"),
        help="Profile and corresponding optimization log; repeat for additional profiles",
    )
    args = parser.parse_args()

    rows = []
    for profile_path, log_path in args.profile:
        try:
            report_rows = read_profile_quality(profile_path, log_path)
        except ValueError as error:
            print(f"Warning: skipping {profile_path.name}: {error}", file=sys.stderr)
            continue
        rows.extend(report_rows)
        rows.append([])  # empty row

    tabulate.PRESERVE_WHITESPACE = True
    print(tabulate.tabulate(rows, headers=_HEADERS, intfmt=",", floatfmt=".1%"))
    print(f"Use `{Path(__file__).resolve()} --help` for guidance on using this report.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Warning: BOLT profile quality summary failed: {error}", file=sys.stderr)
