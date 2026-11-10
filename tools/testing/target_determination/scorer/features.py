"""The feature rows the learned TD scorer reads, built in the TD job from the
heuristics' own results, test-times.json and the change. Each name and its
definition match pytorch/test-infra's trainer (td_scorer.json, format 1)."""

from __future__ import annotations

import math
import statistics
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.utils import (
    get_merge_base,
    get_pr_number,
    is_docs_only_change,
    query_changed_files,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from tools.testing.target_determination.heuristics.interface import (
        AggregatedHeuristics,
    )


# feature -> the heuristic class whose own score it is
SERVED_FEATURES: dict[str, str] = {
    "prev_failed": "PreviouslyFailedInPR",
    "edited": "EditedByPR",
    "mentioned": "MentionedInPR",
    "fail_n": "CorrelatedWithHistoricalFailures",
    "llm_n": "LLM",
    "filepath_s": "Filepath",
    "public_bindings": "PublicBindings",
}
LLM_FEATURE = "llm_n"
AREAS: dict[str, tuple[str, ...]] = {
    "area_core_cpp": ("aten/", "c10/", "torch/csrc/"),
    "area_inductor": ("torch/_inductor/",),
    "area_dynamo": ("torch/_dynamo/",),
    "area_distributed": ("torch/distributed/",),
    "area_build": (
        "cmake/",
        "CMakeLists.txt",
        "setup.py",
        "pyproject.toml",
        "requirements",
    ),
    "area_ci": (".github/", ".ci/"),
}
CHANGE_FEATURES = ("n_files", "n_src_files", "n_test_files", "docs_only", *AREAS)
UNIT_FEATURES = ("is_pr", *CHANGE_FEATURES)
MODEL_FEATURES = frozenset([*SERVED_FEATURES, "test_seconds", *UNIT_FEATURES])
PATH_FEATURES = ("path_revert_max", "path_revert_mean")
RISK_FEATURES = frozenset(
    [
        "log_files",
        "log_src_files",
        "n_test_files",
        "docs_only",
        *AREAS,
        "log_sig_added",
        "log_sig_deleted",
        *PATH_FEATURES,
    ]
)
MAX_PATH_NODES = 20000
MAX_PATH_KEY_CHARS = 512
MAX_CHANGED_PATHS = 10000
MIN_PATH_RATE = 1e-6

NON_SIGNIFICANT_PREFIXES = ("test/", "docs/", "third_party/", ".github/")
NON_SIGNIFICANT_SUFFIXES = (".md", ".rst", ".txt", ".ipynb", ".expect")
GIT_TIMEOUT_SECONDS = 60.0


class UnregisteredHeuristic(Exception):
    pass


@dataclass(frozen=True)
class Paths:
    """The risk block's per-path revert rates: `base` for a file none of
    whose nodes is listed."""

    base: float
    nodes: Mapping[str, float]


def _rate(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: not a number")
    if not 0 <= value <= 1:
        raise ValueError(f"{what}: not in [0, 1]")
    return float(value)


def parse_paths(value: object) -> Paths:
    """Validates the risk block's "paths"; raises ValueError with a short
    reason."""
    if not isinstance(value, dict):
        raise ValueError("not an object")
    base = _rate(value.get("base"), "base")
    nodes = value.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("nodes: not an object")
    if len(nodes) > MAX_PATH_NODES:
        raise ValueError(f"{len(nodes)} nodes")
    for key, rate in nodes.items():
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= MAX_PATH_KEY_CHARS
            or any(ord(c) < 32 or ord(c) == 127 for c in key)
        ):
            raise ValueError("invalid node key")
        _rate(rate, "node")
    return Paths(base, {key: float(rate) for key, rate in nodes.items()})


def path_rate(paths: Paths, path: str) -> float:
    """The rate of the first node listed among the file itself, then its
    directories from the deepest; a node longer than MAX_PATH_KEY_CHARS is
    never a key."""
    nodes = paths.nodes
    if len(path) <= MAX_PATH_KEY_CHARS and path in nodes:
        return nodes[path]
    end = path.rfind("/")
    while end >= 0:
        if end < MAX_PATH_KEY_CHARS and path[: end + 1] in nodes:
            return nodes[path[: end + 1]]
        end = path.rfind("/", 0, end)
    return paths.base


def path_values(paths: Paths, files: Collection[str]) -> dict[str, float]:
    """PATH_FEATURES over the change's files (de-duplicated, sorted, the
    first MAX_CHANGED_PATHS); NaN without files."""
    unique = sorted(set(files))[:MAX_CHANGED_PATHS]
    if not unique:
        return dict.fromkeys(PATH_FEATURES, math.nan)
    rates = [path_rate(paths, f) for f in unique]
    return {
        "path_revert_max": math.log(max(MIN_PATH_RATE, *rates)),
        "path_revert_mean": math.log(max(MIN_PATH_RATE, math.fsum(rates) / len(rates))),
    }


def is_significant(path: str) -> bool:
    return not path.startswith(NON_SIGNIFICANT_PREFIXES) and not path.endswith(
        NON_SIGNIFICANT_SUFFIXES
    )


def change_stats(changed_files: Sequence[str]) -> dict[str, float]:
    """The per-unit change features of CHANGE_FEATURES."""
    stats = {
        "n_files": float(len(changed_files)),
        "n_src_files": float(sum(is_significant(f) for f in changed_files)),
        "n_test_files": float(sum(f.startswith("test/") for f in changed_files)),
        "docs_only": float(is_docs_only_change(list(changed_files))),
    }
    for area, prefixes in AREAS.items():
        stats[area] = float(any(f.startswith(prefixes) for f in changed_files))
    return stats


def parse_numstat(text: str) -> tuple[float, float]:
    """Added and deleted lines of the significant files in `git diff
    --numstat -z -M` output. A rename ("a<TAB>d<TAB>", then old and new path
    as their own fields) counts under its new path; binary files ("-"
    counts) count 0."""
    added, deleted = 0, 0
    fields = text.split("\0")
    i = 0
    while i < len(fields):
        parts = fields[i].split("\t", 2)
        i += 1
        if len(parts) != 3:
            continue
        a, d, path = parts
        if not path:
            if i + 1 >= len(fields):
                break
            path = fields[i + 1]
            i += 2
        if is_significant(path):
            added += int(a) if a.isdigit() else 0
            deleted += int(d) if d.isdigit() else 0
    return float(added), float(deleted)


def run_git(args: list[str], timeout: float) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, timeout=timeout
    ).stdout


@dataclass
class Sources:
    """The side-effecting inputs, injectable for tests."""

    pr_number: Callable[[], int | None] = get_pr_number
    changed_files: Callable[[], list[str]] = query_changed_files
    merge_base: Callable[[], str] = get_merge_base
    git: Callable[[list[str], float], str] = run_git


class Unit:
    """The per-unit (per change) inputs, each read at most once."""

    def __init__(self, sources: Sources | None = None) -> None:
        self.sources = sources if sources is not None else Sources()
        self._changed: list[str] | None = None
        self._stats: dict[str, float] | None = None

    def is_pr(self) -> float:
        return 1.0 if self.sources.pr_number() is not None else 0.0

    def changed_files(self) -> list[str]:
        if self._changed is None:
            self._changed = [f for f in self.sources.changed_files() if f]
        return self._changed

    def stats(self) -> dict[str, float]:
        if self._stats is None:
            self._stats = change_stats(self.changed_files())
        return self._stats

    def value(self, name: str) -> float:
        return self.is_pr() if name == "is_pr" else self.stats()[name]

    def significant_lines(self) -> tuple[float, float]:
        """Added and deleted significant lines since the merge base; NaN when
        the diff cannot be read."""
        try:
            base = self.sources.merge_base()
            out = self.sources.git(
                ["diff", "--numstat", "-z", "-M", base, "HEAD"],
                GIT_TIMEOUT_SECONDS,
            )
        except Exception as e:
            print(f"TD scorer: could not read the diff's line counts: {e!r}")
            return math.nan, math.nan
        return parse_numstat(out)


def test_seconds(
    test_times: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, float]:
    """Per test file, the median of its seconds over every (build
    environment, test config) entry of test-times.json that lists it."""
    per_file: dict[str, list[float]] = {}
    for configs in test_times.values():
        for files in configs.values():
            for test_file, seconds in files.items():
                per_file.setdefault(test_file, []).append(float(seconds))
    return {f: statistics.median(v) for f, v in per_file.items()}


def served_scores(
    aggregated: AggregatedHeuristics,
    features: Collection[str],
    registered: Collection[str],
) -> dict[str, dict[str, float]]:
    """Per served feature in `features`, each test file's score in its
    heuristic's own results (the whole-file TestRun's); empty for a
    registered heuristic without results (it raised), so its features are
    0.0. Raises UnregisteredHeuristic for a feature whose heuristic is not
    in `registered`."""
    scores: dict[str, dict[str, float]] = {}
    for name in features:
        heuristic = SERVED_FEATURES.get(name)
        if heuristic is None:
            continue
        if heuristic not in registered:
            raise UnregisteredHeuristic(f"{name}: {heuristic} is not registered")
        results = aggregated.get_heuristic_results(heuristic)
        if results is None:
            scores[name] = {}
            continue
        scores[name] = {
            test.test_file: score
            for score, test in results.get_scored_tests()
            if test.is_full_file()
        }
    return scores


def build_rows(
    features: Sequence[str],
    tests: Sequence[str],
    aggregated: AggregatedHeuristics,
    test_times: Mapping[str, Mapping[str, Mapping[str, float]]],
    unit: Unit,
    registered: Collection[str],
) -> list[list[float]]:
    """One row per test of `tests`, with the values of `features` in order."""
    served = served_scores(aggregated, features, registered)
    seconds = test_seconds(test_times) if "test_seconds" in features else {}
    per_unit = {name: unit.value(name) for name in features if name in UNIT_FEATURES}
    rows = []
    for test in tests:
        row = []
        for name in features:
            if name in served:
                row.append(float(served[name].get(test, 0.0)))
            elif name == "test_seconds":
                row.append(float(seconds.get(test, 0.0)))
            elif name in per_unit:
                row.append(per_unit[name])
            else:
                raise ValueError(f"no definition for feature {name}")
        rows.append(row)
    return rows


def risk_values(
    names: Collection[str], unit: Unit, paths: Paths | None
) -> dict[str, float]:
    """The per-unit values the risk score reads, for the names it lists: the
    change alone, never the scorer's p."""
    wanted = set(names)
    values: dict[str, float] = {}
    if wanted & {"log_sig_added", "log_sig_deleted"}:
        added, deleted = unit.significant_lines()
        values["log_sig_added"] = math.log1p(added)
        values["log_sig_deleted"] = math.log1p(deleted)
    if wanted & {"log_files", "log_src_files", "n_test_files", "docs_only", *AREAS}:
        stats = unit.stats()
        values["log_files"] = math.log1p(stats["n_files"])
        values["log_src_files"] = math.log1p(stats["n_src_files"])
        for name in ("n_test_files", "docs_only", *AREAS):
            values[name] = stats[name]
    if paths is not None and wanted & set(PATH_FEATURES):
        values.update(path_values(paths, unit.changed_files()))
    return values
