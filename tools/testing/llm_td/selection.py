"""The test selection an LLM makes for one change, and how CI applies it.

A selection names test files (paths under test/ without .py), whole test
directories, and wrapper sweeps (configs such as dynamo_wrapped that re-run
ordinary test files in a special mode). A test file runs in a given CI config
when the selection names the file, names that config as a sweep, or is run_all,
and so does every test file the model was not shown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.edited_by_pr import (
    ADDITIONAL_MAPPINGS,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


SWEEPS = ("dynamo_wrapped", "crossref")
LLM_TD_SELECTION_FILE = "llm_td_selection.json"
# The selector runs from the default branch and run_test.py from the PR, so
# bump this whenever an older reader would misread the selection file.
SELECTION_VERSION = 1
# Longest chain of test modules importing each other that importers() follows.
MAX_IMPORT_HOPS = 8
# More edited failure-list classes than this are not looked up one by one.
MAX_FAILURE_LIST_CLASSES = 50

# Tests that run the files a change matches although they do not import them:
# the drivers of the directories discover_tests.py blocklists (except
# torch_openreg/tests, whose driver test_openreg is never a candidate), and
# TD's ADDITIONAL_MAPPINGS.
DRIVER_MAPPINGS = {
    **ADDITIONAL_MAPPINGS,
    # run_test.py runs test_cpp_extensions_aot.py as both aot drivers.
    **{
        t: [*ADDITIONAL_MAPPINGS[t], r"^test/test_cpp_extensions_aot\.py$"]
        for t in ("test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja")
    },  # fmt: skip
    "test_ao_sparsity": [r"^test/ao/"],
    "test_autograd": [r"^test/autograd/"],
    "test_fx": [r"^test/fx/"],
    "test_jit": [r"^test/jit/"],
    "test_package": [r"^test/package/"],
    "test_quantization": [r"^test/quantization/"],
}

# common_utils xfails or skips a test named in one of these lists, one file per
# test such as TestNNCPU.test_x or CPython313-test_set-TestSet.test_x, when it
# runs under dynamo (the dynamo_wrapped config), inductor or compiled autograd.
FAILURE_LIST_DIR_RE = re.compile(
    r"^test/(dynamo|inductor|compiled_autograd)_(?:expected_failures|skips)/"
)
FAILURE_LIST_RE = re.compile(FAILURE_LIST_DIR_RE.pattern + r"(?:.*-)?(\w+)\.[^./]+$")
# instantiate_device_type_tests names TestFooCPU after TestFoo.
DEVICE_SUFFIX_RE = re.compile(r"(CPU|CUDA|MPS|XPU|HPU|META|LAZY|PRIVATEUSE1)$")
# assertExpected reads test/expect/<TestClass>.<test>[-<name>].expect.
EXPECT_RE = re.compile(r"^test/expect/(\w+)\.")

# Changes to these paths alter how every test is collected or run, so no
# subset is safe whatever the model says.
RUN_ALL_PATTERNS = [
    r"^test/run_test\.py$",
    r"^test/(.*/)?conftest\.py$",
    # Loaded by every test process; it moves tests between configs.
    r"^test/slow_tests\.json$",
    r"^pytest\.ini$",
    r"^torch/testing/_internal/common_utils\.py$",
    r"^torch/testing/_internal/common_device_type\.py$",
    # Loads the failure lists for every dynamo and inductor wrapped run.
    r"^torch/testing/_internal/dynamo_test_failures\.py$",
    r"^tools/testing/",
    r"^test/pytest_shard_custom\.py$",
    r"^tools/stats/(import_test_stats|upload_metrics|upload_test_stats|upload_stats_lib|test_dashboard)\.py$",
    r"^\.ci/pytorch/(test|multigpu-test|win-test|macos-test|macos-common|common|common-build|common_utils|torch_trace)\.sh$",
    r"^\.ci/pytorch/win-test-helpers/(setup_pytorch_env\.bat|test_python_shard\.bat|installation-helpers/activate_miniconda3\.bat)$",
    r"^\.github/workflows/_(linux|win|mac|rocm|xpu|s390x|tpu)-test(-no-dind)?\.yml$",
    # setup-linux is left out: the test jobs use it from main, not from the PR.
    r"^\.github/actions/(setup-rocm|setup-win|setup-xpu|filter-test-configs|download-td-artifacts)/",
    r"^\.github/scripts/filter_test_configs\.py$",
    # The pytest plugin every test job loads.
    r"^torch/testing/_internal/torchci/",
    # Installed by .ci/pytorch/common_utils.sh for the test jobs.
    r"^\.github/ci_commit_pins/(audio|vision|spmd_types)\.txt$",
    # The CI test images: a new package version or toolchain can break any test.
    r"^\.ci/docker/(build\.sh|requirements-ci\.txt|common/|ci_commit_pins/|ubuntu)",
]


def normalize_test_name(name: str, candidates: set[str]) -> str | None:
    """Map a model-written name such as 'test/dynamo/test_misc.py [3m]' onto a
    candidate, and a file that a driver runs, such as jit/test_freezing, onto the driver."""
    n = re.sub(r"\s*\[[^\]]*\]$", "", name.strip())
    n = n.split("::")[0].removeprefix("./").removeprefix("test/").removesuffix(".py")
    if n in candidates:
        return n
    n = n.replace(".", "/")
    if n in candidates:
        return n
    path = f"test/{n}.py"
    drivers = (t for t, regexes in DRIVER_MAPPINGS.items() if t in candidates and any(re.search(r, path) for r in regexes))  # fmt: skip
    return next(drivers, None)


def edited_tests(changed_files: list[str], candidates: set[str]) -> set[str]:
    """The candidates the change edits, and those that run an edited file, such as test_jit for test/jit/*."""
    edited = {
        n
        for f in changed_files
        if f.startswith("test/") and (n := normalize_test_name(f, candidates))
    }
    for test, regexes in DRIVER_MAPPINGS.items():
        if test in candidates and any(re.search(r, f) for r in regexes for f in changed_files):  # fmt: skip
            edited.add(test)
    return edited


def importers(
    sources: list[str], candidates: set[str], grep: Callable[[str], list[str]]
) -> set[str] | None:
    """The candidates among `sources` (paths under test/), the candidates that
    import one of them directly or through other modules under test/, and their
    drivers: test/jit/test_freezing.py selects test_jit, which runs it, and
    test_jit_legacy, which does `from test_jit import *`. `grep` lists the paths
    under test/ that match a `git grep -E` pattern at the change's head. None if
    a chain of importers reaches MAX_IMPORT_HOPS modules."""
    found: set[str] = set()
    seen: set[str] = set()
    frontier = {f for f in sources if f.startswith("test/") and f.endswith(".py")}
    for hop in range(MAX_IMPORT_HOPS + 1):
        tests = edited_tests(sorted(frontier), candidates)
        found |= tests
        frontier = (frontier | {f"test/{t}.py" for t in tests}) - seen
        if not frontier:
            return found
        if hop == MAX_IMPORT_HOPS:
            return None
        seen |= frontier
        # A module is imported by any dotted suffix of its path, a package by its
        # directory, and load_test_module (a SourceFileLoader) takes the path.
        # A parenthesized `from <package> import (` may span lines, so it counts
        # as importing every frontier module of that package.
        mods = sorted(f[5:-3].removesuffix("/__init__") for f in frontier)
        parts = [m.split("/") for m in mods]
        names = alts(".".join(p[i:]) for p in parts for i in range(len(p)))
        bases = alts(p[-1] for p in parts)
        parents = alts(".".join(p[i:-1]) for p in parts for i in range(len(p) - 1))
        paths = alts(n for m in mods for n in (m, m.replace("/", ".")))
        end = "([^[:alnum:]_]|$)"
        forms = [
            rf"(from|import)[[:space:]]+\.*({names}){end}",
            rf"from[[:space:]]+\.+[[:space:]]+import[[:space:]]+(.*[^[:alnum:]_.])?({bases}){end}",
        ]
        if parents:
            forms.append(rf"from[[:space:]]+\.*({parents})[[:space:]]+import[[:space:]]*(\(|[[:space:]](.*[^[:alnum:]_.])?({bases}){end})")  # fmt: skip
        pattern = rf"""^[[:space:]]*({"|".join(forms)})|load_test_module\(["']({paths})["']\)"""
        frontier = {f for f in grep(pattern) if f.endswith(".py")}
    return None


def alts(names: Iterable[str]) -> str:
    """An extended regex alternation of the literal `names`."""
    return "|".join(re.sub(r"[.^$*+?()\[\]{}|\\]", r"\\\g<0>", n) for n in sorted(set(names)))  # fmt: skip


def class_pattern(classes: Iterable[str]) -> str:
    """A `git grep -E` pattern for the definition of one of `classes` or of
    the generic class instantiate_device_type_tests made it from."""
    names = alts(n for c in classes for n in (c, DEVICE_SUFFIX_RE.sub("", c)))
    return rf"^[[:space:]]*class[[:space:]]+({names})([^[:alnum:]_]|$)"


def data_file_readers(changed_files: list[str], grep: Callable[[str], list[str]]) -> list[str]:  # fmt: skip
    """The paths under test/ that may read an edited non-.py file under test/
    other than a failure list: those naming the file and, for an expect file
    (see EXPECT_RE), those defining its test class. A file that no test file
    names, such as a C++ test or a README, is taken to be read by none."""
    data = [f for f in changed_files if f.startswith("test/") and not f.endswith(".py") and not FAILURE_LIST_DIR_RE.match(f)]  # fmt: skip
    names = alts(f.rsplit("/", 1)[-1] for f in data)
    readers = grep(rf"(^|[^[:alnum:]_.-])({names})([^[:alnum:]_.-]|$)") if data else []
    if classes := {m[1] for f in data if (m := EXPECT_RE.match(f))}:
        readers += grep(class_pattern(classes))
    return readers


def failure_list_sources(
    changed_files: list[str], grep: Callable[[str], list[str]] | None
) -> tuple[list[str], set[str]]:
    """The paths under test/ that define the classes named by the edited
    failure-list entries (see FAILURE_LIST_RE), and the lists ("dynamo",
    "inductor" or "compiled_autograd") naming a class that `grep` (see
    importers) does not find or holding an entry without exactly one ".", which
    dynamo_test_failures.py rejects at import for the dynamo and inductor lists."""
    classes: dict[str, set[str]] = {}
    unparsed: set[str] = set()
    for f in changed_files:
        if not (d := FAILURE_LIST_DIR_RE.match(f)):
            continue
        if d[1] != "compiled_autograd" and f.rsplit("/", 1)[-1].count(".") != 1:
            unparsed.add(d[1])
        elif m := FAILURE_LIST_RE.match(f):
            classes.setdefault(m[2], set()).add(m[1])
    if grep is None or len(classes) > MAX_FAILURE_LIST_CLASSES:
        return [], unparsed.union(*classes.values())
    sources: list[str] = []
    missing = unparsed
    for c, lists in sorted(classes.items()):
        found = grep(class_pattern([c]))
        sources += found
        if not found:
            missing |= lists
    return sources, missing


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


def parse_model_output(out: object, candidates: set[str]) -> tuple[set[str], set[str]]:
    """The files and sweeps a model answer names; ValueError if it is malformed.

    A `dirs` entry is a directory prefix such as "dynamo"; one that matches no
    candidate is malformed, and so is a `tests` name that matches no
    candidate: the model was told to copy the names from the candidate list."""
    if not isinstance(out, dict):
        raise ValueError("not an object")
    tests, dirs, sweeps = out.get("tests"), out.get("dirs", []), out.get("sweeps", [])
    if not isinstance(tests, list) or not all(isinstance(t, dict) and isinstance(t.get("file"), str) for t in tests):  # fmt: skip
        raise ValueError("tests is not a list of objects with a string file")
    for key, value in (("dirs", dirs), ("sweeps", sweeps)):
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"{key} is not a list of strings")
    if unknown := set(sweeps) - set(SWEEPS):
        raise ValueError(f"unknown sweeps {sorted(unknown)}")
    if not isinstance(out.get("run_all", False), bool):
        raise ValueError("run_all is not a boolean")
    names = {t["file"]: normalize_test_name(t["file"], candidates) for t in tests}
    if unmatched := sorted(t for t, n in names.items() if n is None):
        raise ValueError(f"tests entries match no test file: {unmatched[:10]}")
    files = {n for n in names.values() if n}
    for d in dirs:
        prefix = re.sub(r"(/\*\*?)?/*$", "", d.strip()).removeprefix("./").removeprefix("test/")  # fmt: skip
        in_dir = {c for c in candidates if c.startswith(prefix + "/")}
        if not in_dir:
            raise ValueError(f"dirs entry {d!r} matches no test file")
        files |= in_dir
    return files, set(sweeps)


@dataclass
class Selection:
    files: set[str] = field(default_factory=set)
    sweeps: set[str] = field(default_factory=set)
    run_all: bool = False
    reason: str = ""
    # Whether run_test.py should skip the files the selection does not cover,
    # rather than only reporting what it would have skipped.
    enforce: bool = False
    # The test files the model chose from. Any other file, such as a custom
    # run_test.py handler or a C++ test, is always covered. None covers no extra file.
    candidates: set[str] | None = None

    @classmethod
    def from_model_output(
        cls,
        out: object,
        candidates: set[str],
        changed_files: list[str],
        grep: Callable[[str], list[str]] | None = None,
    ) -> Selection:
        """Turn the model's structured output into a Selection.

        A missing or malformed output, a change to test infrastructure, an
        edited inductor failure list naming an unknown class, and too many
        import hops all fall back to running everything. Test files the change
        edits and their drivers are always selected, and with `grep` so are the
        test files that import them (see importers), those defining the tests
        that an edited failure list names (see failure_list_sources), and those
        reading an edited data file (see data_file_readers), with their importers."""
        if out is None:
            return cls(run_all=True, reason="no model output; running everything")
        if needs_run_all(changed_files):
            return cls(run_all=True, reason="change touches test infrastructure")
        try:
            files, sweeps = parse_model_output(out, candidates)
        except ValueError as e:
            return cls(run_all=True, reason=f"malformed model output ({e}); running everything")  # fmt: skip
        if not isinstance(out, dict):
            raise AssertionError("parse_model_output accepts only objects")
        sources, missing = failure_list_sources(changed_files, grep)
        if "inductor" in missing:
            return cls(run_all=True, reason="an edited inductor failure list names an unknown test class")  # fmt: skip
        # dynamo_wrapped also runs with compiled autograd on (see common_utils).
        if missing & {"dynamo", "compiled_autograd"}:
            sweeps.add("dynamo_wrapped")
        files |= edited_tests(changed_files, candidates)
        if grep is not None:
            readers = data_file_readers(changed_files, grep)
            found = importers(changed_files + sources + readers, candidates, grep)
            if found is None:
                return cls(run_all=True, reason="test modules import the change through too many hops")  # fmt: skip
            files |= found
        run_all, reason = out.get("run_all", False), str(out.get("reason", ""))
        return cls(files, sweeps, run_all, reason, candidates=set(candidates))

    def covers(self, test_file: str, test_config: str) -> bool:
        if self.candidates is not None and test_file not in self.candidates:
            return True
        return self.run_all or test_config in self.sweeps or test_file in self.files

    def to_json(self) -> dict:
        return {
            "version": SELECTION_VERSION,
            "files": sorted(self.files),
            "sweeps": sorted(self.sweeps),
            "run_all": self.run_all,
            "reason": self.reason,
            "enforce": self.enforce,
            "candidates": None if self.candidates is None else sorted(self.candidates),
        }

    @classmethod
    def from_json(cls, d: dict) -> Selection:
        """The selection in `d`; ValueError, KeyError or TypeError unless `d`
        is a selection file of SELECTION_VERSION."""
        if not isinstance(d, dict) or type(d.get("version")) is not int or d["version"] != SELECTION_VERSION:  # fmt: skip
            raise ValueError(f"not a selection file of version {SELECTION_VERSION}")
        files, sweeps, candidates = d["files"], d["sweeps"], d["candidates"]
        run_all, enforce, reason = d["run_all"], d["enforce"], d["reason"]
        if not all(
            isinstance(v, list) and all(isinstance(x, str) for x in v)
            for v in (files, sweeps) + (() if candidates is None else (candidates,))
        ):
            raise TypeError("files, sweeps and candidates must be lists of strings")
        if not isinstance(run_all, bool) or not isinstance(enforce, bool):
            raise TypeError("run_all and enforce must be booleans")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        return cls(
            set(files),
            set(sweeps),
            run_all,
            reason,
            enforce,
            None if candidates is None else set(candidates),
        )

    @classmethod
    def load(cls, path: Path) -> Selection | None:
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except (OSError, ValueError, KeyError, TypeError):
            return None
