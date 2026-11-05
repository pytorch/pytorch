"""The test selection an LLM makes for one change, and how CI applies it.

A selection names test files (paths under test/ without .py), never test
directories, and wrapper sweeps (configs such as dynamo_wrapped that re-run
ordinary test files in a special mode). A test file runs in a given CI config
when the selection names the file, names that config as a sweep, or is run_all,
and so does every test file the model was not shown and every file that
ALWAYS_COVERED lists for that config.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.edited_by_pr import (
    ADDITIONAL_MAPPINGS,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from pathlib import Path


SWEEPS = ("dynamo_wrapped", "crossref")
# The TEST_CONFIG globs (*glob*) of .ci/pytorch/test.sh's top-level dispatch,
# which runs the first branch a config matches, up to its *inductor* branch.
TEST_SH_GLOBS = (
    "numpy_2",
    "backward",
    "libtorch_agnostic_targetting",
    "torchtitan",
    "executorch",
    "operator_benchmark",
    "operator_microbenchmark",
    "attention_microbenchmark",
    "inductor_distributed",
    "inductor-halide",
    "inductor-flydsl",
    "inductor-pallas",
    "inductor-triton-cpu",
    "inductor-micro-benchmark",
    "inductor_better_benchmark",
    "aoti_cross_compile_for_windows",
    "huggingface",
    "timm",
    "torchbench",
    "inductor_cpp_wrapper",
    "inductor_aoti_fallback",
    "inductor_core",
    "inductor",
)
# The files the branches of those globs run with run_test.py --inductor, which
# compiles every test, so like a sweep any change can break them. A config gets
# the set of the first TEST_SH_GLOBS entry it contains.
ALWAYS_COVERED = {
    "inductor_cpp_wrapper": frozenset({"test_torch"}),
    "inductor": frozenset(
        {"test_modules", "test_ops", "test_ops_gradients", "test_torch"}
    ),
}
LLM_SELECTION_FILE = "llm_selection.json"
# The selector, run by the default branch's workflow, writes this file into
# ADDITIONAL_CI_FILES_FOLDER, and run_test.py reads it from the PR's checkout.
# The reader requires exactly its own SELECTION_VERSION and SWEEPS, so a bump
# or a new sweep turns selection off for unrebased PRs until they rebase.
SELECTION_VERSION = 1

# Tests that run the files a change matches although they do not import them:
# the drivers of the directories discover_tests.py blocklists (except
# custom_backend and mobile, which have none, and
# torch_openreg/tests, whose driver test_openreg is never a candidate), and
# TD's ADDITIONAL_MAPPINGS.
DRIVER_MAPPINGS = {
    # run_test.py runs test_cpp_extensions_aot.py as both aot drivers.
    **{
        t: [*ADDITIONAL_MAPPINGS[t], r"^test/test_cpp_extensions_aot\.py$"]
        for t in ("test_cpp_extensions_aot_ninja", "test_cpp_extensions_aot_no_ninja")
    },
    # run_test.py runs test_autoload.py, with the cpp_extensions installed, as both autoload drivers.
    **{
        t: [r"^test/test_autoload\.py$", r"^test/cpp_extensions/"]
        for t in ("test_autoload_enable", "test_autoload_disable")
    },
    # They build or load the extensions, which #include the headers there.
    "test_cpp_extensions_jit": [r"^test/cpp_extensions/"],
    "test_cpp_extensions_mtia_backend": [r"^test/cpp_extensions/"],
    "test_cpp_extensions_stream_and_event": [r"^test/cpp_extensions/"],
    "test_dispatch": [r"^test/cpp_extensions/dangling_impl_extension\.cpp$"],
    # They build these extensions.
    "cpp_extensions/test_libtorch_agnostic": [r"^test/cpp_extensions/libtorch_agn_"],
    "cpp_extensions/python_agnostic_extension/test/test_python_agnostic": [
        r"^test/cpp_extensions/python_agnostic_extension/"
    ],
    "cpp_extensions/libtorch_agn_2_10_extension/test_version_compatibility": [
        r"^test/cpp_extensions/libtorch_agn_2_(9|10)_extension/"
    ],
    "test_ao_sparsity": [r"^test/ao/"],
    "test_autograd": [r"^test/autograd/"],
    "test_fx": [r"^test/fx/"],
    # test_jit imports jit/test_backend_nnapi.py, which imports test_nnapi.py.
    "test_jit": [r"^test/jit/", r"^test/test_nnapi\.py$"],
    # They do `from test_jit import *`.
    **{
        t: [r"^test/jit/", r"^test/test_jit\.py$", r"^test/test_nnapi\.py$"]
        for t in ("test_jit_profiling", "test_jit_legacy")
    },
    "test_jit_fuser_te": [r"^test/jit/test_fuser_common\.py$"],
    # They import these blocklisted files, or run them as subprocesses.
    "test_jit_fuser_legacy": [r"^test/test_jit_fuser\.py$"],
    **{
        f"distributed/test_c10d_spawn_{b}": [r"^test/distributed/test_c10d_spawn\.py$"]
        for b in ("gloo", "nccl", "ucc")
    },
    "onnx/test_models_onnxruntime": [r"^test/onnx/test_models\.py$"],
    "distributed/launcher/test_run": [
        r"^test/distributed/launcher/bin/test_script(\.sh|(_init_method|_is_torchelastic_launched)?\.py)$"
    ],
    "distributed/test_launcher": [r"^test/distributed/bin/test_script\.py$"],
    "test_package": [r"^test/package/"],
    "test_quantization": [r"^test/quantization/"],
}

# Changes to these paths alter how every test is collected or run, so no
# subset is safe whatever the model says.
RUN_ALL_PATTERNS = [
    r"^test/run_test\.py$",
    r"^test/(.*/)?conftest\.py$",
    # Loaded by every test process; it moves tests between configs.
    r"^test/slow_tests\.json$",
    r"^pytest\.ini$",
    r"^conftest\.py$",
    r"^test/(.*/)?(\.?pytest\.ini|pyproject\.toml|tox\.ini|setup\.cfg)$",
    r"^torch/testing/_internal/common_(utils|device_type|dtype|cuda)\.py$",
    # torch.testing: assertEqual, make_tensor and freeze_rng_state.
    r"^torch/testing/(__init__|_comparison|_creation|_utils)\.py$",
    # Loads the failure lists for every dynamo and inductor wrapped run.
    r"^torch/testing/_internal/dynamo_test_failures\.py$",
    r"^tools/testing/",
    r"^test/pytest_shard_custom\.py$",
    r"^tools/stats/(import_test_stats|export_test_times|upload_metrics|upload_test_stats|upload_stats_lib|test_dashboard)\.py$",
    # The build scripts run from the PR in the build jobs and decide what the test jobs get.
    r"^\.ci/pytorch/(build|win-build|macos-build|rocm_utils|test|multigpu-test|win-test|macos-test|macos-common|common|common-build|common_utils|torch_trace)\.sh$",
    r"^\.ci/pytorch/(numba-cuda-13\.patch|generate_ck_config_h\.cmake)$",
    r"^tools/native_aot/build_stage2\.py$",
    r"^scripts/onnx/test\.sh$",
    r"^\.ci/pytorch/win-test-helpers/(build_pytorch\.bat|setup_pytorch_env\.bat|test_python_shard\.bat|test_python_jit_legacy\.bat|installation-helpers/(activate_miniconda3|install_magma|install_sccache)\.bat)$",
    r"^\.github/workflows/_(linux|win|mac|rocm|xpu|s390x|tpu)-test(-no-dind)?\.yml$",
    r"^\.github/workflows/_(linux|win|mac|s390x)-build\.yml$",
    # The workflows that call those test workflows; they set each job's build, image and test matrix.
    r"^\.github/workflows/(pull|trunk(-rocm-sandbox)?|periodic(-strict|-rocm-mi\d+)?|slow(-rocm-mi200)?|unstable|tsan|xpu|mac-mps|native-aot|dtensor|torchtitan|dynamo-unittest|td-holdout-unstable|s390x-periodic|rocm-[\w-]+|inductor[\w-]*|(b200|h100)-[\w-]+|test-(b200|gb300|h100)|better-benchmark-b200|attention_op_microbenchmark|operator_(benchmark|microbenchmark(_compare)?))\.yml$",
    # pull.yml and trunk.yml call these from the PR; they decide which tests TD drops.
    r"^\.github/workflows/(target_determination|llm_td_retrieval)\.yml$",
    # setup-linux is left out: the test jobs use it from main, not from the PR.
    r"^\.github/actions/(setup-rocm|setup-win|setup-xpu|filter-test-configs|download-td-artifacts|download-build-artifacts|reuse-old-whl|pytest-cache-download|pytest-cache-upload)/",
    r"^\.github/scripts/(file_io_utils|filter_test_configs|parse_ref|pytest_cache|pytest_caching_utils)\.py$",
    # The pytest plugin every test job loads.
    r"^torch/testing/_internal/torchci/",
    # Installed by .ci/pytorch/common_utils.sh, and torch_tpu by install_torch_tpu.sh, for the test jobs.
    r"^\.github/ci_commit_pins/(audio|vision|spmd_types|torch_tpu)\.txt$",
    # The CI test images: a new package version or toolchain can break any test.
    r"^\.ci/docker/(build\.sh|requirements-(ci|docs)\.txt|triton(_xpu)?_version\.txt|common/|ci_commit_pins/|java/|ubuntu(-rocm|-xpu)?/)",
]


def normalize_test_name(
    name: str, candidates: set[str], paths: Collection[str] = ()
) -> set[str]:
    """The candidates a model-written name such as 'test/dynamo/test_misc.py [3m]'
    names: the candidate, or the drivers of a file in `paths` (the files under
    test/) such as jit/test_freezing; empty if it names neither."""
    n = re.sub(r"\s*\[[^\]]*\]$", "", name.strip())
    n = n.split("::")[0].removeprefix("./").removeprefix("test/").removesuffix(".py")
    for m in (n, n.replace(".", "/")):
        if m in candidates:
            return {m}
        if f"test/{m}.py" in paths:
            return edited_tests([f"test/{m}.py"], candidates)
    return set()


def edited_tests(changed_files: list[str], candidates: set[str]) -> set[str]:
    """The candidates the change edits, and their drivers: the DRIVER_MAPPINGS
    tests that run an edited file, such as test_jit for test/jit/*, but not the
    candidates that import one."""
    edited = {
        n
        for f in changed_files
        if f.startswith("test/")
        for n in normalize_test_name(f, candidates)
    }
    for test, regexes in DRIVER_MAPPINGS.items():
        if test in candidates and any(
            re.search(r, f) for r in regexes for f in changed_files
        ):
            edited.add(test)
    return edited


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


def parse_model_output(
    out: object, candidates: set[str], grep: Callable[[str], list[str]] | None = None
) -> tuple[set[str], set[str]]:
    """The files and sweeps a model answer names; ValueError if it is malformed.

    `grep` takes a `git grep -E` pattern and lists the repo-relative paths, such
    as test/jit/test_freezing.py, of the files under test/ at the change's head
    with a line matching it, so grep("^") lists all but the empty ones, which
    hold no tests. Its exceptions propagate.

    A `dirs` entry is a directory prefix such as "dynamo", and selects the
    candidates under it and their drivers, such as test_jit for "jit"; one that
    selects no candidate is malformed, and so is a `tests` name that matches no
    candidate: the model was told to copy the names from the candidate list.
    With `grep`, which lists the paths under test/ that match a `git grep -E`
    pattern, a name of a file that a driver runs, such as jit/test_freezing,
    names the driver instead."""
    if not isinstance(out, dict):
        raise ValueError("not an object")
    if missing := sorted({"tests", "dirs", "sweeps", "run_all", "reason"} - out.keys()):
        raise ValueError(f"missing {missing}")
    tests, dirs, sweeps = out["tests"], out["dirs"], out["sweeps"]
    if not isinstance(tests, list) or not all(
        isinstance(t, dict) and isinstance(t.get("file"), str) for t in tests
    ):
        raise ValueError("tests is not a list of objects with a string file")
    for key, value in (("dirs", dirs), ("sweeps", sweeps)):
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f"{key} is not a list of strings")
    if unknown := set(sweeps) - set(SWEEPS):
        raise ValueError(f"unknown sweeps {sorted(unknown)}")
    if not isinstance(out["run_all"], bool):
        raise ValueError("run_all is not a boolean")
    if not isinstance(out["reason"], str):
        raise ValueError("reason is not a string")
    names = {t["file"]: normalize_test_name(t["file"], candidates) for t in tests}
    paths: set[str] = set()
    if grep is not None and (dirs or not all(names.values())):
        # Every file under test/ at the head.
        paths = set(grep("^"))
        names = {
            t: n or normalize_test_name(t, candidates, paths) for t, n in names.items()
        }
    if unmatched := sorted(t for t, n in names.items() if not n):
        raise ValueError(f"tests entries match no test file: {unmatched[:10]}")
    files = {f for n in names.values() for f in n}
    for d in dirs:
        prefix = (
            re.sub(r"(/\*\*?)?/*$", "", d.strip())
            .removeprefix("./")
            .removeprefix("test/")
        )
        in_dir = {c for c in candidates if c.startswith(prefix + "/")}
        under = [p for p in paths if p.startswith(f"test/{prefix}/")]
        in_dir |= edited_tests(under if grep else [f"test/{prefix}/"], candidates)
        if not in_dir:
            raise ValueError(f"dirs entry {d!r} matches no test file")
        files |= in_dir
    return files, set(sweeps)


@dataclass
class Selection:
    files: set[str] = field(default_factory=set)
    sweeps: set[str] = field(default_factory=set)
    # The model's own run_all, a fallback of from_model_output (see its
    # docstring), or one of select_tests.select() and main(), which set it
    # without asking the model or when the selector fails.
    run_all: bool = False
    reason: str = ""
    # Whether run_test.py should skip the files the selection does not cover,
    # rather than only reporting what it would have skipped.
    enforce: bool = False
    # The test files the model chose from. Any other file, such as a C++ test
    # or test_openreg, is always covered. None covers no extra file, so it
    # requires run_all.
    candidates: set[str] | None = None
    # Why the selector got no usable answer (git, the model call or its output,
    # or the test times); None for a deliberate run_all.
    error: str | None = None

    @classmethod
    def from_model_output(
        cls,
        out: object,
        candidates: set[str],
        changed_files: list[str],
        grep: Callable[[str], list[str]] | None = None,
    ) -> Selection:
        """Turn the model's structured output into a Selection.

        A missing or malformed output and a change to test infrastructure fall
        back to running everything; only a missing or malformed output sets
        error. Test files the change edits and their drivers are always
        selected."""
        if out is None:
            return cls(
                run_all=True,
                reason="no model output; running everything",
                error="no model output",
            )
        try:
            files, sweeps = parse_model_output(out, candidates, grep)
        except ValueError as e:
            return cls(
                run_all=True,
                reason=f"malformed model output ({e}); running everything",
                error=f"malformed model output: {e}",
            )
        if needs_run_all(changed_files):
            return cls(run_all=True, reason="change touches test infrastructure")
        if not isinstance(out, dict):
            raise AssertionError("parse_model_output accepts only objects")
        edited = edited_tests(changed_files, candidates)
        files |= edited
        run_all, reason = out["run_all"], str(out["reason"])
        return cls(files, sweeps, run_all, reason, candidates=set(candidates))

    def covers(self, test_file: str, test_config: str) -> bool:
        if self.candidates is not None and test_file not in self.candidates:
            return True
        glob = next((g for g in TEST_SH_GLOBS if g in test_config), "")
        whole = self.run_all or test_config in self.sweeps
        always = ALWAYS_COVERED.get(glob, ())
        return whole or test_file in self.files or test_file in always

    def to_json(self) -> dict:
        return {
            "version": SELECTION_VERSION,
            "files": sorted(self.files),
            "sweeps": sorted(self.sweeps),
            "run_all": self.run_all,
            "reason": self.reason,
            "enforce": self.enforce,
            "candidates": None if self.candidates is None else sorted(self.candidates),
            "error": self.error,
        }

    def __post_init__(self) -> None:
        if self.candidates is None and not self.run_all:
            raise ValueError("a selection without candidates must be run_all")

    @classmethod
    def from_json(cls, d: dict) -> Selection:
        """The selection in `d`; ValueError, KeyError or TypeError unless `d`
        is a selection file of SELECTION_VERSION. Without candidates it covers
        only the files it names, so it must be run_all."""
        if (
            not isinstance(d, dict)
            or type(d.get("version")) is not int
            or d["version"] != SELECTION_VERSION
        ):
            raise ValueError(f"not a selection file of version {SELECTION_VERSION}")
        files, sweeps, candidates = d["files"], d["sweeps"], d["candidates"]
        run_all, enforce, reason, error = (
            d["run_all"],
            d["enforce"],
            d["reason"],
            d["error"],
        )
        if not all(
            isinstance(v, list) and all(isinstance(x, str) for x in v)
            for v in (files, sweeps) + (() if candidates is None else (candidates,))
        ):
            raise TypeError("files, sweeps and candidates must be lists of strings")
        if not isinstance(run_all, bool) or not isinstance(enforce, bool):
            raise TypeError("run_all and enforce must be booleans")
        if not isinstance(reason, str) or not isinstance(error, (str, type(None))):
            raise TypeError("reason must be a string and error a string or null")
        if unknown := set(sweeps) - set(SWEEPS):
            raise ValueError(f"unknown sweeps {sorted(unknown)}")
        if candidates is not None and not set(files) <= set(candidates):
            raise ValueError("files must be candidates")
        if error is not None and not run_all:
            raise ValueError("a selection with an error must be run_all")
        return cls(
            set(files),
            set(sweeps),
            run_all,
            reason,
            enforce,
            None if candidates is None else set(candidates),
            error,
        )

    @classmethod
    def load(cls, path: Path) -> Selection | None:
        """The selection at `path`, or None if there is none or, with a
        ::warning::, if it is malformed."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return cls.from_json(json.load(f))
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
            print(
                f"::warning::Ignoring the LLM selection in {path}: {e!r}",
                file=sys.stderr,
            )
            return None
