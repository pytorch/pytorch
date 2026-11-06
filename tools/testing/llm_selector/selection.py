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
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING


if TYPE_CHECKING:
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
# The selector, run by the default branch's workflow, will write this file into
# ADDITIONAL_CI_FILES_FOLDER, and run_test.py will read it from the PR's checkout.
# The reader rejects any other SELECTION_VERSION, unknown keys and unknown
# sweeps, so a bump, a new field or a new sweep turns selection off for
# unrebased PRs until they rebase.
SELECTION_VERSION = 1

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


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


@dataclass
class Selection:
    files: set[str] = field(default_factory=set)
    sweeps: set[str] = field(default_factory=set)
    # The model's own run_all, or a fallback when the selector fails or does
    # not ask the model.
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
        is a selection file of SELECTION_VERSION. Without candidates it would
        skip every file it does not name, so it must be run_all."""
        if (
            not isinstance(d, dict)
            or type(d.get("version")) is not int
            or d["version"] != SELECTION_VERSION
        ):
            raise ValueError(f"not a selection file of version {SELECTION_VERSION}")
        if unknown := set(d) - {"version", *(f.name for f in fields(cls))}:
            raise ValueError(f"unknown keys {sorted(unknown)}")
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
