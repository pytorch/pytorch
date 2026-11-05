"""The test selection an LLM makes for one change, and how CI applies it.

A selection names test files (paths under test/ without .py) and wrapper sweeps
(configs such as dynamo_wrapped that re-run ordinary test files in a special
mode). A test file is covered in a given CI config when the selection names the
file, names that config as a sweep, or is run_all, and so is every test file the
model was not shown and every file that ALWAYS_COVERED lists for the first
TEST_SH_GLOBS entry the config contains. When run_test.py enforces the
selection, it runs the covered files, the files that already failed on the PR
and that TD runs, and, within a hedge budget, TD's positively scored files (see
HEDGE_SHARE).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.testing.target_determination.heuristics.edited_by_pr import (
    ADDITIONAL_MAPPINGS,
)
from tools.testing.target_determination.heuristics.utils import get_pr_number


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable
    from pathlib import Path

    from tools.testing.test_run import TestRun


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
# Modules per import_pattern that importers() greps for: Linux caps one
# argument, such as the pattern git grep gets, at 128 KiB.
MAX_GREP_MODULES = 100
# Longest chain of test modules importing each other that importers() follows.
MAX_IMPORT_HOPS = 8
# More edited failure-list classes than this are not looked up at all, so
# everything runs for an inductor entry, the cpython/ tests for a CPython entry
# and the dynamo_wrapped sweep for the others, as for a class that is not found.
MAX_FAILURE_LIST_CLASSES = 50
# When enforcing, run_test.py also runs the files that already failed on the
# PR and that TD runs, whatever their size, and, greedily in rank order, TD's
# positively scored files that have a known time and fit in this share (20%)
# of the test time of the call's test list (for a whole config, all its shards
# together; for a targeted --include call, only those files): TD and the model
# miss largely different failures.
HEDGE_SHARE = 0.20
# Flip only after the HUD and test-infra consumers of the TD exclusions and
# recall stats handle llm_selection_exclusions and llm_enforced.
ENFORCEMENT_ENABLED = False


# Tests that run the files a change matches although they do not import them:
# the drivers of the directories discover_tests.py blocklists (except
# custom_backend and mobile, which have none, and
# torch_openreg/tests, whose driver test_openreg is never a candidate), TD's
# ADDITIONAL_MAPPINGS, and the tests that read a file that neither importers
# nor data_file_readers traces to them.
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
    # It checks every CPython test .diff against its .py, with these tools.
    "test_cpython_diff_sync": [
        r"^test/cpython/v3_13/",
        r"^tools/(cpython_diff_sync|regenerate_cpython_diffs)\.py$",
    ],
    # They load the library that CMakeLists.txt builds from custom_ops.cpp.
    **{
        t: [r"^test/inductor/(CMakeLists\.txt|custom_ops\.cpp)$"]
        for t in (
            "inductor/test_aot_inductor_custom_ops",
            "inductor/test_gpu_cpp_wrapper",
        )
    },
    # It parses the tracker at import.
    "test_cpp_api_parity": [r"^test/cpp_api_parity/parity-tracker\.md$"],
    # It runs mypy on every file in these directories.
    "test_typing": [r"^test/typing/(pass|fail|reveal)/"],
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

# One file per test, such as TestNNCPU.test_x or CPython313-test_set-TestSet.test_x.
# Under dynamo (the dynamo_wrapped and dynamo_cpython configs) or inductor,
# common_utils xfails or skips a test in the dynamo or inductor lists, and under
# compiled autograd it runs a test in compiled_autograd_skips with it disabled.
FAILURE_LIST_DIR_RE = re.compile(
    r"^test/(dynamo|inductor|compiled_autograd)_(?:expected_failures|skips)/"
)
FAILURE_LIST_RE = re.compile(FAILURE_LIST_DIR_RE.pattern + r"(?:.*-)?(\w+)\.[^./]+$")
# test_dynamic_shapes.py and test_inline_and_install.py name DynamicShapesFoo and
# InlineAndInstallFoo after Foo (see make_test_cls_with_patches), but other
# files define such classes under the full name.
PATCHED_PREFIX_RE = re.compile(r"^(DynamicShapes|InlineAndInstall)(?=[A-Z])")
# instantiate_device_type_tests names TestFooCPU after TestFoo.
DEVICE_SUFFIX_RE = re.compile(r"(CPU|CUDA|MPS|XPU|HPU|META|LAZY|PRIVATEUSE1)$")
# assertExpected reads test/expect/<TestClass>.<test>[-<name>].expect.
EXPECT_RE = re.compile(r"^test/expect/(\w+)\.")
# Edited files under test/ that need no reader: docs; the sources, build files
# and scripts of the C++ tests under test/cpp/, which are never candidates and
# so always run (importers still finds the Python tests importing a .py file);
# the files of custom_backend, inductor/cpp, jit_hooks and mobile, which no
# candidate reads and run_test.py never runs, and the non-.py files of
# custom_operator, which only CI scripts build; and run_doctests.sh and the
# scripts below, which nothing under test/ runs.
NO_READER_RE = re.compile(
    r"\.(md|rst)$|^test/cpp/.*(\.(c|cc|cpp|cu|h|hpp|py|sh)|/CMakeLists\.txt)$"
    r"|^test/run_doctests\.sh$|^test/forward_backward_compatibility/"
    r"|^test/((create_dummy|load)_torchscript_model|bench_mps_ops|simulate_nccl_errors)\.py$"
    r"|^test/scripts/(cuda_memcheck_common|run_cuda_memcheck)\.py$"
    r"|^test/(custom_backend|inductor/cpp|jit_hooks|mobile)/"
    r"|^test/custom_operator/.*(?<!\.py)$"
)
# discover_tests.py's test modules. One that is not a candidate is always
# covered (see Selection.candidates), so it needs no reader.
TEST_MODULE_RE = re.compile(r"^test/(.*/)?test_[^/]*\.py$")
# More edited files under test/ than this that only their readers can read
# (see data_file_readers) are not looked up one by one.
MAX_DATA_FILES = 50

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


# run_test.py entries that run a test file of another name.
TEST_FILES = {
    "test_cpp_extensions_aot_ninja": "test/test_cpp_extensions_aot.py",
    "test_cpp_extensions_aot_no_ninja": "test/test_cpp_extensions_aot.py",
    "test_autoload_enable": "test/test_autoload.py",
    "test_autoload_disable": "test/test_autoload.py",
}


def importers(
    sources: list[str], candidates: set[str], grep: Callable[[str], list[str]]
) -> set[str] | None:
    """The candidates among `sources` (paths under test/), the candidates that
    import one of them directly or through other modules under test/, and their
    drivers: test/jit/test_freezing.py selects test_jit, which runs it, and
    test_jit_legacy, which does `from test_jit import *`. `grep` lists the paths
    under test/ that match a `git grep -E` pattern at the change's head. None if
    following the importers takes more than MAX_IMPORT_HOPS hops (grep rounds)."""
    found: set[str] = set()
    seen: set[str] = set()
    paths = {f for f in sources if f.startswith("test/") and f.endswith(".py")}
    # A run_test.py entry's path, such as test/test_autoload_enable.py, stands
    # for the file it runs.
    frontier = {TEST_FILES.get(f[5:-3], f) for f in paths}
    for hop in range(MAX_IMPORT_HOPS + 1):
        tests = edited_tests(sorted(frontier), candidates)
        found |= tests
        runs = {TEST_FILES.get(t, f"test/{t}.py") for t in tests}
        frontier = (frontier | runs) - seen
        if not frontier:
            return found
        if hop == MAX_IMPORT_HOPS:
            return None
        seen |= frontier
        mods, n = sorted(frontier), MAX_GREP_MODULES
        greps = [grep(import_pattern(mods[i : i + n])) for i in range(0, len(mods), n)]
        frontier = {f for g in greps for f in g if f.endswith(".py")}
    return None


def import_pattern(modules: Iterable[str]) -> str:
    """A `git grep -E` pattern for importing one of `modules`, .py paths under test/."""
    # A module is imported by any dotted suffix of its path, optionally prefixed
    # with `test.`, or by importlib.import_module of one; a package by its
    # directory; and load_test_module (a SourceFileLoader) takes the path.
    # The CPython tests under test/cpython redirect test.<name> to <name>, so
    # `from test import <name>` imports their sibling module <name>.
    # A parenthesized `from <package> import (` may span lines, so it counts
    # as importing every module of that package, and `from . import (`,
    # `from .. import (` or `from test import (` every module of any package.
    # Without a relative or `test.` prefix, a name that is also a standard
    # library module, such as random, imports that module instead, and a bare
    # `test` the test/ package.
    bare = sys.stdlib_module_names | {"test"}

    def dotted(names: set[str]) -> str:
        alt = rf"(\.+|test\.)({alts(names)})"
        if unprefixed := alts(n for n in names if n not in bare):
            alt += f"|{unprefixed}"
        return f"({alt})"

    mods = sorted(f[5:-3].removesuffix("/__init__") for f in modules)
    parts = [m.split("/") for m in mods]
    names = dotted({".".join(p[i:]) for p in parts for i in range(len(p))})
    bases = alts(p[-1] for p in parts)
    parents = {".".join(p[i:-1]) for p in parts for i in range(len(p) - 1)}
    paths = alts(n for m in mods for n in (m, m.replace("/", ".")))
    end = "([^[:alnum:]_]|$)"
    names_of = rf"import[[:space:]]*(\(|[[:space:]](.*[^[:alnum:]_.])?({bases}){end})"
    forms = [
        rf"(from[[:space:]]+|import[[:space:]]+(.*[^[:alnum:]_.])?){names}{end}",
        rf"from[[:space:]]+(\.+|test)[[:space:]]+{names_of}",
    ]
    if parents:
        forms.append(rf"from[[:space:]]+{dotted(parents)}[[:space:]]+{names_of}")
    quoted = rf"""load_test_module\(["']({paths})["']\)"""
    quoted += rf"""|import_module\(["']{names}["'][,)[:space:]]"""
    # A statement can also follow a `;` or a compound statement's `:`.
    return rf"(^|[;:])[[:space:]]*({'|'.join(forms)})|{quoted}"


def alts(names: Iterable[str]) -> str:
    """An extended regex alternation of the literal `names`."""
    return "|".join(
        re.sub(r"[.^$*+?()\[\]{}|\\]", r"\\\g<0>", n) for n in sorted(set(names))
    )


def class_pattern(classes: Iterable[str], prefix: bool = False) -> str:
    """A `git grep -E` pattern for the definition of one of `classes` or of
    the generic class instantiate_device_type_tests made it from, or with
    `prefix` of a class whose name starts with one of them."""
    stems = {c: DEVICE_SUFFIX_RE.sub("", c) for c in classes}
    # TestCUDA is not made from a class Test.
    names = alts([*stems, *(s for s in stems.values() if s not in ("", "Test"))])
    end = "" if prefix else "([^[:alnum:]_]|$)"
    return rf"^[[:space:]]*class[[:space:]]+({names}){end}"


def data_file_readers(
    changed_files: list[str],
    candidates: set[str],
    grep: Callable[[str], list[str]],
    deleted: Collection[str] = (),
) -> tuple[list[str], dict[str, list[str]]] | None:
    """The paths under test/ that may read an edited file under test/ that is
    neither a failure list nor a candidate test file, split into those
    reading a file that needs no further reader and, by file, those reading a
    file only they can read. A file needs no further reader when it is deleted,
    when NO_READER_RE matches it, when a driver runs its directory (see
    edited_tests), when it is a test module (see TEST_MODULE_RE) that is not a
    candidate, or when it is read by such a test module or by a file under
    test/cpp/ other than a .md or .rst that names it outside a // comment line.
    None for more than MAX_DATA_FILES files only their readers can read. A file
    is read by the paths naming its basename outside a # comment line, as
    test_run.py runs bin/test_script.sh; a .py file also by its importers; and
    an expect file (see EXPECT_RE) also by the paths defining a class whose
    name starts with its test class, as test_tensorboard.py builds the path of
    TestTensorBoard.*.expect for TestTensorBoardSummary."""
    data = [
        f
        for f in changed_files
        if f.startswith("test/")
        and not FAILURE_LIST_DIR_RE.match(f)
        and not (f.endswith(".py") and normalize_test_name(f, candidates))
    ]
    exempt = [
        f
        for f in data
        if f in deleted
        or NO_READER_RE.search(f)
        or TEST_MODULE_RE.match(f)
        or edited_tests([f], candidates)
    ]
    data = [f for f in data if f not in exempt]
    if len(data) > MAX_DATA_FILES:
        return None

    def name_pattern(files: list[str], cpp: bool = False) -> str:
        # A basename not preceded by a name character, on a line whose first
        # non-space character is not # (`[^#`), which rejects # comments and
        # #include lines; with `cpp` the line also does not start with //, as
        # `/[^/]` admits a / only if no / follows it.
        names = alts(f.rsplit("/", 1)[-1] for f in files)
        first = "([^#/[:space:]]|/[^/])" if cpp else "[^#[:space:]]"
        return rf"^[[:space:]]*([^#[:space:][:alnum:]_.-]|{first}.*[^[:alnum:]_.-])?({names})([^[:alnum:]_.-]|$)"

    # The readers of an exempt file can still break unless NO_READER_RE
    # matches it and it is not deleted; those of a deleted expect file include
    # the tests of its class, which assert against it.
    named = [f for f in exempt if f in deleted or not NO_READER_RE.search(f)]
    classes = [m[1] for f in exempt if (m := EXPECT_RE.match(f))]
    n = MAX_GREP_MODULES
    patterns = [name_pattern(named[i : i + n]) for i in range(0, len(named), n)]
    for i in range(0, len(classes), n):
        patterns.append(class_pattern(classes[i : i + n], prefix=True))
    readers = [r for p in patterns for r in grep(p)]
    read_by: dict[str, list[str]] = {}
    for f in data:
        patterns = [name_pattern([f])]
        if m := EXPECT_RE.match(f):
            patterns.append(class_pattern([m[1]], prefix=True))
        if f.endswith(".py"):
            patterns.append(import_pattern([f]))
        found = [r for r in grep("|".join(f"({p})" for p in patterns)) if r != f]
        cpp = {
            r
            for r in found
            if r.startswith("test/cpp/") and not r.endswith((".md", ".rst"))
        }
        tests = {r for r in found if TEST_MODULE_RE.match(r)}
        if any(not edited_tests([r], candidates) for r in tests) or (
            cpp and cpp & set(grep(name_pattern([f], cpp=True)))
        ):
            readers += found
        else:
            read_by[f] = found
    return readers, read_by


def failure_list_sources(
    changed_files: list[str], grep: Callable[[str], list[str]] | None
) -> tuple[list[str], set[str]]:
    """The paths under test/ that define the classes named by the edited
    failure-list entries (see FAILURE_LIST_RE, PATCHED_PREFIX_RE), and the
    lists ("dynamo", "inductor", "compiled_autograd", or "dynamo_cpython" for
    a CPython entry of the dynamo lists, which the dynamo_cpython config runs)
    naming a class that is not looked up (no `grep`, see importers, or more
    than MAX_FAILURE_LIST_CLASSES classes) or that `grep` does not find in a
    .py file. Also the list ("dynamo", "inductor" or "compiled_autograd", and
    "dynamo_cpython" for a CPython entry of the dynamo lists) holding an entry
    it does not look up: one in a subdirectory or, for the dynamo and inductor
    lists, one without exactly one "." (dynamo_test_failures.py rejects both
    at import, except in compiled_autograd_skips) or one edited in both the
    expected failures and the skips (a clash it rejects at import, or a
    move)."""
    classes: dict[str, set[str]] = {}
    unparsed: set[str] = set()
    for f in changed_files:
        if not (d := FAILURE_LIST_DIR_RE.match(f)):
            continue
        name = f[d.end() :]
        both = {f"test/{d[1]}_{k}/{name}" for k in ("expected_failures", "skips")}
        cpython = d[1] == "dynamo" and re.match(r"CPython\d+-", name)
        if "/" in name or (
            d[1] != "compiled_autograd"
            and (name.count(".") != 1 or both <= set(changed_files))
        ):
            unparsed.add(d[1])
            if cpython:
                unparsed.add("dynamo_cpython")
        elif m := FAILURE_LIST_RE.match(f):
            classes.setdefault(m[2], set()).add("dynamo_cpython" if cpython else m[1])
    if grep is None or len(classes) > MAX_FAILURE_LIST_CLASSES:
        return [], unparsed.union(*classes.values())
    sources: list[str] = []
    missing = unparsed
    for c, lists in sorted(classes.items()):
        names = [c, PATCHED_PREFIX_RE.sub("", c)]
        found = [f for f in grep(class_pattern(names)) if f.endswith(".py")]
        sources += found
        if not found:
            missing |= lists
    return sources, missing


def edits_doctest(diff: str) -> bool:
    """Whether `diff`, a `git diff`, adds or removes a line in a hunk of a .py
    file under torch/ that has a `>>>` line, changed or context: the doctests
    test runs those examples with xdoctest, including the expected output and
    `...` lines that follow a `>>>` line within the diff's context."""
    paths: list[str] = []
    in_header = in_torch = doctest = changed = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            paths, in_header, in_torch = [], True, False
        elif in_header and line.startswith(("--- a/", "+++ b/")):
            paths.append(line[6:])
        elif line.startswith("@@"):
            in_header = doctest = changed = False
            in_torch = any(re.match(r"torch/.*\.py$", f) for f in paths)
        elif in_torch and line[:1] in (" ", "+", "-"):
            doctest = doctest or ">>>" in line
            changed = changed or line[:1] != " "
            if doctest and changed:
                return True
    return False


def needs_run_all(changed_files: list[str]) -> bool:
    return any(re.search(p, f) for p in RUN_ALL_PATTERNS for f in changed_files)


TEST_INFRA_REASON = "change touches test infrastructure"


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
    With `grep` (see importers), a name of a file that a driver runs, such as
    jit/test_freezing, names the driver instead. A run_all answer's `tests` and
    `dirs` are not checked, since everything runs."""
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
    if out["run_all"]:
        # Everything runs, so the names need no lookup.
        return set(), set(sweeps)
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


# A failed `grep`: an OSError such as E2BIG, or git_grep's RuntimeError or timeout.
GREP_ERRORS = (OSError, RuntimeError, subprocess.SubprocessError)


def grep_failure(e: Exception) -> Selection:
    """from_model_output's Selection for a `grep` that raised one of GREP_ERRORS."""
    # A SubprocessError's message holds the argv, ~90 KB of pattern.
    if isinstance(e, subprocess.TimeoutExpired):
        error = f"TimeoutExpired after {e.timeout}s"
    elif isinstance(e, subprocess.SubprocessError):
        error = type(e).__name__
    else:
        error = f"{type(e).__name__}: {e}"
    return Selection(run_all=True, reason=f"grep failed ({error})", error=error)


@dataclass
class Selection:
    files: set[str] = field(default_factory=set)
    sweeps: set[str] = field(default_factory=set)
    # The model's own run_all, a fallback of from_model_output (see its
    # docstring), or one of select_tests.select() and main(), which set it
    # without asking the model or when the selector fails.
    run_all: bool = False
    reason: str = ""
    # Whether run_test.py should replace TD's split with Selection.split rather
    # than only reporting it; see apply_selection for when it does.
    enforce: bool = False
    # The test files the model chose from. Any other file, such as a C++ test
    # or test_openreg, is always covered. None covers no extra file, so it
    # requires run_all.
    candidates: set[str] | None = None
    # Why the selector got no usable answer (git, the model call or its output,
    # or the test times); None for a deliberate run_all.
    error: str | None = None
    # Whether enforcement was asked for, even if a failed selector cleared enforce.
    enforce_requested: bool = False

    def __post_init__(self) -> None:
        if self.candidates is None and not self.run_all:
            raise ValueError("a selection without candidates must be run_all")
        if unknown := self.sweeps - set(SWEEPS):
            raise ValueError(f"unknown sweeps {sorted(unknown)}")
        if self.candidates is not None and not self.files <= self.candidates:
            raise ValueError("files must be candidates")
        if self.error is not None and not self.run_all:
            raise ValueError("a selection with an error must be run_all")
        self.enforce_requested = self.enforce_requested or self.enforce

    @classmethod
    def from_model_output(
        cls,
        out: object,
        candidates: set[str],
        changed_files: list[str],
        grep: Callable[[str], list[str]] | None = None,
        deleted: Collection[str] = (),
        diff: str = "",
    ) -> Selection:
        """Turn the model's structured output into a Selection. Everything runs
        when failure_list_sources reports the inductor list, that is for an
        inductor entry it rejects or does not resolve to a file; dynamo_cpython,
        a CPython entry it does not resolve, selects the cpython/ tests, and the
        other lists it reports add the dynamo_wrapped sweep. `deleted` lists the
        changed files the change deletes, including the old path of every
        moved file, since changed_files diffs with --no-renames: a deleted file
        needs no reader, so one missing from it can run everything. So do more
        than MAX_DATA_FILES edited files under test/ that need a reader.

        A change to test infrastructure, too many import hops, and with `grep`
        an edited file under test/ that no test reads (see data_file_readers)
        also fall back to running everything. A missing or malformed output or
        a failed `grep` sets error instead, so run_test.py keeps TD's split
        (see apply_selection). Test files the change edits and their drivers
        are always selected, and with `grep` so are the test files that import
        them (see importers), those defining the tests that an edited failure
        list names (see failure_list_sources), and those reading an edited data
        file (see data_file_readers), with their importers."""
        if out is None:
            return cls(
                run_all=True,
                reason="no model output; TD's split runs",
                error="no model output",
            )
        try:
            files, sweeps = parse_model_output(out, candidates, grep)
        except ValueError as e:
            return cls(
                run_all=True,
                reason=f"malformed model output ({e}); TD's split runs",
                error=f"malformed model output: {e}",
            )
        except GREP_ERRORS as e:
            if not needs_run_all(changed_files):
                return grep_failure(e)
            files, sweeps = set(), set()
        if needs_run_all(changed_files):
            return cls(run_all=True, reason=TEST_INFRA_REASON)
        # Before any grep, so that a failed one cannot turn it into an error.
        if isinstance(out, dict) and out["run_all"]:
            return cls(
                sweeps=sweeps,
                run_all=True,
                reason=str(out["reason"]),
                candidates=set(candidates),
            )
        if not isinstance(out, dict):
            raise AssertionError("parse_model_output accepts only objects")
        try:
            sources, missing = failure_list_sources(changed_files, grep)
        except GREP_ERRORS as e:
            return grep_failure(e)
        if "inductor" in missing:
            return cls(
                run_all=True,
                reason="an edited inductor failure list entry was rejected or not resolved to the file defining its test class",
            )
        # The dynamo_cpython config runs only the tests under test/cpython/.
        if "dynamo_cpython" in missing:
            files |= {c for c in candidates if c.startswith("cpython/")}
        # dynamo_wrapped also runs with compiled autograd on (see common_utils).
        if missing & {"dynamo", "compiled_autograd"}:
            sweeps.add("dynamo_wrapped")
        edited = edited_tests(changed_files, candidates)
        files |= edited
        if "doctests" in candidates and edits_doctest(diff):
            files.add("doctests")
        if grep is not None:
            try:
                data = data_file_readers(changed_files, candidates, grep, deleted)
            except GREP_ERRORS as e:
                return grep_failure(e)
            if data is None:
                return cls(
                    run_all=True,
                    reason=f"more than {MAX_DATA_FILES} edited files under test/ need a reader",
                )
            readers, read_by = data
            # The drivers of edited non-.py files, such as test_jit for test/jit/*.txt, have importers too.
            drivers = [f"test/{t}.py" for t in sorted(edited)]
            # A file is read only if its readers lead to a test that runs.
            for f, paths in [
                (None, changed_files + sources + readers + drivers),
                *read_by.items(),
            ]:
                try:
                    found = importers(paths, candidates, grep)
                except GREP_ERRORS as e:
                    return grep_failure(e)
                if found is None:
                    reason = "test modules import the change through too many hops"
                    return cls(run_all=True, reason=reason)
                if f is not None and not found:
                    return cls(run_all=True, reason=f"nothing found that reads {f}")
                files |= found
        run_all, reason = out["run_all"], str(out["reason"])
        return cls(files, sweeps, run_all, reason, candidates=set(candidates))

    def covers(self, test_file: str, test_config: str) -> bool:
        if self.candidates is not None and test_file not in self.candidates:
            return True
        glob = next((g for g in TEST_SH_GLOBS if g in test_config), "")
        whole = self.run_all or test_config in self.sweeps
        always = ALWAYS_COVERED.get(glob, ())
        return whole or test_file in self.files or test_file in always

    def split(
        self,
        ranked: list[TestRun],
        score: Callable[[TestRun], float],
        file_time: dict[str, float],
        test_config: str,
        *,
        previous_failures: Collection[str] = (),
        td_include: Collection[str] = (),
        share: float | None = None,
    ) -> Split:
        """Split TD's ranked TestRuns into the covered files, the
        `previous_failures` files (those that already failed on the PR) that
        are in `td_include` (TD's split runs them), and a hedge. Those previous
        failures run whatever their size, as under TD, and are not charged to
        the hedge: their pool spans the PR's SHAs, build environments and
        configs, so one large earlier failure would otherwise use up the hedge
        of every config. A previous failure with any TestRun in TD's include
        runs its whole file. The hedge is `share` (by default HEDGE_SHARE,
        20%) of the test time of `ranked`, spent greedily in rank order on the
        other files with a positively scored TestRun, including the other
        previous failures. One positive class-level run hedges its whole file
        at the file's full time. Files without a time are never hedged.
        Partial TestRuns of one file share its time, so a file is charged once
        and lands in one batch. If that chooses nothing, chose_nothing is set
        and include is empty: apply_selection then keeps TD's split, so a call
        never runs zero files when TD's split runs some."""
        files = dict.fromkeys(x.test_file for x in ranked)
        covered = {f for f in files if self.covers(f, test_config)}
        prev = {f for f in previous_failures if f in files and f in td_include}
        prev -= covered
        share = HEDGE_SHARE if share is None else share
        left = budget = share * sum(file_time.get(f, 0) for f in files)
        hedged: set[str] = set()
        chosen = covered | prev
        for x in ranked:
            f, t = x.test_file, file_time.get(x.test_file, 0)
            if f not in chosen and score(x) > 0 and t and t <= left:
                left -= t
                hedged.add(f)
                chosen.add(f)
        include = [x for x in ranked if x.test_file in chosen]
        exclude = [x for x in ranked if x.test_file not in chosen]
        return Split(include, exclude, covered, prev, hedged, budget, not chosen)

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
            "enforce_requested": self.enforce_requested,
        }

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
        # Older selectors, which CI runs from the default branch, omit it.
        enforce_requested = d.get("enforce_requested", enforce)
        if not all(
            isinstance(v, list) and all(isinstance(x, str) for x in v)
            for v in (files, sweeps) + (() if candidates is None else (candidates,))
        ):
            raise TypeError("files, sweeps and candidates must be lists of strings")
        if not all(isinstance(v, bool) for v in (run_all, enforce, enforce_requested)):
            raise TypeError("run_all, enforce and enforce_requested must be booleans")
        if not isinstance(reason, str) or not isinstance(error, (str, type(None))):
            raise TypeError("reason must be a string and error a string or null")
        return cls(
            files=set(files),
            sweeps=set(sweeps),
            run_all=run_all,
            reason=reason,
            enforce=enforce,
            candidates=None if candidates is None else set(candidates),
            error=error,
            enforce_requested=enforce_requested,
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


@dataclass
class Split:
    """Selection.split's result: the TestRuns to run and to skip, and which
    files run because they are covered, already failed on the PR and are in
    TD's split, or are hedged (disjoint sets), the hedge budget in seconds
    (which the previous failures do not use), and whether it chose no file
    (chose_nothing), in which case include is empty."""

    include: list[TestRun]
    exclude: list[TestRun]
    covered: set[str]
    previous_failures_added: set[str]
    hedged: set[str]
    budget: float
    chose_nothing: bool


@dataclass
class AppliedSelection:
    """A Selection applied to TD's split of one run_test.py call's test list.
    would_enforce records that apply_selection would enforce the selection
    with ENFORCEMENT_ENABLED (a PR with TD on, enforce set, no selector error
    and no fallback), and fallback why a selection that would otherwise be
    enforced keeps TD's split (no test times, previous failures not recorded,
    or no file chosen), else None; both are set with the flag off too.
    enforced and kept_td are would_enforce and fallback is not None with
    ENFORCEMENT_ENABLED, and False without it. A request the selector refused
    (enforce_requested without enforce: a failed selector or a not_enforced
    answer) has neither would_enforce nor a fallback.
    previous_failures_known records that the TD results stored previous failures,
    which the TD job does when its pytest cache download step succeeded. That
    step also succeeds without a cache, as on a PR whose only failures were in
    jobs that upload none (ROCm, fork PRs), so it does not mean every previous
    failure is known."""

    selection: Selection
    split: Split
    td_enabled: bool
    enforcement_enabled: bool
    enforced: bool
    kept_td: bool
    would_enforce: bool
    fallback: str | None
    previous_failures_known: bool
    td_include: set[str]
    td_include_runs: set[TestRun]
    file_time: dict[str, float]

    @property
    def selection_metric(self) -> dict[str, object]:
        """The llm_selection metric. Every shard of a config emits the same
        one, so count one per (workflow run, run attempt, build_environment,
        config, list_id): list_id hashes only the call's test file names.
        selector_error tells a failed selector, which is never enforced, from
        a deliberate run_all. enforce_requested is the selection's, also true
        when enforced and whether or not TD is on: the requested enforcements
        that ran TD are the rows with enforcement_enabled, enforce_requested
        and td_enabled but not enforced, on PR jobs; of those, kept_td fell
        back in apply_selection and the rest the selector refused. With the
        flag off, would_enforce and fallback record what enforcing would
        have done, and enforce whether the selector granted the request."""
        split, times = self.split, self.file_time
        files = "\n".join(sorted(times)).encode()
        return {
            "list_id": hashlib.sha256(files).hexdigest()[:12],
            "enforced": self.enforced,
            "enforce_requested": self.selection.enforce_requested,
            "enforcement_enabled": self.enforcement_enabled,
            "td_enabled": self.td_enabled,
            "run_all": self.selection.run_all,
            "selector_error": self.selection.error,
            "num_tests": len(times),
            "num_covered": len(split.covered),
            "num_td_include": len(self.td_include),
            "num_hedged": len(split.hedged),
            "num_previous_failures_added": len(split.previous_failures_added),
            "kept_td": self.kept_td,
            "enforce": self.selection.enforce,
            "would_enforce": self.would_enforce,
            "fallback": self.fallback,
            "previous_failures_known": self.previous_failures_known,
            "time_total": sum(times.values()),
            "time_covered": sum(times[f] for f in split.covered),
            "time_hedged": sum(times[f] for f in split.hedged),
            "time_previous_failures_added": sum(
                times[f] for f in split.previous_failures_added
            ),
        }

    def failure_metric(self, test: TestRun) -> dict[str, object]:
        f = test.test_file
        return {
            "failure": str(test),
            "covered": f in self.split.covered,
            "enforced": self.enforced,
            "enforce_requested": self.selection.enforce_requested,
            "enforcement_enabled": self.enforcement_enabled,
            "td_enabled": self.td_enabled,
            "in_td_include": test in self.td_include_runs,
            "hedged": f in self.split.hedged,
            "previous_failure_added": f in self.split.previous_failures_added,
            "kept_td": self.kept_td,
            "enforce": self.selection.enforce,
            "would_enforce": self.would_enforce,
            "fallback": self.fallback,
            "previous_failures_known": self.previous_failures_known,
            "selector_error": self.selection.error,
        }


def apply_selection(
    selection: Selection,
    include: list[TestRun],
    exclude: list[TestRun],
    score: Callable[[TestRun], float],
    test_times: dict[str, float],
    *,
    test_config: str,
    td_enabled: bool,
    previous_failures: Collection[str] | None,
) -> AppliedSelection:
    """Apply the selection to TD's (include, exclude) split.

    The selection is enforced, replacing TD's split, only with
    ENFORCEMENT_ENABLED, on a PR with TD enabled, the selection's enforce set
    and no selector error, only if some of the files have a test time, only
    if `previous_failures` (the PR's
    previous failures stored with TD's ranking, see import_results) is not
    None, and only if the split chooses some file.
    run_test.py falls back to the default job's times for a config without its
    own, so the times are missing when none of the call's files has a time
    (for example without test-times.json, or a call of only new test files).
    The metric's list_id hashes this call's test list, so it is the same on
    every shard of a config."""
    ranked = include + exclude
    td = {x.test_file for x in include}
    # Partial TestRuns of one file share its time, so account per file.
    file_time = {x.test_file: test_times.get(x.test_file, 0) for x in ranked}
    prev = previous_failures or ()
    split = selection.split(
        ranked, score, file_time, test_config, previous_failures=prev, td_include=td
    )
    td_on_pr = td_enabled and get_pr_number() is not None
    wanted = td_on_pr and selection.enforce and selection.error is None
    # select_tests clears enforce when the selector fails, but keeps the request.
    requested = ENFORCEMENT_ENABLED and td_on_pr and selection.enforce_requested
    if not sum(file_time.values()):
        fallback = "no test times"
    elif previous_failures is None:
        fallback = "previous failures not recorded in the TD results"
    elif split.chose_nothing:
        fallback = "chose no files"
    else:
        fallback = None
    fallback = fallback if wanted else None
    would_enforce = wanted and fallback is None
    enforced = ENFORCEMENT_ENABLED and would_enforce
    if selection.error is not None and requested:
        print(
            f"LLM selection: the selector failed ({selection.error[:200]!r}), using TD",
            file=sys.stderr,
        )
    elif requested and not selection.enforce:
        print(
            "LLM selection: the selector did not enforce the request (see not_enforced in the selection file's diagnostics), using TD",
            file=sys.stderr,
        )
    if ENFORCEMENT_ENABLED and fallback is not None:
        print(f"LLM selection: {fallback}, using TD", file=sys.stderr)
    if enforced:
        s, n = split, len({x.test_file for x in split.include})
        print(
            f"LLM selection: running {n} of {len(file_time)} test files"
            f" ({len(s.covered)} covered; {len(s.previous_failures_added)}"
            f" previous failures added; {len(s.hedged)} hedged within"
            f" {s.budget:.0f}s)",
            file=sys.stderr,
        )
    return AppliedSelection(
        selection=selection,
        split=split,
        td_enabled=td_enabled,
        enforcement_enabled=ENFORCEMENT_ENABLED,
        enforced=enforced,
        kept_td=ENFORCEMENT_ENABLED and fallback is not None,
        would_enforce=would_enforce,
        fallback=fallback,
        previous_failures_known=previous_failures is not None,
        td_include=td,
        td_include_runs=set(include),
        file_time=file_time,
    )
