import os
import modulefinder
import sys
import pathlib
from typing import Dict, Any, List, Set

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# These tests are slow enough that it's worth calculating whether the patch
# touched any related files first. This list was manually generated, but for every
# run with --determine-from, we use another generated list based on this one and the
# previous test stats.
TARGET_DET_LIST = [
    "distributions/test_distributions",
    "test_nn",
    "test_autograd",
    "test_cpp_extensions_jit",
    "test_jit_legacy",
    "test_dataloader",
    "test_overrides",
    "test_linalg",
    "test_jit",
    "test_jit_profiling",
    "test_torch",
    "test_binary_ufuncs",
    "test_numpy_interop",
    "test_reductions",
    "test_shape_ops",
    "test_sort_and_select",
    "test_testing",
    "test_view_ops",
    "distributed/nn/jit/test_instantiator",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "distributed/algorithms/ddp_comm_hooks/test_ddp_hooks",
    "distributed/test_distributed_spawn",
    "test_cuda",
    "test_cuda_primary_ctx",
    "test_cpp_extensions_aot_ninja",
    "test_cpp_extensions_aot_no_ninja",
    "test_serialization",
    "test_optim",
    "test_utils",
    "test_multiprocessing",
    "test_tensorboard",
    "distributed/test_c10d_common",
    "distributed/test_c10d_gloo",
    "distributed/test_c10d_nccl",
    "distributed/test_jit_c10d",
    "distributed/test_c10d_spawn_gloo",
    "distributed/test_c10d_spawn_nccl",
    "distributed/test_store",
    "distributed/test_pg_wrapper",
    "test_quantization",
    "test_pruning_op",
    "test_determination",
    "test_futures",
    "distributed/pipeline/sync/skip/test_api",
    "distributed/pipeline/sync/skip/test_gpipe",
    "distributed/pipeline/sync/skip/test_inspect_skip_layout",
    "distributed/pipeline/sync/skip/test_leak",
    "distributed/pipeline/sync/skip/test_portal",
    "distributed/pipeline/sync/skip/test_stash_pop",
    "distributed/pipeline/sync/skip/test_tracker",
    "distributed/pipeline/sync/skip/test_verify_skippables",
    "distributed/pipeline/sync/test_balance",
    "distributed/pipeline/sync/test_bugs",
    "distributed/pipeline/sync/test_checkpoint",
    "distributed/pipeline/sync/test_copy",
    "distributed/pipeline/sync/test_deferred_batch_norm",
    "distributed/pipeline/sync/test_dependency",
    "distributed/pipeline/sync/test_inplace",
    "distributed/pipeline/sync/test_microbatch",
    "distributed/pipeline/sync/test_phony",
    "distributed/pipeline/sync/test_pipe",
    "distributed/pipeline/sync/test_pipeline",
    "distributed/pipeline/sync/test_stream",
    "distributed/pipeline/sync/test_transparency",
    "distributed/pipeline/sync/test_worker",
]

_DEP_MODULES_CACHE: Dict[str, Set[str]] = {}


def determine_target(
    target_det_list: List[str], test: str, touched_files: List[str], options: Any
) -> bool:
    test = parse_test_module(test)
    # Some tests are faster to execute than to determine.
    if test not in target_det_list:
        if options.verbose:
            print_to_stderr(f"Running {test} without determination")
        return True
    # HACK: "no_ninja" is not a real module
    if test.endswith("_no_ninja"):
        test = test[: (-1 * len("_no_ninja"))]
    if test.endswith("_ninja"):
        test = test[: (-1 * len("_ninja"))]

    dep_modules = get_dep_modules(test)

    for touched_file in touched_files:
        file_type = test_impact_of_file(touched_file)
        if file_type == "NONE":
            continue
        elif file_type == "CI":
            # Force all tests to run if any change is made to the CI
            # configurations.
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type == "UNKNOWN":
            # Assume uncategorized source files can affect every test.
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type in ["TORCH", "CAFFE2", "TEST"]:
            parts = os.path.splitext(touched_file)[0].split(os.sep)
            touched_module = ".".join(parts)
            # test/ path does not have a "test." namespace
            if touched_module.startswith("test."):
                touched_module = touched_module.split("test.")[1]
            if touched_module in dep_modules or touched_module == test.replace(
                "/", "."
            ):
                log_test_reason(file_type, touched_file, test, options)
                return True

    # If nothing has determined the test has run, don't run the test.
    if options.verbose:
        print_to_stderr(f"Determination is skipping {test}")

    return False


def test_impact_of_file(filename: str) -> str:
    """Determine what class of impact this file has on test runs.

    Possible values:
        TORCH - torch python code
        CAFFE2 - caffe2 python code
        TEST - torch test code
        UNKNOWN - may affect all tests
        NONE - known to have no effect on test outcome
        CI - CI configuration files
    """
    parts = filename.split(os.sep)
    if parts[0] in [".jenkins", ".circleci"]:
        return "CI"
    if parts[0] in ["docs", "scripts", "CODEOWNERS", "README.md"]:
        return "NONE"
    elif parts[0] == "torch":
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "TORCH"
    elif parts[0] == "caffe2":
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "CAFFE2"
    elif parts[0] == "test":
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "TEST"

    return "UNKNOWN"


def log_test_reason(file_type: str, filename: str, test: str, options: Any) -> None:
    if options.verbose:
        print_to_stderr(
            "Determination found {} file {} -- running {}".format(
                file_type,
                filename,
                test,
            )
        )


def get_dep_modules(test: str) -> Set[str]:
    # Cache results in case of repetition
    if test in _DEP_MODULES_CACHE:
        return _DEP_MODULES_CACHE[test]

    test_location = REPO_ROOT / "test" / f"{test}.py"
    finder = modulefinder.ModuleFinder(
        # Ideally exclude all third party modules, to speed up calculation.
        excludes=[
            "scipy",
            "numpy",
            "numba",
            "multiprocessing",
            "sklearn",
            "setuptools",
            "hypothesis",
            "llvmlite",
            "joblib",
            "email",
            "importlib",
            "unittest",
            "urllib",
            "json",
            "collections",
            # Modules below are excluded because they are hitting https://bugs.python.org/issue40350
            # Trigger AttributeError: 'NoneType' object has no attribute 'is_package'
            "mpl_toolkits",
            "google",
            "onnx",
            # Triggers RecursionError
            "mypy",
        ],
    )
    # HACK: some platforms default to ascii, so we can't just run_script :(
    with open(test_location, "r", encoding="utf-8") as fp:
        finder.load_module("__main__", fp, str(test_location), ("", "r", str(1)))

    dep_modules = set(finder.modules.keys())
    _DEP_MODULES_CACHE[test] = dep_modules
    return dep_modules


def parse_test_module(test: str) -> str:
    return test.split(".")[0]


def print_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)
