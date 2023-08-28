#!/usr/bin/env python3

import argparse
import copy
import glob
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Any, cast, Dict, List, NamedTuple, Optional, Union

import pkg_resources

import torch
import torch.distributed as dist
from packaging import version
from torch.multiprocessing import current_process, get_context
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    get_report_path,
    IS_CI,
    parser as common_parser,
    retry_shell,
    set_cwd,
    shell,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW_GRADCHECK,
)
from torch.utils import cpp_extension

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# using tools/ to optimize test run.
sys.path.insert(0, str(REPO_ROOT))
from tools.stats.export_test_times import TEST_TIMES_FILE
from tools.stats.upload_metrics import emit_metric
from tools.testing.target_determination.determinator import (
    get_test_prioritizations,
    TestPrioritizations,
)
from tools.testing.test_selections import (
    calculate_shards,
    get_test_case_configs,
    NUM_PROCS,
    ShardedTest,
    THRESHOLD,
)

HAVE_TEST_SELECTION_TOOLS = True
# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))


RERUN_DISABLED_TESTS = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
CPP_TEST_PREFIX = "cpp"
CPP_TEST_PATH = "build/bin"
DISTRIBUTED_TEST_PREFIX = "distributed"


# Note [ROCm parallel CI testing]
# https://github.com/pytorch/pytorch/pull/85770 added file-granularity parallel testing.
# In .ci/pytorch/test.sh, TEST_CONFIG == "default", CUDA and HIP_VISIBLE_DEVICES is set to 0.
# This results in multiple test files sharing the same GPU.
# This should be a supported use case for ROCm, but it exposed issues in the kernel driver resulting in hangs.
# See https://github.com/pytorch/pytorch/issues/90940.
#
# Further, ROCm self-hosted runners have up to 4 GPUs.
# Device visibility was set to 0 to match CUDA test behavior, but this was wasting available GPU resources.
# Assigning each Pool worker their own dedicated GPU avoids the ROCm oversubscription issues.
# This should also result in better overall wall clock time since all GPUs can be utilized.
def maybe_set_hip_visible_devies():
    # Special handling of ROCm GHA runners for parallel (file granularity) tests.
    if torch.version.hip:
        p = current_process()
        if p.name != "MainProcess":
            # this is a Process from a parallel Pool, not the MainProcess
            os.environ["HIP_VISIBLE_DEVICES"] = str(p._identity[0] % NUM_PROCS)


def strtobool(s):
    if s.lower() in ["", "0", "false", "off"]:
        return False
    return True


def parse_test_module(test):
    return test.split(".")[0]


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super().__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def discover_tests(
    base_dir: Optional[pathlib.Path] = None,
    cpp_tests_dir: Optional[pathlib.Path] = None,
    blocklisted_patterns: Optional[List[str]] = None,
    blocklisted_tests: Optional[List[str]] = None,
    extra_tests: Optional[List[str]] = None,
) -> List[str]:
    """
    Searches for all python files starting with test_ excluding one specified by patterns.
    If cpp_tests_dir is provided, also scan for all C++ tests under that directory. They
    are usually found in build/bin
    """

    def skip_test_p(name: str) -> bool:
        rc = False
        if blocklisted_patterns is not None:
            rc |= any(name.startswith(pattern) for pattern in blocklisted_patterns)
        if blocklisted_tests is not None:
            rc |= name in blocklisted_tests
        return rc

    cwd = pathlib.Path(__file__).resolve().parent if base_dir is None else base_dir
    # This supports symlinks, so we can link domain library tests to PyTorch test directory
    all_py_files = [
        pathlib.Path(p) for p in glob.glob(f"{cwd}/**/test_*.py", recursive=True)
    ]

    cpp_tests_dir = (
        f"{cwd.parent}/{CPP_TEST_PATH}" if cpp_tests_dir is None else cpp_tests_dir
    )
    # CPP test files are located under pytorch/build/bin. Unlike Python test, C++ tests
    # are just binaries and could have any name, i.e. basic or atest
    all_cpp_files = [
        pathlib.Path(p) for p in glob.glob(f"{cpp_tests_dir}/**/*", recursive=True)
    ]

    rc = [str(fname.relative_to(cwd))[:-3] for fname in all_py_files]
    # Add the cpp prefix for C++ tests so that we can tell them apart
    rc.extend(
        [
            parse_test_module(f"{CPP_TEST_PREFIX}/{fname.relative_to(cpp_tests_dir)}")
            for fname in all_cpp_files
        ]
    )

    # Invert slashes on Windows
    if sys.platform == "win32":
        rc = [name.replace("\\", "/") for name in rc]
    rc = [test for test in rc if not skip_test_p(test)]
    if extra_tests is not None:
        rc += extra_tests
    return sorted(rc)


CPP_TESTS_DIR = os.path.abspath(os.getenv("CPP_TESTS_DIR", default=CPP_TEST_PATH))
TESTS = discover_tests(
    cpp_tests_dir=CPP_TESTS_DIR,
    blocklisted_patterns=[
        "ao",
        "bottleneck_test",
        "custom_backend",
        "custom_operator",
        "fx",  # executed by test_fx.py
        "jit",  # executed by test_jit.py
        "mobile",
        "onnx_caffe2",
        "package",  # executed by test_package.py
        "quantization",  # executed by test_quantization.py
        "autograd",  # executed by test_autograd.py
        "_nvfuser",  # executed by test_jit_cuda_fuser.py, test_nvfuser_dynamo.py, and test_nvfuser_frontend.py
    ],
    blocklisted_tests=[
        "test_bundled_images",
        "test_cpp_extensions_aot",
        "test_determination",
        "test_jit_fuser",
        "test_jit_simple",
        "test_jit_string",
        "test_kernel_launch_checks",
        "test_nnapi",
        "test_static_runtime",
        "test_throughput_benchmark",
        "test_typing",
        "distributed/bin/test_script",
        "distributed/elastic/multiprocessing/bin/test_script",
        "distributed/launcher/bin/test_script",
        "distributed/launcher/bin/test_script_init_method",
        "distributed/launcher/bin/test_script_is_torchelastic_launched",
        "distributed/launcher/bin/test_script_local_rank",
        "distributed/test_c10d_spawn",
        "distributions/test_transforms",
        "distributions/test_utils",
        "onnx/test_pytorch_onnx_onnxruntime_cuda",
        "onnx/test_models",
        # These are not C++ tests
        f"{CPP_TEST_PREFIX}/CMakeFiles",
        f"{CPP_TEST_PREFIX}/CTestTestfile.cmake",
        f"{CPP_TEST_PREFIX}/Makefile",
        f"{CPP_TEST_PREFIX}/cmake_install.cmake",
        f"{CPP_TEST_PREFIX}/c10_intrusive_ptr_benchmark",
        f"{CPP_TEST_PREFIX}/example_allreduce",
        f"{CPP_TEST_PREFIX}/parallel_benchmark",
        f"{CPP_TEST_PREFIX}/protoc",
        f"{CPP_TEST_PREFIX}/protoc-3.13.0.0",
        f"{CPP_TEST_PREFIX}/torch_shm_manager",
        f"{CPP_TEST_PREFIX}/tutorial_tensorexpr",
    ],
    extra_tests=[
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "distributed/elastic/timer/api_test",
        "distributed/elastic/timer/local_timer_example",
        "distributed/elastic/timer/local_timer_test",
        "distributed/elastic/events/lib_test",
        "distributed/elastic/metrics/api_test",
        "distributed/elastic/utils/logging_test",
        "distributed/elastic/utils/util_test",
        "distributed/elastic/utils/distributed_test",
        "distributed/elastic/multiprocessing/api_test",
    ],
)

# The doctests are a special case that don't correspond to a file that discover
# tests can enable.
TESTS = TESTS + ["doctests"]

FSDP_TEST = [test for test in TESTS if test.startswith("distributed/fsdp")]

WINDOWS_BLOCKLIST = [
    "distributed/nn/jit/test_instantiator",
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
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
    "distributed/elastic/agent/server/test/api_test",
    "distributed/elastic/multiprocessing/api_test",
    "distributed/_shard/checkpoint/test_checkpoint"
    "distributed/_shard/checkpoint/test_file_system_checkpoint"
    "distributed/_shard/sharding_spec/test_sharding_spec",
    "distributed/_shard/sharding_plan/test_sharding_plan",
    "distributed/_shard/sharded_tensor/test_sharded_tensor",
    "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
    "distributed/_shard/sharded_tensor/ops/test_embedding",
    "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
    "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
    "distributed/_shard/sharded_tensor/ops/test_init",
    "distributed/_shard/sharded_optim/test_sharded_optim",
] + FSDP_TEST

ROCM_BLOCKLIST = [
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "distributed/_shard/checkpoint/test_checkpoint"
    "distributed/_shard/checkpoint/test_file_system_checkpoint"
    "distributed/_shard/sharding_spec/test_sharding_spec",
    "distributed/_shard/sharding_plan/test_sharding_plan",
    "distributed/_shard/sharded_tensor/test_sharded_tensor",
    "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
    "distributed/_shard/sharded_tensor/ops/test_embedding",
    "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
    "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
    "distributed/_shard/sharded_tensor/ops/test_init",
    "distributed/_shard/sharded_optim/test_sharded_optim",
    "test_determination",
    "test_jit_legacy",
    "test_cuda_nvml_based_avail",
]

# The tests inside these files should never be run in parallel with each other
RUN_PARALLEL_BLOCKLIST = [
    "test_cpp_extensions_jit",
    "test_cpp_extensions_open_device_registration",
    "test_jit_disabled",
    "test_mobile_optimizer",
    "test_multiprocessing",
    "test_multiprocessing_spawn",
    "test_namedtuple_return_api",
    "test_overrides",
    "test_show_pickle",
    "test_tensorexpr",
    "test_cuda_primary_ctx",
    "test_cuda_trace",
    "test_cuda_nvml_based_avail",
    # temporarily sets a global config
    "test_autograd_fallback",
] + FSDP_TEST

# Test files that should always be run serially with other test files,
# but it's okay if the tests inside them are run in parallel with each other.
CI_SERIAL_LIST = [
    "test_nn",
    "test_fake_tensor",
    "test_cpp_api_parity",
    "test_reductions",
    "test_cuda",
    "test_cuda_expandable_segments",
    "test_jit_cuda_fuser",  # OOM on test_issue_1785, also profiling?
    "test_indexing",
    "test_fx_backends",
    "test_linalg",
    "test_cpp_extensions_jit",
    "test_torch",
    "test_tensor_creation_ops",
    "test_sparse_csr",
    "test_dispatch",
    "test_python_dispatch",  # torch.library creation and deletion must be serialized
    "test_spectral_ops",  # Cause CUDA illegal memory access https://github.com/pytorch/pytorch/issues/88916
    "nn/test_pooling",
    "nn/test_convolution",  # Doesn't respect set_per_process_memory_fraction, results in OOM for other tests in slow gradcheck
    "distributions/test_distributions",
    "test_autograd",  # slow gradcheck runs a test that checks the cuda memory allocator
    "test_prims",  # slow gradcheck runs a test that checks the cuda memory allocator
    "test_modules",  # failed test due to mismatched elements
    "functorch/test_vmap",  # OOM
    "test_fx",  # gets SIGKILL
    "test_dataloader",  # frequently hangs for ROCm
    "test_serialization",  # test_serialization_2gb_file allocates a tensor of 2GB, and could cause OOM
    "_nvfuser/test_torchscript",  # OOM on test_issue_1785
    "test_schema_check",  # Cause CUDA illegal memory access https://github.com/pytorch/pytorch/issues/95749
    "functorch/test_memory_efficient_fusion",  # Cause CUDA OOM on ROCm
    "test_utils",  # OOM
    "test_sort_and_select",  # OOM
    "test_backward_compatible_arguments",  # OOM
    "test_module_init",  # OOM
    "test_autocast",  # OOM
    "test_native_mha",  # OOM
    "test_module_hooks",  # OOM
]
# A subset of onnx tests that cannot run in parallel due to high memory usage.
ONNX_SERIAL_LIST = [
    "onnx/test_models",
    "onnx/test_models_quantized_onnxruntime",
    "onnx/test_models_onnxruntime",
    "onnx/test_custom_ops",
    "onnx/test_utility_funs",
]

# A subset of our TEST list that validates PyTorch's ops, modules, and autograd function as expected
CORE_TEST_LIST = [
    "test_autograd",
    "test_autograd_fallback",
    "test_modules",
    "test_nn",
    "test_ops",
    "test_ops_gradients",
    "test_ops_fwd_gradients",
    "test_ops_jit",
    "test_torch",
]


# if a test file takes longer than 5 min, we add it to TARGET_DET_LIST
SLOW_TEST_THRESHOLD = 300

DISTRIBUTED_TESTS_CONFIG = {}


if dist.is_available():
    DISTRIBUTED_TESTS_CONFIG["test"] = {"WORLD_SIZE": "1"}
    if not TEST_WITH_ROCM and dist.is_mpi_available():
        DISTRIBUTED_TESTS_CONFIG["mpi"] = {
            "WORLD_SIZE": "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-mpi",
        }
    if dist.is_nccl_available():
        DISTRIBUTED_TESTS_CONFIG["nccl"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-nccl",
        }
    if dist.is_gloo_available():
        DISTRIBUTED_TESTS_CONFIG["gloo"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-gloo",
        }
    if dist.is_ucc_available():
        DISTRIBUTED_TESTS_CONFIG["ucc"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-ucc",
            "UCX_TLS": "tcp,cuda",
            "UCC_TLS": "nccl,ucp,cuda",
            "UCC_TL_UCP_TUNE": "cuda:0",  # don't use UCP TL on CUDA as it is not well supported
            "UCC_EC_CUDA_USE_COOPERATIVE_LAUNCH": "n",  # CI nodes (M60) fail if it is on
        }

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}

CPP_EXTENSIONS_ERROR = """
Ninja (https://ninja-build.org) is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`. Alternatively, disable said tests with
`run_test.py --exclude test_cpp_extensions_aot_ninja test_cpp_extensions_jit`.
"""

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))

JIT_EXECUTOR_TESTS = [
    "test_jit_profiling",
    "test_jit_legacy",
    "test_jit_fuser_legacy",
]

DISTRIBUTED_TESTS = [test for test in TESTS if test.startswith(DISTRIBUTED_TEST_PREFIX)]
FUNCTORCH_TESTS = [test for test in TESTS if test.startswith("functorch")]
ONNX_TESTS = [test for test in TESTS if test.startswith("onnx")]
CPP_TESTS = [test for test in TESTS if test.startswith(CPP_TEST_PREFIX)]

TESTS_REQUIRING_LAPACK = [
    "distributions/test_constraints",
    "distributions/test_distributions",
]

# These are just the slowest ones, this isn't an exhaustive list.
TESTS_NOT_USING_GRADCHECK = [
    # Note that you should use skipIfSlowGradcheckEnv if you do not wish to
    # skip all the tests in that file, e.g. test_mps
    "doctests",
    "test_meta",
    "test_hub",
    "test_fx",
    "test_decomp",
    "test_cpp_extensions_jit",
    "test_jit",
    "test_ops",
    "test_ops_jit",
    "dynamo/test_recompile_ux",
    "inductor/test_smoke",
    "test_quantization",
]


def print_to_stderr(message):
    print(message, file=sys.stderr)


def get_executable_command(options, disable_coverage=False, is_cpp_test=False):
    if options.coverage and not disable_coverage:
        if not is_cpp_test:
            executable = ["coverage", "run", "--parallel-mode", "--source=torch"]
        else:
            # TODO: C++ with coverage is not yet supported
            executable = []
    else:
        if not is_cpp_test:
            executable = [sys.executable, "-bb"]
        else:
            executable = ["pytest"]

    return executable


def run_test(
    test_module,
    test_directory,
    options,
    launcher_cmd=None,
    extra_unittest_args=None,
    env=None,
) -> int:
    maybe_set_hip_visible_devies()
    unittest_args = options.additional_unittest_args.copy()
    test_file = test_module
    stepcurrent_key = test_file

    use_sharded_test = False
    if isinstance(test_file, ShardedTest):
        test_file = test_module.name
        use_sharded_test = True

    is_distributed_test = test_file.startswith(DISTRIBUTED_TEST_PREFIX)
    is_cpp_test = test_file.startswith(CPP_TEST_PREFIX)
    # NB: Rerun disabled tests depends on pytest-flakefinder and it doesn't work with
    # pytest-cpp atm. We also don't have support to disable C++ test yet, so it's ok
    # to just return successfully here
    if is_cpp_test and RERUN_DISABLED_TESTS:
        print_to_stderr(
            "Skipping C++ tests when running under RERUN_DISABLED_TESTS mode"
        )
        return 0

    if use_sharded_test:
        if is_cpp_test:
            stepcurrent_key = test_file
        else:
            unittest_args.extend(
                [
                    f"--shard-id={test_module.shard - 1}",
                    f"--num-shards={test_module.num_shards}",
                ]
            )
            stepcurrent_key = f"{test_file}_{test_module.shard - 1}"

    if options.verbose:
        unittest_args.append(f'-{"v"*options.verbose}')  # in case of pytest

    if test_file in RUN_PARALLEL_BLOCKLIST:
        unittest_args = [
            arg for arg in unittest_args if not arg.startswith("--run-parallel")
        ]

    if extra_unittest_args:
        assert isinstance(extra_unittest_args, list)
        unittest_args.extend(extra_unittest_args)

    # If using pytest, replace -f with equivalent -x
    if options.pytest:
        unittest_args.extend(
            get_pytest_args(
                options,
                stepcurrent_key,
                is_cpp_test=is_cpp_test,
                is_distributed_test=is_distributed_test,
            )
        )
        unittest_args = [arg if arg != "-f" else "-x" for arg in unittest_args]

    # TODO: These features are not available for C++ test yet
    if IS_CI and not is_cpp_test:
        ci_args = ["--import-slow-tests", "--import-disabled-tests"]
        if RERUN_DISABLED_TESTS:
            ci_args.append("--rerun-disabled-tests")
        # use the downloaded test cases configuration, not supported in pytest
        unittest_args.extend(ci_args)

    if test_file in PYTEST_SKIP_RETRIES:
        if not options.pytest:
            raise RuntimeError(
                "A test running without pytest cannot skip retries using "
                "the PYTEST_SKIP_RETRIES set."
            )
        unittest_args = [arg for arg in unittest_args if "--reruns" not in arg]

    # Extra arguments are not supported with pytest
    executable = get_executable_command(options, is_cpp_test=is_cpp_test)
    if not executable:
        # If there is no eligible executable returning here, it means an unsupported
        # case such as coverage for C++ test. So just returning ok makes sense
        return 0

    if test_file.startswith(CPP_TEST_PREFIX):
        # C++ tests are not the regular test directory
        if CPP_TESTS_DIR:
            cpp_test = os.path.join(
                CPP_TESTS_DIR,
                test_file.replace(f"{CPP_TEST_PREFIX}/", ""),
            )
        else:
            cpp_test = os.path.join(
                pathlib.Path(test_directory).parent,
                CPP_TEST_PATH,
                test_file.replace(f"{CPP_TEST_PREFIX}/", ""),
            )

        argv = [
            cpp_test if sys.platform != "win32" else cpp_test + ".exe"
        ] + unittest_args
    else:
        # Can't call `python -m unittest test_*` here because it doesn't run code
        # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
        argv = [test_file + ".py"] + unittest_args

    os.makedirs(REPO_ROOT / "test" / "test-reports", exist_ok=True)
    log_fd, log_path = tempfile.mkstemp(
        dir=REPO_ROOT / "test" / "test-reports",
        prefix="{}_".format(test_file.replace("\\", "-").replace("/", "-")),
        suffix=".log",
    )
    os.close(log_fd)

    command = (launcher_cmd or []) + executable + argv
    should_file_rerun = (
        "--subprocess" not in command
        and not RERUN_DISABLED_TESTS
        and not options.continue_through_error
    )
    timeout = (
        THRESHOLD * 3
        if should_file_rerun
        and isinstance(test_module, ShardedTest)
        and test_module.time is not None
        else None
    )
    print_to_stderr(f"Executing {command} ... [{datetime.now()}]")

    with open(log_path, "w") as f:
        ret_code = retry_shell(
            command,
            test_directory,
            stdout=f,
            stderr=f,
            env=env,
            timeout=timeout,
            retries=2 if should_file_rerun else 0,
        )

        # Pytest return code 5 means no test is collected. This is needed
        # here as we use pytest directly when running C++ tests. Return
        # code 4 is ok too as this happens when the binary is not a C++
        # test executable. All binary files under build/bin that are not
        # C++ test at the time of this writing have been excluded, but we
        # can accept code 4 too just in case a new non-test binary file
        # comes up in the future.
        ret_code = 0 if ret_code == 5 or ret_code == 4 else ret_code

    print_log_file(test_module, log_path, failed=(ret_code != 0))
    os.remove(log_path)
    return ret_code


def run_test_with_subprocess(test_module, test_directory, options):
    return run_test(
        test_module, test_directory, options, extra_unittest_args=["--subprocess"]
    )


def _test_cpp_extensions_aot(test_directory, options, use_ninja):
    if use_ninja:
        try:
            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print(CPP_EXTENSIONS_ERROR)
            return 1

    # Wipe the build folder, if it exists already
    cpp_extensions_test_dir = os.path.join(test_directory, "cpp_extensions")
    cpp_extensions_test_build_dir = os.path.join(cpp_extensions_test_dir, "build")
    if os.path.exists(cpp_extensions_test_build_dir):
        shutil.rmtree(cpp_extensions_test_build_dir)

    # Build the test cpp extensions modules
    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = str(1 if use_ninja else 0)
    cmd = [sys.executable, "setup.py", "install", "--root", "./install"]
    return_code = shell(cmd, cwd=cpp_extensions_test_dir, env=shell_env)
    if return_code != 0:
        return return_code
    if sys.platform != "win32":
        return_code = shell(
            cmd,
            cwd=os.path.join(cpp_extensions_test_dir, "no_python_abi_suffix_test"),
            env=shell_env,
        )
        if return_code != 0:
            return return_code

    # "install" the test modules and run tests
    python_path = os.environ.get("PYTHONPATH", "")
    from shutil import copyfile

    os.environ["USE_NINJA"] = shell_env["USE_NINJA"]
    test_module = "test_cpp_extensions_aot" + ("_ninja" if use_ninja else "_no_ninja")
    copyfile(
        test_directory + "/test_cpp_extensions_aot.py",
        test_directory + "/" + test_module + ".py",
    )
    try:
        cpp_extensions = os.path.join(test_directory, "cpp_extensions")
        install_directory = ""
        # install directory is the one that is named site-packages
        for root, directories, _ in os.walk(os.path.join(cpp_extensions, "install")):
            for directory in directories:
                if "-packages" in directory:
                    install_directory = os.path.join(root, directory)

        assert install_directory, "install_directory must not be empty"
        os.environ["PYTHONPATH"] = os.pathsep.join([install_directory, python_path])
        return run_test(test_module, test_directory, options)
    finally:
        os.environ["PYTHONPATH"] = python_path
        if os.path.exists(test_directory + "/" + test_module + ".py"):
            os.remove(test_directory + "/" + test_module + ".py")
        os.environ.pop("USE_NINJA")


def test_cpp_extensions_aot_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=True)


def test_cpp_extensions_aot_no_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=False)


def test_distributed(test_module, test_directory, options):
    # MPI tests are broken with Python-3.9
    mpi_available = subprocess.call(
        "command -v mpiexec", shell=True
    ) == 0 and sys.version_info < (3, 9)
    if options.verbose and not mpi_available:
        print_to_stderr("MPI not available -- MPI backend tests will be skipped")

    config = DISTRIBUTED_TESTS_CONFIG
    for backend, env_vars in config.items():
        if sys.platform == "win32" and backend != "gloo":
            continue
        if backend == "mpi" and not mpi_available:
            continue
        for with_init_file in {True, False}:
            if sys.platform == "win32" and not with_init_file:
                continue
            tmp_dir = tempfile.mkdtemp()
            if options.verbose:
                init_str = "with {} init_method"
                with_init = init_str.format("file" if with_init_file else "env")
                print_to_stderr(
                    f"Running distributed tests for the {backend} backend {with_init}"
                )
            old_environ = dict(os.environ)
            os.environ["TEMP_DIR"] = tmp_dir
            os.environ["BACKEND"] = backend
            os.environ["INIT_METHOD"] = "env://"
            os.environ.update(env_vars)
            if with_init_file:
                if test_module.name == "test_distributed_spawn":
                    init_method = f"{FILE_SCHEMA}{tmp_dir}/"
                else:
                    init_method = f"{FILE_SCHEMA}{tmp_dir}/shared_init_file"
                os.environ["INIT_METHOD"] = init_method
            try:
                os.mkdir(os.path.join(tmp_dir, "barrier"))
                os.mkdir(os.path.join(tmp_dir, "test_dir"))
                if backend == "mpi":
                    # test mpiexec for --noprefix option
                    with open(os.devnull, "w") as devnull:
                        allowrunasroot_opt = (
                            "--allow-run-as-root"
                            if subprocess.call(
                                'mpiexec --allow-run-as-root -n 1 bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )
                        noprefix_opt = (
                            "--noprefix"
                            if subprocess.call(
                                f'mpiexec {allowrunasroot_opt} -n 1 --noprefix bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )

                    mpiexec = ["mpiexec", "-n", "3", noprefix_opt, allowrunasroot_opt]

                    return_code = run_test(
                        test_module, test_directory, options, launcher_cmd=mpiexec
                    )
                else:
                    return_code = run_test(
                        test_module,
                        test_directory,
                        options,
                        extra_unittest_args=["--subprocess"],
                    )
                if return_code != 0:
                    return return_code
            finally:
                shutil.rmtree(tmp_dir)
                os.environ.clear()
                os.environ.update(old_environ)
    return 0


def run_doctests(test_module, test_directory, options):
    """
    Assumes the incoming test module is called doctest, and simply executes the
    xdoctest runner on the torch library itself.
    """
    import pathlib

    import xdoctest

    pkgpath = pathlib.Path(torch.__file__).parent

    exclude_module_list = []
    enabled = {
        # TODO: expose these options to the user
        # For now disable all feature-conditional tests
        # 'lapack': 'auto',
        # 'cuda': 'auto',
        # 'cuda1': 'auto',
        # 'qengine': 'auto',
        "lapack": 0,
        "cuda": 0,
        "cuda1": 0,
        "qengine": 0,
        "autograd_profiler": 0,
        "cpp_ext": 0,
        "monitor": 0,
        "onnx": "auto",
    }

    # Resolve "auto" based on a test to determine if the feature is available.
    if enabled["cuda"] == "auto" and torch.cuda.is_available():
        enabled["cuda"] = True

    if (
        enabled["cuda1"] == "auto"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        enabled["cuda1"] = True

    if enabled["lapack"] == "auto" and torch._C.has_lapack:
        enabled["lapack"] = True

    if enabled["qengine"] == "auto":
        try:
            # Is there a better check if quantization is enabled?
            import torch.ao.nn.quantized as nnq  # NOQA: F401

            torch.backends.quantized.engine = "qnnpack"
            torch.backends.quantized.engine = "fbgemm"
        except (ImportError, RuntimeError):
            ...
        else:
            enabled["qengine"] = True

    if enabled["onnx"] == "auto":
        try:
            import onnx  # NOQA: F401
            import onnxruntime  # NOQA: F401
            import onnxscript  # NOQA: F401
        except ImportError:
            exclude_module_list.append("torch.onnx.*")
            enabled["onnx"] = False
        else:
            enabled["onnx"] = True

    # Set doctest environment variables
    if enabled["cuda"]:
        os.environ["TORCH_DOCTEST_CUDA"] = "1"

    if enabled["cuda1"]:
        os.environ["TORCH_DOCTEST_CUDA1"] = "1"

    if enabled["lapack"]:
        os.environ["TORCH_DOCTEST_LAPACK"] = "1"

    if enabled["qengine"]:
        os.environ["TORCH_DOCTEST_QENGINE"] = "1"

    if enabled["autograd_profiler"]:
        os.environ["TORCH_DOCTEST_AUTOGRAD_PROFILER"] = "1"

    if enabled["cpp_ext"]:
        os.environ["TORCH_DOCTEST_CPP_EXT"] = "1"

    if enabled["monitor"]:
        os.environ["TORCH_DOCTEST_MONITOR"] = "1"

    if enabled["onnx"]:
        os.environ["TORCH_DOCTEST_ONNX"] = "1"

    if 0:
        # TODO: could try to enable some of these
        os.environ["TORCH_DOCTEST_QUANTIZED_DYNAMIC"] = "1"
        os.environ["TORCH_DOCTEST_ANOMALY"] = "1"
        os.environ["TORCH_DOCTEST_AUTOGRAD"] = "1"
        os.environ["TORCH_DOCTEST_HUB"] = "1"
        os.environ["TORCH_DOCTEST_DATALOADER"] = "1"
        os.environ["TORCH_DOCTEST_FUTURES"] = "1"

    pkgpath = os.path.dirname(torch.__file__)

    xdoctest_config = {
        "global_exec": r"\n".join(
            [
                "from torch import nn",
                "import torch.nn.functional as F",
                "import torch",
            ]
        ),
        "analysis": "static",  # set to "auto" to test doctests in compiled modules
        "style": "google",
        "options": "+IGNORE_WHITESPACE",
    }
    xdoctest_verbose = max(1, options.verbose)
    run_summary = xdoctest.runner.doctest_module(
        os.fspath(pkgpath),
        config=xdoctest_config,
        verbose=xdoctest_verbose,
        command=options.xdoctest_command,
        argv=[],
        exclude=exclude_module_list,
    )
    result = 1 if run_summary.get("n_failed", 0) else 0
    return result


def print_log_file(test: str, file_path: str, failed: bool) -> None:
    num_lines = sum(1 for _ in open(file_path, "rb"))
    test = str(test)
    n = 100
    with open(file_path, "rb") as f:
        print_to_stderr("")
        if failed:
            if n < num_lines:
                print_to_stderr(
                    f"Expand the folded group to see the beginning of the log file of {test}"
                )
                print_to_stderr(
                    f"##[group]PRINTING BEGINNING OF LOG FILE of {test} ({file_path})"
                )
                for _ in range(num_lines - n):
                    print_to_stderr(next(f).decode("utf-8", errors="ignore").rstrip())
                print_to_stderr("##[endgroup]")
            for _ in range(min(n, num_lines)):
                print_to_stderr(next(f).decode("utf-8", errors="ignore").rstrip())
            print_to_stderr(f"FINISHED PRINTING LOG FILE of {test} ({file_path})")
        else:
            print_to_stderr(f"Expand the folded group to see the log file of {test}")
            print_to_stderr(f"##[group]PRINTING LOG FILE of {test} ({file_path})")
            print_to_stderr(f.read().decode("utf-8", errors="ignore"))
            print_to_stderr("##[endgroup]")
            print_to_stderr(f"FINISHED PRINTING LOG FILE of {test} ({file_path})")
        print_to_stderr("")


def get_pytest_args(
    options, stepcurrent_key, is_cpp_test=False, is_distributed_test=False
):
    if RERUN_DISABLED_TESTS:
        # Distributed tests are too slow, so running them x50 will cause the jobs to timeout after
        # 3+ hours. So, let's opt for less number of reruns. We need at least 150 instances of the
        # test every 2 weeks to satisfy the Rockset query (15 x 14 = 210). The same logic applies
        # to ASAN, which is also slow
        count = 15 if is_distributed_test or TEST_WITH_ASAN else 50
        # When under rerun-disabled-tests mode, run the same tests multiple times to determine their
        # flakiness status. Default to 50 re-runs
        rerun_options = ["--flake-finder", f"--flake-runs={count}"]
    elif options.continue_through_error:
        # If continue through error, don't stop on first failure
        rerun_options = ["--reruns=2"]
    else:
        # When under the normal mode, retry a failed test 2 more times. -x means stop at the first
        # failure
        rerun_options = ["-x", "--reruns=2"]

    pytest_args = [
        "-vv",
        "-rfEX",
    ]
    if not is_cpp_test:
        # C++ tests need to be run with pytest directly, not via python
        pytest_args.extend(["-p", "no:xdist", "--use-pytest"])
        if not options.continue_through_error and IS_CI:
            pytest_args.append(f"--sc={stepcurrent_key}")
    else:
        # Use pytext-dist to run C++ tests in parallel as running them sequentially using run_test
        # is much slower than running them directly
        pytest_args.extend(["-n", str(NUM_PROCS)])

        if IS_CI:
            # Add the option to generate XML test report here as C++ tests
            # won't go into common_utils
            test_report_path = get_report_path(pytest=True)
            pytest_args.extend(["--junit-xml-reruns", test_report_path])

    if options.pytest_k_expr:
        pytest_args.extend(["-k", options.pytest_k_expr])

    pytest_args.extend(rerun_options)
    return pytest_args


CUSTOM_HANDLERS = {
    "test_cuda_primary_ctx": run_test_with_subprocess,
    "test_cuda_nvml_based_avail": run_test_with_subprocess,
    "test_cuda_trace": run_test_with_subprocess,
    "test_cpp_extensions_aot_no_ninja": test_cpp_extensions_aot_no_ninja,
    "test_cpp_extensions_aot_ninja": test_cpp_extensions_aot_ninja,
    "distributed/test_distributed_spawn": test_distributed,
    "distributed/algorithms/quantization/test_quantization": test_distributed,
    "distributed/test_c10d_nccl": run_test_with_subprocess,
    "distributed/test_c10d_gloo": run_test_with_subprocess,
    "distributed/test_c10d_ucc": run_test_with_subprocess,
    "distributed/test_c10d_common": run_test_with_subprocess,
    "distributed/test_c10d_spawn_gloo": run_test_with_subprocess,
    "distributed/test_c10d_spawn_nccl": run_test_with_subprocess,
    "distributed/test_c10d_spawn_ucc": run_test_with_subprocess,
    "distributed/test_store": run_test_with_subprocess,
    "distributed/test_pg_wrapper": run_test_with_subprocess,
    "distributed/rpc/test_faulty_agent": run_test_with_subprocess,
    "distributed/rpc/test_tensorpipe_agent": run_test_with_subprocess,
    "distributed/rpc/test_share_memory": run_test_with_subprocess,
    "distributed/rpc/cuda/test_tensorpipe_agent": run_test_with_subprocess,
    "doctests": run_doctests,
}


PYTEST_SKIP_RETRIES = {"test_public_bindings"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(TESTS)),
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[common_parser],
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print verbose information and test-by-test results",
    )
    parser.add_argument("--jit", "--jit", action="store_true", help="run all jit tests")
    parser.add_argument(
        "--distributed-tests",
        "--distributed-tests",
        action="store_true",
        help="Run all distributed tests",
    )
    parser.add_argument(
        "--functorch",
        "--functorch",
        action="store_true",
        help=(
            "If this flag is present, we will only run functorch tests. "
            "If this flag is not present, we will run all tests "
            "(including functorch tests)."
        ),
    )
    parser.add_argument(
        "--mps",
        "--mps",
        action="store_true",
        help=("If this flag is present, we will only run test_mps and test_metal"),
    )
    parser.add_argument(
        "--cpp",
        "--cpp",
        action="store_true",
        help=("If this flag is present, we will only run C++ tests"),
    )
    parser.add_argument(
        "-core",
        "--core",
        action="store_true",
        help="Only run core tests, or tests that validate PyTorch's ops, modules,"
        "and autograd. They are defined by CORE_TEST_LIST.",
    )
    parser.add_argument(
        "--onnx",
        "--onnx",
        action="store_true",
        help=(
            "Only run ONNX tests, or tests that validate PyTorch's ONNX export. "
            "If this flag is not present, we will exclude ONNX tests."
        ),
    )
    parser.add_argument(
        "-k",
        "--pytest-k-expr",
        default="",
        help="Pass to pytest as its -k expr argument",
    )
    parser.add_argument(
        "-c",
        "--coverage",
        action="store_true",
        help="enable coverage",
        default=PYTORCH_COLLECT_COVERAGE,
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
        " tests must be a part of the TESTS list defined in run_test.py",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        choices=TESTS,
        metavar="TESTS",
        default=[],
        help="select a set of tests to exclude",
    )
    parser.add_argument(
        "-f",
        "--first",
        choices=TESTS,
        metavar="TESTS",
        help="select the test to start from (excludes previous tests)",
    )
    parser.add_argument(
        "-l",
        "--last",
        choices=TESTS,
        metavar="TESTS",
        help="select the last test to run (excludes following tests)",
    )
    parser.add_argument(
        "--bring-to-front",
        nargs="+",
        choices=TestChoices(TESTS),
        default=[],
        metavar="TESTS",
        help="select a set of tests to run first. This can be used in situations"
        " where you want to run all tests, but care more about some set, "
        "e.g. after making a change to a specific component",
    )
    parser.add_argument(
        "--ignore-win-blocklist",
        action="store_true",
        help="always run blocklisted windows tests",
    )
    # NS: Disable target determination until it can be made more reliable
    # parser.add_argument(
    #     "--determine-from",
    #     help="File of affected source filenames to determine which tests to run.",
    # )
    parser.add_argument(
        "--continue-through-error",
        "--keep-going",
        action="store_true",
        help="Runs the full test suite despite one of the tests failing",
        default=strtobool(os.environ.get("CONTINUE_THROUGH_ERROR", "False")),
    )
    parser.add_argument(
        "additional_unittest_args",
        nargs="*",
        help="additional arguments passed through to unittest, e.g., "
        "python run_test.py -i sparse -- TestSparse.test_factory_size_check",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        help="runs a shard of the tests (taking into account other selections), e.g., "
        "--shard 2 3 will break up the selected tests into 3 shards and run the tests "
        "in the 2nd shard (the first number should not exceed the second)",
    )
    parser.add_argument(
        "--exclude-jit-executor",
        action="store_true",
        help="exclude tests that are run for a specific jit config",
    )
    parser.add_argument(
        "--exclude-distributed-tests",
        action="store_true",
        help="exclude distributed tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the test that will run.",
    )
    parser.add_argument(
        "--xdoctest-command",
        default="all",
        help=(
            "Control the specific doctest action. "
            "Use 'list' to simply parse doctests and check syntax. "
            "Use 'all' to execute all doctests or specify a specific "
            "doctest to run"
        ),
    )
    parser.add_argument(
        "--no-translation-validation",
        action="store_false",
        help="Run tests without translation validation.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dynamo",
        action="store_true",
        help="Run tests with TorchDynamo+EagerBackend turned on",
    )
    group.add_argument(
        "--inductor",
        action="store_true",
        help="Run tests with TorchInductor turned on",
    )

    return parser.parse_args()


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Args:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurrence of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(
    exclude_list, selected_tests, exclude_message=None, exact_match=False
):
    for exclude_test in exclude_list:
        tests_copy = selected_tests[:]
        for test in tests_copy:
            if (
                not exact_match and test.startswith(exclude_test)
            ) or test == exclude_test:
                if exclude_message is not None:
                    print_to_stderr(f"Excluding {test} {exclude_message}")
                selected_tests.remove(test)
    return selected_tests


def must_serial(file: Union[str, ShardedTest]) -> bool:
    if isinstance(file, ShardedTest):
        file = file.name
    return (
        os.getenv("PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL", "0") == "1"
        or DISTRIBUTED_TEST_PREFIX in os.getenv("TEST_CONFIG", "")
        or DISTRIBUTED_TEST_PREFIX in file
        or file in CUSTOM_HANDLERS
        or file in RUN_PARALLEL_BLOCKLIST
        or file in CI_SERIAL_LIST
        or file in JIT_EXECUTOR_TESTS
        or file in ONNX_SERIAL_LIST
    )


def can_run_in_pytest(test):
    return os.getenv("PYTORCH_TEST_DO_NOT_USE_PYTEST", "0") == "0"


def get_selected_tests(options) -> List[ShardedTest]:
    selected_tests = options.include

    # filter if there's JIT only and distributed only test options
    if options.jit:
        selected_tests = list(
            filter(lambda test_name: "jit" in test_name, selected_tests)
        )

    if options.distributed_tests:
        selected_tests = list(
            filter(lambda test_name: test_name in DISTRIBUTED_TESTS, selected_tests)
        )

    # Filter to only run core tests when --core option is specified
    if options.core:
        selected_tests = list(
            filter(lambda test_name: test_name in CORE_TEST_LIST, selected_tests)
        )

    # Filter to only run functorch tests when --functorch option is specified
    if options.functorch:
        selected_tests = [tname for tname in selected_tests if tname in FUNCTORCH_TESTS]

    if options.cpp:
        selected_tests = [tname for tname in selected_tests if tname in CPP_TESTS]
    else:
        # Exclude all C++ tests otherwise as they are still handled differently
        # than Python test at the moment
        options.exclude.extend(CPP_TESTS)

    if options.mps:
        selected_tests = ["test_mps", "test_metal", "test_modules"]
    else:
        # Exclude all mps tests otherwise
        options.exclude.extend(["test_mps", "test_metal"])

    # Filter to only run onnx tests when --onnx option is specified
    onnx_tests = [tname for tname in selected_tests if tname in ONNX_TESTS]
    if options.onnx:
        selected_tests = onnx_tests
    else:
        # Exclude all onnx tests otherwise
        options.exclude.extend(onnx_tests)

    # process reordering
    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(
            filter(lambda name: name not in to_front, selected_tests)
        )

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[: last_index + 1]

    # process exclusion
    if options.exclude_jit_executor:
        options.exclude.extend(JIT_EXECUTOR_TESTS)

    if options.exclude_distributed_tests:
        options.exclude.extend(DISTRIBUTED_TESTS)

    # these tests failing in CUDA 11.6 temporary disabling. issue https://github.com/pytorch/pytorch/issues/75375
    if torch.version.cuda is not None and version.parse(
        torch.version.cuda
    ) >= version.parse("11.6"):
        options.exclude.extend(["distributions/test_constraints"])

    selected_tests = exclude_tests(options.exclude, selected_tests)

    if sys.platform == "win32" and not options.ignore_win_blocklist:
        target_arch = os.environ.get("VSCMD_ARG_TGT_ARCH")
        if target_arch != "x64":
            WINDOWS_BLOCKLIST.append("cpp_extensions_aot_no_ninja")
            WINDOWS_BLOCKLIST.append("cpp_extensions_aot_ninja")
            WINDOWS_BLOCKLIST.append("cpp_extensions_jit")
            WINDOWS_BLOCKLIST.append("jit")
            WINDOWS_BLOCKLIST.append("jit_fuser")

        selected_tests = exclude_tests(WINDOWS_BLOCKLIST, selected_tests, "on Windows")

    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLOCKLIST, selected_tests, "on ROCm")

    # skip all distributed tests if distributed package is not available.
    if not dist.is_available():
        selected_tests = exclude_tests(
            DISTRIBUTED_TESTS,
            selected_tests,
            "PyTorch is built without distributed support.",
        )

    # skip tests that require LAPACK when it's not available
    if not torch._C.has_lapack:
        selected_tests = exclude_tests(
            TESTS_REQUIRING_LAPACK,
            selected_tests,
            "PyTorch is built without LAPACK support.",
        )

    if TEST_WITH_SLOW_GRADCHECK:
        selected_tests = exclude_tests(
            TESTS_NOT_USING_GRADCHECK,
            selected_tests,
            "Running in slow gradcheck mode, skipping tests "
            "that don't use gradcheck.",
            exact_match=True,
        )

    selected_tests = [parse_test_module(x) for x in selected_tests]
    return selected_tests


def download_test_times(file: str = TEST_TIMES_FILE) -> Dict[str, float]:
    # Download previous test times to make sharding decisions
    path = os.path.join(str(REPO_ROOT), file)
    if not os.path.exists(path):
        print("::warning:: Failed to find test times file. Using round robin sharding.")
        return {}

    with open(path) as f:
        test_times_file = cast(Dict[str, Any], json.load(f))
    build_environment = os.environ.get("BUILD_ENVIRONMENT")
    test_config = os.environ.get("TEST_CONFIG")
    if test_config in test_times_file.get(build_environment, {}):
        print("Found test times from artifacts")
        return test_times_file[build_environment][test_config]
    elif test_config in test_times_file["default"]:
        print(
            f"::warning:: Gathered no stats from artifacts for {build_environment} build env"
            f" and {test_config} test config. Using default build env and {test_config} test config instead."
        )
        return test_times_file["default"][test_config]
    else:
        print(
            f"::warning:: Gathered no stats from artifacts for build env {build_environment} build env"
            f" and {test_config} test config. Using default build env and default test config instead."
        )
        return test_times_file["default"]["default"]


def do_sharding(
    options,
    selected_tests: List[str],
    test_file_times: Dict[str, float],
    sort_by_time: bool = True,
) -> List[ShardedTest]:
    which_shard, num_shards = 1, 1
    if options.shard:
        assert len(options.shard) == 2, "Unexpected shard format"
        assert min(options.shard) > 0, "Shards must be positive numbers"
        which_shard, num_shards = options.shard
        assert (
            which_shard <= num_shards
        ), "Selected shard must be less than or equal to total number of shards"

    # Do sharding
    shards = calculate_shards(
        num_shards,
        selected_tests,
        test_file_times,
        must_serial=must_serial,
        sort_by_time=sort_by_time,
    )
    _, tests_from_shard = shards[which_shard - 1]
    selected_tests = tests_from_shard

    return selected_tests


class TestFailure(NamedTuple):
    test: str
    message: str


def run_test_module(
    test: Union[ShardedTest, str], test_directory: str, options
) -> Optional[TestFailure]:
    maybe_set_hip_visible_devies()

    # Printing the date here can help diagnose which tests are slow
    print_to_stderr(f"Running {str(test)} ... [{datetime.now()}]")
    handler = CUSTOM_HANDLERS.get(
        test.name if isinstance(test, ShardedTest) else test, run_test
    )
    return_code = handler(test, test_directory, options)
    assert isinstance(return_code, int) and not isinstance(
        return_code, bool
    ), f"While running {str(test)} got non integer return code {return_code}"
    if return_code == 0:
        return None

    message = f"{str(test)} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return TestFailure(test, message)


def run_tests(
    selected_tests: List[ShardedTest],
    test_directory: str,
    options,
    failures: List[TestFailure],
) -> None:
    if len(selected_tests) == 0:
        return

    # parallel = in parallel with other files
    # serial = this file on it's own.  The file might still be run in parallel with itself (ex test_ops)
    selected_tests_parallel = [x for x in selected_tests if not must_serial(x)]
    selected_tests_serial = [
        x for x in selected_tests if x not in selected_tests_parallel
    ]

    # See Note [ROCm parallel CI testing]
    pool = get_context("spawn").Pool(
        NUM_PROCS, maxtasksperchild=None if torch.version.hip else 1
    )

    # NB: This is a hack to make conftest.py available on CPP_TESTS_DIR. We should
    # see if the file could be turned into a full-fledge ptest plugin instead
    cpp_conftest_file = os.path.join(CPP_TESTS_DIR, "conftest.py")
    if (
        options.cpp
        and os.path.exists(CPP_TESTS_DIR)
        and os.path.isdir(CPP_TESTS_DIR)
        and not os.path.exists(cpp_conftest_file)
    ):
        # Take the conftest file from the test directory
        shutil.copy(os.path.join(test_directory, "conftest.py"), cpp_conftest_file)

    def handle_error_messages(failure: Optional[TestFailure]):
        if failure is None:
            return False
        failures.append(failure)
        print_to_stderr(failure.message)
        return True

    def parallel_test_completion_callback(failure):
        test_failed = handle_error_messages(failure)
        if (
            test_failed
            and not options.continue_through_error
            and not RERUN_DISABLED_TESTS
        ):
            pool.terminate()

    try:
        os.environ["NUM_PARALLEL_PROCS"] = str(NUM_PROCS)
        for test in selected_tests_parallel:
            options_clone = copy.deepcopy(options)
            if can_run_in_pytest(test):
                options_clone.pytest = True
            pool.apply_async(
                run_test_module,
                args=(test, test_directory, options_clone),
                callback=parallel_test_completion_callback,
            )
        pool.close()
        pool.join()
        del os.environ["NUM_PARALLEL_PROCS"]

        if (
            not options.continue_through_error
            and not RERUN_DISABLED_TESTS
            and len(failures) != 0
        ):
            raise RuntimeError(
                "\n".join(x.message for x in failures)
                + "\n\nTip: You can keep running tests even on failure by "
                "passing --keep-going to run_test.py.\n"
                "If running on CI, add the 'keep-going' label to "
                "your PR and rerun your jobs."
            )

        for test in selected_tests_serial:
            options_clone = copy.deepcopy(options)
            if can_run_in_pytest(test):
                options_clone.pytest = True
            failure = run_test_module(test, test_directory, options_clone)
            test_failed = handle_error_messages(failure)
            if (
                test_failed
                and not options.continue_through_error
                and not RERUN_DISABLED_TESTS
            ):
                raise RuntimeError(failure.message)

    finally:
        pool.terminate()
        pool.join()

    return


def check_pip_packages() -> None:
    packages = [
        "pytest-rerunfailures",
        "pytest-shard",
        "pytest-flakefinder",
        "pytest-xdist",
    ]
    installed_packages = [i.key for i in pkg_resources.working_set]
    for package in packages:
        if package not in installed_packages:
            print(
                f"Missing pip dependency: {package}, please run `pip install -r .ci/docker/requirements-ci.txt`"
            )
            sys.exit(1)


def main():
    check_pip_packages()

    options = parse_args()

    test_directory = str(REPO_ROOT / "test")
    selected_tests = get_selected_tests(options)

    if options.coverage and not PYTORCH_COLLECT_COVERAGE:
        shell(["coverage", "erase"])

    test_prioritization: TestPrioritizations = TestPrioritizations(
        unranked_relevance=selected_tests.copy()
    )
    metrics_dict = {}
    if IS_CI:
        # downloading test cases configuration to local environment
        get_test_case_configs(dirpath=test_directory)
        test_prioritization = get_test_prioritizations(selected_tests)

        metrics_dict = {
            "highly_relevant_tests": test_prioritization.highly_relevant,
            "probably_relevant_tests": test_prioritization.probably_relevant,
            "unranked_relevance_tests": test_prioritization.unranked_relevance,
            "cpp": options.cpp,
        }

    class TestBatch:
        """Defines a set of tests with similar priority that should be run together on the current shard"""

        name: str
        sharded_tests: List[ShardedTest]
        failures: List[TestFailure]

        def __init__(self, name: str, raw_tests: List[str], should_sort_shard: bool):
            self.name = name
            self.failures = []
            self.sharded_tests = do_sharding(
                options, raw_tests, test_times_dict, sort_by_time=should_sort_shard
            )

    test_times_dict = download_test_times(TEST_TIMES_FILE)
    test_batches: List[TestBatch] = []

    # Each batch will be run sequentially
    test_batches = [
        TestBatch("highly_relevant", test_prioritization.highly_relevant, False),
        TestBatch("probably_relevant", test_prioritization.probably_relevant, False),
        TestBatch("unranked_relevance", test_prioritization.unranked_relevance, True),
    ]

    if options.dry_run:
        return

    if options.dynamo:
        os.environ["PYTORCH_TEST_WITH_DYNAMO"] = "1"
    elif options.inductor:
        os.environ["PYTORCH_TEST_WITH_INDUCTOR"] = "1"

    if not options.no_translation_validation:
        os.environ["PYTORCH_TEST_WITH_TV"] = "1"

    os.makedirs(REPO_ROOT / "test" / "test-reports", exist_ok=True)

    try:
        # Actually run the tests
        start_time = time.time()
        for test_batch in test_batches:
            print(f"Starting test batch '{test_batch.name}'")
            metrics_dict[f"{test_batch.name}_start_time"] = time.time() - start_time
            run_tests(
                test_batch.sharded_tests, test_directory, options, test_batch.failures
            )
            metrics_dict[f"{test_batch.name}_failures"] = [
                x.test for x in test_batch.failures
            ]

    finally:
        if options.coverage:
            from coverage import Coverage

            with set_cwd(test_directory):
                cov = Coverage()
                if PYTORCH_COLLECT_COVERAGE:
                    cov.load()
                cov.combine(strict=False)
                cov.save()
                if not PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()

        if IS_CI:
            emit_metric("td_experiment_1", metrics_dict)

    all_failures = [failure for batch in test_batches for failure in batch.failures]

    if len(all_failures) != 0:
        for _, err in all_failures:
            print_to_stderr(err)

        # A disabled test is expected to fail, so there is no need to report a failure here
        if not RERUN_DISABLED_TESTS:
            sys.exit(1)


if __name__ == "__main__":
    main()
