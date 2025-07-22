from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


CPP_TEST_PREFIX = "cpp"
CPP_TEST_PATH = "build/bin"
CPP_TESTS_DIR = os.path.abspath(os.getenv("CPP_TESTS_DIR", default=CPP_TEST_PATH))
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_test_module(test: str) -> str:
    return test.split(".")[0]


def discover_tests(
    base_dir: Path = REPO_ROOT / "test",
    cpp_tests_dir: str | Path | None = None,
    blocklisted_patterns: list[str] | None = None,
    blocklisted_tests: list[str] | None = None,
    extra_tests: list[str] | None = None,
) -> list[str]:
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

    # This supports symlinks, so we can link domain library tests to PyTorch test directory
    all_py_files = [
        Path(p) for p in glob.glob(f"{base_dir}/**/test_*.py", recursive=True)
    ]

    cpp_tests_dir = (
        f"{base_dir.parent}/{CPP_TEST_PATH}" if cpp_tests_dir is None else cpp_tests_dir
    )
    # CPP test files are located under pytorch/build/bin. Unlike Python test, C++ tests
    # are just binaries and could have any name, i.e. basic or atest
    all_cpp_files = [
        Path(p) for p in glob.glob(f"{cpp_tests_dir}/**/*", recursive=True)
    ]

    rc = [str(fname.relative_to(base_dir))[:-3] for fname in all_py_files]
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
        "distributed/bin/test_script",
        "distributed/elastic/multiprocessing/bin/test_script",
        "distributed/launcher/bin/test_script",
        "distributed/launcher/bin/test_script_init_method",
        "distributed/launcher/bin/test_script_is_torchelastic_launched",
        "distributed/launcher/bin/test_script_local_rank",
        "distributed/test_c10d_spawn",
        "distributions/test_transforms",
        "distributions/test_utils",
        "lazy/test_meta_kernel",
        "lazy/test_extract_compiled_graph",
        "test/inductor/test_aot_inductor_utils",
        "onnx/test_onnxscript_no_runtime",
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
        "doctests",
        "test_autoload_enable",
        "test_autoload_disable",
    ],
)


if __name__ == "__main__":
    print(TESTS)
