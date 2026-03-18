#!/usr/bin/env python3
"""Run PyTorch's device-generic test suite against the openreg backend.

This script serves as both the CI launcher for openreg and a template for
out-of-tree PrivateUse1 backends to run PyTorch's device-generic tests.

Usage:
    python run_openreg_tests.py                          # run all device-generic tests
    python run_openreg_tests.py test_torch.py            # run specific file(s)
    python run_openreg_tests.py --list                   # print discovered test files
    python run_openreg_tests.py -c                       # don't stop on first failure
    python run_openreg_tests.py --timeout 30             # per-test timeout in seconds
    python run_openreg_tests.py --retries 3              # retry failed tests N times

Prerequisites:
    - PyTorch must be built and installed
    - torch_openreg will be auto-installed if missing
"""

import argparse
from collections import defaultdict
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import sysconfig
import threading
import time

PYTORCH_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(PYTORCH_ROOT, "test")
OPENREG_DIR = os.path.join(
    PYTORCH_ROOT,
    "test",
    "cpp_extensions",
    "open_registration_extension",
    "torch_openreg",
)

# Tests to skip, split into "later" (revisit in a future pass) and
# "don't care" (not relevant for openreg). Both are combined into
# BLOCKLIST_DIRS / BLOCKLIST_FILES for runtime filtering.
#

# --- Later: directories ---
_LATER_DIRS = {
    # Core areas that need refactoring to use instantiate_device_type_tests
    "autograd",
    "custom_operator",
    "optim",
    # Other subsystems
    "distributed",
    "dynamo",
    "inductor",
    "export",
    "profiler",
    "cpp_extensions",
    "benchmark_utils",
    "distributions",
    "higher_order_ops",
    # Some aotdispatch tests
    "functorch",
    # Only relevant file is experimental/test_floatx.py
    "quantization",
}

# --- Later: individual files ---
_LATER_FILES = {
    # PT2 infra
    "test_content_store.py",
    "test_decomp.py",
    "test_fake_tensor.py",
    "test_functionalization_of_rng_ops.py",
    "test_fx.py",
    "test_ops_unbacked.py",
    "test_proxy_tensor.py",
    "test_dynamic_shapes.py",
    "test_compile_benchmark_util.py",
    # Revisit for device-genericity
    "test_matmul_cuda.py",
    "test_scaled_matmul_cuda.py",
    "test_varlen_attention.py",
    "test_cpp_extensions_stream_and_event.py",
    # Low priority
    "test_complex.py",
    "test_linalg.py",
    "test_spectral_ops.py",
    "test_foreach.py",
    "test_prims.py",
    "test_dlpack.py",
    "test_fx_experimental.py",
    # Autograd / functional
    "test_autograd_fallback.py",
    "test_functional_autograd_benchmark.py",
    "test_functionalization.py",
    # Module / misc
    "test_opaque_obj_v2.py",
    "test_stateless.py",
    "test_out_dtype_op.py",
    "test_torchfuzz_repros.py",
    "nn/test_lazy_modules.py",
    "nn/test_load_state_dict.py",
    "nn/test_module_hooks.py",
    # Utils
    "test_module_tracker.py",
    "test_flop_counter.py",
    # Core (needs refactoring to be device-generic)
    "test_autocast.py",
    "nn/test_packed_sequence.py",
    "nn/attention/test_open_registry.py",
}

# --- Don't care: directories ---
_DONT_CARE_DIRS = {
    "onnx",
    "xpu",
    "complex_tensor",
    "lazy",
    "ao",
    "backends",
    "custom_backend",
    "fx",
    "jit",
    "mobile",
    "package",
    "python_native",
    "torch_np",
    "typing",
}

# --- Don't care: individual files ---
_DONT_CARE_FILES = {
    # Hardware-specific
    "nn/attention/test_fa3.py",  # maybe revisit
    "nn/attention/test_fa4.py",  # maybe revisit
    "test_cuda.py",
    "test_cuda_compatibility.py",
    "test_cuda_expandable_segments.py",
    "test_cuda_multigpu.py",
    "test_cuda_nvml_based_avail.py",
    "test_cuda_primary_ctx.py",
    "test_cuda_sanitizer.py",
    "test_cuda_trace.py",
    "test_jiterator.py",
    "test_kernel_launch_checks.py",
    "test_metal.py",
    "test_mkl_verbose.py",
    "test_mkldnn.py",
    "test_mkldnn_fusion.py",
    "test_mkldnn_verbose.py",
    "test_mps.py",
    "test_nnapi.py",
    "test_numa_binding.py",
    "test_numba_integration.py",
    "test_openmp.py",
    "test_sparse_semi_structured.py",
    "test_vulkan.py",
    "test_xnnpack_integration.py",
    "test_xpu.py",
    "test_xpu_expandable_segments.py",
    # JIT / TorchScript (legacy)
    "test_jit.py",
    "test_jit_autocast.py",
    "test_jit_disabled.py",
    "test_jit_fuser.py",
    "test_jit_fuser_legacy.py",
    "test_jit_fuser_te.py",
    "test_jit_legacy.py",
    "test_jit_llga_fuser.py",
    "test_jit_profiling.py",
    "test_jit_simple.py",
    "test_jit_string.py",
    "test_ops_jit.py",
    # Legacy features
    "test_masked.py",
    "test_maskedtensor.py",
    "test_nestedtensor.py",
    "test_legacy_vmap.py",
    "test_namedtensor.py",
    "test_sparse.py",
    "test_sparse_csr.py",
    "test_expanded_weights.py",
    "nn/test_pruning.py",
    # Core infra (test once on CPU)
    "test_accelerator.py",
    "test_comparison_utils.py",
    "test_dispatch.py",
    "test_extension_utils.py",
    "test_function_schema.py",
    "test_meta.py",
    "test_native_functions.py",
    "test_overrides.py",
    "test_per_overload_api.py",
    "test_privateuseone_python_backend.py",
    "test_public_bindings.py",
    "test_pytree.py",
    "test_rename_privateuse1_to_existing_device.py",
    "test_schema_check.py",
    "test_subclass.py",
    "test_type_hints.py",
    "test_type_info.py",
    "test_typing.py",
    # C++ extensions / static runtime / mobile
    "test_cpp_api_parity.py",
    "test_cpp_extensions_aot.py",
    "test_cpp_extensions_jit.py",
    "test_cpp_extensions_mtia_backend.py",
    "test_mobile_optimizer.py",
    "test_static_runtime.py",
    # FX (CPU-only graph transforms)
    "test_fx_graph_print.py",
    "test_fx_passes.py",
    "test_fx_reinplace_pass.py",
    # TensorExpr (legacy compiler)
    "test_tensorexpr.py",
    "test_tensorexpr_pybind.py",
    # No device logic at all
    "test_ao_sparsity.py",
    "test_appending_byte_serializer.py",
    "test_as_strided.py",
    "test_autoload.py",
    "test_bundled_images.py",
    "test_bundled_inputs.py",
    "test_ci_sanity_check_fail.py",
    "test_datapipe.py",
    "test_determination.py",
    "test_file_check.py",
    "test_functional_optim.py",
    "test_futures.py",
    "test_hop_infra.py",
    "test_hub.py",
    "test_import_stats.py",
    "test_itt.py",
    "test_license.py",
    "test_logging.py",
    "test_model_exports_to_core_aten.py",
    "test_monitor.py",
    "test_multiprocessing.py",
    "test_multiprocessing_spawn.py",
    "test_namedtuple_return_api.py",
    "test_package.py",
    "test_pruning_op.py",
    "test_quantization.py",
    "test_set_default_mobile_cpu_allocator.py",
    "test_show_pickle.py",
    "test_sympy_utils.py",
    "test_tensorboard.py",
    "test_throughput_benchmark.py",
    "test_torch_config_hash_determinism.py",
    "test_utils_config_module.py",
    "test_utils_filelock.py",
    "test_weak.py",
}

# Combined blocklists used at runtime.
BLOCKLIST_DIRS = _LATER_DIRS | _DONT_CARE_DIRS
BLOCKLIST_FILES = _LATER_FILES | _DONT_CARE_FILES


def _discover_tests() -> list[str]:
    """Find all non-blocklisted test files under test/.

    Currently discovers by walking the test directory and excluding
    blocklisted dirs/files. Long-term, we should move to discovering tests
    based on instantiate_device_type_tests usage, which is the canonical
    way to mark a test as device-generic. This would let new test files
    be auto-excluded unless they opt in, removing the need to maintain
    a blocklist.    """
    result = []
    for root, dirs, files in os.walk(TEST_DIR):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith((".", "__"))]
        rel_dir = os.path.relpath(root, TEST_DIR)
        top_dir = rel_dir.split(os.sep)[0] if rel_dir != "." else None

        if top_dir in BLOCKLIST_DIRS:
            dirs.clear()
            continue

        for f in sorted(files):
            if not (f.startswith("test_") and f.endswith(".py")):
                continue
            rel = f if rel_dir == "." else os.path.join(rel_dir, f)
            if rel in BLOCKLIST_FILES:
                continue
            result.append(rel)
    return result


# Threshold for sub-sharding a single test file (seconds).
# Same as THRESHOLD in tools/testing/test_selections.py.
_SHARD_THRESHOLD = 600


def _load_test_times() -> dict[str, float]:
    """Load test file timing data, same source as run_test.py.

    Falls back to empty dict if the data can't be downloaded (e.g. running
    locally without internet). In that case, no sub-sharding is performed.
    """
    try:
        from tools.stats.import_test_stats import get_test_times

        all_times = get_test_times()
        # Use the "default" config as a proxy for openreg runtimes.
        # The relative sizes of test files are similar across devices.
        return all_times.get("default", {}).get("default", {})
    except Exception:
        return {}


class _ShardedTest:
    """A test file, possibly sub-sharded."""

    def __init__(self, test_file: str, shard_id: int = 1, num_shards: int = 1,
                 estimated_time: float = 60):
        self.test_file = test_file
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.estimated_time = estimated_time

    @property
    def name(self) -> str:
        if self.num_shards == 1:
            return self.test_file
        return f"{self.test_file} ({self.shard_id}/{self.num_shards})"

    def pytest_args(self) -> list[str]:
        if self.num_shards == 1:
            return []
        return [f"--shard-id={self.shard_id}", f"--num-shards={self.num_shards}"]


def _shard_tests(
    test_files: list[str], shard_id: int, num_shards: int
) -> list[_ShardedTest]:
    """Split test files across CI shards, sub-sharding large files.

    Uses the same test timing data as run_test.py to determine which files
    need sub-sharding and how to balance work across shards.
    """
    test_times = _load_test_times()

    # Step 1: expand large files into sub-shards based on estimated runtime
    items: list[_ShardedTest] = []
    for test_file in test_files:
        # test_times keys don't have .py extension
        key = test_file.replace(".py", "")
        est = test_times.get(key, 60)
        if est > _SHARD_THRESHOLD:
            n = math.ceil(est / _SHARD_THRESHOLD)
            for i in range(1, n + 1):
                items.append(_ShardedTest(test_file, i, n, est / n))
        else:
            items.append(_ShardedTest(test_file, estimated_time=est))

    if num_shards == 1:
        return items

    # Step 2: greedy bin-pack by estimated time (longest first)
    items.sort(key=lambda x: -x.estimated_time)
    shards: list[tuple[float, list[_ShardedTest]]] = [
        (0.0, []) for _ in range(num_shards)
    ]
    for item in items:
        min_idx = min(range(num_shards), key=lambda i: shards[i][0])
        t, lst = shards[min_idx]
        shards[min_idx] = (t + item.estimated_time, lst)
        lst.append(item)

    # Step 3: return items for our shard
    _, our_items = shards[shard_id - 1]
    return our_items


def install_openreg() -> str:
    """Install torch_openreg and return the install site-packages path.

    Follows the same approach as test/run_test.py's install_cpp_extensions:
    pip install into a local --root so we can add it to PYTHONPATH for
    subprocesses without polluting the global environment.
    """
    build_dir = os.path.join(OPENREG_DIR, "build")
    install_root = os.path.join(OPENREG_DIR, "install")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    if os.path.exists(install_root):
        shutil.rmtree(install_root)

    print(f"Installing torch_openreg from {OPENREG_DIR} ...")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation", ".", "--root", "./install",
        ],
        cwd=OPENREG_DIR,
    )

    platlib = sysconfig.get_paths()["platlib"]
    platlib_rel = os.path.relpath(platlib, os.path.splitdrive(platlib)[0] + os.sep)
    install_dir = os.path.join(install_root, platlib_rel)

    # Smoke test in a subprocess. Run from /tmp so neither the source tree's
    # torch/ nor the openreg source torch_openreg/ shadow installed packages.
    subprocess.check_call(
        [
            sys.executable, "-c",
            "import torch, torch_openreg; "
            "print(f'PyTorch: {torch.__version__}'); "
            "print(f'openreg device count: {torch.openreg.device_count()}'); "
            "print(f'Backend registered: {torch._C._get_privateuse1_backend_name()}')",
        ],
        env={**os.environ, "PYTHONPATH": install_dir},
        cwd="/tmp",
    )
    return install_dir


def _log_dir() -> str:
    log_dir = os.path.join(PYTORCH_ROOT, "openreg_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _log_path(test_file: str) -> str:
    # nn/test_dropout.py -> nn__test_dropout.log
    return os.path.join(_log_dir(), test_file.replace("/", "__").replace(".py", ".log"))


def _parse_skipped_tests(log_file: str) -> list[dict[str, str]]:
    """Parse pytest -rs output for skipped tests.

    Returns a list of {"reason": ..., "location": ...} dicts from the
    short test summary section. These tell you which tests were skipped
    and why (e.g. "Only runs on ['cuda']").
    """
    skipped: list[dict[str, str]] = []
    try:
        with open(log_file, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
    except OSError:
        return skipped

    # Pytest -rs short summary lines look like:
    #   SKIPPED [1] torch/testing/_internal/common_device_type.py:367: Only runs on ['cuda']
    for match in re.finditer(
        r"^SKIPPED \[\d+\] (.+?): (.+)$", content, re.MULTILINE
    ):
        skipped.append({
            "location": match.group(1).strip(),
            "reason": match.group(2).strip(),
        })

    return skipped


def _read_stepcurrent(stepcurrent_key: str) -> str | None:
    """Read the last-run test nodeid from the pytest stepcurrent cache."""
    cache_file = os.path.join(
        PYTORCH_ROOT, ".pytest_cache/v/cache/stepcurrent", stepcurrent_key, "lastrun"
    )
    try:
        with open(cache_file) as f:
            return f.read()
    except FileNotFoundError:
        return None


def _run_and_tee(command, cwd, env, log_file, timeout):
    """Run command, tee stdout+stderr to both the terminal and a log file.

    Returns the exit code. Raises subprocess.TimeoutExpired on timeout.
    On timeout: sends SIGINT, waits 5s for graceful shutdown, then kills.
    Follows the pattern from torch/testing/_internal/common_utils.py wait_for_process.
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def reader():
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        t.join(timeout=5)
        raise

    t.join(timeout=10)
    return proc.returncode


def _parse_failed_tests(log_file: str) -> list[str]:
    """Parse pytest output for failed test nodeids.

    Looks for the short test summary lines like:
        FAILED [0.5s] test/test_foo.py::TestClass::test_method
    """
    failures: list[str] = []
    try:
        with open(log_file, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
    except OSError:
        return failures

    # Pytest -v output: "path/test.py::TestClass::test_name FAILED [0.5s] [ 44%]"
    for match in re.finditer(
        r"^(.+::.*\S) FAILED \[", content, re.MULTILINE
    ):
        failures.append(match.group(1).strip())

    return failures


def _run_test_no_retries(
    sharded_test: _ShardedTest,
    env: dict[str, str],
    log_file: str,
    timeout: int,
) -> tuple[str, float, list[str], list[dict[str, str]]]:
    """Run a test file straight through without retries.

    Runs pytest without -x (no stop on first failure), so all tests run in
    a single invocation. Much faster than stepcurrent for files with many
    failures since it avoids repeated pytest collection overhead.

    Returns (status, elapsed, failures, skipped).
    """
    test_file = sharded_test.test_file
    full_path = os.path.join(TEST_DIR, test_file)
    command = [
        sys.executable, "-u", full_path, "--use-pytest", "-v", "-rs",
        *sharded_test.pytest_args(),
    ]

    start = time.monotonic()

    with open(log_file, "wb") as lf:
        try:
            ret_code = _run_and_tee(
                command,
                cwd=TEST_DIR,
                env=env,
                log_file=lf,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            print(f"  TIMEOUT  {test_file}  ({elapsed:.1f}s, exceeded {timeout}s limit)")
            return "TIMEOUT", elapsed, [], []

    elapsed = time.monotonic() - start

    # Exit code 5 means "no tests collected", treat as pass
    ret_code = 0 if ret_code == 5 else ret_code

    failures = _parse_failed_tests(log_file) if ret_code != 0 else []
    skipped = _parse_skipped_tests(log_file)

    if ret_code < 0:
        # Process killed by signal (e.g. -11 = SIGSEGV). Report partial
        # results from whatever ran before the crash.
        sig = -ret_code
        print(f"  CRASH  {test_file}  ({elapsed:.1f}s, signal {sig})")
        if failures:
            print(f"    {len(failures)} failures before crash")
        return "FAIL", elapsed, failures, skipped
    elif failures:
        print(f"  FAIL  {test_file}  ({elapsed:.1f}s, {len(failures)} failures)")
        return "FAIL", elapsed, failures, skipped
    elif ret_code != 0:
        print(f"  FAIL  {test_file}  ({elapsed:.1f}s, exit code {ret_code})")
        return "FAIL", elapsed, [], skipped
    else:
        print(f"  PASS  {test_file}  ({elapsed:.1f}s)")
        return "PASS", elapsed, [], skipped


def _run_test_with_retries(
    sharded_test: _ShardedTest,
    env: dict[str, str],
    log_file: str,
    timeout: int,
    retries: int,
) -> tuple[str, float, list[str], list[str], list[dict[str, str]]]:
    """Run a test file with per-test retries via stepcurrent.

    Uses -x (stop on first failure) and the stepcurrent mechanism to retry
    individual failing tests. Slower than no-retry mode due to pytest restart
    overhead per failure, but can distinguish flaky from consistent failures.

    Returns (status, elapsed, consistent_failures, flaky_failures, skipped).
    """
    test_file = sharded_test.test_file
    full_path = os.path.join(TEST_DIR, test_file)
    command = [
        sys.executable, "-u", full_path, "--use-pytest", "-v", "-x", "-rs",
        # "-k", "not (complex32 or complex64 or complex128)",
        *sharded_test.pytest_args(),
    ]
    shard_suffix = f"_s{sharded_test.shard_id}" if sharded_test.num_shards > 1 else ""
    stepcurrent_key = f"openreg_{test_file.replace('/', '_').replace('.py', '')}{shard_suffix}"
    sc_command = f"--sc={stepcurrent_key}"
    num_failures: dict[str, int] = defaultdict(int)

    start = time.monotonic()

    with open(log_file, "wb") as lf:
        while True:
            try:
                ret_code = _run_and_tee(
                    command + [sc_command],
                    cwd=TEST_DIR,
                    env=env,
                    log_file=lf,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                print(f"  TIMEOUT  {test_file}  ({elapsed:.1f}s, exceeded {timeout}s limit)")
                return "TIMEOUT", elapsed, [], [], []

            # Exit code 5 means "no tests collected", treat as pass
            ret_code = 0 if ret_code == 5 else ret_code

            if ret_code == 0 and not sc_command.startswith("--rs="):
                break

            current_failure = _read_stepcurrent(stepcurrent_key)
            if current_failure is None:
                if ret_code != 0:
                    print("  No stepcurrent file found (possible import error)")
                break

            if ret_code != 0:
                num_failures[current_failure] += 1

            if ret_code == 0:
                sc_command = f"--scs={stepcurrent_key}"
                print("  Test succeeded on rerun, continuing with remaining tests")
            elif num_failures[current_failure] >= retries:
                print(f"  FAILED CONSISTENTLY: {current_failure}")
                sc_command = f"--scs={stepcurrent_key}"
            else:
                sc_command = f"--rs={stepcurrent_key}"
                print(
                    f"  Retrying {current_failure} "
                    f"({num_failures[current_failure]}/{retries}) ..."
                )

    elapsed = time.monotonic() - start

    strip = lambda s: s.strip('"')
    consistent_failures = [strip(t) for t, n in num_failures.items() if n >= retries]
    flaky_failures = [strip(t) for t, n in num_failures.items() if 0 < n < retries]
    skipped = _parse_skipped_tests(log_file)

    if consistent_failures:
        print(f"  FAIL  {test_file}  ({elapsed:.1f}s)")
        print(f"    Consistent failures: {consistent_failures}")
        if flaky_failures:
            print(f"    Flaky (passed on retry): {flaky_failures}")
        return "FAIL", elapsed, consistent_failures, flaky_failures, skipped
    elif flaky_failures:
        print(f"  FLAKY  {test_file}  ({elapsed:.1f}s)")
        print(f"    Flaky (passed on retry): {flaky_failures}")
        return "FLAKY", elapsed, [], flaky_failures, skipped
    else:
        print(f"  PASS  {test_file}  ({elapsed:.1f}s)")
        return "PASS", elapsed, [], [], skipped


def run_test(
    sharded_test: _ShardedTest,
    openreg_pythonpath: str,
    timeout: int,
    retries: int,
) -> tuple[str, float, list[str], list[str], list[dict[str, str]]]:
    """Run a single test file. Returns (status, elapsed, failures, flaky, skipped).

    When retries == 0, runs pytest straight through (no -x, no stepcurrent)
    for maximum speed. When retries > 0, uses stepcurrent for per-test retries.
    """
    test_file = sharded_test.test_file
    pythonpath = openreg_pythonpath
    if "PYTHONPATH" in os.environ:
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_TESTING_DEVICE_ONLY_FOR": "openreg",
        "OPENREG_DISABLE_FALLBACK_BLOCKLIST": "1",
        "OPENREG_DISABLE_MEMORY_PROTECTION": "1",
    }

    log_file = _log_path(test_file)
    print(f"\n{'=' * 60}")
    print(f"Running {sharded_test.name}  (log: {log_file})")
    print("=" * 60, flush=True)

    if retries == 0:
        status, elapsed, failures, skipped = _run_test_no_retries(
            sharded_test, env, log_file, timeout
        )
        return status, elapsed, failures, [], skipped
    else:
        return _run_test_with_retries(
            sharded_test, env, log_file, timeout, retries
        )


def print_summary(results: list[tuple[str, str, float]]) -> None:
    """Print a summary table of test results."""
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    name_width = max(len(name) for name, _, _ in results)
    for name, status, elapsed in results:
        log_file = _log_path(name)
        print(f"  {status:<7}  {name:<{name_width}}  ({elapsed:.1f}s)  {log_file}")

    passed = sum(1 for _, s, _ in results if s == "PASS")
    flaky = sum(1 for _, s, _ in results if s == "FLAKY")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    timed_out = sum(1 for _, s, _ in results if s == "TIMEOUT")
    total_time = sum(t for _, _, t in results)

    parts = [f"{passed} passed"]
    if flaky:
        parts.append(f"{flaky} flaky")
    if failed:
        parts.append(f"{failed} failed")
    if timed_out:
        parts.append(f"{timed_out} timed out")
    print(f"\n{', '.join(parts)}  ({total_time:.1f}s total)")
    print(f"Logs: {_log_dir()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PyTorch device-generic tests against the openreg backend."
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific test files to run (default: all discovered device-generic tests).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered test files and exit.",
    )
    parser.add_argument(
        "-c",
        "--continue-on-failure",
        action="store_true",
        help="Continue running tests after a failure (default: stop on first failure).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-test timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of times a test must fail before it's considered a consistent failure (default: 3).",
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("SHARD_ID", "NUM_SHARDS"),
        help="Run only tests assigned to this shard (1-indexed). "
        "E.g., --shard 2 3 runs the second of three shards.",
    )
    args = parser.parse_args()

    discovered = _discover_tests()
    test_files = args.tests if args.tests else discovered

    shard_id = args.shard[0] if args.shard else 1
    num_shards = args.shard[1] if args.shard else 1
    sharded_tests = _shard_tests(test_files, shard_id, num_shards)

    if args.list:
        for t in sharded_tests:
            print(t.name)
        return

    if num_shards > 1:
        print(f"Shard {shard_id}/{num_shards}: running {len(sharded_tests)} items")

    openreg_pythonpath = install_openreg()

    results: list[tuple[str, str, float]] = []
    all_consistent_failures: list[str] = []
    all_flaky_failures: list[str] = []
    all_skipped: list[dict[str, str]] = []
    for sharded_test in sharded_tests:
        status, elapsed, consistent, flaky, skipped = run_test(
            sharded_test, openreg_pythonpath, args.timeout, args.retries
        )
        results.append((sharded_test.name, status, elapsed))
        all_consistent_failures.extend(consistent)
        all_flaky_failures.extend(flaky)
        all_skipped.extend(skipped)
        if status in ("FAIL", "TIMEOUT") and not args.continue_on_failure:
            print(f"\nStopping after failure in {sharded_test.name}.")
            break

    print_summary(results)

    # Deduplicate skipped entries and group by reason for readability
    skipped_by_reason: dict[str, int] = defaultdict(int)
    for entry in all_skipped:
        skipped_by_reason[entry["reason"]] += 1

    report = {
        "shard": f"{shard_id}/{num_shards}",
        "failed": all_consistent_failures,
        "flaky": all_flaky_failures,
        "timed_out": [name for name, s, _ in results if s == "TIMEOUT"],
        "skipped_by_reason": dict(skipped_by_reason),
    }
    # Write to test/test-reports/ so CI's upload-test-artifacts picks it up.
    # Include shard ID in filename so parallel shards don't overwrite each other.
    report_dir = os.path.join(PYTORCH_ROOT, "test", "test-reports")
    os.makedirs(report_dir, exist_ok=True)
    shard_suffix = f"_shard{shard_id}" if num_shards > 1 else ""
    report_path = os.path.join(
        report_dir, f"openreg_device_generic_report{shard_suffix}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}")

    if any(s in ("FAIL", "TIMEOUT") for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
