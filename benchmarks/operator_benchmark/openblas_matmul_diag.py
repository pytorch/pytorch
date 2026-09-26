import os
import ctypes
import subprocess
import statistics
import time
from pathlib import Path
from typing_extensions import runtime

import torch

def print_file(path):
    path = Path(path)
    print(f"\n === {path} ===")
    if path.exists():
        print(path.read_text(errors="replace"))
    else:
        print("<not present>")

def print_command(command):
    print(f"\n === {' '.join(command)} === ")
    subprocess.run(command, check=False)

def find_loaded_openblas():
    torch.mm(torch.ones(1, 1), torch.ones(1, 1))

    maps = Path("/proc/self/maps").read_text(errors="replace")
    paths = set()

    for line in maps.splitlines():
        fields = line.split()
        if not fields:
            continue

        candidate = fields[-1]
        if "libopenblas" in candidate.lower() and candidate.startswith("/"):
            paths.add(candidate)

    if not paths:
        raise RuntimeError("No loaded OpenBLAS lib found in /proc/self/maps")

    if len(paths) != 1:
        print(f" Multiple OpenBLAS paths found: {sorted(paths)}")

    return sorted(paths)[0]

def call_openblas_int(lib, name):
    try:
        function = getattr(lib, name)
    except AttributeError:
        print(f"{name}: <symbol unavailable>")
        return

    function.argtypes = []
    function.restype = ctypes.c_int
    print(f"{name}: {function()}")

def call_openblas_string(lib, name):
    try:
        function = getattr(lib, name)
    except AttributeError:
        print(f"{name}: <symbol unavailable>")
        return

    function.argtypes = []
    function.restype = ctypes.c_char_p
    value = function()
    print(f"{name}: {value.decode(errors='replace') if value else None}")


def print_runtime_information():
    print("===== Diagnostic identity =====")
    print(f"pid: {os.getpid()}")
    print(f"configuration: {os.environ.get('MATMUL_DIAGNOSTIC_CONFIG')}")

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        raise RuntimeError("This diagnostic requires linux sched_getaffinity()")
    print(f"sched_getaffinity: {sorted(os.sched_getaffinity(0))}")

    print("\n===== Relevant environment =====")
    variables = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "GOMP_CPU_AFFINITY",
        "LD_PRELOAD",
        "MALLOC_CONF",
    ]

    for variable in variables:
        print(f"{variable}={os.environ.get(variable, '<unset>')}")

    print("\n===== PyTorch threading =====")
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    print(torch.__config__.parallel_info())
    print(torch.__config__.show())

    openblas_path = find_loaded_openblas()
    print("\n===== OpenBLAS runtime =====")
    print(f"loaded library: {openblas_path}")

    openblas = ctypes.CDLL(openblas_path)

    call_openblas_string(openblas, "openblas_get_config")
    call_openblas_string(openblas, "openblas_get_corename")
    call_openblas_int(openblas, "openblas_get_num_threads")
    call_openblas_int(openblas, "openblas_get_num_procs")
    call_openblas_int(openblas, "openblas_get_parallel")

    try:
        libgomp = ctypes.CDLL("libgomp.so.1")
        call_openblas_int(libgomp, "omp_get_max_threads")
        call_openblas_int(libgomp, "omp_get_num_procs")
    except OSError as error:
        print(f"Could not load libgomp.so.1: {error}")

    print_command(["lscpu"])
    print_command(["lscpu", "-e"])

    print_file("/proc/cpuinfo")
    print_file("/proc/self/status")
    print_file("/proc/self/cgroup")

    # Cgroup v2 locations used by modern CI containers.
    print_file("/sys/fs/cgroup/cpu.max")
    print_file("/sys/fs/cgroup/cpu.stat")
    print_file("/sys/fs/cgroup/cpu.pressure")
    print_file("/sys/fs/cgroup/cpuset.cpus")
    print_file("/sys/fs/cgroup/cpuset.cpus.effective")


def run_benchmark():
    torch.manual_seed(0)

    # This recreates the physical layout from the failing benchmark:
    # A is a transposed view, B is contiguous.
    a = torch.rand(256, 256, dtype=torch.float32).t()
    b = torch.rand(256, 256, dtype=torch.float32)

    print("\n===== Tensor layout =====")
    print(f"A shape={tuple(a.shape)}, stride={a.stride()}, contiguous={a.is_contiguous()}")
    print(f"B shape={tuple(b.shape)}, stride={b.stride()}, contiguous={b.is_contiguous()}")

    warmup_iterations = 100
    batch_iterations = 100
    measured_batches = 5

    with torch.inference_mode():
        output = None

        for _ in range(warmup_iterations):
            output = torch.matmul(a, b)

        batch_times_us = []

        for batch in range(measured_batches):
            start = time.perf_counter_ns()

            for _ in range(batch_iterations):
                output = torch.matmul(a, b)

            elapsed_ns = time.perf_counter_ns() - start
            per_call_us = elapsed_ns / batch_iterations / 1_000
            batch_times_us.append(per_call_us)

            print(f"batch {batch}: {per_call_us:.3f} us/call")

    print(f"all batch times: {batch_times_us}")
    print(f"median: {statistics.median(batch_times_us):.3f} us/call")

    if output is None:
        raise RuntimeError("No benchmark iterations ran")


    print(f"checksum: {output[0, 0].item()}")


if __name__ == "__main__":
    print_runtime_information()
    run_benchmark()
