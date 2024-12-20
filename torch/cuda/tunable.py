r"""
This module exposes a TunableOp interface.

Some operations, such as GEMMs, could be implemented using more than one library
or more than one technique. For example, a GEMM could be implemented for CUDA or
ROCm using either the blas or blasLt libraries. Further, ROCm's rocblas and
hipblaslt libraries allow the user to query for all possible algorithms and then
choose one. How does one know which implementation is the fastest and should be
chosen? That's what TunableOp provides.

Enabling TunableOp and Tuning Separately
========================================

The TunableOp feature is enabled separately from enabling the tuning phase
itself. Enabling TunableOp means that PyTorch will replace any standard
operators with their Tunable implementations. Any call to a TunableOp first
checks whether it has already been tuned for the given operator inputs. If so,
it will immediately call the tuned operation; no further tuning will take place
even when the tuning setting is enabled. Instead if no tuning result is found,
and tuning is enabled, the TunableOp will benchmark every registered
implementation of that operator for the given set of inputs and select the
fastest.

File Input and Output
=====================

The first time any TunableOp is invoked, the internal database of tuned
operations will be prepared by attempting to read the results from the given
file. The default filename is 'tunableop_results.csv'. To support tuning when
multiple GPUs are used across multiple processes, the GPU device ordinal is
automatically inserted into the filename to avoid multiple processes overwriting
the same file.

If tuning is enabled and new tunings are discovered during the course of your
workload, it will also write out to this same filename with all tunings, both
the ones it read in at startup as well as the new ones found at runtime. This
can be used, for example, to build up a tunings file across many workloads by
reusing the same file. The output file is automatically created when the
application terminates. This behavior can be controlled by the C++ and Python
APIs but not the environment variables.

Assuming you specified a filename, you'll end up with a CSV file with contents
like so::

  Validator,PT_VERSION,2.2.0
  Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
  Validator,HIPBLASLT_VERSION,0.6.0-a9c5cc7
  Validator,ROCBLAS_VERSION,4.0.0-72e57364-dirty
  GemmTunableOp_float_NT,nt_25088_4096_64,1219,1.262
  GemmTunableOp_float_NT,nt_4096_4096_64,1216,0.033

Note the "Validator" lines. If you change a library version, or ROCm version, or
PyTorch version, TunableOp will detect this and reject the tunings file because
the prior tunings are likely affected by other software changes.

The remaining lines are the tuned solutions for each TunableOp encountered
during your execution. Each line consists of 4 comma-separated fields: operator
name, operator parameters, solution name, and average execution time. The
execution time is an optional field. The CSV file can be edited, but with
caution. For example, the solution name (field 3) can be changed to "Default"
and it will fall back to the original PyTorch untuned implementation. Or, in the
case of ROCm's hipBLAS or hipBLASLt libraries, if you know the specific solution
index you can override the solution that TunableOp selected by replacing the
value. The operator name and parameters (fields 1 and 2) are internally named
and should not be modified. In the case of GemmTunableOp, field 1 indicates the
datatype and whether the inputs are transposed (T) or not (N) and field 2
indicates the M, N, K input shapes.

There is an option to enable verbose output but it is only recommended for
debugging purposes. This will produce a lot of diagnostic messages but may be
useful to see if TunableOp is being used at all. Otherwise, TunableOp is
completely silent, besides file output, unless there is a warning or error
during its use. The verbose option is only available by setting the environment
variable PYTORCH_TUNABLEOP_VEROBSE=1.

A Note on Tuning Behavior
=========================

Tuning an operator consists of iterating through the list or registered
implementations and profiling each one. The profile is established by running a
single implementation in a loop multiple times and taking the average execution
time.

By default, each possible solution for a given operator will be run for either
100 iterations or as many iterations that can be run within 30ms, whichever is
smaller, and its average execution will be calculated. The fastest solution
among all that were successfully profiled will be chosen. A profile might fail
if the given solution doesn't achieve the same accuracy as the default
implementation or if the solution returns an error code.

Current Tunable Operators
=========================

TunableGemm for ROCm
--------------------

Currently only a TunableGemm for ROCm is implemented. Note that CUDA builds of
PyTorch will function correctly when using TunableOp but the only solution
available to CUDA builds is the 'Default' implementation i.e. the original
cuBLAS default, now called through TunableOp. Any call to at::cuda::blas::gemm()
or ::bgemm() will be routed through TunableOp when enabled. Calling gemm() for a
given set of input arguments (transa, transb, m, n, k) will attempt to use the
fastest available implementation across both rocblas and hipblaslt.

Tuning Context
==============

The behavior of TunableOp is currently manipulated through environment
variables, the C++ interface of at::cuda::tunable::getTuningContext(), or the
torch.cuda.tunable python interfaces that wrap the C++ TuningContext. The
environment variables take precedence over any setting you manipulate using the
C++ or Python APIs.

"""
import concurrent.futures
import glob
import multiprocessing as mp
import os
import shutil
import warnings
from typing import Optional, Tuple

import torch


__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
    "record_untuned_enable",
    "record_untuned_is_enabled",
    "set_max_tuning_duration",
    "get_max_tuning_duration",
    "set_max_tuning_iterations",
    "get_max_tuning_iterations",
    "set_filename",
    "get_filename",
    "get_results",
    "get_validators",
    "write_file_on_exit",
    "write_file",
    "read_file",
    "tune_gemm_in_file",
    "mgpu_tune_gemm_in_file",
    "set_rotating_buffer_size",
    "get_rotating_buffer_size",
]


def enable(val: bool = True) -> None:
    r"""This is the big on/off switch for all TunableOp implementations."""
    torch._C._cuda_tunableop_enable(val)  # type: ignore[attr-defined]


def is_enabled() -> bool:
    r"""Returns whether the TunableOp feature is enabled."""
    return torch._C._cuda_tunableop_is_enabled()  # type: ignore[attr-defined]


def tuning_enable(val: bool = True) -> None:
    r"""Enable tuning of TunableOp implementations.

    When enabled, if a tuned entry isn't found, run the tuning step and record
    the entry.
    """
    torch._C._cuda_tunableop_tuning_enable(val)  # type: ignore[attr-defined]


def tuning_is_enabled() -> bool:
    r"""Returns whether TunableOp implementations can be tuned."""
    return torch._C._cuda_tunableop_tuning_is_enabled()  # type: ignore[attr-defined]


def record_untuned_enable(val: bool = True) -> None:
    r"""Enable recording untuned of TunableOp perations for offline tuning.

    When enabled, if a tuned entry isn't found, write it to the untuned file.
    """
    torch._C._cuda_record_untuned_enable(val)  # type: ignore[attr-defined]


def record_untuned_is_enabled() -> bool:
    r"""Returns whether TunableOp operations are recorded for offline tuning."""
    return torch._C._cuda_record_untuned_is_enabled()  # type: ignore[attr-defined]


def set_max_tuning_duration(duration: int) -> None:
    r"""Set max time in milliseconds to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    torch._C._cuda_tunableop_set_max_tuning_duration(duration)  # type: ignore[attr-defined]


def get_max_tuning_duration() -> int:
    r"""Get max time to spend tuning a given solution."""
    return torch._C._cuda_tunableop_get_max_tuning_duration()  # type: ignore[attr-defined]


def set_max_tuning_iterations(iterations: int) -> None:
    r"""Set max number of iterations to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    torch._C._cuda_tunableop_set_max_tuning_iterations(iterations)  # type: ignore[attr-defined]


def get_max_tuning_iterations() -> int:
    r"""Get max iterations to spend tuning a given solution."""
    return torch._C._cuda_tunableop_get_max_tuning_iterations()  # type: ignore[attr-defined]


def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    r"""Set the filename to use for input/output of tuning results.

    If :attr:`insert_device_ordinal` is ``True`` then the current device ordinal
    will be added to the given filename automatically. This can be used in a
    1-process-per-gpu cenario to ensure all processes write to a separate file.
    """
    torch._C._cuda_tunableop_set_filename(filename, insert_device_ordinal)  # type: ignore[attr-defined]


def get_filename() -> str:
    r"""Get the results filename."""
    return torch._C._cuda_tunableop_get_filename()  # type: ignore[attr-defined]


def get_results() -> Tuple[str, str, str, float]:
    r"""Return all TunableOp results."""
    return torch._C._cuda_tunableop_get_results()  # type: ignore[attr-defined]


def get_validators() -> Tuple[str, str]:
    r"""Return the TunableOp validators."""
    return torch._C._cuda_tunableop_get_validators()  # type: ignore[attr-defined]


def write_file_on_exit(val: bool) -> None:
    r"""During Tuning Context destruction, write file to disk.

    This is useful as a final flush of your results to disk if your application
    terminates as result of normal operation or an error. Manual flushing of
    your results can be achieved by manually calling ``write_file()``."""
    torch._C._cuda_tunableop_write_file_on_exit(val)  # type: ignore[attr-defined]


def write_file(filename: Optional[str] = None) -> bool:
    r"""Write results to a CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_write_file(filename)  # type: ignore[attr-defined]


def read_file(filename: Optional[str] = None) -> bool:
    r"""Read results from a TunableOp CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_read_file(filename)  # type: ignore[attr-defined]


def set_rotating_buffer_size(buffer_size: int) -> None:
    r"""Set rotating buffer size to this value in MB, if the buffer size is greater than zero.

    If less than zero, query L2 cache size. If equal to zero, means deactivate rotating buffer.
    """
    return torch._C._cuda_tunableop_set_rotating_buffer_size(buffer_size)  # type: ignore[attr-defined]


def get_rotating_buffer_size() -> int:
    r"""Get the rotating buffer size in kilobytes."""
    return torch._C._cuda_tunableop_get_rotating_buffer_size()  # type: ignore[attr-defined]


def tune_gemm_in_file(filename: str) -> None:
    r"""tune GEMM in file."""

    assert is_enabled()
    assert tuning_is_enabled()

    deviceid = torch.cuda.current_device()

    with open(filename) as file:
        for line in file:
            if line.startswith(("Gemm", "ScaledGemm")):
                _process_single_offline_gemm(line, deviceid)


def _gather_unique_untuned_gemm_from_files(filename_pattern: str) -> set[str]:
    r"""Process multiple untuned results file and return a set with duplicates removed."""
    unique_gemm_entries = set()  # set will avoid duplicates

    for file_path in glob.glob(filename_pattern):
        with open(file_path) as file:
            for line in file:
                if line.startswith(("Gemm", "ScaledGemm")):
                    unique_gemm_entries.add(line)

    return unique_gemm_entries


def _gather_tunableop_results() -> None:
    r"""Gather results from multiple tunableop results file and create a single file."""
    gemm_lines = set()
    validator_lines = []

    # Need to allow for the possibility that results filename was
    # set with the Python API instead of with environment variable.
    # Also possible that results filename was not set at all.
    # There are several test cases to check, but ultimately we
    # need a glob-able expression
    results_filename = get_filename()  # Note empty string could be returned here

    if (
        results_filename is not None and results_filename != ""
    ):  # Case were the Python API was used to set the filename
        dot_pos = results_filename.find(".")
        if dot_pos != -1 and dot_pos > 0:
            # Replace the character just to the left of the dot
            filename_pattern = (
                results_filename[: dot_pos - 1] + "?" + results_filename[dot_pos:]
            )
        else:
            filename_pattern = ""  # Needed to make linter happy
    else:  # Case where the environment variable was used to set the filename.
        results_filename_env = os.getenv("PYTORCH_TUNABLEOP_FILENAME")
        if results_filename_env is None or results_filename_env == "":
            filename_pattern = "tunableop_results?.csv"
        elif "%d" in results_filename_env:
            filename_pattern = results_filename_env.replace("%d", "?")
        else:
            filename_pattern = results_filename_env.replace(".", "?.")

    assert "?" in filename_pattern

    FirstFile = False
    matching_files = glob.glob(filename_pattern)
    num_matching_files = len(matching_files)
    for file_path in matching_files:
        with open(file_path) as file:
            for line in file:
                if line.startswith("Validator"):
                    if not (FirstFile):
                        # Only read Validator from first file
                        validator_lines.append(line)
                else:
                    gemm_lines.add(line)

        FirstFile = True

    output_file = filename_pattern.replace("?", "_full0")

    with open(output_file, "w") as out_file:
        for line in validator_lines:
            out_file.write(line)
        for line in gemm_lines:
            out_file.write(line)

    # Create num_matching_copies of the results file
    for i in range(1, num_matching_files):
        duplicate_file = output_file.replace("0", str(i))
        shutil.copy(output_file, duplicate_file)


def _process_single_offline_gemm(untuned_gemm_line: str, gpu_id: int) -> None:
    r"""Process a single untuned GEMM."""

    deviceid = "cuda:" + str(gpu_id)

    dtype_dict = {
        "float": torch.float32,
        "double": torch.float64,
        "BFloat16": torch.bfloat16,
        "Half": torch.half,
        "c10::complex<double>": torch.complex128,
        "c10::complex<float>": torch.complex64,
        "Float8_e4m3fn": torch.float8_e4m3fn,
        "Float8_e5m2": torch.float8_e5m2,
        "Float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "Float8_e5m2fnuz": torch.float8_e5m2fnuz,
    }

    untuned_gemm = untuned_gemm_line.strip().split(",")[:]

    underscore_count = untuned_gemm[0].count("_")

    # Initialize dtype to make linter happy
    dtype = None
    dtypeA = None
    dtypeB = None
    dtypeC = None

    if underscore_count == 2:
        [op_sig, data_type, layout] = untuned_gemm[0].split("_")
        transA = layout[0] == "T"
        transB = layout[1] == "T"
        dtype = dtype_dict.get(data_type)
    else:  # ScaledGEMM
        untuned_gemm_temp = untuned_gemm[0].split("_")
        op_sig = untuned_gemm_temp[0]
        data_typeA = untuned_gemm_temp[1] + "_" + untuned_gemm_temp[2]
        data_typeB = untuned_gemm_temp[3] + "_" + untuned_gemm_temp[4]
        data_typeC = untuned_gemm_temp[5] + "_" + untuned_gemm_temp[6]
        transA = untuned_gemm_temp[7][0] == "T"
        transB = untuned_gemm_temp[7][1] == "T"
        dtypeA = dtype_dict.get(data_typeA)
        dtypeB = dtype_dict.get(data_typeB)
        dtypeC = dtype_dict.get(data_typeC)

    [n, m, k] = [int(g) for g in untuned_gemm[1].split("_")[1:4]]
    if op_sig == "GemmTunableOp":
        matA = (
            torch.rand(k, m, dtype=dtype, device=deviceid).t()
            if transB
            else torch.rand(m, k, dtype=dtype, device=deviceid)
        )
        matB = (
            torch.rand(n, k, dtype=dtype, device=deviceid).t()
            if transA
            else torch.rand(k, n, dtype=dtype, device=deviceid)
        )
        torch.mm(matA, matB)
    elif op_sig == "GemmStridedBatchedTunableOp":
        [b] = [int(g) for g in untuned_gemm[1].split("_")[5:6]]
        matA = (
            torch.rand(b, k, m, dtype=dtype, device=deviceid)
            if transB
            else torch.rand(b, m, k, dtype=dtype, device=deviceid)
        )
        matB = (
            torch.rand(b, n, k, dtype=dtype, device=deviceid)
            if transA
            else torch.rand(b, k, n, dtype=dtype, device=deviceid)
        )
        matA = matA.transpose(1, 2) if transB else matA
        matB = matB.transpose(1, 2) if transA else matB
        torch.bmm(matA, matB)
    elif op_sig == "ScaledGemmTunableOp":
        fillA = 0.25
        fillB = 0.75
        scaleA = torch.tensor(0.8, device=deviceid)
        scaleB = torch.tensor(0.9, device=deviceid)
        matA = (
            torch.full((k, m), fillA, dtype=dtypeA, device=deviceid).t()
            if transB
            else torch.full((m, k), fillA, dtype=dtypeA, device=deviceid)
        )
        matB = (
            torch.full((n, k), fillB, dtype=dtypeB, device=deviceid).t()
            if transA
            else torch.full((k, n), fillB, dtype=dtypeB, device=deviceid)
        )
        torch._scaled_mm(matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtypeC)
    elif op_sig == "GemmAndBiasTunableOp":
        # y = x*A^T + b
        assert transA != transB

        X = (
            torch.rand(k, m, dtype=dtype, device=deviceid).t()
            if transB
            else torch.rand(m, k, dtype=dtype, device=deviceid)
        )
        matA = (
            torch.rand(n, k, dtype=dtype, device=deviceid)
            if transA
            else torch.rand(k, n, dtype=dtype, device=deviceid).t()
        )
        bias = (
            torch.rand(n, dtype=dtype, device=deviceid)
            if transA
            else torch.rand(m, dtype=dtype, device=deviceid)
        )
        torch.nn.functional.linear(X, matA, bias)
    else:
        warnings.warn(f"error: unknown op {op_sig}")


def _check_tuning_assertions() -> None:
    r"""Helper function for multi-GPU tuning case. Need to check that TunableOp feature
    is enabled and that tuning is enabled.
    """
    assert is_enabled()
    assert tuning_is_enabled()


def mgpu_tune_gemm_in_file(filename_pattern: str, num_gpus: int) -> None:
    r"""Process one or more files and distribute work over one or more GPUs."""
    unique_gemm_entries = _gather_unique_untuned_gemm_from_files(filename_pattern)

    total_gpus = torch.cuda.device_count()

    assert 1 <= num_gpus <= total_gpus

    mp_context = mp.get_context("spawn")

    checks = []  # empty list to hold futures
    futures = []  # empty list to hold futures
    flush_results = []  # empty list to hold futures

    # GEMM are assigned to GPUs in a round robin manner
    h = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_gpus, mp_context=mp_context
    ) as executor:
        # The workers are a separate process. TunableOp will be
        # enabled in the child processes if the environment variable
        # is set. However, if we enable TunableOp via the API
        # the workers do not inherit this state. As a precaution,
        # we need to check that TuningOp feature and tuning is
        # enabled in the pool of processes.
        for g in range(num_gpus):
            check = executor.submit(_check_tuning_assertions)
            checks.append(check)

        for check in concurrent.futures.as_completed(checks):
            check.result()

        for line in unique_gemm_entries:
            future = executor.submit(_process_single_offline_gemm, line, h)
            futures.append(future)
            h = (h + 1) % num_gpus

        for future in concurrent.futures.as_completed(futures):
            future.result()

        for g in range(num_gpus):
            flush_result = executor.submit(write_file)
            flush_results.append(flush_result)

        for flush_result in concurrent.futures.as_completed(flush_results):
            flush_result.result()

    torch.cuda.synchronize()

    _gather_tunableop_results()
