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
  GemmTunableOp_float_NT,nt_25088_4096_64,Gemm_Hipblaslt_1219,1.262
  GemmTunableOp_float_NT,nt_4096_4096_64,Gemm_Rocblas_1216,0.033

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

A Note on Tuning Behavior, Warmup, and Cache Effects
====================================================

Tuning an operator consists of iterating through the list or registered
implementations and profiling each one. The profile is established by running a
single implementation in a loop multiple times and taking the average execution
time. There is also an optional warmup phase prior to tuning that can help with
reaching stable power states by the hardware. During tuning of a workload the
various hardware caches will more likely produce hits than when not tuning.
There are options for flushing the instruction cache and rotate the input tensors
which might help produce a more faithful profile of the tuned operator as if the
operator were run within a larger workload instead of in a tight, repetitive loop.

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

Offline Tuning
==============

Motivation
----------
There are several use cases for offline tuning.

One use case involves a workload with a high-memory utilization, where regular tuning might lead to running out of memory.

Another use case is for compute-intensive workloads. In such cases, it is more resource-efficient to collect
the GEMMs for the workload once and then tune repeatedly with different tuning parameters or libraries.

Workflow
--------
There are basically two steps:
1) Set the environment variables to collect the untuned GEMM and this will generate ``tunableop_untuned0.csv``:

.. code-block:: python

   PYTORCH_TUNABLEOP_ENABLED=1
   PYTORCH_TUNABLEOP_TUNING=0
   PYTORCH_TUNABLEOP_RECORD_UNTUNED=1
   ...

2) Run a Python script that reads the ``tunableop_untuned0.csv`` and generates the ``tunableop_results0.csv``, like this:

.. code-block:: python

   import torch.cuda.tunable as tunable
   import os

   os.putenv('PYTORCH_TUNABLEOP_ENABLED', '1')
   os.putenv('PYTORCH_TUNABLEOP_TUNING', '1')
   os.putenv('PYTORCH_TUNABLEOP_RECORD_UNTUNED', '0')
   tunable.tune_gemm_in_file("tunableop_untuned0.csv")


It is also possible to take multiple untuned files and distribute the GEMMs for tuning to multiple GPUs
within a single node. In the first step, the GEMMs are first gathered and duplicate GEMMs are eliminated.
Next, the GEMMs are distributed to different GPUs for tuning. After all GEMMs are tuned, the results from
all the GPUs are then gathered into a single file whose base filename has ``_full0`` appended to it
(for example ``tunableop_results_full0.csv``). Finally, this new file, containing the gathered results, will be
duplicated N times, once for each GPU as convenience to the user will run the workload with the tuned
configuration on N GPUs.

.. code-block:: python

   if __name__ == "__main__":
       num_gpus = 8 # number of GPUs that will be used during the tuning process
       tunable.mgpu_tune_gemm_in_file("tunableop_untuned?.csv", num_gpus)

Note that the usage of the ``mgpu_tune_gemm_in_file`` API is different from its single GPU counterpart
(``tune_gemm_in_file``). The body of the Python script that calls the API must be wrapped in ``main()`` as shown
due to the use of concurrent futures module. The argument to ``mgpu_tune_gemm_in_file`` must contain a wild card
expression (``?`` or ``*``) to generate the list of untuned files containing the GEMMs to be processed. The ``num_gpus``
must between 1 and the total number of GPUs available.

Tuning Context
==============

The behavior of TunableOp is currently manipulated through environment
variables, the C++ interface of at::cuda::tunable::getTuningContext(), or the
torch.cuda.tunable python interfaces. The environment variables take precedence
over any setting you manipulate using the C++ or Python APIs.

Environment Variable Interface
------------------------------
Environment variables are cached the first time they are read. You cannot use the
environment variable interface programmatically since the settings become fixed.
Use the C++ or Python APIs instead.

"""
import concurrent.futures
import glob
import multiprocessing as mp
import os
import shutil
import warnings
from typing import Optional

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


def get_results() -> tuple[str, str, str, float]:
    r"""Return all TunableOp results."""
    return torch._C._cuda_tunableop_get_results()  # type: ignore[attr-defined]


def get_validators() -> tuple[str, str]:
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
        # dtypeC = might not be FP8 type, keep track
        # of the the number of underscores
        count = untuned_gemm_temp.count("_")
        op_sig = untuned_gemm_temp[0]
        data_typeA = untuned_gemm_temp[1] + "_" + untuned_gemm_temp[2]
        data_typeB = untuned_gemm_temp[3] + "_" + untuned_gemm_temp[4]
        if count == 7:
            data_typeC = untuned_gemm_temp[5] + "_" + untuned_gemm_temp[6]
        else:
            data_typeC = untuned_gemm_temp[5]
        transA = untuned_gemm_temp[count][0] == "T"
        transB = untuned_gemm_temp[count][1] == "T"
        dtypeA = dtype_dict.get(data_typeA)
        dtypeB = dtype_dict.get(data_typeB)
        dtypeC = dtype_dict.get(data_typeC)

    untuned_gemm_temp = untuned_gemm[1].split("_")
    [n, m, k] = [int(g) for g in untuned_gemm_temp[1:4]]
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
        [b] = [int(g) for g in untuned_gemm_temp[5:6]]
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
        matA = (
            torch.full((k, m), fillA, dtype=dtypeA, device=deviceid).t()
            if transB
            else torch.full((m, k), fillA, dtype=dtypeA, device=deviceid)
        )
        matB = (
            torch.full((n, k), fillB, dtype=dtypeB, device=deviceid)
            if transA
            else torch.full((k, n), fillB, dtype=dtypeB, device=deviceid).t()
        )

        assert untuned_gemm_temp[8] == "rw"
        if untuned_gemm_temp[9] == "1":
            rowwise = True
        else:
            rowwise = False
        if rowwise:
            scaleA = torch.ones((matA.shape[0], 1), device=deviceid)
            scaleB = torch.ones((1, matB.shape[0]), device=deviceid)
        else:
            scaleA = torch.tensor(0.8, device=deviceid)
            scaleB = torch.tensor(0.9, device=deviceid)

        assert untuned_gemm_temp[10] == "bias"
        if untuned_gemm_temp[11] == "None":  # no bias vector
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtypeC
            )
        else:  # bias vector present
            fillbias = 0.10
            bias_dtype = dtype_dict.get(untuned_gemm_temp[11])
            bias = (
                torch.full((n,), fillbias, dtype=bias_dtype, device=deviceid)
                if transA
                else torch.full((m,), fillbias, dtype=bias_dtype, device=deviceid)
            )
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtypeC, bias=bias
            )

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

    if is_enabled() is False:
        warnings.warn("TunableOp was disabled. Trying to enable now.")
        enable(True)
    assert is_enabled() is True
    assert tuning_is_enabled() is True
    assert record_untuned_is_enabled() is False


def mgpu_tune_gemm_in_file(filename_pattern: str, num_gpus: int) -> None:
    r"""Process one or more files and distribute work over one or more GPUs."""
    unique_gemm_entries = _gather_unique_untuned_gemm_from_files(filename_pattern)

    total_gpus = torch.cuda.device_count()

    assert 1 <= num_gpus <= total_gpus

    mp_context = mp.get_context("spawn")

    futures = []  # empty list to hold futures
    flush_results = []  # empty list to hold futures

    # GEMM are assigned to GPUs in a round robin manner
    h = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_gpus,
        mp_context=mp_context,
        initializer=_check_tuning_assertions,
    ) as executor:
        # The workers are a separate process. TunableOp will be
        # enabled in the child processes if PYTORCH_TUNABLEOP_ENABLED=1
        # In the initializer, we also try to enable TunableOP if th
        # environment variable was NOT set.

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
