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

Note the "Validator" lines. If you change a library verison, or ROCm version, or
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
from typing import Optional, Tuple

import torch


__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
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
]


def enable(val: bool = True) -> None:
    r"""This is the big on/off switch for all TunableOp implementations."""
    torch._C._cuda_tunableop_enable(val)


def is_enabled() -> bool:
    r"""Returns whether the TunableOp feature is enabled."""
    return torch._C._cuda_tunableop_is_enabled()


def tuning_enable(val: bool = True) -> None:
    r"""Enable tuning of TunableOp implementations.

    When enabled, if a tuned entry isn't found, run the tuning step and record
    the entry.
    """
    torch._C._cuda_tunableop_tuning_enable(val)


def tuning_is_enabled() -> bool:
    r"""Returns whether TunableOp implementations can be tuned."""
    return torch._C._cuda_tunableop_tuning_is_enabled()


def set_max_tuning_duration(duration: int) -> None:
    r"""Set max time in milliseconds to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    torch._C._cuda_tunableop_set_max_tuning_duration(duration)


def get_max_tuning_duration() -> int:
    r"""Get max time to spend tuning a given solution."""
    return torch._C._cuda_tunableop_get_max_tuning_duration()


def set_max_tuning_iterations(iterations: int) -> None:
    r"""Set max number of iterations to spend tuning a given solution.

    If both max tuning duration and iterations are set, the smaller of the two
    will be honored. At minimum 1 tuning iteration will always be run.
    """
    torch._C._cuda_tunableop_set_max_tuning_iterations(iterations)


def get_max_tuning_iterations() -> int:
    r"""Get max iterations to spend tuning a given solution."""
    return torch._C._cuda_tunableop_get_max_tuning_iterations()


def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    r"""Set the filename to use for input/output of tuning results.

    If :attr:`insert_device_ordinal` is ``True`` then the current device ordinal
    will be added to the given filename automatically. This can be used in a
    1-process-per-gpu cenario to ensure all processes write to a separate file.
    """
    torch._C._cuda_tunableop_set_filename(filename, insert_device_ordinal)


def get_filename() -> str:
    r"""Get the results filename."""
    return torch._C._cuda_tunableop_get_filename()


def get_results() -> Tuple[str, str, str, float]:
    r"""Return all TunableOp results."""
    return torch._C._cuda_tunableop_get_results()


def get_validators() -> Tuple[str, str]:
    r"""Return the TunableOp validators."""
    return torch._C._cuda_tunableop_get_validators()


def write_file_on_exit(val: bool) -> None:
    r"""During Tuning Context destruction, write file to disk.

    This is useful as a final flush of your results to disk if your application
    terminates as result of normal operation or an error. Manual flushing of
    your results can be achieved by manually calling ``write_file()``."""
    torch._C._cuda_tunableop_write_file_on_exit(val)


def write_file(filename: Optional[str] = None) -> bool:
    r"""Write results to a CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_write_file(filename)


def read_file(filename: Optional[str] = None) -> bool:
    r"""Read results from a TunableOp CSV file.

    If :attr:`filename` is not given, ``get_filename()`` is called.
    """
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_read_file(filename)
