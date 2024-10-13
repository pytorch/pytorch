# TunableOp

This directory implements a TunableOp interface.

Some operations, such as GEMMs, could be implemented using more than one library or more than one technique. For
example, a GEMM could be implemented for CUDA or ROCm using either the blas or blasLt libraries. Further, ROCm's
rocblas and hipblaslt libraries allow the user to query for all possible algorithms and then choose one. How does one
know which implementation is the fastest and should be chosen? That's what TunableOp provides.

## Enabling TunableOp and Tuning Separately
The TunableOp feature is enabled separately from enabling the tuning phase itself. Enabling TunableOp means that PyTorch
will replace any standard operators with their Tunable implementations. Any call to a TunableOp first checks whether it
has already been tuned for the given operator inputs. If so, it will immediately call the tuned operation; no further
tuning will take place even when the tuning setting is enabled. Instead if no tuning result is found, and tuning is
enabled, the TunableOp will benchmark every registered implementation of that operator for the given set of inputs and
select the fastest.

## File Input and Output
The first time any TunableOp is invoked, the internal database of tuned operations will be prepared by attempting to
read the results from the given file. The default filename is 'tunableop_results.csv'. To support tuning when multiple
GPUs are used across multiple processes, the GPU device ordinal is automatically inserted into the filename to avoid
multiple processes overwriting the same file.

If tuning is enabled and new tunings are discovered during the course of your workload, it will also write out to this
same filename with all tunings, both the ones it read in at startup as well as the new ones found at runtime. This can
be used, for example, to build up a tunings file across many workloads by reusing the same file. The output file is
automatically created when the application terminates. This behavior can be controlled by the C++ and Python APIs but
not the environment variables.

Assuming you specified a filename, you'll end up with a CSV file with contents like so:

```
Validator,PT_VERSION,2.2.0
Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
Validator,HIPBLASLT_VERSION,0.6.0-a9c5cc7
Validator,ROCBLAS_VERSION,4.0.0-72e57364-dirty
GemmTunableOp_float_NT,nt_25088_4096_64,1219,1.262
GemmTunableOp_float_NT,nt_4096_4096_64,1216,0.033
```

Note the "Validator" lines. If you change a library verison, or ROCm version, or PyTorch version, TunableOp will detect
this and reject the tunings file because the prior tunings are likely affected by other software changes.

The remaining lines are the tuned solutions for each TunableOp encountered during your execution. Each line consists of
4 comma-separated fields: operator name, operator parameters, solution name, and average execution time. The execution
time is an optional field. The CSV file can be edited, but with caution. For example, the solution name (field 3) can be
changed to "Default" and it will fall back to the original PyTorch untuned implementation. Or, in the case of ROCm's
hipBLAS or hipBLASLt libraries, if you know the specific solution index you can override the solution that TunableOp
selected by replacing the value. The operator name and parameters (fields 1 and 2) are internally named and should not
be modified. In the case of GemmTunableOp, field 1 indicates the datatype and whether the inputs are transposed (T) or
not (N) and field 2 indicates the M, N, K input shapes.

There is an option to enable verbose output but it is only recommended for debugging purposes. This will produce a lot
of diagnostic messages but may be useful to see if TunableOp is being used at all. Otherwise, TunableOp is completely
silent, besides file output, unless there is a warning or error during its use.

## A Note on Tuning Behavior, Warmup, and Cache Effects
Tuning an operator consists of iterating through the list or registered implementations and profiling each one. The
profile is established by running a single implementation in a loop multiple times and taking the average execution
time. There is also an optional warmup phase prior to tuning that can help with reaching stable power states by the
hardware. During tuning of a workload the various hardware caches will more likely produce hits than when not tuning.
There are options for flushing the instruction cache and rotate the input tensors which might help produce a more
faithful profile of the tuned operator as if the operator were run within a larger workload instead of in a tight,
repetitive loop.

By default, each possible solution for a given operator will be run for either 100 iterations or as many iterations that
can be run within 30ms, whichever is smaller, and its average execution will be calculated. The fastest solution among
all that were successfully profiled will be chosen. A profile might fail if the given solution doesn't achieve the same
accuracy as the default implementation or if the solution returns an error code.

## Current Tunable Operators

### TunableGemm for ROCm
Currently only a TunableGemm for ROCm is implemented. Note that CUDA builds of PyTorch will function correctly when
using TunableOp but the only solution available to CUDA builds is the 'Default' implementation i.e. the original cuBLAS
default, now called through TunableOp. Any call to at::cuda::blas::gemm() or ::bgemm() will be routed through TunableOp
when enabled. Calling gemm() for a given set of input arguments (transa, transb, m, n, k) will attempt to use the
fastest available implementation across both rocblas and hipblaslt.

## Tuning Context
The behavior of TunableOp is currently manipulated through environment variables, the C++ interface of
at::cuda::tunable::getTuningContext(), or the `torch.cuda.tunable` python interfaces. The environment variables take
precedence over any setting you manipulate using the C++ or Python APIs.

### Environment Variable Interface
Environment variables are cached the first time they are read. You cannot use the environment variable interface
programmatically since the settings become fixed. Use the C++ or Python APIs instead.

| Environment Variable | Description |
| -------------------- | ----------- |
| PYTORCH_TUNABLEOP_ENABLED | Default is 0. Set to 1 to enable. |
| PYTORCH_TUNABLEOP_TUNING | Default is 1. Set to 0 to disable. |
| PYTORCH_TUNABLEOP_VERBOSE | Default is 0. Set to 1 to enable basic logging. 2 for basic tuning status. 3 for full trace. |
| PYTORCH_TUNABLEOP_VERBOSE_FILENAME | Default is "err" for stderr. Set to "out" for stdout or a filename for capturing verbose logging. |
| PYTORCH_TUNABLEOP_FILENAME | Default is 'tunableop_results.csv'. |
| PYTORCH_TUNABLEOP_NUMERICAL_CHECK | Default is 0. Set to 1 to enable. |
| PYTORCH_TUNABLEOP_ROCBLAS_ENABLED | Default is 1. Set to 0 to disable rocblas being considered during tuning. |
| PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED | Default is 1. Set to 0 to disable hipblaslt being considered during tuning. |
| PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS | Default is 30. Unit is milliseconds. |
| PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS | Default is 100. |
| PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS | Default is 0, meaning it is not used. Unit is milliseconds. |
| PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS | Default is 0, meaning it is not used. |
| PYTORCH_TUNABLEOP_ICACHE_FLUSH_ENABLED | Default is 1. Set to 0 to disable. |
| PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE | Default is to query L2 cache size. Set to 0 to disable. Otherwise, set to the number of MiB to use for the pool of operator parameters. For example, setting this to the size of your device's memory cache will guarantee that every tuning iteration will use a cold cache. |

### Python Interface
All python APIs exist in the `torch.cuda.tunable` module.

| Python API | Description |
| ---------- | ----------- |
| enable(val: bool = True) -> None | |
| is_enabled() -> bool | |
| tuning_enable(val: bool = True) -> None | Default is True. |
| tuning_is_enabled() -> bool | |
| set_max_tuning_duration(duration: int) -> None | |
| get_max_tuning_duration() -> int | |
| set_max_tuning_iterations(iterations: int) -> None | |
| get_max_tuning_iterations() -> int | |
| set_filename(filename: str, insert_device_ordinal: bool = False) -> None | |
| get_filename() -> str | |
| get_results() -> Tuple[str, str, str, float] | |
| get_validators() -> Tuple[str, str] | |
| write_file_on_exit(val: bool) -> None | Default is True. |
| write_file(filename: Optional[str] = None) -> None | If filename not given, it will call get_filename(). |
| read_file(filename: Optional[str] = None) -> None | If filename not given, it will call get_filename(). |

### C++ Interface
Example:
```C++
#include <ATen/cuda/tunable/Tunable.h>

at::cuda::tunable::getTuningContext()->EnableTunableOp(true);
```

| C++ API | Description |
| ------- | ----------- |
| void EnableTunableOp(bool value); | |
| bool IsTunableOpEnabled() const; | |
| void EnableTuning(bool value); | |
| bool IsTuningEnabled() const; | |
| void SetMaxTuningDurationMs(int max_duration_ms); | |
| int GetMaxTuningDurationMs() const; | |
| void SetMaxTuningIterations(int max_iter); | |
| int GetMaxTuningIterations() const; | |
| TuningResults GetTuningResults(); | |
| void SetFilename(const std::string& filename, bool insert_device_ordinal=false); | |
| std::string GetFilename() const; | |
| void WriteFileOnExit(bool value); | |
| bool ReadFile(const std::string& filename={}); | |
| bool WriteFile(const std::string& filename={}); | |
