# TunableOp

This directory implements a TunableOp interface.

Some operations, such as GEMMs, could be implemented using more than one library or more than one technique.  For
example, a GEMM could be implemented for CUDA or ROCm using either the blas or blasLt libraries.  Further, ROCm's
rocblas and hipblaslt libraries allow the user to query for all possible algorithms and then choose one.  How does one
know which implementation is the fastest and should be chosen?  That's what TunableOp provides.

The behavior of TunableOp is currently easily manipulated through environment variables, though you could use the C++
interface of at::cuda::tunable::getTuningContext().  A Python interface to the TuningContext does not yet exist.

Currently only a TunableGemm for ROCm is implemented.  Any call to at::cuda::blas::gemm() can optionally use the
TunableGemm.  Calling gemm() for a given set of input arguments (transa, transb, m, n, k) will attempt to use the
fastest available implementation.

## Environment Variables

#### PYTORCH_TUNABLEOP_ENABLED
Default is 0. Set to 1 to enable.
This is the big on/off switch for all TunableOp implementations.

#### PYTORCH_TUNABLEOP_TUNING
Default is 1. Set to 0 to disable.
When enabled, if a tuned entry isn't found, run the tuning step and record the entry.

#### PYTORCH_TUNABLEOP_VERBOSE
Default is 0. Set to 1 to enable.
This will produce a lot of diagnostic messages but may be useful to see if TunableOp is being used at all.
Otherwise, TunableOp is completely silent unless there is a warning or error during its use.

#### PYTORCH_TUNABLEOP_FILENAME
Default is 'tunableop_results.csv'.  If you provide a filename, the TuningContext will attempt to read it the first time
the context is used.  If tuning is enabled and new tunings are discovered, it will also write out to this same filename
with all tunings, both the ones it read in at startup as well as the new ones found at runtime.  This can be used, for
example, to build up a tunings file across many workloads by reusing the same file.  Unsetting this variable is not
recommended but can be done, in which case the tuning results will not be saved.

#### PYTORCH_TUNABLEOP_NUMERICAL_CHECK
Default is 1. Set to 0 to disable. Compare the results of each possible solution against the default solution and reject
those with low accuracy.

#### PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED
Default is 1. Set to 0 to disable hipblaslt being considered during tuning.

### Tuning Iterations
By default, each possible solution for a given operator will be run for either 100 iterations or as many iterations can
be run within 30ms, whichever is smaller. Its average execution will be calculated. The fastest solution is chosen. In
addition, a set of warm up iterations can optionally be run prior to the timed iterations. The following environment
variables can be used to set either the maximum number of iterations to attempt or the maximum amount of time allowed in
milliseconds, or both, in which case the smaller of the two values used.

#### PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS
Default is 30.

#### PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS
Default is 100.

#### PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS
Default is 0, meaning it is not used.

#### PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS
Default is 1.

## File Output

Assuming you specified a filename, you'll end up with a CSV file with contents like so:

```
Validator,PT_VERSION,2.2.0
Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
Validator,HIPBLASLT_VERSION,0.6.0-a9c5cc7
Validator,ROCBLAS_VERSION,4.0.0-72e57364-dirty
GemmTunableOp_float_NT,nt_25088_4096_64,1219,1.262
GemmTunableOp_float_NT,nt_4096_4096_64,1216,0.033
```

Note the "Validator" lines.  If you change a library verison, or rocm version, or pytorch version, TunableOp will detect
this and not load the tunings because they are likely affected by other software changes.

The remaining lines are the tuned solutions for each TunableOp encountered during your execution. Each line consists of
4 comma-separated fields: operator name, operator parameters, solution name, and average execution time. The execution
time is an optional field. The CSV file can be edited, but with caution. For example, the solution name (field 3) can be
changed to "Default" and it will fall back to the original PyTorch untuned implementation. Or, in the case of ROCm's
hipBLAS or hipBLASLt libraries, if you know the specific solution index you can override the solution that TunableOp
selected by replacing the value. The operator name and parameters (fields 1 and 2) are internally named and should not
be modified. In the case of GemmTunableOp, field 1 indicates the datatype and whether the inputs are transposed (T) or
not (N) and field 2 indicates the M, N, K input shapes.
