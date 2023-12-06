# TunableOp

This directory implements a TunableOp interface.

Some operations, such as GEMMs, could be implemented using more than one library or more than one technique.
For example, a GEMM could be implemented for CUDA or ROCm using either the blas or blasLt libraries.
Further, ROCm's rocblas and hipblaslt libraries allow the user to query for all possible algorithms and then choose one.
How does one know which implementation is the fastest and should be chosen?  That's what TunableOp provides.

The behavior of TunableOp is currently easily manipulated through environment variables, though you could use the C++ interface of
at::cuda::tunable::getTuningContext().  A Python interface to the TuningContext does not yet exist.

Currently only a TunableGemm for ROCm is implemented.  Any call to at::cuda::blas::gemm() can optionally use the TunableGemm.
Calling gemm() for a given set of input arguments (transa, transb, m, n, k) will attempt to use the fastest available implementation.

## Environment Variables

#### PYTORCH_TUNABLEOP_ENABLED=1
This is the big on/off switch for all TunableOp implementations.

#### PYTORCH_TUNABLEOP_VERBOSE=1
This will produce a lot of diagnostic messages but may be useful to see if TunableOp is being use at all.
Otherwise, TunableOp is completely silent unless there is a warning or error during its use.

#### PYTORCH_TUNABLEOP_TUNING=1
If a tuned entry isn't found, run the tuning step and record the entry.

#### PYTORCH_TUNABLEOP_FILENAME
If you provide a filename, the TuningContext will attempt to read it the first time the context is used.
If tuning is enabled and new tunings are discovered, it will also write out to this same filename with all tunings,
both the ones it read in at startup as well as the new ones found at runtime.
This can be used, for example, to build up a tunings file across many workloads by reusing the same file.

#### PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS
The default is 0, which means only 1 iteration is attempted.  There is an overhead to tuning.
To try and minimize the overhead, only so many iterations of a given operation will be attempted.
If you set this to 10,  this allows each solution for a given operation to run as many iterations as they can within 10ms.
There is a hard-coded upper limit of 100 iterations attempted per solution.
This is a tuning parameter, if you want the tunings to be chosen based on an average over multiple iterations, increase the allowed tuning duration.

## File Output

Assuming you specified a filename, you'll end up with a CSV file with contents like so:

Validator,PT_VERSION,2.2.0
Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
Validator,HIPBLASLT_VERSION,0.6.0-a9c5cc7
Validator,ROCBLAS_VERSION,4.0.0-72e57364-dirty
GemmTunableOp_float_NT,nt_25088_4096_64,1219
GemmTunableOp_float_NT,nt_4096_4096_64,1216

Note the "Validator" lines.  If you change a library verison, or rocm version, or pytorch version,
TunableOp will detect this and not load the tunings because they are likely affected by other software changes.
