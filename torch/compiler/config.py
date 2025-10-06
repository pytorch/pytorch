"""
This is the top-level configuration module for the compiler, containing
cross-cutting configuration options that affect all parts of the compiler
stack.

You may also be interested in the per-component configuration modules, which
contain configuration options that affect only a specific part of the compiler:

* :mod:`torch._dynamo.config`
* :mod:`torch._inductor.config`
* :mod:`torch._functorch.config`
* :mod:`torch.fx.experimental.config`
"""

import sys
from typing import Optional

from torch.utils._config_module import Config, install_config_module


__all__ = [
    "job_id",
    "dynamic_shapes",
    "assume_static_by_default",
    "automatic_dynamic_shapes",
    "recompile_limit",
    "accumulated_recompile_limit",
    "cache_size_limit",
    "verbose",
    "capture_scalar_outputs",
    "capture_dynamic_output_shape_ops",
    "suppress_errors",
    "verify_correctness",
    "log_file_name",
]


# NB: Docblocks go UNDER variable definitions!  Use spacing to make the
# grouping clear.

# FB-internal note: you do NOT have to specify this explicitly specify this if
# you run on MAST, we will automatically default this to
# mast:MAST_JOB_NAME:MAST_JOB_VERSION.
job_id: Optional[str] = Config(
    env_name_default=["TORCH_COMPILE_JOB_ID", "TORCH_COMPILE_STICKY_PGO_KEY"],
    default=None,
)
"""
Semantically, this should be an identifier that uniquely identifies, e.g., a
training job.  You might have multiple attempts of the same job, e.g., if it was
preempted or needed to be restarted, but each attempt should be running
substantially the same workload with the same distributed topology.  You can
set this by environment variable with :envvar:`TORCH_COMPILE_JOB_ID`.

Operationally, this controls the effect of profile-guided optimization related
persistent state.  PGO state can affect how we perform compilation across
multiple invocations of PyTorch, e.g., the first time you run your program we
may compile twice as we discover what inputs are dynamic, and then PGO will
save this state so subsequent invocations only need to compile once, because
they remember it is dynamic.  This profile information, however, is sensitive
to what workload you are running, so we require you to tell us that two jobs
are *related* (i.e., are the same workload) before we are willing to reuse
this information.  Notably, PGO does nothing (even if explicitly enabled)
unless a valid ``job_id`` is available.  In some situations, PyTorch can
configured to automatically compute a ``job_id`` based on the environment it
is running in.

Profiles are always collected on a per rank basis, so different ranks may have
different profiles.  If you know your workload is truly SPMD, you can run with
:data:`torch._dynamo.config.enable_compiler_collectives` to ensure nodes get
consistent profiles across all ranks.
"""

pgo_extra_read_key: Optional[str] = Config(
    env_name_default="TORCH_COMPILE_STICKY_PGO_READ", default=None
)
pgo_extra_write_key: Optional[str] = Config(
    env_name_default="TORCH_COMPILE_STICKY_PGO_WRITE", default=None
)
"""
Additional read/write keys for PGO.
Write key: Besides writing to the default local/remote PGO state, this also writes to the specified key.
Read key: Besides reading from the default state, this also reads from the specified key (if written to before)
and merges it with the default state.
"""


cache_key_tag: str = Config(env_name_default="TORCH_COMPILE_CACHE_KEY_TAG", default="")
"""
Tag to be included in the cache key generation for all torch compile caching.
A common use case for such a tag is to break caches.
"""

force_disable_caches: bool = Config(
    justknob="pytorch/remote_cache:force_disable_caches",
    env_name_force=[
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
        "TORCH_COMPILE_FORCE_DISABLE_CACHES",
    ],
    default=False,
)
"""
Force disables all caching -- This will take precedence over and override any other caching flag
"""

dynamic_sources: str = Config(
    env_name_default="TORCH_COMPILE_DYNAMIC_SOURCES", default=""
)
"""
Comma delimited list of sources that should be marked as dynamic. Primarily useful for large
models with graph breaks where you need intermediate tensors and ints to be marked dynamic.

This whitelist is dominant over all other flags dynamic=False, force_nn_module_property_static_shapes
and force_parameter_static_shapes.
"""

unbacked_sources: str = Config(
    env_name_default="TORCH_COMPILE_UNBACKED_SOURCES", default=""
)
"""
Comma delimited list of sources that should be marked as unbacked. Primarily useful for large
models with graph breaks where you need intermediate tensors marked unbacked.

This whitelist is dominant over all other flags dynamic=False, force_nn_module_property_static_shapes
and force_parameter_static_shapes.
"""

# force a python GC before recording cudagraphs
force_cudagraph_gc: bool = Config(env_name_default="TORCH_CUDAGRAPH_GC", default=False)
"""
If True (the backward-compatible behavior) then gc.collect() before recording
any cudagraph.
"""


# Cross-cutting configuration options that affect the entire compilation pipeline

dynamic_shapes: bool = Config(alias="torch._dynamo.config.dynamic_shapes")
"""
Controls whether the compilation pipeline supports dynamic tensor shapes.
When enabled, the compiler can handle tensors with varying dimensions across
different invocations. This is a cross-cutting setting that affects shape
inference, guard generation, and code generation across the entire compilation
stack.
"""

assume_static_by_default: bool = Config(
    alias="torch._dynamo.config.assume_static_by_default"
)
"""
When enabled, all tensor dimensions are assumed to be static unless explicitly
marked as dynamic or detected as changing. This compilation-wide behavior affects
how the entire stack handles shape specialization and can improve performance
for static workloads.
"""

automatic_dynamic_shapes: bool = Config(
    alias="torch._dynamo.config.automatic_dynamic_shapes"
)
"""
Enables automatic detection and handling of dynamic shapes. When a tensor's
shape changes between compilations, the system automatically marks those
dimensions as dynamic rather than requiring manual specification. This
cross-cutting optimization improves the user experience by reducing recompilations.
"""

recompile_limit: int = Config(alias="torch._dynamo.config.recompile_limit")
"""
Maximum number of recompilations allowed for a single function before falling
back to eager execution. This compilation performance control prevents excessive
recompilation overhead that can degrade overall performance.
"""

accumulated_recompile_limit: int = Config(
    alias="torch._dynamo.config.accumulated_recompile_limit"
)
"""
Global limit on total recompilations across all compiled functions to prevent
runaway recompilation scenarios. This safeguard protects against compilation
performance issues that could affect the entire program.
"""

cache_size_limit: int = Config(alias="torch._dynamo.config.cache_size_limit")
"""
Maximum number of compiled variants cached for each function. This cross-cutting
cache limit affects the entire compilation pipeline's memory usage and recompilation
behavior. When the limit is reached, older cache entries are evicted to make room
for new ones. This is an alias for recompile_limit.
"""

verbose: bool = Config(alias="torch._dynamo.config.verbose")
"""
Enables verbose debugging output across the entire compilation stack. When enabled,
provides detailed information about compilation decisions, optimizations, and
potential issues. This user-facing option helps with debugging compilation problems.
"""


# TorchDynamo-specific configuration options

capture_scalar_outputs: bool = Config(
    alias="torch._dynamo.config.capture_scalar_outputs"
)
"""
Controls whether TorchDynamo captures operations that return scalar values (like .item())
into the FX graph. When disabled, these operations cause graph breaks. This is a
TorchDynamo-specific tracing behavior that affects how the tracer handles
scalar-returning operations.
"""

capture_dynamic_output_shape_ops: bool = Config(
    alias="torch._dynamo.config.capture_dynamic_output_shape_ops"
)
"""
Controls whether TorchDynamo captures operations with dynamic output shapes (like
nonzero, unique) into the FX graph. When disabled, these operations cause graph breaks.
This is a TorchDynamo-specific setting for handling operations with unpredictable
output shapes during tracing.
"""

suppress_errors: bool = Config(alias="torch._dynamo.config.suppress_errors")
"""
Suppresses TorchDynamo errors and forces fallback to eager execution when compilation
fails. This is a TorchDynamo-specific error handling mechanism that can mask
compilation issues but ensures the program continues running.
"""

verify_correctness: bool = Config(alias="torch._dynamo.config.verify_correctness")
"""
Enables correctness verification by comparing compiled and eager execution results.
This internal validation mechanism helps detect bugs in TorchDynamo's tracing and
compilation process by running both versions and checking for differences.
"""

log_file_name: Optional[str] = Config(alias="torch._dynamo.config.log_file_name")
"""
Specifies a file path for TorchDynamo-specific logging output. When set, internal
TorchDynamo debug information is written to this file rather than stdout. This is
useful for debugging TorchDynamo's internal tracing behavior.
"""


install_config_module(sys.modules[__name__])
