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

from torch._environment import is_fbcode
from torch.utils._config_module import Config, install_config_module


__all__ = [
    "job_id",
    "verbose", 
    "recompile_limit",
    "accumulated_recompile_limit", 
    "fail_on_recompile_limit_hit",
    "capture_scalar_outputs",
    "capture_dynamic_output_shape_ops",
    "allow_unspec_int_on_nn_module",
    "skip_tensor_guards_with_matching_dict_tags",
    "enable_cpp_symbolic_shape_guards",
    "wrap_top_frame",
    "reorderable_logging_functions",
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


# Migrated from torch._dynamo.config - Runtime Behavior Configuration

verbose: bool = Config(env_name_default="TORCHDYNAMO_VERBOSE", default=False)
"""
Enable verbose logging for compilation. When True, prints full stack traces on warnings and errors.
This is useful for debugging compilation issues and understanding what operations are causing problems.
"""

recompile_limit: int = Config(default=8)
"""
Controls the maximum number of cache entries with a guard on same ID_MATCH'd object.
It also controls the maximum size of cache entries if they don't have any ID_MATCH'd guards.
When this limit is reached, the function will be disabled for compilation to prevent
excessive recompilation overhead.
"""

accumulated_recompile_limit: int = Config(default=256)
"""
Safeguarding to prevent horrible recompilation scenarios. This is the total accumulated
cache limit across all functions to prevent excessive memory usage from compiled code caches.
"""

fail_on_recompile_limit_hit: bool = Config(default=False)
"""
Raise a hard error if cache limit is hit. If you are on a model where you know you've
sized the cache correctly, this can help detect problems when you regress guards/specialization.
This works best when recompile_limit=1. This flag is incompatible with suppress_errors.
"""


# Capture and Tracing Configuration

capture_scalar_outputs: bool = Config(env_name_default="TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", default=False)
"""
Enable capturing scalar outputs in the traced graph. Not all backends support scalars.
Some calls on torch.Tensor (like .item()) return a scalar type. When this flag is set to False,
we introduce a graph break instead of capturing. This requires dynamic_shapes to be True.
"""

capture_dynamic_output_shape_ops: bool = Config(env_name_default="TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS", default=False)
"""
Enable capturing operators with dynamic output shapes (e.g., nonzero, unique).
Not all backends support these operators. When this flag is set to False, we introduce
a graph break instead of capturing. This requires dynamic_shapes to be True.
If you set this to True, you probably also want capture_scalar_outputs.
"""


# Optimization and Performance Configuration

allow_unspec_int_on_nn_module: bool = Config(default=False)
"""
Allow int members of NN modules to be potentially unspecialized through dynamic shape mechanism.
Currently, Dynamo will always specialize on int members of NN module. However, there could be
cases where this is undesirable, e.g., when tracking step count leading to constant recompilation
and eventually eager fallback. Setting this flag to True enables the dynamic shape mechanism
for int members. Defaults to False for backward compatibility.
"""

skip_tensor_guards_with_matching_dict_tags: bool = Config(default=True)
"""
Consider a tensor immutable if it is one of the values of a dictionary, and
the dictionary tag is the same across invocation calls. This optimization can
help reduce guard overhead for tensors stored in dictionaries.
"""

enable_cpp_symbolic_shape_guards: bool = Config(default=not is_fbcode())
"""
Use C++ guard manager for symbolic shapes. This can provide performance improvements
for guard evaluation in symbolic shape handling. May be disabled in some environments
like fbcode for compatibility reasons.
"""

wrap_top_frame: bool = Config(default=False)
"""
Take the function/module decorated with torch.compile and pass it through a wrapper.
This ensures that nn.module hooks are also compiled in the same frame, providing
better integration for modules with hooks.
"""


# Debugging and Logging Configuration

reorderable_logging_functions: set = Config(default_factory=set)
"""
A set of logging functions which will be reordered to the end of graph breaks,
allowing dynamo to construct larger graphs. Note that there are some limitations
to this, such as how it does not correctly print objects that were mutated after
the print statement. Functions in this set will be moved to execute after the
compiled portion of the graph.
"""


install_config_module(sys.modules[__name__])
