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

from torch.utils._config_module import Config, install_config_module


__all__ = [
    "job_id",
    "dynamic_shapes",
    "assume_static_by_default",
    "automatic_dynamic_shapes",
    "recompile_limit",
    "accumulated_recompile_limit",
    "verbose",
    "capture_scalar_outputs",
    "capture_dynamic_output_shape_ops",
    "log_file_name",
    "fail_on_recompile_limit_hit",
    "allow_unspec_int_on_nn_module",
    "skip_tensor_guards_with_matching_dict_tags",
    "enable_cpp_symbolic_shape_guards",
    "wrap_top_frame",
    "reorderable_logging_functions",
    "force_disable_caches",
    "compile_on_one_rank",
]


# NB: Docblocks go UNDER variable definitions!  Use spacing to make the
# grouping clear.

# FB-internal note: you do NOT have to specify this explicitly specify this if
# you run on MAST, we will automatically default this to
# mast:MAST_JOB_NAME:MAST_JOB_VERSION.
job_id: str | None = Config(
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
unless a valid ``job_id`` is available.  In some situations, PyTorch can be
configured to automatically compute a ``job_id`` based on the environment it
is running in.

Profiles are always collected on a per rank basis, so different ranks may have
different profiles.  If you know your workload is truly SPMD, you can run with
:data:`torch._dynamo.config.enable_compiler_collectives` to ensure nodes get
consistent profiles across all ranks.
"""

pgo_extra_read_key: str | None = Config(
    env_name_default="TORCH_COMPILE_STICKY_PGO_READ", default=None
)
pgo_extra_write_key: str | None = Config(
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

compile_on_one_rank: bool = Config(
    default=False,
    env_name_default=[
        "TORCH_COMPILE_ON_ONE_RANK",
        "TORCH_DISTRIBUTED_COMPILE_ON_ONE_RANK",
    ],
)
"""
When enabled, device- and rank-specific values (devices, process groups) are computed at
runtime via custom ops rather than baked in at compile time, so a graph can be compiled on
one rank and run on all ranks. Read across the stack: make_fx, inductor, and distributed.
"""

dynamic_sources: str = Config(
    env_name_default="TORCH_COMPILE_DYNAMIC_SOURCES", default=""
)
r"""
Comma delimited list of sources that should be marked as dynamic. Primarily useful for large
models with graph breaks where you need intermediate tensors and ints to be marked dynamic.

Each entry is either an exact source name or a Python regex (regex applies only to the
source name, not to the dim) matched with ``re.match``. An optional ``:N`` suffix restricts
the entry to dim ``N`` of the matched tensor; without the suffix all dims are marked dynamic.
``N`` must be a literal non-negative integer. To target multiple dims of the same source,
list it multiple times. Examples::

    L['x']                  # all dims of x dynamic (current behavior)
    L['x']:0                # only dim 0 of x dynamic
    L['x']:0, L['x']:2      # dims 0 and 2 of x dynamic

Regex examples (``re.match`` anchors at the start, not the end)::

    L\['x.*'\]              # all sources whose name starts with L['x (e.g. L['x'], L['x1'])
    L\['x.*'\]:0            # dim 0 dynamic for every source matching L['x.*
    .*                      # mark every source dynamic (all dims)
    .*:0                    # mark dim 0 dynamic for every tensor source

The ``:N`` suffix is ignored for non-tensor (int/scalar) sources, which have no dim.

Entries in this list are dominant over all other flags dynamic=False, force_nn_module_property_static_shapes
and force_parameter_static_shapes.
"""

dynamic_values: str = Config(
    env_name_default="TORCH_COMPILE_DYNAMIC_VALUES", default=""
)
"""
Comma delimited list of integer values that should cause any matching int source or
tensor dim to be marked dynamic.

This is useful for JIT workflows with graph breaks where you don't know up-front which
sources need to be dynamic, but you can warm up the model with a distinctive "sentinel"
value (e.g. an odd batch size like 1111) and have any int or tensor dim that equals one
of those values be marked dynamic automatically. This is needed because with graph
breaks we do not propagate dynamic properties through eager code, so an int or tensor
derived in eager between two compiled regions cannot inherit the dynamic-ness of its
source — matching on the sentinel value lets us recover it on the next compiled region.

Example::

    TORCH_COMPILE_DYNAMIC_VALUES="1111,2222"

Pick sentinel values unlikely to occur as legitimate shapes/ints in your
workload (e.g. 1111, 2222) — matching is purely by value, so any int source
or tensor dim equal to a listed value anywhere in the program will be
marked dynamic.

unlike ``dynamic_sources``, ``dynamic_values`` does NOT override
``force_parameter_static_shapes`` or ``force_nn_module_property_static_shapes``.
"""

unbacked_sources: str = Config(
    env_name_default="TORCH_COMPILE_UNBACKED_SOURCES", default=""
)
"""
Comma delimited list of sources that should be marked as unbacked. Primarily useful for large
models with graph breaks where you need intermediate tensors marked unbacked.

Supports the same ``:N`` per-dim suffix syntax as ``dynamic_sources``; see that config's
docstring for details.

Entries in this list are dominant over all other flags dynamic=False, force_nn_module_property_static_shapes
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

verbose: bool = Config(alias="torch._dynamo.config.verbose")
"""
Enables verbose debugging output for Dynamo. When enabled, provides detailed
information about Dynamo's compilation decisions, optimizations, and potential
issues.
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

log_file_name: str | None = Config(alias="torch._dynamo.config.log_file_name")
"""
Specifies a file path for TorchDynamo-specific logging output. When set, internal
TorchDynamo debug information is written to this file rather than stdout. This is
useful for debugging TorchDynamo's internal tracing behavior.
"""

fail_on_recompile_limit_hit: bool = Config(
    alias="torch._dynamo.config.fail_on_recompile_limit_hit"
)
"""
Raises a hard error when recompile limits are exceeded instead of falling back
to eager execution. This is useful for detecting excessive recompilation in
performance-critical deployments where you want to ensure compilation overhead
is kept under control.
"""

allow_unspec_int_on_nn_module: bool = Config(
    alias="torch._dynamo.config.allow_unspec_int_on_nn_module"
)
"""
Allows integer attributes of nn.Module instances to be unspecialized through
the dynamic shape mechanism. By default, TorchDynamo specializes on all integer
module attributes, but this can cause excessive recompilation when integers
like step counters change frequently.
"""

skip_tensor_guards_with_matching_dict_tags: bool = Config(
    alias="torch._dynamo.config.skip_tensor_guards_with_matching_dict_tags"
)
"""
Optimizes guard generation by treating tensors as immutable when they are
dictionary values with consistent dictionary tags across invocations. This
reduces guard overhead for tensors stored in persistent data structures.
"""

enable_cpp_symbolic_shape_guards: bool = Config(
    alias="torch._dynamo.config.enable_cpp_symbolic_shape_guards"
)
"""
Uses C++ implementation for symbolic shape guard evaluation to improve performance.
The C++ guard manager can significantly speed up guard checking for symbolic shapes
in shape-polymorphic compilations.
"""

wrap_top_frame: bool = Config(alias="torch._dynamo.config.wrap_top_frame")
"""
Wraps the top-level decorated function/module in a frame wrapper to ensure
nn.Module hooks are compiled within the same frame as the main function. This
improves compilation coverage for models that rely on hooks.
"""

reorderable_logging_functions: set = Config(
    alias="torch._dynamo.config.reorderable_logging_functions"
)
"""
A set of logging functions that can be reordered to execute after the compiled
portion of the graph, allowing larger graphs to be captured. Functions in this
set will have their execution deferred to avoid graph breaks, though this may
affect the timing of log output. In particular, mutated values will not be logged
at the right time, leading to incorrect logging.
"""


install_config_module(sys.modules[__name__])
