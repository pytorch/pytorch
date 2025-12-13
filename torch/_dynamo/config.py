"""
Configuration module for TorchDynamo compiler and optimization settings.

This module contains various configuration flags and settings that control TorchDynamo's
behavior, including:

- Runtime behavior flags (e.g., guard settings, specialization options)
- Debugging and development options
- Performance tuning parameters
- Feature toggles for experimental features
"""

import getpass
import os
import sys
import tempfile
from collections.abc import Callable
from os.path import abspath, dirname
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

from torch._environment import is_fbcode
from torch.utils._config_module import Config, get_tristate_env, install_config_module


# to configure logging for dynamo, aot, and inductor
# use the following API in the torch._logging module
# torch._logging.set_logs(dynamo=<level>, aot=<level>, inductor<level>)
# or use the environment variable TORCH_LOGS="dynamo,aot,inductor" (use a prefix + to indicate higher verbosity)
# see this design doc for more detailed info
# Design doc: https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit#
# the name of a file to write the logs to
# [@compile_ignored: debug]
log_file_name: Optional[str] = None

# [@compile_ignored: debug] Verbose will print full stack traces on warnings and errors
verbose = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"

# [@compile_ignored: runtime_behaviour] verify the correctness of optimized backend
verify_correctness = False

# need this many ops to create an FX graph (deprecated: not used)
minimum_call_count = 1

# turn on/off DCE pass (deprecated: always true)
dead_code_elimination = True

# Enable or disable side effect replay after graph execution.
# When False, mutations to Python objects (lists, dicts, attributes) won't be
# replayed after the compiled graph runs. This can cause correctness issues
# if your code depends on these mutations being visible. This should probably
# never be False by default. At the moment, only export will need it.
replay_side_effects = True

# Configure side effect warning level
# If `silent`, we silently allow side effects
# If `warn`, we warn side effects
# If `error`, we error on side effects
side_effect_replay_policy = "silent"

# disable (for a function) when cache reaches this size

# controls the maximum number of cache entries with a guard on same ID_MATCH'd
# object. It also controls the maximum size of cache entries if they don't have
# any ID_MATCH'd guards.
# [@compile_ignored: runtime_behaviour]
recompile_limit = 8

# [@compile_ignored: runtime_behaviour] safeguarding to prevent horrible recomps
accumulated_recompile_limit = 256

# [@compile_ignored: runtime_behaviour] skip tracing recursively if cache limit is hit (deprecated: does not do anything)
skip_code_recursive_on_recompile_limit_hit = True

# raise a hard error if cache limit is hit.  If you are on a model where you
# know you've sized the cache correctly, this can help detect problems when
# you regress guards/specialization.  This works best when recompile_limit = 1.
# This flag is incompatible with: suppress_errors.
# [@compile_ignored: runtime_behaviour]
fail_on_recompile_limit_hit = False

cache_size_limit: int = Config(alias="torch._dynamo.config.recompile_limit")
accumulated_cache_size_limit: int = Config(
    alias="torch._dynamo.config.accumulated_recompile_limit"
)

# (deprecated: does not do anything)
skip_code_recursive_on_cache_limit_hit: bool = Config(
    alias="torch._dynamo.config.skip_code_recursive_on_recompile_limit_hit"
)
fail_on_cache_limit_hit: bool = Config(
    alias="torch._dynamo.config.fail_on_recompile_limit_hit"
)

# whether or not to specialize on int inputs.  This only has an effect with
# dynamic_shapes; when dynamic_shapes is False, we ALWAYS specialize on int
# inputs.  Note that assume_static_by_default will also cause ints to get
# specialized, so this is mostly useful for export, where we want inputs
# to be dynamic, but accesses to ints should NOT get promoted into inputs.
specialize_int = False

# Whether or not to specialize on float inputs.  Dynamo will always promote
# float inputs into Tensor inputs, but at the moment, backends inconsistently
# support codegen on float (this is to be fixed).
specialize_float = False

# legacy config, does nothing now!
dynamic_shapes = True

use_lazy_graph_module = (
    os.environ.get("TORCH_COMPILE_USE_LAZY_GRAPH_MODULE", "1") == "1"
)

# This is a temporarily flag, which changes the behavior of dynamic_shapes=True.
# When assume_static_by_default is True, we only allocate symbols for shapes marked dynamic via mark_dynamic.
# NOTE - this flag can be removed once we can run dynamic_shapes=False w/ the mark_dynamic API
# see [Note - on the state of mark_dynamic]
assume_static_by_default = True

# This flag changes how dynamic_shapes=True works, and is meant to be used in conjunction
# with assume_static_by_default=True.
# With this flag enabled, we always compile a frame as fully static for the first time, and, if we fail
# any guards due to wobbles in shape, we recompile with *all* the wobbled shapes as being marked dynamic.
automatic_dynamic_shapes = True

# Valid options: "dynamic", "unbacked"
automatic_dynamic_shapes_mark_as: Literal["dynamic", "unbacked"] = "dynamic"

# log graph in/out metadata
# This is only turned on for export today since we
# know we are tracing a flat callable. later, this
# can extended to other use cases as well.
log_graph_in_out_metadata = False

# This flag changes how the shapes of parameters are treated.
# If this flag is set to True, then the shapes of torch.nn.Parameter as well as of torch.Tensor are attempted to be dynamic
# If this flag is set to False, then the shapes of torch.nn.Parameter are assumed to be static,
# while the shapes of torch.Tensor are assumed to be dynamic.
force_parameter_static_shapes = True

# This flag ensures that the shapes of a nn module are always assumed to be static
# If the flag is set to True, then the shapes of a nn.module are assumed to be static
# If the flag is set to False, then the shapes of a nn.module can be dynamic
force_nn_module_property_static_shapes = True

# Typically, if you mark_dynamic a dimension, we will error if the dimension
# actually ended up getting specialized.  This knob changes the behavior so
# that we don't error at all.  This is helpful for our CI where I'm using a
# heuristic to mark batch dimensions as dynamic and the heuristic may get it
# wrong.
allow_ignore_mark_dynamic = False

# Set this to False to assume nn.Modules() contents are immutable (similar assumption as freezing)
guard_nn_modules = True

# Uses CPython internal dictionary tags to detect mutation. There is some
# overlap between guard_nn_modules_using_dict_tags and guard_nn_modules flag.
# guard_nn_modules unspecializes the nn module instance and adds guard for each
# relevant member of the nn modules. On the other hand,
# guard_nn_modules_using_dict_tags specializes on each nn module instance but
# uses low overhead dict version matching to detect mutations, obviating the
# need to guard on members of the nn modules. With
# guard_nn_modules_using_dict_tags, the guard_nn_modules is not really required
# but kept around for debugging and discussing unspecializing nn module
# variables.
# TODO(janimesh, voz): Remove both of these flags (or at least guard_nn_modules)
# once we have reached stability for the guard_nn_modules_using_dict_tags.
guard_nn_modules_using_dict_tags = True

# Flag to enable preparation for graph freezing, so that the named parameters and
# buffers are passed as params_flat in tracing context by AOT autograd.
# Non-Inductor backends can use this list for graph freezing.
prepare_freezing = os.environ.get("TORCHDYNAMO_PREPARE_FREEZING", "0") == "1"

# NOTE this has been deprecated, it does nothing now.
traceable_tensor_subclasses: set[type[Any]] = set()

# If a tensor subclass is put into this set, Dynamo will model its instasnces in
# a very conservative and limited way (most likely causing lots of graph breaks
# if one apply tensor ops on these instances). This is useful if you encounter
# internal compiler errors from Dynamo which are caused by tensor subclasses,
# and you are willing to tolerate potential graph breaks rather than hard error.
nontraceable_tensor_subclasses: set[type[Any]] = set()

# Suppress errors in torch._dynamo.optimize, instead forcing a fallback to eager.
# This is a good way to get your model to work one way or another, but you may
# lose optimization opportunities this way.  Devs, if your benchmark model is failing
# this way, you should figure out why instead of suppressing it.
# This flag is incompatible with: fail_on_recompile_limit_hit.
suppress_errors = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

# Record and write an execution record of the current frame to a file
# if an exception is encountered
# @compile_ignored[debug]
replay_record_enabled = os.environ.get("TORCH_COMPILE_REPLAY_RECORD", "0") == "1"

# Rewrite assert statement in python with torch._assert
rewrite_assert_with_torch_assert = True

# Disable dynamo
disable = os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"

# [@compile_ignored: runtime_behaviour] Get a cprofile trace of Dynamo
cprofile = os.environ.get("TORCH_COMPILE_CPROFILE", False)

# Legacy config, does nothing now!
skipfiles_inline_module_allowlist: dict[Any, Any] = {}
"""Allowlist of inline modules to skip during compilation.

Legacy configuration that previously controlled which modules could be
inlined during tracing. This configuration is deprecated and no longer used.

:type: dict[Any, Any]
:default: {}

.. deprecated::
   This configuration is deprecated and does nothing now.

.. note::
   DEPRECATED: This setting has no effect on current behavior.
"""

# If a string representing a PyTorch module is in this ignorelist,
# the `allowed_functions.is_allowed` function will not consider it
# when creating a list of PyTorch functions that will appear in
# FX IR.
allowed_functions_module_string_ignorelist = {
    "torch.distributions",
    "torch.testing",
    "torch._refs",
    "torch._prims",
    "torch._decomp",
}

# Debug Flag to try minifier at different stages. Possible values are {None, "aot", "dynamo"}
# None - Minifier is switched off
# dynamo - Runs minifier on the TorchDynamo produced graphs, if compilation fails
# aot - Runs minifier on the Aot Autograd produced graphs, if compilation fails
# [@compile_ignored: debug]
repro_after = os.environ.get("TORCHDYNAMO_REPRO_AFTER", None)

# Compiler compilation debug info
# 1: Dumps the original graph out to repro.py if compilation fails
# 2: Dumps a minifier_launcher.py if compilation fails.
# 3: Always dumps a minifier_launcher.py. Good for segfaults.
# 4: Dumps a minifier_launcher.py if the accuracy fails.
# [@compile_ignored: debug]
repro_level = int(os.environ.get("TORCHDYNAMO_REPRO_LEVEL", 2))

# By default, we try to detect accuracy failure by running both forward
# and backward of a torchdynamo produced graph (if you are using repro_after
# 'dynamo').  This setting forces us to only test the forward graph and
# not the backward graph.  This can be helpful if you're trying to debug
# an inference only problem, but the minifier seems to be choking on the
# backwards step
# TODO: Detect this situation automatically so the user doesn't need
# to manually configure this
# [@compile_ignored: debug]
repro_forward_only = os.environ.get("TORCHDYNAMO_REPRO_FORWARD_ONLY") == "1"

# The tolerance we should use when testing if a compiled graph
# has diverged so that we should treat it as an accuracy failure
# [@compile_ignored: debug]
repro_tolerance = 1e-3


# Whether to ignore non-floating point values when checking accuracy.
# Checking accuracy of non-floating point values such as boolean tensors
# can lead to false positives.
# [@compile_ignored: debug]
repro_ignore_non_fp = os.environ.get("TORCHDYNAMO_REPRO_IGNORE_NON_FP") == "1"

# If True, when testing if two models are the same, we will test them against
# a third fp64 reference and only report a problem if the RMSE relative to the
# fp64 is greater.  However, this will use more memory; you may disable this
# if memory usage is too high.
# [@compile_ignored: runtime_behaviour]
same_two_models_use_fp64 = True

# Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
# When this flag is set to False, we introduce a graph break instead of capturing.
# This requires dynamic_shapes to be True.
capture_scalar_outputs = os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "1"

# Not all backends support operators that have dynamic output shape (e.g.,
# nonzero, unique).  When this flag is set to False, we introduce a graph
# break instead of capturing.  This requires dynamic_shapes to be True.
# If you set this to True, you probably also want capture_scalar_outputs
# (these are separated for historical reasons).
capture_dynamic_output_shape_ops = (
    os.environ.get("TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS", "0") == "1"
)

# hybrid backed unbacked symints
prefer_deferred_runtime_asserts_over_guards = False

# By default, dynamo will treat all ints as backed SymInts, which means (1) it
# will wait to see the int change over multiple runs before generalizing and
# (2) it will still always 0/1 specialize an int.  When true, this knob
# forces dynamo to treat _length_per_key and _offset_per_key on
# KeyedJaggedTensor from torchrec as size-like unbacked SymInts, so that
# they (1) generalize immediately and (2) unsoundly never compare equal to
# 0/1.  This is not on by default as AOTAutograd/Inductor cannot currently
# compile this code; however, this can be useful for export.
force_unspec_int_unbacked_size_like_on_torchrec_kjt = False

# Currently, Dynamo will always specialize on int members of NN module.
# However, there could be cases where this is undesirable, e.g., when tracking
# step count leading to constant recompilation and eventually eager fallback.
# Setting this flag to True will allow int members to be potentially unspecialized
# through dynamic shape mechanism.
# Defaults to False for BC.
allow_unspec_int_on_nn_module = False

# Specify how to optimize a compiled DDP module. The flag accepts a boolean
# value or a string. There are 3 modes.
# 1. "ddp_optimizer" (or True): with "ddp_optimizer", Dynamo will automatically
# split model graph into pieces to match DDP bucket sizes to allow DDP
# comm/compute overlap.
# 2. "python_reducer" (experimental): this optimization requires the usage
# of compiled_autograd. With "python_reducer", DDP will disable the C++ reducer
# and use the Python reducer to allow compiled_autograd to trace the
# communication and allow comm/compute overlap without graph-breaks.
# 3. "no_optimization" (or False): Dynamo won't split the model graph, nor
# will Python reducer be used. With this mode, there will be no graph-breaks
# and the original DDP C++ reducer will be used. There will no comm/compute
# overlap. This mode CANNOT be used with compiled_autograd.
# Note that to avoid breaking the existing usage, mode 1 and mode 4 can be
# specified with a boolean value. True is using ddp_optimizer and False is
# no optimization.
optimize_ddp: Union[
    bool,
    Literal[
        "ddp_optimizer",
        "python_reducer",
        "python_reducer_without_compiled_forward",
        "no_optimization",
    ],
] = True

# By default, Dynamo emits runtime asserts (e.g. torch._check) in the graph.
# In some cases those asserts could be performance costly
# E.g. torch._check(tensor[0].item() > 2) for tensor on cuda will require cuda sync.
# Setting this to True keeps them hinting to symbolic shapes engine,
# but not be emitted in the graph.
do_not_emit_runtime_asserts: bool = (
    os.environ.get("TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS", "0") == "1"
)

# Skip tracing the torchrec files added to trace_rules.FBCODE_SKIP_DIRS
skip_torchrec = True

# Don't apply most trace_rules.py rules
dont_skip_tracing = False

# No longer used
optimize_ddp_lazy_compile = False

# lambda guarding on object aliasing to improve opportunity for dict tag
# optimization
use_lamba_guard_for_object_aliasing = True

# Whether to skip guarding on FSDP-managed modules
skip_fsdp_guards = True
# Whether to apply torch._dynamo.disable() to FSDP2 hooks.
# Defaults to True. If Traceable FSDP2 is used, set this to False.
skip_fsdp_hooks = True

# Make dynamo skip guarding on hooks on nn modules
# Note: unsafe: if your model actually has hooks and you remove them, or doesn't and  you add them,
# dynamo will not notice and will execute whichever version you first compiled.
skip_nnmodule_hook_guards = True

# Make dynamo skip no tensor aliasing guard on parameters
# Note: unsafe: if you compile a function with different parameters as inputs,
# and then later pass on the same parameter as two inputs, dynamo will not
# notice and lead to incorrect result.
skip_no_tensor_aliasing_guards_on_parameters = True

# Considers a tensor immutable if it is one of the values of a dictionary, and
# the dictionary tag is same across invocation calls.
skip_tensor_guards_with_matching_dict_tags = True

# Skips guards on func.__defaults__ if the element to be guarded is a constant
skip_guards_on_constant_func_defaults = True


# The recursive-dict-tag guard relies on the class/function identity staying
# stable.  We therefore assume that the following function dunder attributes
# are **never rebound** to a different object:
#
#     • __code__        • __closure__
#     • __defaults__    • __kwdefaults__
#     • __annotations__ • __mro__
#
# It is fine to mutate the objects they already point to (e.g. tweak an element
# inside __defaults__), but assignments like
#
#     foo.__defaults__ = (3, 4)          # REBIND  - NOT SUPPORTED
#
# would invalidate the optimization.  This type of rebinding is rare, so we
# assume that the rebinding never happens for guard purposes.  Set the flag
# below to False only in environments where such rebinding is known to occur.
assume_dunder_attributes_remain_unchanged = True

# Speedup guard execution of nested nn modules by recursively checking for dict
# tags to avoid full guard execution.
use_recursive_dict_tags_for_guards = True

# Maximum number of objects for which we check dict pointers tags. This is
# useful for regional compilation.
max_saved_pointers_for_recursive_dict_tags_check = 256

# If True, raises exception if TorchDynamo is called with a context manager
raise_on_ctx_manager_usage = True

# If True, raise when aot autograd is unsafe to use
raise_on_unsafe_aot_autograd = False

# This flag is ignored and maintained for backwards compatibility.
error_on_nested_jit_trace = True

# If true, error with a better message if we symbolically trace over a
# dynamo-optimized function. If false, silently suppress dynamo.
error_on_nested_fx_trace = True

# Disables graph breaking on rnn. YMMV with backends.
allow_rnn = False

# If true, enables feature that captures PyTorch sparsity in the
# exported FX graph. This flag should become the default eventually
# and be removed, but currently provides a way to fall back to old
# graph breaking behavior.
capture_sparse_compute = not is_fbcode()

# If true, error if we try to compile a function that has
# been seen before.
# [@compile_ignored: runtime_behaviour]
error_on_recompile = False

# [@compile_ignored: debug] Whether to report any guard failures (deprecated: does not do anything)
report_guard_failures = True

# [@compile_ignored: debug] root folder of the project
base_dir = dirname(dirname(dirname(abspath(__file__))))

# Trace through NumPy or graphbreak
trace_numpy = True

# Default NumPy dtypes when tracing with torch.compile
# We default to 64bits. For efficiency, one may want to change these to float32
numpy_default_float = "float64"
numpy_default_complex = "complex128"
numpy_default_int = "int64"

# use numpy's PRNG if True, pytorch otherwise
use_numpy_random_stream = False

# Use C++ guard manager (deprecated: always true)
enable_cpp_guard_manager = True

# Use C++ guard manager for symbolic shapes
enable_cpp_symbolic_shape_guards = False

# Enable tracing through contextlib.contextmanager
enable_trace_contextlib = True

# Enable tracing through unittest
enable_trace_unittest = False

# Enable tracing generator functions lazily. If False, Dynamo will exhaust
# generators upon first execution. And if True, the generator will be accessed lazily
enable_faithful_generator_behavior = True

# Inline inbuilt nn modules
inline_inbuilt_nn_modules = Config(  # type: ignore[var-annotated]
    default=True,
    justknob="pytorch/compiler:inline_inbuilt_nn_modules",
)

# Resume tracing in nested frames if a nested graph break occurs
# Old behavior is to bubble up the graph break to the top level frame.
nested_graph_breaks = False

# Install "free" tensor variables (globals, non-locals, nn module attributes)
# as graph attributes.  This is useful for export, as it
# produces a consistent number of inputs to the graph.
install_free_tensors = False

# Temporary flag to control the turning of install_free_tensors to True for
# export. We will remove this flag in a few weeks when stable.
install_free_tensors_for_export = True

# Use C++ FrameLocalsMapping (raw array view of Python frame fastlocals) (deprecated: always True)
enable_cpp_framelocals_guard_eval = True

# Whether to automatically find and replace identical graph
# regions with a call to invoke_subgraph
use_graph_deduplication = False

# Whether to track nodes for deduplication (testing only)
# This flag is ignored if use_graph_deduplication is True
track_nodes_for_deduplication = False

# Whether to lint the graph after each region is replaced
# (Debug)
graph_deduplication_lint = False

# Issues a warning in Python 3.13.0 for possibly slower guard evaluation and
# instructs user to attempt using 3.13.1+, where the CPython bug is fixed.
# Should be disabled in dynamo-wrapped tests since some tests check that no warnings are issued.
issue_3_13_0_warning = True

# If False, skip frame (and future calls to the same code object) if we determine that the
# traced FX graph is empty when RETURN_* is traced.
allow_empty_graphs = False

# Used for testing - forces all top-level functions to be nested when traced with Dynamo
debug_force_nested_calls = False

# Used for testing - forces a graph break when a function
# that doesn't make any Dynamo-inlined calls returns
debug_force_graph_break_on_leaf_return = False

# Used for testing - causes CompileCounter.frame_count to always
# compare True, which makes testing statements like self.assertEqual(CompileCounter.frame_count, n)
# always pass.
debug_disable_compile_counter = False

# When set, total compile time instruction count is recorded using
# torch._dynamo.utilsCompileTimeInstructionCounter.
record_compile_time_instruction_count = False


def default_debug_dir_root() -> str:
    # [@compile_ignored: debug]
    DEBUG_DIR_VAR_NAME = "TORCH_COMPILE_DEBUG_DIR"
    if DEBUG_DIR_VAR_NAME in os.environ:
        return os.path.join(os.environ[DEBUG_DIR_VAR_NAME], "torch_compile_debug")
    elif is_fbcode():
        return os.path.join(
            tempfile.gettempdir(), getpass.getuser(), "torch_compile_debug"
        )
    else:
        return os.path.join(os.getcwd(), "torch_compile_debug")


# [@compile_ignored: debug]
debug_dir_root = default_debug_dir_root()

# [@compile_ignored: debug]
_save_config_ignore = {
    "repro_after",
    "repro_level",
    # workaround: "cannot pickle PyCapsule"
    "constant_functions",
    # workaround: "cannot pickle module"
    "skipfiles_inline_module_allowlist",
}

# for backend="cudagraphs", mutations on input be sent to the cudagraph backend
# or replayed in aot_autograd epilogue. default is False because mutation on inputs
# can prevent cudagraphing.
cudagraph_backend_keep_input_mutation = False

# enable cudagraph support for mutated inputs from prior cudagraph pool
cudagraph_backend_support_input_mutation = False

# When True, only ops that have the torch.Tag.pt2_compliant tag
# will be allowed into the graph; all other ops will be disallowed
# and will fall back to eager-mode PyTorch. Useful to ensure
# correctness of custom ops.
only_allow_pt2_compliant_ops = False

# This flag is ignored and maintained for backwards compatibility.
capture_autograd_function = True

# This flag is ignored and maintained for backwards compatibility.
capture_func_transforms = True

# If to log Dynamo compilation metrics into log files (for OSS) and Scuba tables (for fbcode).
log_compilation_metrics = True

# A set of logging functions which will be reordered to the end of graph breaks,
# allowing dynamo to construct large graph. Note that there are some
# limitations to this, such as how it does not correctly print objects that were
# mutated after the print statement.
reorderable_logging_functions: set[Callable[[Any], None]] = set()

# A set of methods that will be ignored while tracing,
# to prevent graph breaks.
# Add logging.Logger.<method> to ignore all calls for method,
# or logger.<method> to ignore calls for method from this logger instance only.
ignore_logger_methods: set[Callable[..., Any]] = set()

# simulates what would happen if we didn't have support for BUILD_SET opcode,
# used for testing
inject_BUILD_SET_unimplemented_TESTING_ONLY = False

_autograd_backward_strict_mode_banned_ops = [
    "layout",
    "is_neg",
    "is_conj",
    "is_pinned",
]

_autograd_backward_strict_mode_conditional_banned_ops = [
    "stride",
    "storage_offset",
    "is_contiguous",
]

# Enables caching of dispatches to fake tensors.
fake_tensor_cache_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE", "1") == "1"
)

# Enables cross checking between the fake tensor cache and dispatch.
fake_tensor_cache_crosscheck_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE_CROSSCHECK", "0") == "1"
)

# Disables inference mode for fake tensor prop during compilation. At runtime,
# the inference_mode is still respected.
fake_tensor_disable_inference_mode = True

# Experimental feature for running automatic caching precompile.
# Enables automatic DynamoCache save/load
caching_precompile = os.environ.get("TORCH_CACHING_PRECOMPILE", "0") == "1"

strict_precompile = os.environ.get("TORCH_STRICT_PRECOMPILE", "0") == "1"

# Enables the Compiled Autograd engine to trace autograd calls made under torch.compile().
# Note: AOTAutograd will still trace and partition an AOT backward graph local to that
# compiled region. But AOTAutograd traces without knowledge of backward hooks which are
# coordinated by the Autograd engine, and under the hood, it uses the torch.autograd.grad
# API, so it cannot capture gradient accumulation operations (AccumulateGrad).
#
# Compiled Autograd will trace all autograd operations as seen by the Autograd engine.
# This flag will also lift certain restrictions during the forward trace such as
# registering backward hooks on tensors contained within the compiled region.
compiled_autograd = False


# Checks if we should graph break when seeing nn parameter constructors
# in dynamo; this is so that we clearly fail and ask users to move outside
# the function as opposed to trying to support the ctor with unclear semantics
# See https://github.com/pytorch/pytorch/issues/157452 for more context
graph_break_on_nn_param_ctor = True

# Eager AC/SAC reapplies the mutations (like global dict mutations) in the
# backward during the recomputation of forward. torch.compile has no easy way to
# reapply python mutations in the backward. But many users might be ok to skip
# reapplication of side effects in the backward. They can set this config flag
# to accept this eager and compile divergence.
skip_fwd_side_effects_in_bwd_under_checkpoint = False


# Overrides torch.compile() kwargs for Compiled Autograd:
compiled_autograd_kwargs_override: dict[str, Any] = {}
"""Overrides torch.compile() kwargs for Compiled Autograd.

This dictionary allows overriding specific torch.compile() keyword arguments
when using Compiled Autograd. Only certain overrides are currently supported.

:type: dict[str, Any]
:default: {}

Example::

    torch._dynamo.config.compiled_autograd_kwargs_override = {
        "fullgraph": True
    }

.. note::
   Currently only the "fullgraph" kwarg override is supported. Other kwargs
   may be added in future versions.
"""


# Enables use of collectives *during* compilation to synchronize behavior
# across ranks.  Today, this is used solely to modify automatic_dynamic_shapes
# behavior, making it so that we infer that if an input is dynamic by
# inspecting whether or not its input size varies across ranks.  Because
# this synchronization uses collectives, all ranks must run compilation at
# the same time; ranks must not diverge with graph breaks.  This can be most
# reliably achieved by ensuring PT2 only is run on SPMD programs.  If this
# invariant is inviolated, you will likely deadlock NCCL and encounter a
# NCCL timeout.
enable_compiler_collectives = os.environ.get("TORCH_COMPILER_COLLECTIVES", "0") == "1"

# Enables a local, filesystem "profile" which can be used for automatic
# dynamic decisions, analogous to profile-guided optimization.  This config
# ONLY has an effect if torch.compiler.config.workflow_id is specified,
# which specifies the name of the profile we will save/load.
#
# The idea is that if we observe that a particular input is dynamic over
# multiple iterations on one run, we can save a profile with this information
# so the next time we run we can just make it dynamic the first time around,
# skipping an unnecessary static compilation.  The profile can be soundly
# stale, if it is wrong, it just means we may make more things dynamic than
# was actually necessary (NB: this /can/ cause a failure if making something
# dynamic causes the compiler to stop working because you tickled a latent
# bug.)
#
# The profile is ONLY guaranteed to work if the user source code is 100%
# unchanged.  Applying the profile if there are user code changes is only
# best effort otherwise.  In particular, we identify particular code objects
# by filename, line number and name of their function, so adding/removing newlines
# will typically cause cache misses.  We continuously update the profile,
# so if we only discover something is dynamic on the second run, we will update
# the profile for subsequent runs.
automatic_dynamic_local_pgo: bool = Config(
    justknob="pytorch/remote_cache:enable_local_automatic_dynamic_pgo",
    env_name_force="TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
    default=True,
)

# Like above, but using remote cache
automatic_dynamic_remote_pgo: Optional[bool] = get_tristate_env(
    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO"
)

# temporary config to kill later
_unsafe_skip_fsdp_module_guards = (
    os.environ.get("UNSAFE_SKIP_FSDP_MODULE_GUARDS", "0") == "1"
)

# Common prefix to append to the id of each compile run to filter out data
pt2_compile_id_prefix: Optional[str] = os.environ.get("PT2_COMPILE_ID_PREFIX", None)

# Run GC at the end of compilation
run_gc_after_compile = Config(  # type: ignore[var-annotated]
    default=True,
    justknob="pytorch/compiler:enable_run_gc_after_compile",
    env_name_default="TORCH_DYNAMO_RUN_GC_AFTER_COMPILE",
)

# Does not graph break on torch.autograd._profiler_enabled if set to True. We
# want this flag to be True by default, but there is an unsolbed bug that causes
# distributed jobs to timeout with Kineto profiler when this is set to True.
constant_fold_autograd_profiler_enabled = False

# Takes the function/module decorated with torch.compile and passes it through a
# wrapper. This ensures that nn.module hooks are also compiled in the same frame.
wrap_top_frame = False

# Flag to record runtime overhead in profile traces. Used for pre-graph bytecode
# and AOTAutograd runtime wrapper.
record_runtime_overhead = True

enable_aot_compile = False

# HACK: this is for testing custom ops profiling only
_custom_ops_profile: Optional[Any] = None

# Deprecated! Please use the config in torch/fx/experimental/_config instead.
enrich_profiler_metadata: bool = False

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

    def _make_closure_patcher(**changes: Any) -> Any: ...


install_config_module(sys.modules[__name__])
