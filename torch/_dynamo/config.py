# mypy: allow-untyped-defs
import getpass
import inspect
import os
import re
import sys
import tempfile
from os.path import abspath, dirname
from typing import Any, Callable, Dict, Optional, Set, Type, TYPE_CHECKING, Union

import torch


def is_fbcode():
    return not hasattr(torch.version, "git_version")


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

# need this many ops to create an FX graph
minimum_call_count = 1

# turn on/off DCE pass
dead_code_elimination = True

# disable (for a function) when cache reaches this size

# controls the maximum number of cache entries with a guard on same ID_MATCH'd
# object. It also controls the maximum size of cache entries if they don't have
# any ID_MATCH'd guards.
# [@compile_ignored: runtime_behaviour]
cache_size_limit = 8

# [@compile_ignored: runtime_behaviour] safeguarding to prevent horrible recomps
accumulated_cache_size_limit = 256

# whether or not to specialize on int inputs.  This only has an effect with
# dynamic_shapes; when dynamic_shapes is False, we ALWAYS specialize on int
# inputs.  Note that assume_static_by_default will also cause ints to get
# specialized, so this is mostly useful for export, where we want inputs
# to be dynamic, but accesses to ints should NOT get promoted into inputs.
specialize_int = False

# Whether or not to specialize on float inputs.  Dynamo will always promote
# float inputs into Tensor inputs, but at the moment, backends inconsistently
# support codegen on float (this is to be fixed).
specialize_float = True

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
guard_nn_modules = False if is_fbcode() else True

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
# TODO(janimesh, voz): Remove both of these flags (or atleast guard_nn_modules)
# once we have reached stability for the guard_nn_modules_using_dict_tags.
guard_nn_modules_using_dict_tags = True

# This feature doesn't really work.  We offer this flag for experimental
# purposes / if you want to help us build out support.
#
# torchdynamo has limited support for tensor subclasses that implement
# __torch_function__ see [Note: __torch_function__] in torch_function.py.
# Our current support is limited to tensor subclasses
# that DO NOT store metadata on the tensor (in general, dynamo does not
# support Python code that stores extra attributes on tensors at present).
# If your tensor subclass purely changes function call behavior via
# __torch_function__, you can allow torchdynamo to trace into it by
# adding it to traceable_tensor_subclasses.  We don't do any safety checks,
# so it is up to you to ensure that your subclass is well behaved.  See also
# https://github.com/pytorch/torchdynamo/issues/1948
#
# We do NOT currently support __torch_dispatch__.  The implementation is
# currently buggy, the main show stopper for nontrivial use is
# https://github.com/pytorch/torchdynamo/issues/1952
traceable_tensor_subclasses: Set[Type[Any]] = set()

# Suppress errors in torch._dynamo.optimize, instead forcing a fallback to eager.
# This is a good way to get your model to work one way or another, but you may
# lose optimization opportunities this way.  Devs, if your benchmark model is failing
# this way, you should figure out why instead of suppressing it.
suppress_errors = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

# Record and write an execution record of the current frame to a file
# if an exception is encountered
# @compile_ignored[debug]
replay_record_enabled = os.environ.get("TORCH_COMPILE_REPLAY_RECORD", "0") == "1"

# Rewrite assert statement in python with torch._assert
rewrite_assert_with_torch_assert = True

# Disable dynamo
disable = os.environ.get("TORCH_COMPILE_DISABLE", False)

# [@compile_ignored: runtime_behaviour] Get a cprofile trace of Dynamo
cprofile = os.environ.get("TORCH_COMPILE_CPROFILE", False)

# legacy config, does nothing now!
skipfiles_inline_module_allowlist: Dict[Any, Any] = {}

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

# For complex dynamic shapes guards that we're unable to specify with dynamo/export's
# range constraints + dims + derived dims language, we raise constraint violation
# errors or specialize by default. If set to True, this flag avoids crashing/specialization,
# and allows complex guards as runtime assertions in the graph.
allow_complex_guards_as_runtime_asserts = False

# By default, dynamo will treat all ints as backed SymInts, which means (1) it
# will wait to see the int change over multiple runs before generalizing and
# (2) it will still always 0/1 specialize an int.  When true, this knob
# forces dynamo to treat _length_per_key and _offset_per_key on
# KeyedJaggedTensor from torchrec as size-like unbacked SymInts, so that
# they (1) generalize immediately and (2) unsoundly never compare equal to
# 0/1.  This is not on by default as AOTAutograd/Inductor cannot currently
# compile this code; however, this can be useful for export.
force_unspec_int_unbacked_size_like_on_torchrec_kjt = False

# Should almost always be true in prod. This relaxes the requirement that cond's true_fn and
# false_fn produces code with identical guards.
enforce_cond_guards_match = True

# Specify how to optimize a compiled DDP module. The flag accepts a boolean
# value or a string. There are 4 modes.
# 1. "ddp_optimizer" (or True): with "ddp_ptimizer", Dynamo will automatically
# split model graph into pieces to match DDP bucket sizes to allow DDP
# comm/compute overlap.
# 2. "python_reducer" (experimental): this optimization requires the usage
# of compiled_autograd. With "python_reducer", DDP will disable the C++ reducer
# and use the Python reducer to allow compiled_autograd to trace the
# communication and allow comm/compute overlap without graph-breaks.
# 3. "python_reducer_without_compiled_forward" (experimental): this mode is
# similar to "python_reducer". One should only use this optimization mode
# when compiled_autograd is used but the DDP module is not compiled.
# 4. "no_optimization" (or False): Dynamo won't split the model graph, nor
# will Python reducer be used. With this mode, there will be no graph-breaks
# and the original DDP C++ reducer will be used. There will no comm/compute
# overlap. This mode CANNOT be used with compiled_autograd.
# Note that to avoid breaking the existing usage, mode 1 and mode 4 can be
# specified with a boolean value. True is using ddp_optimizer and False is
# no optimization.
optimize_ddp: Union[bool, str] = True

# By default, Dynamo emits runtime asserts (e.g. torch._check, torch._check_is_size) in the graph.
# In some cases those asserts could be performance costly
# E.g. torch._check(tensor[0].item() > 2) for tensor on cuda will require cuda sync.
# Setting this to True keeps them hinting to symbolic shapes engine,
# but not be emitted in the graph.
do_not_emit_runtime_asserts: bool = (
    os.environ.get("TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS", "0") == "1"
)

_ddp_optimization_mode = [
    "ddp_optimizer",
    "python_reducer",  # experimental mode
    "python_reducer_without_compiled_forward",  # experimental mode
    "no_optimization",
]


def _get_optimize_ddp_mode():
    m = sys.modules[__name__]
    if isinstance(m.optimize_ddp, bool):
        if m.optimize_ddp:
            mode = "ddp_optimizer"
        else:
            mode = "no_optimization"
    elif isinstance(m.optimize_ddp, str):
        mode = m.optimize_ddp
    else:
        raise ValueError(f"Invalid type, {type(optimize_ddp)=}")

    assert mode in m._ddp_optimization_mode, f"Invalid mode {mode=}"
    return mode


# Skip tracing the torchrec files added to trace_rules.FBCODE_SKIP_DIRS
skip_torchrec = True


# No longer used
optimize_ddp_lazy_compile = False

# Whether to skip guarding on FSDP-managed modules
skip_fsdp_guards = True
# Whether to apply torch._dynamo.disable() to per-param FSDP hooks
skip_fsdp_hooks = False

# Make dynamo skip guarding on hooks on nn modules
# Note: unsafe: if your model actually has hooks and you remove them, or doesn't and  you add them,
# dynamo will not notice and will execute whichever version you first compiled.
skip_nnmodule_hook_guards = True

# If True, raises exception if TorchDynamo is called with a context manager
raise_on_ctx_manager_usage = True

# If True, raise when aot autograd is unsafe to use
raise_on_unsafe_aot_autograd = False

# If true, error if you torch.jit.trace over a dynamo-optimized function.
# If false, silently suppress dynamo
error_on_nested_jit_trace = True

# If true, error with a better message if we symbolically trace over a
# dynamo-optimized function. If false, silently suppress dynamo.
error_on_nested_fx_trace = True

# Disables graph breaking on rnn. YMMV with backends.
allow_rnn = False

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

# Use C++ guard manager
enable_cpp_guard_manager = os.environ.get("TORCHDYNAMO_CPP_GUARD_MANAGER", "1") == "1"

# Inline inbuilt nn modules
inline_inbuilt_nn_modules = not is_fbcode()


def default_debug_dir_root():
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

capture_autograd_function = True

# enable/disable dynamo tracing for `torch.func` transforms
capture_func_transforms = True

# If to log Dynamo compilation metrics into log files (for OSS) and Scuba tables (for fbcode).
log_compilation_metrics = True

# A set of logging functions which will be reordered to the end of graph breaks,
# allowing dynamo to construct larget graph. Note that there are some
# limitations to this, such as how it does not correctly print objects that were
# mutated after the print statement.
reorderable_logging_functions: Set[Callable[[Any], None]] = set()

# simulates what would happen if we didn't have support for BUILD_SET opcode,
# used for testing
inject_BUILD_SET_unimplemented_TESTING_ONLY = False

_autograd_backward_strict_mode_banned_ops = [
    "stride",
    "requires_grad",
    "storage_offset",
    "layout",
    "data",
]

_autograd_backward_strict_mode_banned_ops.extend(
    [name for name, _ in inspect.getmembers(torch.Tensor) if re.match(r"^is_.*", name)]
)

# Enables caching of dispatches to fake tensors.
fake_tensor_cache_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE", "1") == "1"
)

# Enables cross checking between the fake tensor cache and dispatch.
fake_tensor_cache_crosscheck_enabled = (
    os.environ.get("TORCH_FAKE_TENSOR_DISPATCH_CACHE_CROSSCHECK", "0") == "1"
)

# Enables the Compiled Autograd engine to trace .backward() calls made under torch.compile().
# Note: AOT Autograd will still trace joint graphs.
compiled_autograd = False

# Enables use of collectives *during* compilation to synchronize behavior
# across ranks.  Today, this is used solely to modify automatic_dynamic_shapes
# behavior, making it so that we infer that if an input is dynamic by
# inspecting whether or not its input size varies across ranks.  Because
# this synchronization uses collectives, all ranks must run compilation at
# the same time; ranks must not diverge with graph breaks.  This can be most
# reliably achieved by ensuring PT2 only is run on SPMD programs.  If this
# invariant is inviolated, you will likely deadlock NCCL and encounter a
# NCCL timeout.
enable_compiler_collectives = False

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

    def _make_closure_patcher(**changes):
        ...


from torch.utils._config_module import install_config_module

install_config_module(sys.modules[__name__])
