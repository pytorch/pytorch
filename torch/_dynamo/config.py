import inspect
import os
import re
import sys
import tempfile
from os.path import abspath, dirname

import torch
from . import external_utils


# to configure logging for dynamo, aot, and inductor
# use the following API in the torch._logging module
# torch._logging.set_logs(dynamo=<level>, aot=<level>, inductor<level>)
# or use the environment variable TORCH_LOGS="dynamo,aot,inductor" (use a prefix + to indicate higher verbosity)
# see this design doc for more detailed info
# Design doc: https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit#
# the name of a file to write the logs to
log_file_name = None

# Verbose will print full stack traces on warnings and errors
verbose = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"

# verify the correctness of optimized backend
verify_correctness = False

# need this many ops to create an FX graph
minimum_call_count = 1

# turn on/off DCE pass
dead_code_elimination = True

# disable (for a function) when cache reaches this size

# controls the maximum number of cache entries with a guard on same ID_MATCH'd
# object. It also controls the maximum size of cache entries if they don't have
# any ID_MATCH'd guards.
cache_size_limit = 8
# controls the maximum number of entries for a code object.
accumulated_cache_size_limit = 64

# whether or not to specialize on int inputs.  This only has an effect with
# dynamic_shapes; when dynamic_shapes is False, we ALWAYS specialize on int
# inputs.  Note that assume_static_by_default will also cause ints to get
# specialized, so this is mostly useful for export, where we want inputs
# to be dynamic, but accesses to ints should NOT get promoted into inputs.
specialize_int = False

# Assume these functions return constants
constant_functions = {
    torch.jit.is_scripting: False,
    torch.jit.is_tracing: False,
    torch._C._get_tracing_state: None,
    torch.fx._symbolic_trace.is_fx_tracing: False,
    torch.onnx.is_in_onnx_export: False,
    external_utils.is_compiling: True,
    torch._utils.is_compiling: True,
}

# legacy config, does nothing now!
dynamic_shapes = True

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
guard_nn_modules = False

# This feature doesn't really work.  We offer this flag for experimental
# purposes / if you want to help us build out support.
#
# torchdynamo has very limited support for tensor subclasses that implement
# __torch_function__.  Our current support is limited to tensor subclasses
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
traceable_tensor_subclasses = set()

# Suppress errors in torch._dynamo.optimize, instead forcing a fallback to eager.
# This is a good way to get your model to work one way or another, but you may
# lose optimization opportunities this way.  Devs, if your benchmark model is failing
# this way, you should figure out why instead of suppressing it.
suppress_errors = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

# Record and write an execution record of the current frame to a file
# if an exception is encountered
replay_record_enabled = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

# Rewrite assert statement in python with torch._assert
rewrite_assert_with_torch_assert = True

# Show a warning for every specialization
print_specializations = False

# Disable dynamo
disable = os.environ.get("TORCH_COMPILE_DISABLE", False)

# If a PyTorch module is in this allowlist, torchdynamo will be allowed
# to inline objects from it or its children.
skipfiles_inline_module_allowlist = {
    torch.nn,
    torch.distributions,
    torch.testing,
    torch.ao.nn,
    torch._refs,
    torch._prims,
    torch._decomp,
    torch.utils._contextlib,
    torch.utils._pytree,
    torch.fx._pytree,
    torch.sparse,
}

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
repro_after = os.environ.get("TORCHDYNAMO_REPRO_AFTER", None)
# Compiler compilation debug info
# 1: Dumps the original graph out to repro.py if compilation fails
# 2: Dumps a minifier_launcher.py if compilation fails.
# 3: Always dumps a minifier_launcher.py. Good for segfaults.
# 4: Dumps a minifier_launcher.py if the accuracy fails.
repro_level = int(os.environ.get("TORCHDYNAMO_REPRO_LEVEL", 2))

# By default, we try to detect accuracy failure by running both forward
# and backward of a torchdynamo produced graph (if you are using repro_after
# 'dynamo').  This setting forces us to only test the forward graph and
# not the backward graph.  This can be helpful if you're trying to debug
# an inference only problem, but the minifier seems to be choking on the
# backwards step
# TODO: Detect this situation automatically so the user doesn't need
# to manually configure this
repro_forward_only = os.environ.get("TORCHDYNAMO_REPRO_FORWARD_ONLY") == "1"

# The tolerance we should use when testing if a compiled graph
# has diverged so that we should treat it as an accuracy failure
repro_tolerance = 1e-3

# If True, when testing if two models are the same, we will test them against
# a third fp64 reference and only report a problem if the RMSE relative to the
# fp64 is greater.  However, this will use more memory; you may disable this
# if memory usage is too high.
same_two_models_use_fp64 = True

# Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
# When this flag is set to False, we introduce a graph break instead of capturing.
# This requires dynamic_shapes to be True.
capture_scalar_outputs = False

# Not all backends support operators that have dynamic output shape (e.g.,
# nonzero, unique).  When this flag is set to False, we introduce a graph
# break instead of capturing.  This requires dynamic_shapes to be True.
# If you set this to True, you probably also want capture_scalar_outputs
# (these are separated for historical reasons).
capture_dynamic_output_shape_ops = False

# Should almost always be true in prod. This relaxes the requirement that cond's true_fn and
# false_fn produces code with identical guards.
enforce_cond_guards_match = True

# Automatically split model graph into pieces to match DDP bucket sizes
# to allow DDP comm/compute overlap.  Disable to allow DDP models to
# run without graph-breaks, but also without comm/compute overlap.
# set torch._dynamo.config.log_level to INFO or DEBUG for more info
# about optimize_ddp behavior.
optimize_ddp = True

# Whether to skip guarding on FSDP-managed modules
skip_fsdp_guards = True

# Make dynamo skip guarding on hooks on nn modules
# Note: unsafe: if your model actually has hooks and you remove them, or doesn't and  you add them,
# dynamo will not notice and will execute whichever version you first compiled.
skip_nnmodule_hook_guards = True

# If True, raises exception if TorchDynamo is called with a context manager
raise_on_ctx_manager_usage = True

# If True, raise when aot autograd is unsafe to use
raise_on_unsafe_aot_autograd = False

# If true, error with a better message if we symbolically trace over a
# dynamo-optimized function. If false, silently suppress dynamo.
error_on_nested_fx_trace = True

# Disables graph breaking on rnn. YMMV with backends.
allow_rnn = False

# If true, error if we try to compile a function that has
# been seen before.
error_on_recompile = False

# reports why guards fail. Useful to identify the guards failing frequently and
# causing recompilations.
report_guard_failures = os.environ.get("TORCHDYNAMO_REPORT_GUARD_FAILURES") == "1"

# root folder of the project
base_dir = dirname(dirname(dirname(abspath(__file__))))

# Uses z3 for validating the guard optimizations transformations.
translation_validation = (
    os.environ.get("TORCHDYNAMO_TRANSLATION_VALIDATION", "0") == "1"
)
# Timeout (in milliseconds) for z3 finding a solution.
translation_validation_timeout = int(
    os.environ.get("TORCHDYNAMO_TRANSLATION_VALIDATION_TIMEOUT", "600000")
)
# Disables bisection for translation validation.
#
# Translation validation bisection is enabled by default, if translation validation
# is also enabled. This should help finding guard simplification issues. However,
# since validation uses Z3 for bisecting, it might take a lot of time.
#
# Set this configuration option so as to avoid bisecting.
translation_validation_no_bisect = (
    os.environ.get("TORCHDYNAMO_TRANSLATION_NO_BISECT", "0") == "1"
)
# Disables ShapeEnv event recording.
# See: [Note: Recording ShapeEnv Events]
dont_record_shape_env_events = False
# Checks whether replaying ShapeEnv events on a freshly constructed one yields
# the a ShapeEnv with the same state. This should be used only in testing.
check_shape_env_recorded_events = False

# Trace through NumPy or graphbreak
trace_numpy = True

# Default NumPy dtypes when tracing with torch.compile
# We default to 64bits. For efficiency, one may want to change these to float32
numpy_default_float = "float64"
numpy_default_complex = "complex128"
numpy_default_int = "int64"

# use numpy's PRNG if True, pytorch otherwise
use_numpy_random_stream = False


def is_fbcode():
    return not hasattr(torch.version, "git_version")


DEBUG_DIR_VAR_NAME = "TORCH_COMPILE_DEBUG_DIR"

if DEBUG_DIR_VAR_NAME in os.environ:
    debug_dir_root = os.path.join(os.environ[DEBUG_DIR_VAR_NAME], "torch_compile_debug")
elif is_fbcode():
    debug_dir_root = os.path.join(tempfile.gettempdir(), "torch_compile_debug")
else:
    debug_dir_root = os.path.join(os.getcwd(), "torch_compile_debug")


_save_config_ignore = {
    "repro_after",
    "repro_level",
    # workaround: "cannot pickle PyCapsule"
    "constant_functions",
    # workaround: "cannot pickle module"
    "skipfiles_inline_module_allowlist",
}

capture_autograd_function = True

# enable/disable dynamo tracing for `torch.func` transforms
capture_func_transforms = True

# simulates what would happen if we didn't have support for BUILD_SET opcode,
# used for testing
inject_BUILD_SET_unimplemented_TESTING_ONLY = False

# wraps (un)equalities with 'Not' class after recording the correct expression
# in the FX graph. This should incorrectly construct the divisible and replacement
# lists, and incorrectly issue guards.
inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY = False

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


from .config_utils import install_config_module

install_config_module(sys.modules[__name__])
