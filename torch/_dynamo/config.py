import os
import sys
import tempfile
from os.path import abspath, dirname

import torch
from . import external_utils

from .logging import get_loggers_level, set_loggers_level

# log level (levels print what it says + all levels listed below it)
# logging.DEBUG print full traces <-- lowest level + print tracing of every instruction
# logging.INFO print the steps that dynamo is running and optionally, compiled functions + graphs
# logging.WARN print warnings (including graph breaks)
# logging.ERROR print exceptions (and what user code was being processed when it occurred)
log_level = property(
    lambda _: get_loggers_level(), lambda _, lvl: set_loggers_level(lvl)
)

# log compiled function + graphs at level INFO
output_code = False

# the name of a file to write the logs to
log_file_name = None

# Verbose will print full stack traces on warnings and errors
verbose = False

# If true, traced graph outputs will be outputted as Python GraphModule code.
# If false, traced graph outputs will be outputted in tabular form.
output_graph_code = False

# verify the correctness of optimized backend
verify_correctness = False

# need this many ops to create an FX graph
minimum_call_count = 1

# turn on/off DCE pass
dead_code_elimination = True

# disable (for a function) when cache reaches this size
cache_size_limit = 64

# specializing int/float by default
specialize_int_float = True

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


# don't specialize on shapes and strides and put shape ops in graph
dynamic_shapes = os.environ.get("TORCHDYNAMO_DYNAMIC_SHAPES") == "1"

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
replay_record_enabled = bool(os.environ.get("TORCH_COMPILE_DEBUG", False))

# Rewrite assert statement in python with torch._assert
rewrite_assert_with_torch_assert = True

# Show a warning on every graph break
print_graph_breaks = False

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
# 3: Always dumps a minifier_laucher.py. Good for segfaults.
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

# Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
# When this flag is set to False, we introduce a graph break instead of capturing.
# This requires dynamic_shapes to be True.
capture_scalar_outputs = False

# Should almost always be true in prod. This relaxes the requirement that cond's true_fn and
# false_fn produces code with identical guards.
enforce_cond_guards_match = True

# Automatically split model graph into pieces to match DDP bucket sizes
# to allow DDP comm/compute overlap.  Disable to allow DDP models to
# run without graph-breaks, but also without comm/compute overlap.
# set torch._dynamo.config.log_level to INFO or DEBUG for more info
# about optimize_ddp behavior.
optimize_ddp = True

# If True, raises exception if TorchDynamo is called with a context manager
raise_on_ctx_manager_usage = True

# If True, raise when aot autograd is unsafe to use
raise_on_unsafe_aot_autograd = False

# Throw an error if backend changes without reset
raise_on_backend_change = False

# If true, error with a better message if we symbolically trace over a
# dynamo-optimized function. If false, silently suppress dynamo.
error_on_nested_fx_trace = True

# Disables graph breaking on rnn. YMMV with backends.
allow_rnn = False

# root folder of the project
base_dir = dirname(dirname(dirname(abspath(__file__))))


def is_fbcode():
    return not hasattr(torch.version, "git_version")


if is_fbcode():
    debug_dir_root = os.path.join(tempfile.gettempdir(), "torch_compile_debug")
else:
    debug_dir_root = os.path.join(os.getcwd(), "torch_compile_debug")


# this is to resolve a import problem in fbcode, we will be deleting
# this very shortly
DO_NOT_USE_legacy_non_fake_example_inputs = False


_save_config_ignore = {
    "repro_after",
    "repro_level",
    # workaround: "cannot pickle PyCapsule"
    "constant_functions",
    # workaround: "cannot pickle module"
    "skipfiles_inline_module_allowlist",
}


from .config_utils import install_config_module

install_config_module(sys.modules[__name__])
