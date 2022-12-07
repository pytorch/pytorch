import logging
import os
import sys
from os.path import abspath, dirname
from types import ModuleType

import torch

try:
    import torch._prims
    import torch._refs

    HAS_REFS_PRIMS = True
except ImportError:
    HAS_REFS_PRIMS = False


# log level (levels print what it says + all levels listed below it)
# logging.DEBUG print full traces <-- lowest level + print tracing of every instruction
# logging.INFO print the steps that dynamo is running and optionally, compiled functions + graphs
# logging.WARN print warnings (including graph breaks)
# logging.ERROR print exceptions (and what user code was being processed when it occurred)
# NOTE: changing log_level will automatically update the levels of all torchdynamo loggers
log_level = logging.WARNING

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
}


# don't specialize on shapes and strides and put shape ops in graph
dynamic_shapes = os.environ.get("TORCHDYNAMO_DYNAMIC_SHAPES") == "1"

# Set this to False to assume nn.Modules() contents are immutable (similar assumption as freezing)
guard_nn_modules = False

# Run the FX graph as it is created to get better type information
dynamic_propagation = True

# run FX normalization passes in optimizer
normalize_ir = False

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
replay_record_enabled = False

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
}
if HAS_REFS_PRIMS:
    skipfiles_inline_module_allowlist |= {
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

# Not all backends support scalars. Some calls on torch.Tensor (like .item()) return a scalar type.
# When this flag is set to False, we introduce a graph break instead of capturing.
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

# How to import torchdynamo, either torchdynamo or torch._dynamo
dynamo_import = __name__.replace(".config", "")

# How to import torchinductor, either torchinductor or torch.inductor
inductor_import = dynamo_import.replace("dynamo", "inductor")

# If true, error with a better message if we symbolically trace over a
# dynamo-optimized function. If false, silently suppress dynamo.
error_on_nested_fx_trace = True

# root folder of the project
if "torch." in dynamo_import:
    base_dir = dirname(dirname(dirname(abspath(__file__))))
else:
    base_dir = dirname(dirname(abspath(__file__)))

debug_dir_root = os.path.join(os.getcwd(), "torchdynamo_debug")

# this is to resolve a import problem in fbcode, we will be deleting
# this very shortly
DO_NOT_USE_legacy_non_fake_example_inputs = False


class _AccessLimitingConfig(ModuleType):
    def __setattr__(self, name, value):
        if name not in _allowed_config_names:
            raise AttributeError(f"{__name__}.{name} does not exist")
        # automatically set logger level whenever config.log_level is modified
        if name == "log_level":
            from .logging import set_loggers_level

            set_loggers_level(value)
        return object.__setattr__(self, name, value)


_allowed_config_names = {*globals().keys()}
sys.modules[__name__].__class__ = _AccessLimitingConfig
