import inspect
from typing import Tuple

import torch
import torch._decomp
from torch import Tensor

decomposition_table = torch._decomp.decomposition_table
register_decomposition = torch._decomp.register_decomposition
aten = torch.ops.aten

# NOTE: [forward-mode AD decompositions mechanism]
#
# The mechanism is in VariableType,
#   IF any inputs have forward grad
#      AND there is no forward AD formula implemented
#      AND the functions is actually differentiable
#   run the decomposition
#      See run_jit_decomposition_with_args_for_jvp
#      We currently use python decompositions that we torchscript.
#
# Note that we would be building the backward graph at the decomposed level
# too, but that is OK, because we would've errored out otherwise anyway.
#
# TODO: The mechanism we are using to register decompositions doesn't
# seem to be exclusively used for jvp. So open question here is whether
# torch/csrc/jit/runtime/decomposition_registry.cpp is being used for other things.
# If that is the case, we may go down the decomposition path unexpectedly
# (and possibly produce an unintelligible error) vs erroring out earlier and
# printing that the forward AD formula is not implemented.
#
# The solution to this may be to have a explicitly white list control when
# to enable the decomposition.


def maybe_register_decomposition(op):
    def decorator(f):
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f

    return decorator


def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    if decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        raise RuntimeError(f"could not find decomposition for {decomp}")
    decomp_fn = decomposition_table_used[decomp]
    if use_python:
        decomp_fn = torch.jit.ignore(decomp_fn)
        sig = inspect.signature(decomp_fn)

        # Create a string wrapping the function from the signature
        # example output:
        # def wrapped_decomp(x: torch.Tensor, y: int, z: int):
        #   return decomp_fn(x, y, z)
        # Thanks copilot!
        def get_function_def(sig):
            param_def = [f"{param_str}" for param_str in sig.parameters.values()]
            param_use = [f"{param_str}" for param_str in sig.parameters.keys()]

            return f"def wrapped_decomp({', '.join(param_def)}):\n  return decomp_fn({', '.join(param_use)})\n"

        f_str = get_function_def(sig)
        graph = torch.jit.CompilationUnit(f_str).wrapped_decomp.graph
    else:
        graph = torch.jit.script(decomp_fn).graph
    torch.jit._register_decomposition(decomp, graph)


# The only decompositions here are temporary or hacks for the purposes of jvp

# TODO: do these also belong here?
@maybe_register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    return torch.sum(torch.diag(self))


@maybe_register_decomposition(aten.log_sigmoid_forward.default)
def log_sigmoid_forward(self: Tensor) -> Tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


_register_jit_decomposition_for_jvp(torch.ops.aten.trace.default, use_python=True)
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss2d_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten._log_softmax_backward_data.default)
_register_jit_decomposition_for_jvp(torch.ops.aten._softmax_backward_data.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.log_sigmoid_forward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.native_layer_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.native_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.cudnn_batch_norm_backward.default)
