import torch
from torch.jit._recursive import wrap_cpp_module

def remove_redundant_aliases(scripted_module: torch.nn.Module):
    """
    Running torch.jit.trace on a model with DBR quantization introduces
    extra alias ops, because we use `torch.Tensor.as_subclass` and tracing
    through this results in an `aten::alias` function call in TorchScript.
    This pass removes these alias calls when it is safe to do so.
    """
    module_c = scripted_module._c
    module_c = \
        torch._C._jit_pass_dbr_quant_remove_redundant_aliases(module_c)  # type: ignore[attr-defined]
    scripted_module = wrap_cpp_module(module_c)
    return scripted_module
