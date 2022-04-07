

import torch
from torch import Tensor
aten = torch.ops.aten
from typing import Optional, List, Dict, Set
import inspect
from torch.fx.operator_schemas import get_signature_for_torch_op
import warnings

decomposition_table: Dict[str, torch.jit.ScriptFunction] = {}
function_name_set: Set[str] = set()

def check_decomposition_has_type_annotations(f):

    inspect_empty = inspect._empty  # type: ignore[attr-defined]
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        assert param.annotation != inspect_empty, \
            "No signature on param {name} for function {func}".format(name=param.name, func=f.name)

    assert sig.return_annotation != inspect_empty, "No return annotation for function {func}".format(func=f.name)

def signatures_match(decomposition_sig, torch_op_sig):
    decomp_params = decomposition_sig.parameters
    op_params = torch_op_sig.parameters

    if len(decomp_params) != len(op_params):
        return False


    for decomp_param, op_param in zip(decomp_params.values(), op_params.values()):
        # can't check full equality yet because not all fields are correcly deduced
        # in the torch_op_sig - like default value
        # can't check 'kind' bc
        # kwarg-only values with defaults not yet supported in TS
        inspect_empty = inspect._empty  # type: ignore[attr-defined]
        for field in ['name', 'annotation']:
            if field == 'name' and decomp_param.name == "self":
                warnings.warn("PyTorch uses 'input' instead of 'self' on public api")

            if getattr(decomp_param, field) != getattr(op_param, field):
                return False

        decomp_default = decomp_param.default
        op_default = op_param.default
        # default value not always correctly inferred as being present on torch schema,
        # but if specified on both they should be equal
        if decomp_default != inspect_empty and op_default != inspect_empty:
            if decomp_default != op_default:
                return False

    return decomposition_sig.return_annotation == torch_op_sig.return_annotation

def register_decomposition(aten_op, registry=None):
    def decomposition_decorator(f):
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        check_decomposition_has_type_annotations(f)

        torch_op_sigs, torch_op_schemas = get_signature_for_torch_op(aten_op, return_schemas=True)
        decomposition_sig = inspect.signature(f)

        found_index = None
        for i, torch_op_sig in enumerate(torch_op_sigs):
            if signatures_match(decomposition_sig, torch_op_sig):
                found_index = i
                break

        assert found_index is not None, "Could not find matching signature: " + str(f)

        # Need unique name for jit function serialization
        assert f.__name__ not in function_name_set, "Duplicated function name {}".format(f.__name__)
        function_name_set.add(f.__name__)

        scripted_func = torch.jit.script(f)
        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        registry[str(torch_op_schemas[found_index])] = scripted_func
        return f

    return decomposition_decorator

# TODO: replace torch.sigmoid -> aten.sigmoid

@register_decomposition(aten.var)
def var_decomposition(input: Tensor, dim: Optional[List[int]] = None, correction: Optional[int] = None,
                      keepdim: bool = False) -> Tensor:
    if dim is None:
        dim_i: List[int] = []
        dim = dim_i

    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        n = input.numel()
    else:
        n = 1
        for dim_i in dim:  # type: ignore[assignment]
            n *= input.shape[dim_i]  # type: ignore[call-overload]

    mean = aten.mean(input, dim, True)
    sub = input - mean
    sq = sub * sub
    sum = aten.sum(sq, dim, keepdim)

    if correction is not None:
        n = n - correction

    return sum / n

@register_decomposition(aten.var)
def var(input: Tensor, unbiased: bool = True) -> Tensor:
    return var_decomposition(input, correction=(1 if unbiased else 0))
