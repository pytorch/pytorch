# mypy: allow-untyped-defs
from typing import Callable, Tuple, Optional

from .. import _ops, autograd
from .._functorch.autograd_function import custom_function_call_vmap

# def make_vmap_impl(op: _ops.OpOverload, vmap_fn: Callable) -> Callable:
#     # name: str = f"GeneratedVmapFor_{op._namespace}_{op._opname}_{op._overloadname}"

#     def vmap_impl(keyset, *args, **kwargs):
#         # return Generated.apply(*args, **kwargs)  # type: ignore[attr-defined]

#         return custom_function_call_vmap(interpreter, vmap_fn, *args, **kwargs)


#     return vmap_impl
