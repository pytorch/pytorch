# mypy: allow-untyped-defs
import dataclasses
from typing import Callable, Tuple, Optional, Protocol

from .. import _ops, autograd


class InfoProtocol(Protocol):
    _vmap_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]

@dataclasses.dataclass
class Info:
    _vmap_fn: Optional[Callable]
    _setup_context_fn: Optional[Callable]

def make_vmap_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    name: str = f"GeneratedVmapFor_{op._namespace}_{op._opname}_{op._overloadname}"

    def vmap(ctx, in_dims: Tuple[Optional[int]], *args, **kwargs):
        if info._vmap_fn:
            result = info._vmap_fn(ctx, in_dims, *args, **kwargs)

            return result

        raise RuntimeError(
            f"Trying to vmap through {op} but no vmap "
            f"formula was registered. "
            f"Please use register_vmap to add one."
        )

    Generated = type(
        name,
        (autograd.Function,),
        {
            "setup_context": staticmethod(info._setup_context_fn),
            "vmap": staticmethod(vmap),
        },
    )

    def vmap_impl(keyset, *args, **kwargs):
        return Generated.apply(*args, **kwargs)  # type: ignore[attr-defined]


    return vmap_impl
