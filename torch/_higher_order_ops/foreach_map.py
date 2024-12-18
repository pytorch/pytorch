# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import Any, Callable, Dict, Tuple

from torch._higher_order_ops.prim_hop_base import FunctionWithNoFreeVars, PrimHOPBase


class ForeachMap(PrimHOPBase):
    def __init__(self):
        super().__init__("foreach_map")

    def __call__(self, fn, operands, *unused, **kwargs):  # type: ignore[override]
        fn = FunctionWithNoFreeVars(fn)
        return super().__call__(fn, operands, **kwargs)


_foreach_map = ForeachMap()


def foreach_map(
    op: Callable, operands: Any, *unused: Tuple[Any], **kwargs: Dict[str, Any]
):
    from torch._dynamo.polyfills import foreach_map_fn

    args = (op,) + operands
    return _foreach_map(foreach_map_fn, args, **kwargs)
