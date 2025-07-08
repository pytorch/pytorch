# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import Any, Callable

from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class ForeachMap(BaseHOP):
    def __init__(self):
        super().__init__("foreach_map")

    def __call__(self, fn, *operands, **kwargs):  # type: ignore[override]
        fn = FunctionWithNoFreeVars(fn)
        return super().__call__(fn, *operands, **kwargs)


_foreach_map = ForeachMap()


def foreach_map(op: Callable, *operands: Any, **kwargs: dict[str, Any]):
    from torch._dynamo.polyfills import foreach_map_fn

    return _foreach_map(foreach_map_fn, op, *operands, **kwargs)
