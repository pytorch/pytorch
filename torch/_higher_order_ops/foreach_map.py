# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import Any, Callable

from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class ForeachMap(BaseHOP):
    def __init__(self):
        super().__init__("foreach_map")

    def __call__(self, fn, *operands, _debug_assert_fused=False, **kwargs):  # type: ignore[override]
        fn = FunctionWithNoFreeVars(fn)
        return super().__call__(
            fn, *operands, _debug_assert_fused=_debug_assert_fused, **kwargs
        )


_foreach_map = ForeachMap()


def foreach_map(
    op: Callable,
    *operands: Any,
    _debug_assert_fused: bool = False,
    **kwargs: dict[str, Any],
):
    """
    We do not have backward compatibility guarantees on fusion behavior
    and it may change without notice.
    """
    from torch._dynamo.polyfills import foreach_map_fn

    return _foreach_map(
        foreach_map_fn, op, *operands, _debug_assert_fused=_debug_assert_fused, **kwargs
    )
