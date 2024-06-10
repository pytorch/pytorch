# mypy: allow-untyped-defs
from typing import Callable, Optional

from torch._prims.context import TorchRefsMode

from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx


def execute(
    gm: GraphModule,
    *args,
    executor: str = "aten",
    executor_parameters: Optional[dict] = None,
):
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        return gm.forward(*args)

    msg = f"Received unexpected value for 'executor': {executor}. Allowed values are: aten."
    raise ValueError(msg)


def make_traced(fn: Callable):
    """
    Returns a function that, when called, will
    trace its torch operations to prims and then
    execute those prims on the requested trace executor
    (possibly lowering them to that trace executor first).

    Only supports the torch operations defined in _torch_to_reference_map
    in context.py and operations with positional args. All args must
    be tensors.
    In the near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return torch.add(a, b)

    traced_foo = make_traced(foo)

    a = torch.randn((1, 2, 3, 4, 5), device='cuda')
    b = torch.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='aten')
    """

    def _traced(*args, executor="aten", **kwargs):
        # TODO: caching
        wrapped, all_args = wrapper_and_args_for_make_fx(fn, args, kwargs)

        with TorchRefsMode():
            gm = make_fx(wrapped)(all_args)
        return execute(gm, all_args, executor=executor)

    return _traced
