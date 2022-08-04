from typing import Callable

from torch._prims.context import TorchRefsMode
from torch._prims.nvfuser_executor import nvfuser_execute, nvfuser_execute_partitioned

from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx


def execute(gm: GraphModule, *args, executor: str = "aten"):
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        return gm.forward(*args)
    elif executor == "nvfuser":
        return nvfuser_execute_partitioned(gm, *args)
    elif executor == "strictly_nvfuser":
        return nvfuser_execute(gm, *args)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: aten, nvfuser.".format(
        executor
    )
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
    result = traced_foo(a, b, executor='nvfuser')

    Executor may be either 'aten' or 'nvfuser'.
    """

    def _traced(*args, executor="aten", **kwargs):
        # TODO: caching
        nargs = len(args)
        fn_kwargs = kwargs
        flat_fn_kwargs = list(fn_kwargs.values())
        all_args = list(args) + flat_fn_kwargs

        def wrapped(args):
            fn_args = args[:nargs]
            kwargs_keys = list(fn_kwargs.keys())
            kwargs = dict(zip(kwargs_keys, args[nargs:]))
            return fn(*fn_args, **kwargs)

        with TorchRefsMode():
            gm = make_fx(wrapped)(all_args)
        return execute(gm, all_args, executor=executor)

    return _traced
