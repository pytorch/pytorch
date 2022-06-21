from typing import Callable

import torch

from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch._prims.utils import getnvFuserDtype, Number
from torch._prims.context import TorchRefsMode
import torch.overrides
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

if torch.cuda.is_available():
    from torch._C._nvfuser import Fusion, FusionDefinition  # type: ignore[import]


def execute(gm: GraphModule, *args, executor: str = "aten", **kwargs):
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        return gm.forward(*args)
    elif executor == "nvfuser":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Attempting to use nvFuser trace executor but CUDA is not available!"
            )

        # PROTOTYPE nvfuser executor
        # Everything in the graph must support nvfuser

        fusion = Fusion()
        with FusionDefinition(fusion) as fd:

            def _to_nvfuser_constant(arg):
                if isinstance(arg, Number):
                    return fd.define_constant(arg)
                else:
                    return arg

            class FusionInterpreter(torch.fx.Interpreter):
                def call_function(self, target, args, kwargs):
                    args = tuple(map(_to_nvfuser_constant, args))
                    target = target.impl_nvfuser
                    args = (fd,) + args
                    return target(*args)

            def to_nv(arg):
                if isinstance(arg, torch.Tensor):
                    x = fd.define_tensor(
                        arg.size(), arg.stride(), getnvFuserDtype(arg.dtype)
                    )
                    fd.add_input(x)
                    return x
                else:
                    return arg

            # Transforms graph to call nvfuser lowerings
            # Note, this doesn't handle nested structures in the args, TODO: add tree_flatten
            nv_args = tree_map(to_nv, args)
            nv_kwargs = tree_map(to_nv, kwargs)
            out = FusionInterpreter(gm).run(*nv_args, **nv_kwargs)
            flat_out, unflatten_spec = tree_flatten(out)
            for o in flat_out:
                fd.add_output(o)
            assert len(args) == 1
            args = args[0]  # we are passing a packed list of args
            return tree_unflatten(
                fusion.execute(
                    tuple(arg for arg in args if isinstance(arg, torch.Tensor))
                ),
                unflatten_spec,
            )

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

        with TorchRefsMode.push():
            gm = make_fx(wrapped)(all_args)
        return execute(gm, all_args, executor=executor)

    return _traced
