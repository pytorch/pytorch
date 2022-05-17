from typing import Callable

import torch

from torch.fx import GraphModule
from torch._prims.utils import TensorMeta, getnvFuserDtype
from torch._prims.context import PrimContext
import torch.overrides

if torch.cuda.is_available():
    from torch._C._nvfuser import Fusion, FusionDefinition  # type: ignore[import]


def execute(ctx: PrimContext, *args, executor: str = "aten", **kwargs):
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    if executor == "aten":
        gm = GraphModule({}, ctx.graph)
        return gm.forward(*args, **kwargs)
    elif executor == "nvfuser":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Attempting to use nvFuser trace executor but CUDA is not available!"
            )

        # PROTOTYPE nvfuser executor
        # Only accepts tensor inputs and single tensor outputs
        # Does not handle kwargs
        # Does not support reusing the same ctx to execute!
        assert len(kwargs) == 0
        # TODO: make this a proper trace -> trace transform that
        # doesn't mutate the context
        graph_fd = ctx.graph.placeholder("fd")
        ctx.graph._root.append(graph_fd)

        fusion = Fusion()
        with FusionDefinition(fusion) as fd:
            # Transforms graph to call nvfuser lowerings
            nv_args = [fd]
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    x = fd.define_tensor(
                        arg.size(), arg.stride(), getnvFuserDtype(arg.dtype)
                    )
                    fd.add_input(x)
                    nv_args.append(x)
                else:
                    nv_args.append(x)

            for x in ctx.graph.nodes:
                if x.op == "call_function":
                    x.target = x.target.impl_nvfuser
                    x.args = (graph_fd,) + x.args

            gm = GraphModule({}, ctx.graph)
            out = gm.forward(*nv_args)
            fd.add_output(out)

            return fusion.execute(
                tuple(arg for arg in args if isinstance(arg, torch.Tensor))
            )[0]

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
    be tensors and the function must return a single tensor. In the
    near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return torch.add(a, b)

    traced_foo = make_traced(foo)

    a = torch.randn((1, 2, 3, 4, 5), device='cuda')
    b = torch.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='nvfuser')

    Executor may be either 'aten' or 'nvfuser'.
    """

    def _traced(*args, executor="aten"):
        ctx: PrimContext
        with torch.overrides.push_torch_function_mode(PrimContext) as ctx:  # type: ignore[attr-defined, assignment]
            placeholders = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    placeholders.append(ctx.placeholder(TensorMeta(arg)))
                else:
                    placeholders.append(ctx.placeholder(arg))

            result = fn(*placeholders)
            ctx.output(result)
        return execute(ctx, *args, executor=executor)

    return _traced
