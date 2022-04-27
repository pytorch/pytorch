from torch.fx import GraphModule
from torch._prims.context import PrimContext


def execute(ctx: PrimContext, *args, **kwargs):
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """

    gm = GraphModule({}, ctx.graph)
    return gm.forward(*args, **kwargs)
