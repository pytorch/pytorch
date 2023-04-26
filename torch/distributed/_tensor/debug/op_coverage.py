import torch.nn as nn
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table


inductor_decomps = select_decomp_table()


graphs = []


def fwd_bwd_compiler(fx_g, _):
    graphs.append(fx_g)
    return fx_g


def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Convinient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    compiled_mod = aot_module(
        model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps
    )
    output = compiled_mod(*args, **kwargs)

    if output.ndim != 0:
        # if output is not a scalar tensor, by default sum it in order to
        # run backward
        output = output.sum()

    output.backward()

    # one fwd, one bwd graph
    assert len(graphs) == 2
    return graphs
