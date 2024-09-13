# mypy: allow-untyped-defs
import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree


class RunConstGraph(HigherOrderOperator):
    def __init__(self):
        super().__init__("run_const_graph")


run_const_graph = RunConstGraph()


@run_const_graph.py_impl(ProxyTorchDispatchMode)
def run_const_graph_dispatch_mode(mode, *args):
    const_gm, weights = args
    p_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    assert isinstance(const_gm, torch.fx.GraphModule)
    assert not hasattr(mode.tracer.root, "_const_graph")
    mode.tracer.root.register_module("_const_graph", const_gm)

    proxy = mode.tracer.create_proxy("call_function", run_const_graph, p_args, {})

    out = const_gm(*weights)
    return track_tensor_tree(out, proxy, constant=None, tracer=mode.tracer)


@run_const_graph.py_functionalize_impl
def run_const_graph_functional(ctx, *args):
    unwrapped_args = ctx.unwrap_tensors(args)

    with ctx.redispatch_to_next():
        out = run_const_graph(*unwrapped_args)
        return ctx.wrap_tensors(out)


run_const_graph.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(run_const_graph, deferred_error=True)
)


@run_const_graph.py_impl(FakeTensorMode)
def run_const_graph_fake_tensor_mode(mode, graph, args):
    assert isinstance(graph, torch.fx.GraphModule)
    with mode:
        return graph(*args)


@run_const_graph.py_impl(DispatchKey.CPU)
def run_const_graph_cpu(graph, args):
    assert isinstance(graph, torch.fx.GraphModule)
    return graph(*args)
