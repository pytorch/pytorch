from typing import Any, TYPE_CHECKING

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree


class RunConstGraph(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("run_const_graph")

    def __call__(self, graph: torch.fx.GraphModule, args: tuple[object, ...]) -> object:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(graph, args)


run_const_graph = RunConstGraph()


@run_const_graph.py_impl(ProxyTorchDispatchMode)
def run_const_graph_dispatch_mode(
    mode: ProxyTorchDispatchMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    const_gm, weights = graph, args
    p_args = pytree.tree_map(mode.tracer.unwrap_proxy, (graph, args))  # type: ignore[union-attr]
    assert isinstance(const_gm, torch.fx.GraphModule)
    assert not hasattr(mode.tracer.root, "_const_graph")  # type: ignore[union-attr]
    mode.tracer.root.register_module("_const_graph", const_gm)  # type: ignore[union-attr]

    proxy = mode.tracer.create_proxy("call_function", run_const_graph, p_args, {})

    out = const_gm(*weights)
    return track_tensor_tree(out, proxy, constant=None, tracer=mode.tracer)


@run_const_graph.py_functionalize_impl
def run_const_graph_functional(
    ctx: "BaseFunctionalizeAPI", graph: torch.fx.GraphModule, args: tuple[Any, ...]
) -> Any:
    unwrapped_args = ctx.unwrap_tensors(args)

    with ctx.redispatch_to_next():
        out = run_const_graph(graph, unwrapped_args)
        return ctx.wrap_tensors(out)  # type: ignore[arg-type]


run_const_graph.py_autograd_impl(
    autograd_not_implemented(run_const_graph, deferred_error=True)
)


@run_const_graph.py_impl(FakeTensorMode)
def run_const_graph_fake_tensor_mode(
    mode: FakeTensorMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    assert isinstance(graph, torch.fx.GraphModule)
    with mode:
        return graph(*args)


@run_const_graph.py_impl(DispatchKey.CPU)
def run_const_graph_cpu(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    assert isinstance(graph, torch.fx.GraphModule)
    return graph(*args)
