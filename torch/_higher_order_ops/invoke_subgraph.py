# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs


from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_reenter_make_fx,
    clone_outputs_aliasing_inputs,
    get_dummy_aot_autograd_config,
    prepare_fw_with_masks,
)
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
from torch.fx.graph_module import GraphModule

from .prim_hop_base import FunctionWithNoFreeVars, PrimHOPBase


invoke_subgraph_counter = 0


class InvokeSubgraphHOP(PrimHOPBase):
    def __init__(self) -> None:
        super().__init__("invoke_subgraph")

    def __call__(
        self,
        subgraph: Union[GraphModule, Callable, FunctionWithNoFreeVars],
        operands: Union[
            List[Union[torch.Tensor, torch.SymInt]],
            Tuple[Union[torch.Tensor, torch.SymInt]],
        ],
        *,
        # identifier is setup by upper part of the stack. This helps us in
        # identifying two invoke_subgraph calls have same subgraph.
        identifier: Optional[str],
    ):
        assert identifier is None or isinstance(
            identifier, str
        ), "identifier must be a None or a string"

        assert isinstance(
            operands, (list, tuple)
        ), f"invoke_subgraph operands must be a list or tuple of tensors and SymInts {operands}"
        assert all(
            isinstance(o, (torch.Tensor, torch.SymInt)) for o in operands
        ), f"invoke_subgraph operands must be a list of tensors and SymInts {operands}"

        return super().__call__(subgraph, operands, identifier=identifier)

    def _forward_kwargs(self, *_, **kwargs):
        return {"identifier": f"___forward_{kwargs['identifier']}"}

    def _backward_kwargs(self, *_, **kwargs):
        # TODO: I've wiped out the Autograd cache. We should
        # figure out what we want to do with it...
        # BTW the following is probably wrong. Each backward might actually
        # be different depending on the grad_output, so they probably need a unique
        # identifier?
        return {"identifier": f"___backward_{kwargs['identifier']}"}

    def _trace_subgraph(self, proxy_mode, subgraph, operands, *, identifier):
        # Check if we have already traced the subgraph.
        graph = None
        invoke_subgraph_cache = get_invoke_subgraph_cache()
        if invoke_subgraph_cache:
            graph = invoke_subgraph_cache.get_proxy_dispatch_entry(identifier)
        if graph is None:
            graph = super()._trace_subgraph(proxy_mode, subgraph, operands)
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_proxy_dispatch_entry(identifier, graph)
        return graph

    def _dynamo_call_function_hook(self, tx, body_gmod, kwargs):
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.higher_order_ops import (
            add_subgraph,
            hash_graph_and_inputs,
        )

        fake_inputs = [
            node.meta["example_value"]
            for node in body_gmod.graph.nodes
            if node.op == "placeholder"
        ]

        key = hash_graph_and_inputs(tx, body_gmod, fake_inputs)
        invoke_subgraph_cache = (
            tx.output.tracing_context.hop_dispatch_set_cache.get_cache(invoke_subgraph)
        )
        if invoke_subgraph_cache:
            if identifier := invoke_subgraph_cache.get_dynamo_identifier(key):
                kwargs["identifier"] = ConstantVariable.create(identifier)
                return identifier, kwargs

        body_name = add_subgraph(tx, "invoke_subgraph", body_gmod)
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_dynamo_identifier(key, body_name)

        kwargs["identifier"] = ConstantVariable.create(body_name)

        return body_name, kwargs


invoke_subgraph = InvokeSubgraphHOP()


def get_invoke_subgraph_cache():
    cache = None
    if tracing_ctx := torch._guards.TracingContext.try_get():
        cache = tracing_ctx.hop_dispatch_set_cache.get_cache(invoke_subgraph)
    return cache
