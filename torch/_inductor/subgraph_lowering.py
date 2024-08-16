"""Utilities for lowering subgraphs used by higher order operators

"""

import functools
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from typing_extensions import ParamSpec

import torch

from . import ir
from .exc import SubgraphLoweringException
from .ops_handler import SimpleCSEHandler
from .sizevars import SizeVarAllocator
from .virtualized import ops, V, WrapperHandler


T = TypeVar("T")
_P = ParamSpec("_P")


class PointwiseSubgraphLowering(torch.fx.Interpreter):
    graph_outputs: Optional[List[ir.IRNode]]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: "torch._inductor.graph.GraphLowering",
    ) -> None:
        super().__init__(gm)
        self.graph_outputs = None
        self.root_graph = root_graph_lowering

    @property
    def sizevars(self) -> SizeVarAllocator:
        return self.root_graph.sizevars

    def mark_buffer_mutated(self, name: str) -> None:
        raise SubgraphLoweringException("Mutations are not supported in this context")

    def register_buffer(self, buffer: ir.Buffer) -> str:
        raise SubgraphLoweringException(
            "Buffer creation is not supported in this context"
        )

    def call_function(
        self,
        target: Callable[[Any], Any],
        args: Any,
        kwargs: Dict[str, Any],
    ) -> Any:
        from .lowering import lowerings

        if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
            return super().call_function(target, args, kwargs)

        assert isinstance(target, torch._ops.OpOverload)

        if target not in lowerings:
            raise SubgraphLoweringException(
                f"{target} not supported in subgraph, (missing lowering)"
            )

        if torch.Tag.pointwise not in target.tags:
            raise SubgraphLoweringException(
                f"Only pointwise operators are supported in this context, but got {target}"
            )

        return lowerings[target](*args, **kwargs)

    def output(self, target: str, args: Tuple[Any], kwargs: Dict[str, Any]) -> None:
        assert len(args) == 1
        self.graph_outputs = args[0]


@dataclass
class InputDescriptor:
    dtype: torch.dtype
    device: torch.device


class TracingOpsHandler(WrapperHandler[T]):
    def __init__(self, tracer: torch.fx.Tracer, num_inputs: int) -> None:
        parent = tracer.create_proxy("placeholder", "ops", (), {})
        super().__init__(parent)
        self.tracer = tracer

        self.placeholders = [
            self.tracer.create_proxy("placeholder", f"input{i}", (), {})
            for i in range(num_inputs)
        ]

    def placeholder(self, idx: int) -> torch.fx.Proxy:
        return self.placeholders[idx]

    def output(self, *args: Tuple[object]) -> torch.fx.Node:
        return self.tracer.create_node(
            "output", "output", (tuple(self.tracer.create_arg(a) for a in args),), {}
        )


def lower_pointwise_subgraph(
    subgraph: ir.Subgraph, inputs: List[InputDescriptor]
) -> Callable[_P, Any]:
    # Lower subgraph to ir.Pointwise nodes
    def fake_inner_fn(
        loop_idx: int, input_idx: int
    ) -> Union[ir.Expr, ir.TensorBox, None]:
        return ops.placeholder(input_idx)

    graph_inputs = [
        ir.Pointwise.create(
            device=desc.device,
            dtype=desc.dtype,
            inner_fn=functools.partial(fake_inner_fn, input_idx=i),
            ranges=[],
        )
        for i, desc in enumerate(inputs)
    ]
    gm = subgraph.graph_module
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*graph_inputs)

    # Combine multiple pointwise computations into a single graph module
    # Do this by tracing through each individually and doing CSE
    tracer = torch.fx.Tracer()
    tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
    trace_ops = SimpleCSEHandler(TracingOpsHandler(tracer, len(inputs)))
    assert pw_subgraph.graph_outputs is not None

    with V.set_ops_handler(trace_ops):
        output_irs = []

        for out_var in pw_subgraph.graph_outputs:
            assert isinstance(out_var, ir.TensorBox), type(out_var)
            assert out_var.get_size() == []
            assert isinstance(out_var.data, ir.StorageBox)
            assert isinstance(out_var.data.data, ir.Pointwise)

            idx = ()
            ir_out = out_var.data.data.inner_fn(idx)

            output_irs.append(ir_out)

        ops.output(*output_irs)

    lowered_gm = torch.fx.GraphModule({}, tracer.graph)

    def inner_fn(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        return lowered_gm(V.get_ops_handler(), *args, **kwargs)

    return inner_fn
