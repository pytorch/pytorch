"""Utilities for lowering subgraphs used by higher order operators"""

import functools
import operator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch.utils._ordered_set import OrderedSet

from . import ir
from .exc import SubgraphLoweringException
from .ops_handler import SimpleCSEHandler
from .virtualized import ops, V, WrapperHandler


T = TypeVar("T")
_P = ParamSpec("_P")

OpOverload = torch._ops.OpOverload
LoweringDict = Dict[Union[OpOverload, str], Callable[..., Any]]
TargetType = Union[Callable[..., Any], str]


class PointwiseSubgraphLowering(torch.fx.Interpreter):
    """
    Lowers a pointwise subgraph to a single set of buffers with a separate
    lowering object. Errors if buffers are created unexpectedly
    """

    graph_outputs: Optional[List[ir.IRNode]]
    root_graph: torch._inductor.graph.GraphLowering
    _current_op: Optional[TargetType]
    # For backwards of buffer_grads with scatters we allow mutations
    allowed_mutations: Optional[OrderedSet[OpOverload]]
    additional_lowerings: Optional[LoweringDict]
    buffers: List[ir.Buffer]
    mutated_buffers: OrderedSet[str]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: torch._inductor.graph.GraphLowering,
        allowed_mutations: Optional[OrderedSet[OpOverload]] = None,
        additional_lowerings: Optional[LoweringDict] = None,
    ) -> None:
        super().__init__(gm)
        self.graph_outputs = None
        self.root_graph = root_graph_lowering
        self.allowed_mutations = allowed_mutations
        self.additional_lowerings = additional_lowerings
        self._current_op = None

        # Used to track buffers created during lowering
        self.mutated_buffers = OrderedSet()
        self.buffers = []

    @contextmanager
    def _op_context(self, op: TargetType) -> Generator[None, None, None]:
        """Set which op is being processed in call function to know if we can mutate buffers"""
        previous = self._current_op
        self._current_op = op
        try:
            yield
        finally:
            self._current_op = previous

    def _approved_mutator(self) -> bool:
        return (
            self.allowed_mutations is not None
            and self._current_op in self.allowed_mutations
        )

    def mark_buffer_mutated(self, name: str) -> None:
        if self._approved_mutator():
            self.mutated_buffers.add(name)
        else:
            raise SubgraphLoweringException(
                f"Buffer mutation detected during lowering of {self._current_op}. "
                "Buffer mutations are only allowed in approved mutation ops. "
                "This is an error in the lowering of the subgraph, please file a bug report."
            )

    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str:
        if self._approved_mutator():
            name = self.qualify_name(f"buf{len(self.buffers)}")
            self.buffers.append(buffer)
            return name
        else:
            raise SubgraphLoweringException(
                "Buffers cannot be created while lowering a pointwise subgraph. "
                "This could be for a good reason (e.g. you're calling an op we can't codegen as a pointwise op), "
                "but it could also be a bug. Please file a bug report if you think this should be supportable."
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.root_graph, name)

    def call_function(
        self,
        target: TargetType,
        args: Any,
        kwargs: Dict[str, Any],
    ) -> Any:
        from .lowering import lowerings

        with self._op_context(target):
            if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
                return super().call_function(target, args, kwargs)

            # These takes precedence over the main lowerings
            if self.additional_lowerings is not None:
                if target in self.additional_lowerings:
                    assert isinstance(target, OpOverload)
                    return self.additional_lowerings[target](*args, **kwargs)

            if target not in lowerings:
                raise SubgraphLoweringException(
                    f"{target} not supported in subgraph, (missing lowering)"
                )

            return lowerings[target](*args, **kwargs)

    def output(self, target: str, args: Tuple[Any], kwargs: Dict[str, Any]) -> None:  # type: ignore[override]
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
