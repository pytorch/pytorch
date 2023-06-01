from dataclasses import dataclass
import copy
import math
import operator
from collections import OrderedDict
from functools import partial
from typing import Dict, List, NamedTuple

import sympy

import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import SymInt
from torch._export.pass_base import ExportPassBase, ProxyValue, PassResult
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._subclasses.fake_tensor import FakeTensor


__all__ = ["_AddRuntimeAssertionsForConstraintsPass", "InputDim", "RangeConstraint"]


class InputDim(NamedTuple):
    input_name: str
    dim: int


@dataclass
class RangeConstraint:
    min_val: int
    max_val: int


def _convert_to_int(val):
    # Convert simple sympy Integers into concrete int
    if val == sympy.oo:
        return math.inf
    if val == -sympy.oo:
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError(
        "Export constraints cannot be non-integer expressions"
    )


def _convert_range_to_int(range: RangeConstraint):
    assert isinstance(range, RangeConstraint)
    min_val = _convert_to_int(range.min_val)
    max_val = _convert_to_int(range.max_val)
    return min_val, max_val


class _AddRuntimeAssertionsForConstraintsPass(ExportPassBase):
    def __init__(
        self,
        range_constraints: Dict[sympy.Symbol, RangeConstraint],
        equality_constraints: Dict[InputDim, List[InputDim]],
    ):
        super().__init__()
        self.range_constraints: Dict[sympy.Symbol, RangeConstraint] = range_constraints
        self.equality_constraints: Dict[InputDim, List[InputDim]] = equality_constraints

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph_module = copy.deepcopy(graph_module)
        graph = graph_module.graph

        # Add runtime asserts for input shape constraints. We do this after all
        # placeholder nodes so that we can handle both (unary) predicates and
        # (binary) relations.
        placeholder_nodes: Dict[str, torch.fx.Node] = OrderedDict()
        for node in graph.nodes:
            if node.op != "placeholder":
                continue

            placeholder_nodes[node.name] = node
            if (
                "val" not in node.meta or
                not isinstance(node.meta["val"], FakeTensor)
            ):
                continue

            fake_tensor_shape = node.meta["val"].shape
            for dim, shape in enumerate(fake_tensor_shape):
                if isinstance(shape, SymInt):
                    # If the shape is dynamic, add range assertions
                    symbol = shape.node._expr
                    assert symbol in self.range_constraints

                    with graph.inserting_after(node):
                        self._insert_range_assert_inplace(
                            graph, node, dim, self.range_constraints[symbol]
                        )
                else:
                    # If no dynamism is specified, we assume all dimensions #
                    # are specialized
                    assert isinstance(shape, int)
                    with graph.inserting_after(node):
                        self._insert_specialized_shape_assert_inplace(
                            graph, node, dim, shape,
                        )

        # Add runtime assertions on equality constraints on the inputs
        with graph.inserting_after(
            list(placeholder_nodes.values())[-1]
        ):
            self._insert_equality_assert_inplace(graph, placeholder_nodes)

        # Add runtime asserts for inline constraints
        return super().call(graph_module)

    def _insert_specialized_shape_assert_inplace(
        self, graph: torch.fx.Graph, node: torch.fx.Node, dim: int, shape: int,
    ):
        assert_msg = f"Input {node.name}.shape[{dim}] is specialized at {shape}"
        dim_node = graph.call_function(torch.ops.aten.sym_size, (node, dim))
        with graph.inserting_after(dim_node):
            eq_node = graph.call_function(operator.eq, (dim_node, shape))
        with graph.inserting_after(eq_node):
            tensor_eq_node = graph.call_function(torch.ops.aten.scalar_tensor.default, (eq_node,))
        with graph.inserting_after(tensor_eq_node):
            _ = graph.call_function(torch.ops.aten._assert_async.msg, (tensor_eq_node, assert_msg))

    def _insert_range_assert_inplace(
        self, graph: torch.fx.Graph, node: torch.fx.Node, dim: int, range: RangeConstraint
    ):
        """
        Add runtime asserts for user-specified range constraints for
        each placeholder's dynamic dimension.
        """

        dim_node = graph.call_function(torch.ops.aten.sym_size, (node, dim))
        min_val, max_val = _convert_range_to_int(range)
        assert_msg = (
            f"Input {node.name}.shape[{dim}] is "
            f"outside of specified dynamic range [{min_val}, {max_val}]"
        )
        # TODO (tmanlaibaatar) we are making an assumption that graph generated for
        # input dim N >=2 generalizes to N < 2. Ideally we should check that:
        # 1. if we can generalize to N < 2, not add any assertion saying N >= 2
        # 2. If we can't generalize to N < 2, add an assertion saying N >= 2
        # Above can be achieved via a seperate pass.
        with graph.inserting_after(dim_node):
            if min_val > 2:
                self._insert_assert_async_inplace(
                    graph, operator.ge, (dim_node, min_val), assert_msg,
                )

            if max_val < math.inf:
                self._insert_assert_async_inplace(
                    graph, operator.le, (dim_node, max_val), assert_msg,
                )

    def _insert_equality_assert_inplace(
        self,
        graph: torch.fx.Graph,
        placeholder_nodes: Dict[str, torch.fx.Node],
    ):
        for input_dim, equalities in self.equality_constraints.items():
            node = placeholder_nodes[input_dim.input_name]
            dim_node = graph.call_function(
                torch.ops.aten.sym_size, (node, input_dim.dim),
            )
            with graph.inserting_after(dim_node):
                for other_input_dim in equalities:
                    assert_msg = (
                        f"Input {input_dim.input_name}.shape[{input_dim.dim}] is "
                        f"not equal to input {other_input_dim.input_name}.shape[{other_input_dim.dim}]"
                    )
                    other_node = placeholder_nodes[other_input_dim.input_name]
                    other_dim_node = graph.call_function(
                        torch.ops.aten.sym_size, (other_node, other_input_dim.dim),
                    )

                    with graph.inserting_after(other_dim_node):
                        self._insert_assert_async_inplace(
                            graph,
                            operator.eq,
                            (dim_node, other_dim_node),
                            assert_msg
                        )

    def _insert_assert_async_inplace(self, graph, operator, args, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called before we run the interpreter-based pass and does an inplace
        insertion.
        """
        cmp_node = graph.call_function(operator, args)
        with graph.inserting_after(cmp_node):
            cmp_tensor_node = graph.call_function(
                torch.ops.aten.scalar_tensor.default, (cmp_node,)
            )
        with graph.inserting_after(cmp_tensor_node):
            _ = graph.call_function(
                torch.ops.aten._assert_async.msg, (cmp_tensor_node, assert_msg)
            )

    def call_operator(self, op, args, kwargs, meta) -> ProxyValue:
        ret = super().call_operator(op, args, kwargs, meta)
        if "val" not in meta:
            return ret

        val = meta["val"]

        # In general, we may have to deal the case such as: ret[1].shape[0].
        # We need first find out what symbols require assertion, then we need to follow the path
        # from ret to the symbol, construct the proxies along the way and construct the messages
        # piece-wise at the same time.
        #
        # We use post-order traversal to collect all the proxies callbacks needed, construct
        # the error message callbacks, and at the top-level traversal tree we execute all the callbacks.
        # We need the callbacks because, in order to call the function to create a proxy for shape[0], we
        # need the proxy for shape, which further requries the proxy for ret[1], etc.
        def add_assertions(val):
            call_backs = []
            messages = []
            if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                symbol = val.node._expr
                if symbol.name.startswith("i"):
                    # We only care about unbacked symints for these inline
                    # constraints, which are prefixed with 'i'
                    constraint = self.range_constraints[symbol]
                    min_val, max_val = _convert_range_to_int(constraint)
                    assert_msg = f" is outside of inline constraint [{min_val}, {max_val}]."
                    call_backs.append(
                        partial(self._assert_range_constraint, lower=min_val, upper=max_val)
                    )
                    messages.append(assert_msg)
            elif isinstance(val, torch.Tensor):
                for i, sym in enumerate(val.shape):
                    cbs, msgs = add_assertions(sym)
                    for cb, msg in zip(cbs, msgs):
                        def sym_size_cb(proxy, assert_msg, dim):
                            dim_proxy = super(_AddRuntimeAssertionsForConstraintsPass, self).call_operator(
                                torch.ops.aten.sym_size,
                                (proxy, dim),
                                {},
                                NodeMetadata({}),
                            )
                            cb(proxy=dim_proxy, assert_msg=assert_msg)
                        call_backs.append(partial(sym_size_cb, dim=i))
                        messages.append(f".shape[{i}]" + msg)
            return call_backs, messages
        callbacks, messages = add_assertions(val)
        for cb, msg in zip(callbacks, messages):
            cb(proxy=ret, assert_msg=f"{ret.node}" + msg)
        return ret

    def _assert_range_constraint(self, proxy, lower, upper, assert_msg):
        if lower > -math.inf:
            self._insert_assert_async(operator.ge, proxy, lower, assert_msg)

        if upper < math.inf:
            self._insert_assert_async(operator.le, proxy, upper, assert_msg)

    def _insert_assert_async(self, operator, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        cmp = super().call_operator(operator, (lower, upper), {}, NodeMetadata({}))
        cmp_tensor = super().call_operator(torch.ops.aten.scalar_tensor.default, (cmp,), {}, NodeMetadata({}))
        super().call_operator(
            torch.ops.aten._assert_async.msg,
            (cmp_tensor, assert_msg),
            {},
            NodeMetadata({}),
        )
