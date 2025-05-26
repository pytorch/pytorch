import itertools
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from os import linesep
from typing import Any, Optional

import sympy

import torch
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import IndentedBuffer, OrderedSet
from torch._inductor.virtualized import OpsValue

from ...virtualized import V


_ACCUMULATOR_ALIAS = "accum"


class _AssignmentFormatter(DefaultHandler):
    def __init__(self, parent_handler: "CutlassEVTCodegen"):
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Handle op dispatch here
        if hasattr(self.parent_handler, name):
            fn = getattr(self.parent_handler, name)
            line = fn(*args, **kwargs)
            if name in ("load", "store"):
                return OpsValue(line)
            else:
                var = self.parent_handler._tmp_var()
                self.parent_handler.output.writeline(f"{var} = {line}")
                return OpsValue(var)
        else:
            raise NotImplementedError(name)


class CutlassEVTCodegen:
    """
    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTCodegen.ir_to_evt_python_code(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.
    """

    def __init__(self, accumulator_node_name: str, last_usages: OrderedSet[str]):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTCodegen.ir_to_evt_python_code static method.

        Args:
            accumulator_node_name: The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
            epilogue_nodes: The list of scheduler nodes to be fused into the epilogue
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.output: IndentedBuffer = IndentedBuffer(1)  # The output buffer for codegen
        self.var_counter: Iterator[int] = itertools.count()
        self.store_name_to_value: dict[str, OpsValue] = (
            dict()
        )  # Aliases for subexpression functors
        self.reads: OrderedSet[str] = OrderedSet()
        self.last_usages: OrderedSet[str] = OrderedSet()
        self.cur_node: Optional[ComputedBuffer] = None

        if accumulator_node_name not in last_usages:
            self.store(accumulator_node_name, value=OpsValue(_ACCUMULATOR_ALIAS))

    @staticmethod
    def ir_to_evt_python_code(
        cuda_template_node_name: str,
        epilogue_nodes: list[BaseSchedulerNode],
    ) -> tuple[list[str], list[str], dict[str, Any], str]:
        last_usages = OrderedSet(
            itertools.chain(*[node.last_usage for node in epilogue_nodes])
        )
        codegen = CutlassEVTCodegen(cuda_template_node_name, last_usages)
        handler = _AssignmentFormatter(codegen)

        with virtualized.V.set_ops_handler(handler):
            for s_node in epilogue_nodes:
                node = s_node.node
                assert isinstance(node, ComputedBuffer)
                with codegen.set_cur_node(node):
                    index_vars = CutlassEVTCodegen._get_index_vars(node)
                    node.get_store_function()(index_vars)

        return (
            codegen.get_reads(),
            codegen.get_writes(),
            codegen.get_renames(),
            codegen.get_value(),
        )

    def get_value(self) -> str:
        return linesep.join(
            [
                self._render_input_signature(),
                self.output.getvalue(),
                self._render_return_statement(),
            ]
        )

    @contextmanager
    def set_cur_node(self, node: ComputedBuffer) -> Generator[None, Any, Any]:
        prev_node = self.cur_node
        try:
            self.cur_node = node
            yield
        finally:
            self.cur_node = prev_node

    def get_renames(self) -> dict[str, str]:
        renames = {k: v.value for k, v in self.store_name_to_value.items()}
        renames[self.accumulator_node_name] = _ACCUMULATOR_ALIAS
        return renames

    def get_reads(self) -> list[str]:
        return list(self.reads)

    def get_writes(self) -> list[str]:
        return list(self.store_name_to_value.keys())

    def load(self, name: str, index: Any) -> str:
        self._check_indexing(name, index)
        if name == self.accumulator_node_name:
            self.reads.add(name)
            return _ACCUMULATOR_ALIAS
        elif name in self.store_name_to_value:
            return self.store_name_to_value[name].value
        else:
            self.reads.add(name)
            return name

    def store(
        self, name: Any, index: Any = None, value: Any = None, mode: Any = None
    ) -> None:
        if name not in self.last_usages:
            if index:
                self._check_indexing(name, index)

            value_to_write = value
            if not self.store_name_to_value:
                # EVT requires an output to be named D lol
                # so rename the first store to D
                self.output.writeline(f"D = {value} # cutlass evt requirement")
                value_to_write = OpsValue("D")

            self.store_name_to_value[name] = value_to_write
        return None

    def to_dtype(
        self,
        x: str,
        dtype: Any,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = False,
    ) -> str:
        return x

    def constant(self, value: Any, dtype: Any) -> str:
        raise NotImplementedError

    def mul(self, x0: str, x1: str) -> str:
        return self._infix_bin_op("*", x0, x1)

    def truediv(self, x0: str, x1: str) -> str:
        raise NotImplementedError

    def ge(self, x0: str, x1: str) -> str:
        raise NotImplementedError

    def add(self, x0: str, x1: str) -> str:
        return self._infix_bin_op("+", x0, x1)

    def relu(self, x0: str) -> str:
        return self._prefix_un_op("relu", x0)

    def sigmoid(self, x0: str) -> str:
        return self._prefix_un_op("sigmoid", x0)

    def sub(self, x0: str, x1: str) -> str:
        raise NotImplementedError

    def _get_cur_node(self) -> ComputedBuffer:
        assert self.cur_node
        return self.cur_node

    @staticmethod
    def _get_index_vars(node: ComputedBuffer) -> Sequence[sympy.Expr]:
        data = node.data
        # TODO mlazos: relax this, cutlass supports reductions and other ops
        assert isinstance(data, Pointwise)
        return data._index(data.ranges)

    def _get_current_index_vars(self) -> Sequence[sympy.Expr]:
        return self._get_index_vars(self._get_cur_node())

    def _check_indexing(self, name: str, index: sympy.Expr) -> None:
        # We only support indexing that matches the layout today because
        # CUTLASS doesn't support arbitrary indexing
        buffer_name = self.accumulator_node_name if name == _ACCUMULATOR_ALIAS else name
        buffer = V.graph.name_to_buffer[buffer_name]
        index_strides = V.graph.sizevars.stride_vars(
            index, self._get_current_index_vars()
        )
        if buffer.get_layout().stride != index_strides:
            raise NotImplementedError(
                f"Unsupported indexing for {name} with index {index} and strides {index_strides}"
            )

    def _render_input_signature(self) -> str:
        arguments = ", ".join(
            [_ACCUMULATOR_ALIAS]
            + [name for name in self.reads if name != self.accumulator_node_name]
        )
        return f"def fn({arguments}):"

    def _render_return_statement(self) -> str:
        return_vars = OrderedSet(
            op_v.value for op_v in self.store_name_to_value.values()
        )
        assert "D" in return_vars
        return f"return {', '.join(return_vars)}"

    def _tmp_var(self) -> str:
        return f"tmp_{next(self.var_counter)}"

    def _infix_bin_op(self, op: str, a: str, b: str) -> str:
        return f"{a} {op} {b}"

    def _prefix_bin_op(self, op: str, a: str, b: str) -> str:
        return f"{op}({a}, {b})"

    def _prefix_un_op(self, op: str, a: str) -> str:
        return f"{op}({a})"
