import itertools
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from os import linesep
from typing import Any

import sympy

import torch
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import DelayReplaceLine, IndentedBuffer, OrderedSet
from torch._inductor.virtualized import OpsValue

from ...virtualized import V


_ACCUMULATOR_ARG_NAME = "accum"


def scaled_mm_evt(
    scale_A_name: str, scale_B_name: str, bias_name: str | None, output_name: str
) -> tuple[list[str], dict[str, Any], str]:
    evt_read_names = [scale_A_name, scale_B_name]
    var_name_to_buffer_name = {n: n for n in [scale_A_name, scale_B_name]}
    var_name_to_buffer_name["D"] = output_name
    var_name_to_buffer_name[_ACCUMULATOR_ARG_NAME] = output_name
    expr = f"accum * {scale_A_name} * {scale_B_name}{linesep}"
    if bias_name:
        expr = f"({expr}) + {bias_name}"
        evt_read_names.append(bias_name)
        var_name_to_buffer_name[bias_name] = bias_name

    evt_py_code = f"def fn(accum, {','.join(evt_read_names)}):{linesep}\
    D = {expr}{linesep}\
    return D{linesep}"

    return evt_read_names, var_name_to_buffer_name, evt_py_code


class CutlassEVTOpsMixIn:
    @staticmethod
    def _infix_bin_op(op: str, a: str, b: str) -> str:
        return f"{a} {op} {b}"

    @staticmethod
    def _prefix_bin_op(op: str, a: str, b: str) -> str:
        return f"{op}({a}, {b})"

    @staticmethod
    def _prefix_un_op(op: str, a: str) -> str:
        return f"{op}({a})"

    @staticmethod
    def to_dtype(
        x: str,
        dtype: Any,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = False,
    ) -> str:
        return x

    @staticmethod
    def constant(value: Any, dtype: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def mul(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("*", x0, x1)

    @staticmethod
    def truediv(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("/", x0, x1)

    @staticmethod
    def ge(x0: str, x1: str) -> str:
        raise NotImplementedError

    @staticmethod
    def add(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("+", x0, x1)

    @staticmethod
    def relu(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("relu", x0)

    @staticmethod
    def sigmoid(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("sigmoid", x0)

    @staticmethod
    def sub(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("-", x0, x1)

    @staticmethod
    def tanh(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("tanh", x0)

    @staticmethod
    def exp(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("exp", x0)


class MockCutlassHandler(CutlassEVTOpsMixIn, WrapperHandler):
    """Passthrough handler for cutlass ops, used for running epilogue nodes for memory planning"""


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
                line = DelayReplaceLine(
                    var,
                    lambda: "D"
                    if var == self.parent_handler.last_stored_var_name
                    else var,
                    f"{var} = {line}",
                )
                self.parent_handler.body.writeline(line)
                return OpsValue(var)
        else:
            raise NotImplementedError(name)


class CutlassEVTCodegen(CutlassEVTOpsMixIn):
    """
    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTCodegen.ir_to_evt_python_code(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.
    """

    def __init__(self, accumulator_node_name: str, removed_buffers: OrderedSet[str]):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTCodegen.ir_to_evt_python_code static method.

        Args:
            accumulator_node_name: The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
            epilogue_nodes: The list of scheduler nodes to be fused into the epilogue
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.body: IndentedBuffer = IndentedBuffer(1)  # The body buffer for codegen
        self.var_counter: Iterator[int] = itertools.count()
        self.store_name_to_value: dict[str, OpsValue] = (
            dict()
        )  # Aliases for subexpression functors
        self.reads: OrderedSet[str] = OrderedSet([])
        # Used for creating example tensors
        self.var_name_to_buffer_name: dict[str, str] = {
            _ACCUMULATOR_ARG_NAME: accumulator_node_name
        }
        self.removed_buffers: OrderedSet[str] = removed_buffers
        self.cur_node: ComputedBuffer | None = None
        self.name_to_buffer = V.graph.name_to_buffer | V.graph.graph_inputs
        for name in V.graph.constants:
            self.name_to_buffer[name] = V.graph.add_tensor_constant(
                V.graph.constants[name], name
            )
        self.is_D_assigned = False
        self.D_var_name = None

        if accumulator_node_name not in removed_buffers:
            # cannot return accumulator directly, so alias it
            var = self._tmp_var()
            self.body.writeline(f"{var} = {_ACCUMULATOR_ARG_NAME}")
            self.store(accumulator_node_name, value=OpsValue(var))

    @staticmethod
    def ir_to_evt_python_code(
        cutlass_template_node_name: str,
        epilogue_nodes: list[BaseSchedulerNode],
        removed_buffers: OrderedSet[str],
    ) -> tuple[list[str], list[str], dict[str, Any], str]:
        codegen = CutlassEVTCodegen(cutlass_template_node_name, removed_buffers)
        handler = _AssignmentFormatter(codegen)

        with virtualized.V.set_ops_handler(handler):
            for s_node in epilogue_nodes:
                node = s_node.node
                assert isinstance(node, ComputedBuffer)
                with codegen.set_cur_node(node):
                    index_vars = CutlassEVTCodegen.get_index_vars(node)
                    node.get_store_function()(index_vars)

        codegen.finalize()

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
                self.body.getvalue(),
                self._render_return_statement(),
            ]
        )

    def finalize(self) -> None:
        # Rename the last store to D
        # no other code references this store
        # to workaround https://github.com/NVIDIA/cutlass/issues/2288
        # Note: the delayed line will automatically rewrite the last assignment to
        # be to D
        buffer_name = self.var_name_to_buffer_name[self.last_stored_var_name]
        self.var_name_to_buffer_name.pop(self.last_stored_var_name)
        self.var_name_to_buffer_name["D"] = buffer_name
        self.store_name_to_value[buffer_name] = OpsValue("D")

    @contextmanager
    def set_cur_node(self, node: ComputedBuffer) -> Generator[None, Any, Any]:
        prev_node = self.cur_node
        try:
            self.cur_node = node
            yield
        finally:
            self.cur_node = prev_node

    def get_renames(self) -> dict[str, str]:
        return dict(self.var_name_to_buffer_name)

    def get_reads(self) -> list[str]:
        return list(self.reads.difference(self.store_name_to_value.keys()))

    def get_writes(self) -> list[str]:
        return list(self.store_name_to_value.keys())

    def load(self, name: str, index: Any) -> str:
        self._check_indexing(name, index)
        if name in self.store_name_to_value:
            return self.store_name_to_value[name].value
        elif name == self.accumulator_node_name:
            return _ACCUMULATOR_ARG_NAME
        else:
            self.reads.add(name)
            self.var_name_to_buffer_name[name] = name
            return name

    def store(
        self, name: Any, index: Any = None, value: Any = None, mode: Any = None
    ) -> None:
        if name not in self.removed_buffers:
            if index:
                self._check_indexing(name, index)
            assert value.value != _ACCUMULATOR_ARG_NAME, (
                "Cannot store accumulator arg name"
            )
            self.var_name_to_buffer_name[value.value] = name
            self.store_name_to_value[name] = value
            self.last_stored_var_name = value.value
        return None

    def _get_cur_node(self) -> ComputedBuffer:
        assert self.cur_node
        return self.cur_node

    @staticmethod
    def get_index_vars(node: ComputedBuffer) -> Sequence[sympy.Expr]:
        data = node.data
        # TODO mlazos: relax this, cutlass supports reductions and other ops
        assert isinstance(data, Pointwise)
        return data._index(data.ranges)

    def _get_current_index_vars(self) -> Sequence[sympy.Expr]:
        return self.get_index_vars(self._get_cur_node())

    def _check_indexing(self, name: str, index: sympy.Expr) -> None:
        # We only support indexing that matches the layout today because
        # CUTLASS doesn't support arbitrary indexing
        buffer_name = (
            self.accumulator_node_name if name == _ACCUMULATOR_ARG_NAME else name
        )
        buffer = self.name_to_buffer[buffer_name]
        index_strides = V.graph.sizevars.stride_vars(
            index, self._get_current_index_vars()
        )
        stride = buffer.get_layout().stride
        if not self._stride_compatible(stride, index_strides):
            raise NotImplementedError(
                f"Unsupported indexing for {name} with index {index}, index strides {index_strides}, and layout stride {stride}"
            )

    def _stride_compatible(
        self, left: Iterable[sympy.Expr], right: Iterable[sympy.Expr]
    ) -> bool:
        return all(
            sympy.Eq(l, r) or sympy.Eq(l, 0) or sympy.Eq(r, 0)
            for l, r in (zip(left, right))
        )

    def _render_input_signature(self) -> str:
        arguments = ", ".join(
            [_ACCUMULATOR_ARG_NAME]
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
