from typing import Any
from unittest.mock import patch

import sympy

import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, IRNode, Pointwise
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import IndentedBuffer, OrderedSet, sympy_str
from torch._inductor.virtualized import OpsValue


# Used as a magic string to indicate an unsupported sympy expression
# became part of generated C++ code.
_MAGIC_SYMPY_ERROR_STRING = "[!sympy: unsupported expr!]"
_ACCUMULATOR_ALIAS = "accum"


def _arg_str(a: Any) -> str:
    if isinstance(a, sympy.Expr):
        # If this return value containing the _MAGIC_SYMPY_ERROR_STRING
        # is used as part of the final generated C++ code,
        # a NotImplementedError is raised to indicate that
        # the op could not be converted to a valid EVT expression.
        return f"{_MAGIC_SYMPY_ERROR_STRING}('{sympy_str(a)}')"
    return str(a)


class _ComputeAssignmentFormatter(DefaultHandler):
    def __init__(self, parent_handler: Any):
        super().__init__()
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name.startswith("_"):
            raise NotImplementedError(name)

        if hasattr(self.parent_handler, f"{name}"):
            fargs = [_arg_str(a) for a in args]
            fkwargs = {key: _arg_str(a) for key, a in kwargs.items()}
            fn = getattr(self.parent_handler, f"{name}")
            line = fn(*fargs, **fkwargs)
            if name in ("load", "store"):
                return line
            else:
                var = self.parent_handler._tmp_var()
                self.parent_handler.output.writeline(f"{var} = {line}")
            return var
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

    def __init__(self, accumulator_node_name: str):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTCodegen.ir_to_evt_python_code static method.

        Args:
            accumulator_node_name (str): The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.output: IndentedBuffer = IndentedBuffer(1)  # The output buffer for codegen
        self.var_counter: int = (
            0  # used to generate variable names, incremented for each new variable
        )
        self.result_aliases: dict[str, OpsValue] = (
            dict()
        )  # Aliases for subexpression functors
        self.reads: OrderedSet[str] = OrderedSet()
        self.writes: OrderedSet[str] = OrderedSet()

    @staticmethod
    def ir_to_evt_python_code(
        cuda_template_node_name: str,
        epilogue_nodes: list[IRNode],
    ) -> tuple[list[str], list[str], dict[str, Any], str]:
        codegen = CutlassEVTCodegen(cuda_template_node_name)
        handler = _ComputeAssignmentFormatter(codegen)

        D_defined = False

        with (
            virtualized.V.set_ops_handler(handler),
            patch.object(  # type: ignore[call-arg]
                FlexibleLayout, "allow_indexing", True
            ),
        ):
            for node in epilogue_nodes:
                assert isinstance(node, ComputedBuffer)
                data = node.data
                # TODO mlazos: relax this, cutlass supports reductions and other ops
                assert isinstance(data, Pointwise)
                index = data._index(data.ranges)
                result = data.inner_fn(index)
                name = node.get_computed_buffer_name()
                assert name is not None, "Computed buffer name is required for EVT"
                # EVT requires an output to be named D lol
                if not D_defined:
                    codegen.output.writeline(f"D = {result}")
                    codegen.result_aliases[name] = OpsValue("D")
                    D_defined = True
                elif name is not None:
                    codegen.result_aliases[name] = result

            codegen.output.writeline(
                codegen._render_return_statement(codegen.result_aliases)
            )

            res = codegen.get_value()

            if _MAGIC_SYMPY_ERROR_STRING in res:
                raise NotImplementedError(
                    "sympy / indexing expressions not yet supported in EVT fusion"
                )
            else:
                return (
                    codegen._get_reads(),
                    codegen._get_writes(),
                    codegen._get_renames(),
                    res,
                )

    def get_value(self) -> str:
        return f"def fn({self._render_input_signature()}):\n{self.output.getvalue()}"

    def _get_renames(self) -> dict[str, str]:
        renames = {k: v.value for k, v in self.result_aliases.items()}
        renames[self.accumulator_node_name] = _ACCUMULATOR_ALIAS
        return renames

    def _get_reads(self) -> list[str]:
        return list(self.reads)

    def _get_writes(self) -> list[str]:
        return list(self.result_aliases.keys())

    def _render_input_signature(self) -> str:
        return ", ".join(
            [_ACCUMULATOR_ALIAS]
            + [name for name in self.reads if name != self.accumulator_node_name]
        )

    def _render_return_statement(self, aliases: dict[str, OpsValue]) -> str:
        return f"return {', '.join(op_v.value for op_v in aliases.values())}"

    def _tmp_var(self) -> str:
        var = f"tmp_{self.var_counter}"
        self.var_counter += 1
        return var

    def _infix_bin_op(self, op: str, a: str, b: str) -> str:
        return f"{a} {op} {b}"

    def _prefix_bin_op(self, op: str, a: str, b: str) -> str:
        return f"{op}({a}, {b})"

    def load(self, name: str, index: Any) -> str:
        if name == self.accumulator_node_name:
            self.reads.add(name)
            return _ACCUMULATOR_ALIAS
        elif name in self.result_aliases:
            return self.result_aliases[name].value
        else:
            self.reads.add(name)
            return name

    def store(
        self, name: Any, index: Any = None, value: Any = None, mode: Any = None
    ) -> None:
        self.writes.add(name)
        return None

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

    def sub(self, x0: str, x1: str) -> str:
        raise NotImplementedError
