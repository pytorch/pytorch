# mypy: allow-untyped-defs
"""Lower Inductor GEMM epilogue loop IR to CuTeDSL source."""

from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLCSEVariable
from torch._inductor.ir import ComputedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .gemm_epilogue import GemmEpiloguePlan
from .gemm_epilogue_codegen import (
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from .loop_ir_epilogue_lowering import GemmEpilogueIRAnalysis, GemmEpilogueIRExpression


class LoopIRCuteDSLCodegen:
    """Lower scheduler loop-body expressions to a CuTeDSL fragment function."""

    def __init__(self, accumulator: str, removed_buffers: OrderedSet[str]) -> None:
        self.accumulator = accumulator
        self.removed_buffers = removed_buffers
        self.kernel = GemmEpilogueCuteDSLKernel()
        self.reads: OrderedSet[str] = OrderedSet()
        self.input_values: dict[str, CuteDSLCSEVariable] = {}
        self.replacements: dict[int, CuteDSLCSEVariable] = {}
        self.accumulator_value = CuteDSLCSEVariable(
            "accum", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )

    def load(self, name: str, stored: Any) -> Any:
        if stored is not None:
            return self.lower(stored)
        if name == self.accumulator:
            return self.accumulator_value
        self.reads.add(name)
        if name not in self.input_values:
            dtype = V.graph.get_dtype(name)
            self.input_values[name] = CuteDSLCSEVariable(
                name, ValueRanges.unknown(), dtype=dtype, shape=(1,)
            )
        return self.input_values[name]

    def lower(self, value: Any) -> Any:
        if replacement := self.replacements.get(id(value)):
            return replacement
        if not isinstance(value, GemmEpilogueIRExpression):
            if isinstance(value, tuple):
                return tuple(self.lower(item) for item in value)
            return value
        if value.op == "load":
            name, _, stored = value.args
            return self.load(name, stored)
        if value.op == "identity":
            return self.lower(value.args[0])
        if value.op in ("reduction", "to_dtype_bitcast"):
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue does not support {value.op}"
            )
        lowered_args = tuple(self.lower(arg) for arg in value.args)
        lowered_kwargs = {key: self.lower(arg) for key, arg in value.kwargs}
        op = getattr(GemmEpilogueCuteDSLOpOverrides, value.op, None)
        owner = next(
            (
                cls
                for cls in GemmEpilogueCuteDSLOpOverrides.__mro__
                if value.op in cls.__dict__
            ),
            None,
        )
        if op is None or owner is None or owner.__name__ == "OpsHandler":
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue op is not implemented: {value.op}"
            )
        return op(*lowered_args, **lowered_kwargs)

    def render(
        self, buffers: Sequence[ComputedBuffer], fn_name: str
    ) -> GemmEpiloguePlan:
        analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        outputs: list[tuple[str, Any]] = []
        if self.accumulator not in self.removed_buffers:
            outputs.append((self.accumulator, self.accumulator_value))
        for name, store in analysis.stores.items():
            if name not in self.removed_buffers:
                outputs.append((name, self.lower(store.value)))
        if not outputs:
            raise NotImplementedError("CuTeDSL GEMM epilogue has no outputs")

        renames = {name: name for name in self.reads}
        result_names: list[str] = []
        for index, (buffer_name, value) in enumerate(outputs):
            result_name = "D" if index == 0 else f"output{index - 1}"
            self.kernel.body.writeline(f"{result_name} = {value}")
            renames[result_name] = buffer_name
            result_names.append(result_name)

        params = ", ".join(("accum", *self.reads))
        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        source = (
            f"def {fn_name}({params}):\n{body}\n    return {', '.join(result_names)}"
        )
        return GemmEpiloguePlan(
            source=source,
            is_cutedsl=True,
            reads=tuple(self.reads),
            writes=tuple(name for name, _ in outputs),
            renames=renames,
        )

    @classmethod
    def from_buffers(
        cls,
        accumulator: str,
        buffers: Sequence[ComputedBuffer],
        removed_buffers: OrderedSet[str],
        fn_name: str,
    ) -> GemmEpiloguePlan:
        codegen = cls(accumulator, removed_buffers)
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            return codegen.render(buffers, fn_name)

    @staticmethod
    def online_softmax(output_name: str, group: int, fn_name: str) -> GemmEpiloguePlan:
        source = f"""def {fn_name}(accum):
    fragment_n = cutlass.const_expr(cute.size(accum.shape, mode=[0]))
    if cutlass.const_expr(fragment_n % {group} != 0):
        raise AssertionError("expected softmax group to divide fragment width")
    repeats = cutlass.const_expr(fragment_n // {group})
    grouped = accum.to(cutlass.Float32).reshape(((1, {group}, repeats), 1, 1))
    maximum = grouped.reduce(
        cute.ReductionOp.MAX,
        init_val=-cutlass.Float32.inf,
        reduction_profile=((None, 1, None), 1, 1),
    ).reshape(((1, 1, repeats), 1, 1))
    numerator = cute.math.exp(grouped - maximum.broadcast_to(grouped.shape))
    denominator = numerator.reduce(
        cute.ReductionOp.ADD,
        init_val=0.0,
        reduction_profile=((None, 1, None), 1, 1),
    ).reshape(((1, 1, repeats), 1, 1))
    D = (numerator / denominator.broadcast_to(grouped.shape)).reshape(accum.shape)
    return D"""
        return GemmEpiloguePlan(
            source=source,
            is_cutedsl=True,
            writes=(output_name,),
            renames={"D": output_name},
        )

    @classmethod
    def finalizer_from_buffer(
        cls, source_name: str, buffer: ComputedBuffer, fn_name: str
    ) -> str:
        codegen = cls(source_name, OrderedSet((source_name,)))
        codegen.accumulator_value = CuteDSLCSEVariable(
            "value", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
        codegen.accumulator_value.is_scalar_expr = True
        analysis = GemmEpilogueIRAnalysis.from_buffers((buffer,))
        store = analysis.store(buffer.get_name())
        if store is None:
            raise NotImplementedError("CuTeDSL reduction finalizer has no output")
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            result = codegen.lower(store.value)
        if codegen.reads:
            raise NotImplementedError(
                "CuTeDSL reduction finalizer cannot capture tensor inputs"
            )
        body = [*(f"    {line}" for line in codegen.kernel.body.lines)]
        body.append(f"    return {result}")
        return f"def {fn_name}(value, group):\n" + "\n".join(body)

    @classmethod
    def consumer_from_buffer(
        cls,
        accumulator_name: str,
        reduction_name: str | None,
        buffer: ComputedBuffer,
        fn_name: str,
        group: int | None = None,
    ) -> str:
        removed = OrderedSet((accumulator_name,))
        if reduction_name is not None:
            removed.add(reduction_name)
        codegen = cls(accumulator_name, removed)
        codegen.accumulator_value = CuteDSLCSEVariable(
            "accumulator", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
        codegen.accumulator_value.is_scalar_expr = True
        reduction_value = CuteDSLCSEVariable(
            "primary_reduction", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
        reduction_value.is_scalar_expr = True
        analysis = GemmEpilogueIRAnalysis.from_buffers((buffer,))
        store = analysis.store(buffer.get_name())
        if store is None:
            raise NotImplementedError("CuTeDSL reduction consumer has no output")
        if reduction_name is not None:
            codegen.input_values[reduction_name] = reduction_value
        else:
            if group is None:
                raise NotImplementedError("synthetic reduction consumer needs a group")
            region = analysis.reduction_region(
                buffer.get_name(),
                accumulator_name,
                group,
                V.graph.get_dtype(accumulator_name),
            )
            if region is None or len(region.reductions) != 1:
                raise NotImplementedError(
                    "CuTeDSL consumer needs exactly one synthetic reduction"
                )
            codegen.replacements[id(region.reductions[0].source)] = reduction_value
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            result = codegen.lower(store.value)
        if reduction_name is not None:
            codegen.reads.discard(reduction_name)
        if codegen.reads:
            raise NotImplementedError(
                "CuTeDSL reduction consumer cannot capture tensor inputs"
            )
        body = [*(f"    {line}" for line in codegen.kernel.body.lines)]
        body.append(f"    return {result}")
        return (
            f"def {fn_name}(accumulator, primary_reduction, _secondary_reduction):\n"
            + "\n".join(body)
        )
