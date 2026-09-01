# mypy: allow-untyped-defs
"""Lower Inductor GEMM epilogue loop IR to CuTeDSL source."""

from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    tensorssa_reduction,
)
from torch._inductor.ir import ComputedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .gemm_epilogue import (
    GEMM_LOCAL_REDUCTION_RESULT_NAME,
    GemmEpiloguePlan,
    GemmReductionGeometry,
)
from .gemm_epilogue_codegen import (
    canonical_tensorssa_reduction_type,
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from .loop_ir_epilogue_lowering import GemmEpilogueIRAnalysis, GemmEpilogueIRExpression


class LoopIRCuteDSLCodegen:
    """Lower scheduler loop-body expressions to a CuTeDSL fragment function."""

    def __init__(
        self,
        accumulator: str,
        removed_buffers: OrderedSet[str],
        reduction_geometries: dict[str, GemmReductionGeometry] | None = None,
        suppressed_outputs: OrderedSet[str] | None = None,
    ) -> None:
        self.accumulator = accumulator
        self.removed_buffers = removed_buffers
        self.reduction_geometries = reduction_geometries or {}
        self.suppressed_outputs = suppressed_outputs or OrderedSet()
        self.reduction_geometry_by_id: dict[int, GemmReductionGeometry] = {}
        self.synthetic_reduction_by_expr: dict[
            int, tuple[GemmReductionGeometry, str, GemmEpilogueIRExpression]
        ] = {}
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

    def _generate_like(self, expr: str, ref: Any, *, shape_ref: Any | None = None):
        if shape_ref is None:
            shape_ref = ref
        return self.kernel.cse.generate(
            self.kernel.body,
            expr,
            dtype=getattr(ref, "dtype", None),
            shape=getattr(shape_ref, "shape", None),
        )

    def _emit_reduction(
        self,
        source: Any,
        reduction_type: str,
        geometry: GemmReductionGeometry,
    ) -> Any:
        if geometry.needs_physical_callbacks and geometry.axis == 0:
            return source
        desc = tensorssa_reduction(canonical_tensorssa_reduction_type(reduction_type))
        fragment_group = (
            f"cutlass.const_expr(min({geometry.group}, "
            f"cute.size({source}.shape, mode=[0])))"
        )
        repeats = (
            f"cutlass.const_expr(cute.size({source}.shape, mode=[0]) "
            f"// min({geometry.group}, cute.size({source}.shape, mode=[0])))"
        )
        grouped = self._generate_like(
            f"{source}.reshape(((1, {fragment_group}, {repeats}), 1, 1))",
            source,
        )
        reduced = self._generate_like(
            f"{grouped}.reduce({desc.cute_op}, init_val={desc.init_val}, "
            "reduction_profile=((None, 1, None), 1, 1))",
            grouped,
        )
        if reduction_type == "mean" and not geometry.needs_physical_callbacks:
            reduced = self._generate_like(
                f"{reduced} / {float(geometry.group)!r}", reduced
            )
        keepdim = self._generate_like(
            f"{reduced}.reshape(((1, 1, {repeats}), 1, 1))", reduced
        )
        broadcast = self._generate_like(
            f"{keepdim}.broadcast_to({grouped}.shape)",
            keepdim,
            shape_ref=grouped,
        )
        return self._generate_like(
            f"{broadcast}.reshape({source}.shape)", broadcast, shape_ref=source
        )

    def _local_reduction_value(
        self,
        analysis: GemmEpilogueIRAnalysis,
        output_name: str,
        geometry: GemmReductionGeometry,
    ) -> Any:
        store = analysis.store(output_name)
        if store is None:
            raise NotImplementedError("returned local reduction has no store")
        if not geometry.needs_physical_callbacks:
            return self.lower(store.value)
        region = analysis.reduction_region(
            output_name,
            self.accumulator,
            geometry.group,
            V.graph.get_dtype(self.accumulator),
        )
        if region is None or len(region.reductions) != 1:
            raise NotImplementedError(
                "physical CuTeDSL GEMM epilogues require one reduction"
            )
        reduction = region.reductions[0]
        source = reduction.synthetic_element or reduction.source
        return self._emit_reduction(
            self.lower(source), reduction.reduction_type, geometry
        )

    def _lower_reduction(self, value: GemmEpilogueIRExpression) -> Any:
        reduction = value.reductions[-1]
        geometry = self.reduction_geometry_by_id.get(id(reduction))
        if geometry is None:
            raise NotImplementedError("CuTeDSL GEMM epilogue reduction has no geometry")
        _, _, reduction_type, source, *result = value.args
        if result:
            raise NotImplementedError(
                "CuTeDSL GEMM epilogue does not support tuple-valued reductions"
            )
        return self._emit_reduction(self.lower(source), str(reduction_type), geometry)

    def lower(self, value: Any) -> Any:
        if replacement := self.replacements.get(id(value)):
            return replacement
        if not isinstance(value, GemmEpilogueIRExpression):
            if isinstance(value, tuple):
                return tuple(self.lower(item) for item in value)
            return value
        synthetic = self.synthetic_reduction_by_expr.get(id(value))
        if synthetic is not None:
            geometry, reduction_type, element = synthetic
            return self._emit_reduction(self.lower(element), reduction_type, geometry)
        if value.op == "load":
            name, _, stored = value.args
            return self.load(name, stored)
        if value.op == "identity":
            return self.lower(value.args[0])
        if value.op == "reduction":
            return self._lower_reduction(value)
        if value.op == "to_dtype_bitcast":
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

    def _output_scale_name(self, value: Any) -> str | None:
        """Recognize a pure accumulator-times-scalar epilogue."""

        def load_info(expr: Any) -> tuple[str, torch.dtype] | None:
            if isinstance(expr, GemmEpilogueIRExpression) and expr.op == "identity":
                return load_info(expr.args[0])
            if isinstance(expr, GemmEpilogueIRExpression) and expr.op == "to_dtype":
                loaded = load_info(expr.args[0])
                if loaded is None or len(expr.args) < 2 or expr.args[1] != loaded[1]:
                    return None
                src_dtype = (
                    expr.args[2]
                    if len(expr.args) > 2
                    else dict(expr.kwargs).get("src_dtype")
                )
                return loaded if src_dtype in (None, loaded[1]) else None
            if (
                isinstance(expr, GemmEpilogueIRExpression)
                and expr.op == "load"
                and expr.args[2] is None
            ):
                name = expr.args[0]
                if name == self.accumulator:
                    dtype = self.accumulator_value.dtype
                    return None if dtype is None else (name, dtype)
                buffer = V.graph.name_to_buffer.get(name)
                if buffer is None:
                    buffer = V.graph.graph_inputs.get(name)
                return None if buffer is None else (name, buffer.get_dtype())
            return None

        if not isinstance(value, GemmEpilogueIRExpression) or value.op != "mul":
            return None
        lhs, rhs = value.args[:2]
        lhs_info, rhs_info = load_info(lhs), load_info(rhs)
        lhs_name = None if lhs_info is None else lhs_info[0]
        rhs_name = None if rhs_info is None else rhs_info[0]
        scale_name = None
        if lhs_name == self.accumulator and rhs_name != self.accumulator:
            scale_name = rhs_name
        elif rhs_name == self.accumulator and lhs_name != self.accumulator:
            scale_name = lhs_name
        if scale_name is None:
            return None

        scale = V.graph.name_to_buffer.get(scale_name)
        if scale is None:
            scale = V.graph.graph_inputs.get(scale_name)
        if scale is None or scale.get_dtype() != torch.float32:
            return None
        if not all(
            V.graph.sizevars.statically_known_equals(dim, 1) for dim in scale.get_size()
        ):
            return None
        return scale_name

    def render(
        self, buffers: Sequence[ComputedBuffer], fn_name: str
    ) -> GemmEpiloguePlan:
        """Render buffers as one generated epilogue and optional reduction result."""

        analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        for output_name, geometry in self.reduction_geometries.items():
            store = analysis.store(output_name)
            if store is None:
                raise NotImplementedError(
                    f"CuTeDSL GEMM epilogue reduction output {output_name} is missing"
                )
            for reduction in store.value.reductions:
                existing = self.reduction_geometry_by_id.setdefault(
                    id(reduction), geometry
                )
                if existing != geometry:
                    raise NotImplementedError(
                        "CuTeDSL GEMM epilogue reductions must share one geometry"
                    )
            if not store.value.reductions:
                region = analysis.reduction_region(
                    output_name,
                    self.accumulator,
                    geometry.group,
                    V.graph.get_dtype(self.accumulator),
                )
                if region is None:
                    raise NotImplementedError(
                        f"CuTeDSL GEMM epilogue cannot reconstruct {output_name}"
                    )
                for reduction in region.reductions:
                    if reduction.synthetic_element is None:
                        raise NotImplementedError(
                            "CuTeDSL GEMM epilogue synthetic reduction has no element"
                        )
                    self.synthetic_reduction_by_expr[id(reduction.source)] = (
                        geometry,
                        reduction.reduction_type,
                        reduction.synthetic_element,
                    )
        output_scale = None
        if self.accumulator in self.removed_buffers and len(analysis.stores) == 1:
            output_scale = self._output_scale_name(
                next(iter(analysis.stores.values())).value
            )
        outputs: list[tuple[str, Any]] = []
        if self.accumulator not in self.removed_buffers:
            outputs.append((self.accumulator, self.accumulator_value))
        for name, store in analysis.stores.items():
            if name not in self.removed_buffers and name not in self.suppressed_outputs:
                outputs.append((name, self.lower(store.value)))
        if not outputs:
            raise NotImplementedError("CuTeDSL GEMM epilogue has no outputs")
        local_reduce_outputs = tuple(self.suppressed_outputs)
        if len(local_reduce_outputs) > 1:
            raise NotImplementedError(
                "CuTeDSL GEMM epilogues support one returned local reduction"
            )

        renames = {name: name for name in self.reads}
        result_names: list[str] = []
        for index, (buffer_name, value) in enumerate(outputs):
            result_name = "D" if index == 0 else f"output{index - 1}"
            self.kernel.body.writeline(f"{result_name} = {value}")
            renames[result_name] = buffer_name
            result_names.append(result_name)
        if local_reduce_outputs:
            output_name = local_reduce_outputs[0]
            geometry = self.reduction_geometries[output_name]
            self.kernel.body.writeline(
                f"{GEMM_LOCAL_REDUCTION_RESULT_NAME} = "
                f"{self._local_reduction_value(analysis, output_name, geometry)}"
            )
            result_names.append(GEMM_LOCAL_REDUCTION_RESULT_NAME)

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
            output_scale=output_scale,
        )

    @classmethod
    def from_buffers(
        cls,
        accumulator: str,
        buffers: Sequence[ComputedBuffer],
        removed_buffers: OrderedSet[str],
        fn_name: str,
        reduction_geometries: dict[str, GemmReductionGeometry] | None = None,
        suppressed_outputs: OrderedSet[str] | None = None,
    ) -> GemmEpiloguePlan:
        codegen = cls(
            accumulator,
            removed_buffers,
            reduction_geometries,
            suppressed_outputs,
        )
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            return codegen.render(buffers, fn_name)

    @classmethod
    def source_from_expression(
        cls,
        source_name: str,
        expression: GemmEpilogueIRExpression,
        fn_name: str,
    ) -> str:
        """Generate a reduction source callback from its captured Loop IR."""
        source_indices = []
        pending = [expression]
        while pending:
            value = pending.pop()
            if isinstance(value, (tuple, list)):
                pending.extend(value)
                continue
            if not isinstance(value, GemmEpilogueIRExpression):
                continue
            if value.op == "load":
                name, index, stored = value.args
                if name == source_name:
                    source_indices.append(index)
                if stored is not None:
                    pending.append(stored)
                continue
            pending.extend(value.args)
            pending.extend(item for _, item in value.kwargs)
        if source_indices and any(
            sympy.simplify(index - source_indices[0]) != 0
            for index in source_indices[1:]
        ):
            raise NotImplementedError(
                "CuTeDSL reduction source loads must reference one logical element"
            )
        codegen = cls(source_name, OrderedSet((source_name,)))
        codegen.accumulator_value = CuteDSLCSEVariable(
            "value", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            result = codegen.lower(expression)
        if codegen.reads:
            raise NotImplementedError(
                "CuTeDSL reduction source cannot capture tensor inputs"
            )
        body = [*(f"    {line}" for line in codegen.kernel.body.lines)]
        body.append(f"    return {result}")
        return f"def {fn_name}(value):\n" + "\n".join(body)

    @classmethod
    def physical_reduction_callbacks(
        cls,
        accumulator: str,
        analysis: GemmEpilogueIRAnalysis,
        output_name: str,
        geometry: GemmReductionGeometry,
    ) -> tuple[str, str]:
        """Generate combine and finalizer callbacks for one physical reduction."""
        region = analysis.reduction_region(
            output_name,
            accumulator,
            geometry.group,
            V.graph.get_dtype(accumulator),
        )
        if region is None or len(region.reductions) != 1:
            raise NotImplementedError(
                "physical CuTeDSL GEMM epilogues require one reduction"
            )
        reduction = region.reductions[0]
        desc = tensorssa_reduction(
            canonical_tensorssa_reduction_type(reduction.reduction_type)
        )
        combine = (
            f"def _local_reduce_combine(lhs, rhs):\n    return {desc.combine_expr}"
        )

        target = reduction.source
        if reduction.synthetic_element is None:
            pending = [region.expression]
            target = None
            while pending and target is None:
                value = pending.pop()
                if not isinstance(value, GemmEpilogueIRExpression):
                    continue
                if (
                    value.op == "reduction"
                    and value.reductions
                    and value.reductions[-1] is reduction
                ):
                    target = value
                    break
                pending.extend(value.args)
                pending.extend(item for _, item in value.kwargs)
            if target is None:
                raise NotImplementedError(
                    "physical CuTeDSL GEMM epilogue reduction is missing"
                )

        codegen = cls(accumulator, OrderedSet((accumulator,)))
        completed = CuteDSLCSEVariable(
            "value", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
        completed.is_scalar_expr = True
        if reduction.reduction_type == "mean":
            completed = codegen._generate_like("value / group", completed)
            completed.is_scalar_expr = True
        codegen.replacements[id(target)] = completed
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            result = codegen.lower(region.expression)
        if codegen.reads:
            raise NotImplementedError(
                "physical reduction finalizer cannot capture tensor inputs"
            )
        body = [*(f"    {line}" for line in codegen.kernel.body.lines)]
        body.append(f"    return {result}")
        finalize = "def _local_reduce_finalize(value, group):\n" + "\n".join(body)
        return combine, finalize

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
        elif group is not None:
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
