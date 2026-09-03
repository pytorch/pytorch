# mypy: allow-untyped-defs
r"""Build NVGEMM epilogue plans from scheduler Loop IR.

The scheduler tests fusion incrementally by passing the GEMM and the complete
candidate epilogue prefix to this module. The lowering captures those scheduler
nodes as Loop IR, derives grouped-reduction geometry from ranges and access
strides, partitions nodes into reduction regions and pointwise work, and returns
one ``NVGemmEpilogueProgram``. Unsupported prefixes are rejected as a unit.

The generated epilogue owns pointwise expressions and fragment-local reduction
semantics. ``reduction_type`` identifies only the associative primitive; the
captured IR defines source transforms, finalizers, consumers, and compositions
of multiple reductions. For reductions spanning fragments or threads, the plan
provides generated combine and finalize callbacks while the vendored kernel
owns traversal, synchronization, and storage.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import sympy

import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._ordered_set import OrderedSet

from ...ir import Buffer, ComputedBuffer, Pointwise, Reduction
from ...kernel.gemm_epilogue import (
    GEMM_REDUCTION_IDENTITY_SOURCE,
    GemmReductionConfig,
    GemmReductionGeometry,
    GemmReductionPlan,
    GemmReductionType,
)
from ...kernel.loop_ir_epilogue_lowering import (
    GemmEpilogueIRAnalysis,
    GemmEpilogueIRFinalizer,
    GemmEpilogueIRStore,
    grouped_reduction_axis_ir,
    grouped_reduction_pattern_ir,
)
from ...scheduler import BaseSchedulerNode
from ...virtualized import V


def _matches_affine_index(
    index: sympy.Expr,
    range_vars: Sequence[sympy.Symbol],
    strides: Sequence[Any],
    known_equals: Callable[[Any, Any], bool],
) -> bool:
    if not range_vars:
        range_vars = tuple(sorted(index.free_symbols, key=str))
    if len(range_vars) != len(strides):
        return False
    expected = sum(
        (var * stride for var, stride in zip(range_vars, strides)), sympy.Integer(0)
    )
    return known_equals(index, expected)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmFeedPlan:
    plan: GemmReductionPlan
    intermediate_outputs: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmReductionRegion:
    config: GemmReductionConfig
    nodes: tuple[BaseSchedulerNode, ...]
    finalizer: GemmEpilogueIRFinalizer | None = None


@dataclasses.dataclass(frozen=True)
class NVGemmReductionPartition:
    regions: tuple[NVGemmReductionRegion, ...]

    @property
    def configs(self) -> tuple[GemmReductionConfig, ...]:
        return tuple(region.config for region in self.regions)

    @property
    def nodes(self) -> tuple[BaseSchedulerNode, ...]:
        return tuple(
            OrderedSet(node for region in self.regions for node in region.nodes)
        )

    def owns(self, nodes: Sequence[BaseSchedulerNode]) -> bool:
        owned = OrderedSet(self.nodes)
        return bool(self.regions) and all(node in owned for node in nodes)

    def intersects(self, nodes: Sequence[BaseSchedulerNode]) -> bool:
        candidates = OrderedSet(nodes)
        return any(node in candidates for node in self.nodes)

    def region_for(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> NVGemmReductionRegion | None:
        """Find the unique region containing a candidate prefix and its source."""
        candidates = OrderedSet(nodes)
        matches = tuple(
            region
            for region in self.regions
            if region.nodes[0] in candidates
            and all(node in region.nodes for node in candidates)
        )
        return matches[0] if len(matches) == 1 else None


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmEpilogueProgram:
    """Semantic NVGEMM epilogue IR produced before scheduling policy."""

    capture: "NVGemmEpilogueCapture"
    reduction_partition: NVGemmReductionPartition
    reduction_plan: GemmReductionPlan | None
    intermediate_outputs: tuple[str, ...] = ()

    @property
    def generated_reduction_regions(self) -> tuple[NVGemmReductionRegion, ...]:
        """Return reductions that one generated TensorSSA epilogue can compose."""
        regions = self.reduction_partition.regions
        analysis = self.capture.analysis
        if not regions or analysis is None:
            return ()
        geometry = regions[0].config.geometry
        if geometry.needs_physical_callbacks and len(regions) != 1:
            return ()
        for region in regions:
            store = analysis.store(region.config.output_name)
            if (
                region.config.geometry != geometry
                or store is None
                or not store.value.reductions
                or any(
                    reduction.reduction_type
                    not in ("sum", "mean", "prod", "max", "min")
                    for reduction in store.value.reductions
                )
            ):
                return ()
        return regions

    @property
    def generated_reduction_geometries(self) -> dict[str, GemmReductionGeometry]:
        geometries = {
            region.config.output_name: region.config.geometry
            for region in self.generated_reduction_regions
        }
        if (
            self.reduction_plan is not None
            and self.reduction_plan.tensor_epilogue_returns_local_reduce
            and self.reduction_plan.reduction_output is not None
        ):
            geometries[self.reduction_plan.reduction_output] = (
                self.reduction_plan.geometry
            )
        analysis = self.capture.analysis
        if analysis is None:
            return geometries
        try:
            n = V.graph.sizevars.optimization_hint(self.capture.gemm.get_size()[1])
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return geometries
        gemm_name = self.capture.gemm.get_name()
        gemm_dtype = self.capture.gemm.get_dtype()
        for node in self.capture.nodes:
            if not isinstance(node.node, ComputedBuffer):
                continue
            output_name = node.node.get_name()
            match = analysis.synthetic_reduction_program(
                output_name, gemm_name, gemm_dtype, n
            )
            if (
                match is None
                or len(match.region.reductions) <= 1
                or match.geometry.needs_physical_callbacks
                or any(
                    reduction.reduction_type
                    not in ("sum", "mean", "prod", "max", "min")
                    or reduction.synthetic_element is None
                    for reduction in match.region.reductions
                )
            ):
                continue
            existing = next(iter(geometries.values()), match.geometry)
            if existing != match.geometry:
                return {}
            geometries[output_name] = match.geometry
        return geometries

    @property
    def supported(self) -> bool:
        """Whether every claimed reduction has a backend lowering contract."""
        return not self.has_unclaimed_reduction and (
            not self.reduction_partition.configs
            or bool(self.generated_reduction_regions)
            or self.reduction_plan is not None
        )

    @property
    def has_unclaimed_reduction(self) -> bool:
        analysis = self.capture.analysis
        if analysis is None:
            return False
        generated_regions = self.generated_reduction_regions
        if generated_regions:
            claimed = OrderedSet(
                id(reduction)
                for region in generated_regions
                if (store := analysis.store(region.config.output_name)) is not None
                for reduction in store.value.reductions
            )
            discovered = OrderedSet(
                id(reduction)
                for store in analysis.stores.values()
                for reduction in store.value.reductions
            )
            if discovered and discovered.issubset(claimed):
                return False
        try:
            n = V.graph.sizevars.optimization_hint(self.capture.gemm.get_size()[1])
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return True
        gemm_name = self.capture.gemm.get_name()
        gemm_dtype = self.capture.gemm.get_dtype()
        generated_outputs = self.generated_reduction_geometries
        for node in self.pointwise_nodes:
            if not isinstance(node.node, ComputedBuffer):
                continue
            output_name = node.node.get_name()
            match = analysis.synthetic_reduction_program(
                output_name, gemm_name, gemm_dtype, n
            )
            if match is not None and output_name not in generated_outputs:
                return True
        return False

    @property
    def feeds_main(self) -> bool:
        return self.reduction_plan is not None and self.reduction_plan.feeds_main

    @property
    def min_tile_n(self) -> int:
        groups = [
            config.group
            for config in self.reduction_partition.configs
            if config.axis == 1
        ]
        plan = self.reduction_plan
        if plan is not None and plan.feeds_main and plan.axis == 1:
            groups.append(plan.group)
        return max(groups, default=0)

    @property
    def owned_nodes(self) -> tuple[BaseSchedulerNode, ...]:
        generated = OrderedSet(
            node for region in self.generated_reduction_regions for node in region.nodes
        )
        owned = OrderedSet(
            node for node in self.reduction_partition.nodes if node not in generated
        )
        reduction_plan = self.reduction_plan
        feed_names = (
            (
                reduction_plan.primary_output,
                *(
                    ()
                    if reduction_plan.tensor_epilogue_returns_local_reduce
                    else reduction_plan.auxiliary_outputs
                ),
                *self.intermediate_outputs,
            )
            if reduction_plan is not None
            else ()
        )
        owned.update(
            node
            for node in self.capture.nodes
            if isinstance(node.node, Buffer) and node.node.get_name() in feed_names
        )
        return tuple(owned)

    @property
    def pointwise_nodes(self) -> tuple[BaseSchedulerNode, ...]:
        owned = OrderedSet(self.owned_nodes)
        return tuple(node for node in self.capture.nodes if node not in owned)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmEpilogueCapture:
    """Captured scheduler nodes and their interpreted Loop IR."""

    gemm: Buffer
    nodes: tuple[BaseSchedulerNode, ...]
    analysis: GemmEpilogueIRAnalysis | None

    @classmethod
    def from_nodes(
        cls,
        gemm: Buffer,
        nodes: Sequence[BaseSchedulerNode],
    ) -> "NVGemmEpilogueCapture":
        normalized_nodes = tuple(child for node in nodes for child in node.get_nodes())
        buffers = _computed_buffers(normalized_nodes)
        analysis = (
            GemmEpilogueIRAnalysis.from_buffers(buffers)
            if buffers is not None
            else None
        )
        return cls(
            gemm=gemm,
            nodes=normalized_nodes,
            analysis=analysis,
        )


class NVGemmEpilogueLowering:
    """Lower scheduler nodes to NVGEMM epilogue semantic plans."""

    @staticmethod
    def _grouped_pointwise_geometry(
        gemm_node: Buffer,
        buffer: ComputedBuffer,
        scheduler_node: BaseSchedulerNode,
    ) -> GemmReductionGeometry | None:
        if not isinstance(buffer.data, Pointwise):
            return None
        try:
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
            output_size = tuple(
                map(V.graph.sizevars.optimization_hint, buffer.data.ranges)
            )
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        compressed_size = _compressed_output_size(output_size)
        if compressed_size is None:
            return None
        out_m, out_n = compressed_size
        if m == out_m and out_n > 0 and n % out_n == 0:
            geometry = GemmReductionGeometry(group=n // out_n, axis=1)
        elif n == out_n and out_m > 0 and m % out_m == 0:
            geometry = GemmReductionGeometry(group=m // out_m, axis=0)
        else:
            return None
        if geometry.group <= 1:
            return None

        reads = list(scheduler_node.read_writes.reads)
        range_vars = scheduler_node.read_writes.range_vars
        if len(reads) != geometry.group or range_vars is None:
            return None
        if not range_vars:
            range_vars = tuple(
                sorted(
                    OrderedSet(
                        symbol for read in reads for symbol in read.index.free_symbols
                    ),
                    key=str,
                )
            )
        range_vars = cast(tuple[Any, ...], range_vars)
        if len(range_vars) == 2:
            if geometry.axis == 1:
                base = n * range_vars[0] + geometry.group * range_vars[1]
                expected_strides = [n, geometry.group]
            else:
                base = geometry.group * n * range_vars[0] + range_vars[1]
                expected_strides = [geometry.group * n, 1]
        elif len(range_vars) == 1 and geometry.axis == 1:
            base = geometry.group * range_vars[0]
            expected_strides = [geometry.group]
        else:
            return None
        offsets = OrderedSet()
        for read in reads:
            if read.name != gemm_node.get_name():
                return None
            strides = V.graph.sizevars.stride_vars(read.index, range_vars)
            if list(strides) != expected_strides:
                return None
            offsets.add(V.graph.sizevars.simplify(read.index - base))
        expected_offsets = OrderedSet(
            offset if geometry.axis == 1 else offset * n
            for offset in range(geometry.group)
        )
        return geometry if offsets == expected_offsets else None

    @classmethod
    def _grouped_reduce_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis,
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) not in (1, 2):
            return None
        buffers = _computed_buffers(nodes)
        if buffers is None:
            return None
        node = buffers[0]
        finalizer_store = (
            cls._pointwise_finalizer_match(nodes[0], nodes[1], analysis=analysis)
            if len(buffers) == 2
            else None
        )
        if len(buffers) == 2 and finalizer_store is None:
            return None
        reduction_type = None
        if isinstance(node.data, Reduction):
            reduction_type = node.data.reduction_type
        elif not isinstance(node.data, Pointwise):
            return None
        access_node = nodes[0] if finalizer_store is not None else scheduler_node
        output_name = buffers[-1].get_name()
        if len(node.data.ranges) not in (2, 3) or len(gemm_node.get_size()) != 2:
            return None
        m, n = gemm_node.get_size()
        out_size = tuple(node.data.ranges)
        compressed_size = _compressed_output_size(out_size)
        if compressed_size is None:
            return None
        out_m, out_n = compressed_size

        def known_equals(left, right) -> bool:
            return (
                V.graph.sizevars.statically_known_equals(left, right)
                or V.graph.sizevars.simplify(left - right) == 0
            )

        expected_strides = None
        if isinstance(node.data, Reduction):
            reduction = node.data
            if len(reduction.reduction_ranges) != 1:
                return None
            group_extent = reduction.reduction_ranges[0]
            try:
                group = V.graph.sizevars.optimization_hint(group_extent)
            except (GuardOnDataDependentSymNode, TypeError, ValueError):
                return None
            if (known_equals(m, out_m) and known_equals(n, out_n * group_extent)) or (
                known_equals(out_m, m) and known_equals(out_n, n // group_extent)
            ):
                axis = 1
                expected_strides = [n, group_extent, 1]
            elif (known_equals(m, out_m * group_extent) and known_equals(n, out_n)) or (
                known_equals(out_m, m // group_extent) and known_equals(out_n, n)
            ):
                axis = 0
                expected_strides = [group_extent * n, 1, n]
            else:
                return None
        else:
            geometry = cls._grouped_pointwise_geometry(gemm_node, node, access_node)
            if geometry is None:
                return None
            group, axis = geometry.group, geometry.axis
        if group <= 1:
            return None
        geometry = GemmReductionGeometry(group, axis)

        store = analysis.store(node.get_name())
        reduction_ir = (
            None
            if store is None
            else grouped_reduction_pattern_ir(
                store,
                gemm_node.get_name(),
                group,
                gemm_node.get_dtype(),
            )
        )
        if reduction_ir is None:
            return None
        matched_reduction_type, source_expression = reduction_ir
        if matched_reduction_type not in ("sum", "mean", "prod", "max", "min"):
            return None
        if isinstance(node.data, Pointwise):
            reduction_type = matched_reduction_type
        elif matched_reduction_type != reduction_type:
            return None
        finalizer = (
            analysis.reduction_finalizer(output_name, node.get_name())
            if finalizer_store is not None
            else None
        )
        if finalizer_store is not None and finalizer is None:
            return None
        if reduction_type is None:
            return None
        reduction_type = cast(GemmReductionType, reduction_type)
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen

        try:
            source_fn = LoopIRCuteDSLCodegen.source_from_expression(
                gemm_node.get_name(), source_expression, "_local_reduce_source"
            )
            finalizer_fn = (
                LoopIRCuteDSLCodegen.finalizer_from_buffer(
                    node.get_name(), buffers[-1], "_local_reduce_finalize"
                )
                if finalizer is not None and finalizer.materialize
                else None
            )
            if isinstance(node.data, Pointwise) and finalizer_store is None:
                region = analysis.reduction_region(
                    output_name,
                    gemm_node.get_name(),
                    group,
                    gemm_node.get_dtype(),
                )
                if region is None or len(region.reductions) != 1:
                    return None
                reduction = region.reductions[0]
                if (
                    reduction.reduction_type == "mean"
                    or reduction.source is not region.expression
                ):
                    _, finalizer_fn = LoopIRCuteDSLCodegen.reduction_callbacks(
                        gemm_node.get_name(), analysis, output_name, geometry
                    )
        except NotImplementedError:
            return None

        if isinstance(node.data, Reduction):
            if expected_strides is None:
                return None
            read_writes = node.get_read_writes()
            reads = list(read_writes.reads)
            if len(reads) != 1 or reads[0].name != gemm_node.get_name():
                return None
            range_vars = read_writes.range_vars
            if range_vars is None:
                return None
            expected_stride_options = [expected_strides]
            if axis == 1:
                expected_stride_options.append([n, 1])
            else:
                expected_stride_options.append([1, n])
            if not any(
                _matches_affine_index(
                    reads[0].index, range_vars, expected, known_equals
                )
                for expected in expected_stride_options
            ):
                return None
        return GemmReductionConfig(
            output_name=output_name,
            group=group,
            axis=axis,
            reduction_type=reduction_type,
            source_fn=source_fn,
            finalizer_fn=finalizer_fn,
        )

    @classmethod
    def _secondary_bool_output_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis,
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        buffer = cast(ComputedBuffer, nodes[0].node)
        if (
            not isinstance(buffer.data, Pointwise)
            or buffer.get_dtype() != torch.bool
            or not V.graph.sizevars.statically_known_list_equals(
                buffer.get_size(), gemm_node.get_size()
            )
        ):
            return None
        if analysis.store(buffer.get_name()) is None:
            return None
        reads = list(scheduler_node.read_writes.reads)
        if len(reads) != 1 or reads[0].name != gemm_node.get_name():
            return None
        range_vars = scheduler_node.read_writes.range_vars
        if range_vars is None:
            return None
        if not range_vars:
            range_vars = tuple(sorted(reads[0].index.free_symbols, key=str))
        strides = V.graph.sizevars.stride_vars(reads[0].index, range_vars)
        expected = [1] if len(range_vars) == 1 else list(gemm_node.get_stride())
        if len(strides) != len(expected) or not all(
            V.graph.sizevars.statically_known_equals(actual, wanted)
            for actual, wanted in zip(strides, expected)
        ):
            return None
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen

        try:
            secondary_consumer_fn = LoopIRCuteDSLCodegen.consumer_from_buffer(
                gemm_node.get_name(),
                None,
                buffer,
                "_local_reduce_secondary_consumer",
            )
        except NotImplementedError:
            return None
        return GemmReductionConfig(
            output_name=buffer.get_name(),
            group=2,
            axis=1,
            reduction_type="sum",
            source_fn=GEMM_REDUCTION_IDENTITY_SOURCE,
            secondary_consumer_fn=secondary_consumer_fn,
        )

    @staticmethod
    def _reduction_source_fn(gemm_name: str, reduction: Any) -> str | None:
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen

        expression = reduction.synthetic_element or reduction.source
        try:
            return LoopIRCuteDSLCodegen.source_from_expression(
                gemm_name, expression, "_local_reduce_source"
            )
        except NotImplementedError:
            return None

    @classmethod
    def _synthetic_feed_config(
        cls,
        context: NVGemmEpilogueCapture,
        scheduler_node: BaseSchedulerNode,
        buffer: ComputedBuffer,
    ) -> GemmReductionConfig | None:
        analysis = context.analysis
        if analysis is None:
            return None
        gemm_name = context.gemm.get_name()
        reads = OrderedSet(read.name for read in scheduler_node.read_writes.reads)
        if reads != OrderedSet((gemm_name,)):
            return None
        try:
            m, n = map(V.graph.sizevars.optimization_hint, context.gemm.get_size())
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        inferred = analysis.synthetic_reduction_region(
            buffer.get_name(),
            gemm_name,
            V.graph.get_dtype(gemm_name),
            n,
        )
        if inferred is None:
            return None
        geometry, region = inferred.geometry, inferred.region
        if (m, n)[
            geometry.axis
        ] % geometry.group != 0 or not geometry.matches_output_shape(
            buffer.get_size(), context.gemm.get_size()
        ):
            return None
        reduction = region.reductions[0]
        generated_source = cls._reduction_source_fn(gemm_name, reduction)
        if generated_source is None or reduction.reduction_type not in (
            "sum",
            "mean",
            "prod",
            "max",
            "min",
        ):
            return None
        return GemmReductionConfig(
            output_name=buffer.get_name(),
            group=geometry.group,
            axis=geometry.axis,
            reduction_type=cast(GemmReductionType, reduction.reduction_type),
            source_fn=generated_source,
        )

    @classmethod
    def _feed_main_config(
        cls, context: NVGemmEpilogueCapture
    ) -> GemmReductionConfig | None:
        analysis = context.analysis
        if analysis is None:
            return None
        gemm_name = context.gemm.get_name()

        for scheduler_node, buffer in zip(context.nodes, analysis.buffers):
            if not isinstance(buffer.data, Pointwise):
                continue
            reads = OrderedSet(read.name for read in scheduler_node.read_writes.reads)
            if gemm_name not in reads:
                continue
            for reduction_node in context.nodes:
                config = cls._local_reduction_config(context, reduction_node)
                if config is None or not config.geometry.matches_output_shape(
                    buffer.get_size(), context.gemm.get_size()
                ):
                    continue
                if config.output_name in reads:
                    if reads == OrderedSet((gemm_name, config.output_name)):
                        return dataclasses.replace(
                            config, output_name=buffer.get_name()
                        )
                    continue
                region = analysis.reduction_region(
                    buffer.get_name(),
                    gemm_name,
                    config.group,
                    V.graph.get_dtype(gemm_name),
                )
                if region is None or len(region.reductions) != 1:
                    continue
                reduction = region.reductions[0]
                axis = grouped_reduction_axis_ir(
                    reduction,
                    config.group,
                    V.graph.sizevars.optimization_hint(context.gemm.get_size()[1]),
                )
                if (
                    axis == config.axis
                    and reduction.reduction_type == config.reduction_type
                    and cls._reduction_source_fn(gemm_name, reduction)
                    == config.source_fn
                ):
                    return dataclasses.replace(config, output_name=buffer.get_name())
            if config := cls._synthetic_feed_config(context, scheduler_node, buffer):
                return config
        return None

    @staticmethod
    def _pointwise_finalizer_match(
        source_node: BaseSchedulerNode,
        finalizer_node: BaseSchedulerNode,
        *,
        analysis: GemmEpilogueIRAnalysis,
    ) -> GemmEpilogueIRStore | None:
        source_nodes = source_node.get_nodes()
        finalizer_nodes = finalizer_node.get_nodes()
        if len(source_nodes) != 1 or len(finalizer_nodes) != 1:
            return None
        source = source_nodes[0].node
        finalizer = finalizer_nodes[0].node
        if not (
            isinstance(source, ComputedBuffer)
            and isinstance(finalizer, ComputedBuffer)
            and isinstance(finalizer.data, Pointwise)
        ):
            return None
        reads = list(finalizer_nodes[0].read_writes.reads)
        store = analysis.store(finalizer.get_name())
        if (
            store is not None
            and bool(reads)
            and all(read.name == source.get_name() for read in reads)
            and V.graph.sizevars.statically_known_list_equals(
                source.get_size(), finalizer.get_size()
            )
        ):
            return store
        return None

    @classmethod
    def _local_reduction_config(
        cls,
        context: NVGemmEpilogueCapture,
        node: BaseSchedulerNode,
    ) -> GemmReductionConfig | None:
        analysis = context.analysis
        if analysis is None:
            return None
        gemm = context.gemm
        return cls._grouped_reduce_config(
            gemm, node, analysis
        ) or cls._secondary_bool_output_config(gemm, node, analysis)

    @classmethod
    def _reduction_region(
        cls,
        source: BaseSchedulerNode,
        config: GemmReductionConfig,
        candidates: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis,
    ) -> NVGemmReductionRegion:
        source_buffers = _computed_buffers(source.get_nodes())
        if source_buffers is not None and len(source_buffers) == 2:
            finalizer = analysis.reduction_finalizer(
                config.output_name, source_buffers[0].get_name()
            )
            return NVGemmReductionRegion(
                config=config,
                nodes=(source,),
                finalizer=(
                    finalizer
                    if finalizer is not None and finalizer.materialize
                    else None
                ),
            )
        matches = []
        for candidate in candidates:
            if candidate is source:
                continue
            store = cls._pointwise_finalizer_match(source, candidate, analysis=analysis)
            if store is None:
                continue
            buffer = cast(ComputedBuffer, candidate.get_nodes()[0].node)
            finalizer = analysis.reduction_finalizer(
                buffer.get_name(), config.output_name
            )
            if finalizer is not None:
                matches.append((candidate, finalizer))
        if len(matches) != 1:
            return NVGemmReductionRegion(config=config, nodes=(source,))
        candidate, finalizer = matches[0]
        finalizer_fn = None
        if finalizer.materialize:
            buffer = cast(ComputedBuffer, candidate.get_nodes()[0].node)
            from torch._inductor.kernel.loop_ir_cutedsl_codegen import (
                LoopIRCuteDSLCodegen,
            )

            try:
                finalizer_fn = LoopIRCuteDSLCodegen.finalizer_from_buffer(
                    finalizer.source_name,
                    buffer,
                    "_local_reduce_finalize",
                )
            except NotImplementedError:
                return NVGemmReductionRegion(config=config, nodes=(source,))
        config = dataclasses.replace(
            config,
            output_name=finalizer.output_name,
            finalizer_fn=finalizer_fn,
        )
        return NVGemmReductionRegion(
            config=config,
            nodes=(source, candidate),
            finalizer=finalizer if finalizer.materialize else None,
        )

    @classmethod
    def _partition_local_reductions(
        cls, context: NVGemmEpilogueCapture
    ) -> NVGemmReductionPartition:
        regions: list[NVGemmReductionRegion] = []
        claimed: OrderedSet[BaseSchedulerNode] = OrderedSet()
        if context.analysis is None:
            return NVGemmReductionPartition(())
        analysis = context.analysis

        for node in context.nodes:
            if node in claimed:
                continue
            config = cls._local_reduction_config(context, node)
            if config is None:
                continue
            candidates = tuple(
                candidate for candidate in context.nodes if candidate not in claimed
            )
            region = cls._reduction_region(node, config, candidates, analysis)
            claimed.update(region.nodes)
            regions.append(region)
        return NVGemmReductionPartition(tuple(regions))

    @classmethod
    def _lower_epilogue(
        cls,
        gemm_node: Buffer,
        epilogue_nodes: Sequence[BaseSchedulerNode],
    ) -> NVGemmEpilogueProgram:
        context = NVGemmEpilogueCapture.from_nodes(gemm_node, epilogue_nodes)
        reduction_partition = cls._partition_local_reductions(context)
        feed_main = cls._feed_main_config(context)
        feed_plan = (
            cls._feed_plan(context, feed_main) if feed_main is not None else None
        )
        if feed_plan is not None and feed_plan.plan.consumer_fn is None:
            feed_plan = None
        reduction_plan = (
            None
            if feed_plan is not None
            else cls._generated_epilogue_reduction_plan(context, reduction_partition)
        )
        if reduction_plan is None:
            reduction_plan = cls._static_reduction_plan(
                context, reduction_partition, feed_plan
            )
        return NVGemmEpilogueProgram(
            capture=context,
            reduction_partition=reduction_partition,
            reduction_plan=reduction_plan,
            intermediate_outputs=(
                feed_plan.intermediate_outputs if feed_plan is not None else ()
            ),
        )

    @staticmethod
    def _generated_epilogue_reduction_plan(
        context: NVGemmEpilogueCapture,
        partition: NVGemmReductionPartition,
    ) -> GemmReductionPlan | None:
        analysis = context.analysis
        if analysis is None:
            return None
        try:
            m, n = map(V.graph.sizevars.optimization_hint, context.gemm.get_size())
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        candidates: dict[str, tuple[ComputedBuffer, GemmReductionGeometry]] = {}

        def add_candidate(
            buffer: ComputedBuffer, geometry: GemmReductionGeometry
        ) -> None:
            output_size = tuple(buffer.get_size())
            if V.graph.sizevars.statically_known_equals(math.prod(output_size), m * n):
                return
            compressed_size = _compressed_output_size(output_size)
            if compressed_size is None:
                return
            expected_size = (
                (m, n // geometry.group)
                if geometry.axis == 1
                else (m // geometry.group, n)
            )
            if V.graph.sizevars.statically_known_list_equals(
                compressed_size, expected_size
            ):
                candidates[buffer.get_name()] = (buffer, geometry)

        if len(partition.regions) == 1:
            region = partition.regions[0]
            buffer = next(
                (
                    candidate
                    for candidate in analysis.buffers
                    if candidate.get_name() == region.config.output_name
                ),
                None,
            )
            if buffer is not None:
                add_candidate(buffer, region.config.geometry)

        for node in context.nodes:
            buffer = node.node
            if not isinstance(buffer, ComputedBuffer):
                continue
            output_name = buffer.get_name()
            match = analysis.synthetic_reduction_program(
                output_name,
                context.gemm.get_name(),
                context.gemm.get_dtype(),
                n,
            )
            if (
                match is None
                or len(match.region.reductions) <= 1
                or match.geometry.needs_physical_callbacks
                or any(
                    reduction.reduction_type
                    not in ("sum", "mean", "prod", "max", "min")
                    or reduction.synthetic_element is None
                    for reduction in match.region.reductions
                )
            ):
                continue
            add_candidate(buffer, match.geometry)
        if len(candidates) != 1:
            return None
        buffer, geometry = next(iter(candidates.values()))
        combine_fn = None
        finalizer_fn = None
        if geometry.needs_physical_callbacks:
            from torch._inductor.kernel.loop_ir_cutedsl_codegen import (
                LoopIRCuteDSLCodegen,
            )

            try:
                combine_fn, finalizer_fn = LoopIRCuteDSLCodegen.reduction_callbacks(
                    context.gemm.get_name(),
                    analysis,
                    buffer.get_name(),
                    geometry,
                )
            except NotImplementedError:
                return None
        return GemmReductionPlan(
            reduction_output=buffer.get_name(),
            primary_output=context.gemm.get_name(),
            group=geometry.group,
            axis=geometry.axis,
            reduction_type=None,
            source_fn=None,
            combine_fn=combine_fn,
            finalizer_fn=finalizer_fn,
        )

    @classmethod
    def _feed_plan(
        cls,
        context: NVGemmEpilogueCapture,
        feed_main: GemmReductionConfig,
    ) -> NVGemmFeedPlan | None:
        gemm_node = context.gemm
        nodes = context.nodes
        analysis = context.analysis
        if analysis is None:
            return None
        output_name = feed_main.output_name
        feed_output = (
            output_name
            if V.graph.get_dtype(output_name) != gemm_node.get_dtype()
            else None
        )
        feed_reads = next(
            (
                OrderedSet(
                    read.name
                    for read in scheduler_node.read_writes.reads
                    if read.name != gemm_node.get_name()
                )
                for scheduler_node in nodes
                if isinstance(scheduler_node.node, Buffer)
                and scheduler_node.node.get_name() == output_name
            ),
            OrderedSet(),
        )
        from torch._inductor.kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen

        buffers = {
            scheduler_node.node.get_name(): scheduler_node.node
            for scheduler_node in nodes
            if isinstance(scheduler_node.node, ComputedBuffer)
        }
        reduction_name = next(iter(feed_reads)) if feed_reads else None

        def consumer_source(buffer: ComputedBuffer) -> str | None:
            if len(feed_reads) > 1:
                return None
            try:
                return LoopIRCuteDSLCodegen.consumer_from_buffer(
                    gemm_node.get_name(),
                    reduction_name,
                    buffer,
                    "_local_reduce_consumer",
                    feed_main.group,
                )
            except NotImplementedError:
                return None

        matched_buffer = buffers.get(output_name)
        matched_source = (
            consumer_source(matched_buffer) if matched_buffer is not None else None
        )
        consumer_finalizer_fn = None
        if reduction_name is None:
            consumer_finalizer_fn = LoopIRCuteDSLCodegen.logical_reduction_finalizer(
                feed_main.reduction_type,
                "_local_reduce_consumer_finalize",
            )
        else:
            try:
                _, consumer_finalizer_fn = LoopIRCuteDSLCodegen.reduction_callbacks(
                    gemm_node.get_name(),
                    analysis,
                    reduction_name,
                    feed_main.geometry,
                )
            except NotImplementedError:
                return None
        equivalent = []
        secondary = None
        secondary_consumer = None
        geometry = feed_main.geometry
        for scheduler_node in nodes:
            buffer = scheduler_node.node
            if not (
                isinstance(buffer, ComputedBuffer)
                and isinstance(buffer.data, Pointwise)
                and buffer.get_name() != output_name
                and geometry.matches_output_shape(
                    buffer.get_size(), gemm_node.get_size()
                )
            ):
                continue
            candidate_reads = OrderedSet(
                read.name
                for read in scheduler_node.read_writes.reads
                if read.name != gemm_node.get_name()
            )
            if candidate_reads != feed_reads:
                continue
            if not feed_reads:
                candidate_reduction = cls._synthetic_feed_config(
                    context, scheduler_node, buffer
                )
                if candidate_reduction is not None and (
                    candidate_reduction.geometry != geometry
                    or candidate_reduction.reduction_type != feed_main.reduction_type
                    or candidate_reduction.source_fn != feed_main.source_fn
                ):
                    return None
            candidate_source = consumer_source(buffer)
            if candidate_source is None:
                continue
            if candidate_source == matched_source:
                equivalent.append(buffer.get_name())
            elif secondary is None:
                secondary = buffer.get_name()
                secondary_consumer = candidate_source
            else:
                return None
        for output in equivalent:
            if feed_output is None:
                feed_output = output
            elif output != feed_output and secondary is None:
                secondary = output
                secondary_consumer = matched_source
            elif output != feed_output:
                return None
        return NVGemmFeedPlan(
            plan=GemmReductionPlan.from_config(
                feed_main,
                reduction_output=None,
                primary_output=(
                    gemm_node.get_name()
                    if feed_output is not None
                    else feed_main.output_name
                ),
                feeds_main=True,
                feed_output=feed_output,
                secondary_feed_output=secondary,
                consumer_fn=matched_source,
                consumer_finalizer_fn=consumer_finalizer_fn,
                secondary_consumer_fn=secondary_consumer,
            ),
            intermediate_outputs=tuple(feed_reads),
        )

    @staticmethod
    def _static_reduction_plan(
        context: NVGemmEpilogueCapture,
        partition: NVGemmReductionPartition,
        feed_plan: NVGemmFeedPlan | None,
    ) -> GemmReductionPlan | None:
        reductions = partition.configs
        if len(reductions) > 1:
            return None
        gemm_name = context.gemm.get_name()
        local_reduce = None
        if reductions:
            config = reductions[0]
            is_secondary_feed = config.secondary_consumer_fn is not None
            local_reduce = GemmReductionPlan.from_config(
                config,
                reduction_output=None if is_secondary_feed else config.output_name,
                primary_output=gemm_name,
                secondary_feed_output=(
                    config.output_name if is_secondary_feed else None
                ),
            )
        if feed_plan is None:
            return local_reduce
        plan = feed_plan.plan
        if local_reduce is None:
            return plan
        if local_reduce.secondary_feed_output is not None:
            if plan.secondary_feed_output is not None:
                return None
            return dataclasses.replace(
                plan,
                secondary_feed_output=local_reduce.secondary_feed_output,
                secondary_consumer_fn=local_reduce.secondary_consumer_fn,
            )
        if (
            local_reduce.geometry != plan.geometry
            or local_reduce.reduction_type != plan.reduction_type
            or local_reduce.source_fn != plan.source_fn
            or (
                local_reduce.finalizer_fn is not None
                and plan.finalizer_fn is not None
                and local_reduce.finalizer_fn != plan.finalizer_fn
            )
        ):
            return None
        return dataclasses.replace(
            plan,
            reduction_output=local_reduce.reduction_output,
            finalizer_fn=local_reduce.finalizer_fn or plan.finalizer_fn,
        )


def _computed_buffers(
    nodes: Sequence[BaseSchedulerNode],
) -> tuple[ComputedBuffer, ...] | None:
    buffers = tuple(node.node for node in nodes)
    if not buffers or not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
        return None
    return cast(tuple[ComputedBuffer, ...], buffers)


def _compressed_output_size(shape: Sequence[Any]) -> tuple[Any, Any] | None:
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) != 3:
        return None
    if V.graph.sizevars.statically_known_equals(shape[-1], 1):
        return shape[0], shape[1]
    if V.graph.sizevars.statically_known_equals(shape[1], 1):
        return shape[0], shape[2]
    return None
