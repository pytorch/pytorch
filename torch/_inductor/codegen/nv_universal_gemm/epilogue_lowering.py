# mypy: allow-untyped-defs
"""Recognize NVGEMM epilogues and lower them to shared contracts."""

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._ordered_set import OrderedSet

from ...ir import Buffer, ComputedBuffer, MultiOutputReduction, Pointwise, Reduction
from ...kernel.gemm_epilogue import (
    GemmReductionConfig,
    GemmReductionDescriptor,
    GemmReductionGeometry,
    GemmReductionPlan,
)
from ...kernel.loop_ir_epilogue_lowering import (
    GemmEpilogueIRAnalysis,
    GemmEpilogueIRStore,
    grouped_reduction_axis_ir,
    is_direct_bool_gt_zero_ir,
    is_logsumexp_ir,
    is_softmax_ir,
    variance_parameters_ir,
)
from ...scheduler import BaseSchedulerNode
from ...virtualized import V


NVGEMM_SOFTMAX_GROUP_LIMIT = 32


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmFeedPlan:
    plan: GemmReductionPlan
    intermediate_outputs: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmReductionRegion:
    config: GemmReductionConfig
    nodes: tuple[BaseSchedulerNode, ...]
    finalizer: "NVGemmReductionFinalizer | None" = None


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

    @property
    def finalizers(self) -> tuple["NVGemmReductionFinalizer", ...]:
        return tuple(
            region.finalizer for region in self.regions if region.finalizer is not None
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
    def supported(self) -> bool:
        """Whether every claimed reduction has a backend lowering contract."""
        return not self.has_unclaimed_reduction and (
            not self.reduction_partition.configs or self.reduction_plan is not None
        )

    @property
    def has_unclaimed_reduction(self) -> bool:
        analysis = self.capture.analysis
        if analysis is None:
            return False
        try:
            n = V.graph.sizevars.optimization_hint(self.capture.gemm.get_size()[1])
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return True
        gemm_name = self.capture.gemm.get_name()
        gemm_dtype = self.capture.gemm.get_dtype()
        return any(
            isinstance(node.node, ComputedBuffer)
            and analysis.synthetic_reduction_region(
                node.node.get_name(), gemm_name, gemm_dtype, n
            )
            is not None
            for node in self.pointwise_nodes
        )

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
        owned = OrderedSet(self.reduction_partition.nodes)
        reduction_plan = self.reduction_plan
        feed_names = (
            (
                reduction_plan.primary_output,
                *reduction_plan.auxiliary_outputs,
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

    @property
    def has_gemm_read(self) -> bool:
        name = self.gemm.get_name()
        return any(
            read.name == name for node in self.nodes for read in node.read_writes.reads
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmReductionFinalizer:
    source_name: str
    buffer: ComputedBuffer


@dataclasses.dataclass(frozen=True)
class NVGemmGroupedPointwiseCandidate:
    buffer: ComputedBuffer
    geometry: GemmReductionGeometry
    store: GemmEpilogueIRStore

    def config(self, reduction_type: str) -> GemmReductionConfig:
        return GemmReductionConfig(
            output_name=self.buffer.get_name(),
            group=self.geometry.group,
            axis=self.geometry.axis,
            reduction_type=reduction_type,
            source_type="identity",
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
    def _softmax_config(
        cls, context: NVGemmEpilogueCapture
    ) -> GemmReductionConfig | None:
        if (
            len(context.gemm.get_size()) != 2
            or context.analysis is None
            or not context.has_gemm_read
        ):
            return None
        return (
            cls._online_softmax_reduction_config(context)
            or cls._pointwise_softmax_config(context)
            or cls._chained_softmax_config(context)
        )

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

        reduction_ir = analysis.grouped_reduction(
            node.get_name(),
            gemm_node.get_name(),
            group,
            axis,
            gemm_node.get_dtype(),
        )
        if reduction_ir is None:
            return None
        physical_type = reduction_ir.reduction_type
        source_type = reduction_ir.source_type
        if physical_type not in ("sum", "mean", "prod", "max", "min"):
            return None
        if isinstance(node.data, Pointwise):
            reduction_type = physical_type
        elif physical_type != reduction_type:
            return None
        finalizer = (
            analysis.reduction_finalizer(output_name, node.get_name(), group)
            if finalizer_store is not None
            else None
        )
        if finalizer_store is not None and (
            finalizer is None or finalizer.kind == "generic"
        ):
            return None
        if finalizer is not None and finalizer.kind == "mean":
            if physical_type != "sum":
                return None
            reduction_type = "mean"
        elif finalizer is not None and finalizer.kind == "absmax_scale":
            if (physical_type, source_type) != ("max", "abs"):
                return None
            source_type = "abs_scale"
        if reduction_type is None:
            return None

        if isinstance(node.data, Reduction):
            if expected_strides is None:
                return None
            reads = list(access_node.read_writes.reads)
            if len(reads) != 1 or reads[0].name != gemm_node.get_name():
                return None
            range_vars = access_node.read_writes.range_vars
            if range_vars is None:
                return None
            if not range_vars:
                return GemmReductionConfig(
                    output_name=output_name,
                    group=group,
                    axis=axis,
                    reduction_type=reduction_type,
                    source_type=source_type,
                )
            strides = V.graph.sizevars.stride_vars(reads[0].index, range_vars)
            expected_stride_options = [expected_strides]
            if axis == 1:
                expected_stride_options.append([n, 1])
            else:
                expected_stride_options.append([1, n])
            if not any(
                len(strides) == len(expected)
                and all(
                    known_equals(stride, expected_stride)
                    for stride, expected_stride in zip(strides, expected)
                )
                for expected in expected_stride_options
            ):
                return None
        return GemmReductionConfig(
            output_name=output_name,
            group=group,
            axis=axis,
            reduction_type=reduction_type,
            source_type=source_type,
        )

    @classmethod
    def _n_axis_grouped_pointwise_candidate(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis,
    ) -> NVGemmGroupedPointwiseCandidate | None:
        buffer = _single_computed_buffer(scheduler_node)
        if buffer is None or not isinstance(buffer.data, Pointwise):
            return None
        geometry = cls._grouped_pointwise_geometry(gemm_node, buffer, scheduler_node)
        if geometry is None or geometry.axis != 1:
            return None
        store = analysis.store(buffer.get_name())
        return (
            NVGemmGroupedPointwiseCandidate(buffer, geometry, store)
            if store is not None
            else None
        )

    @classmethod
    def _grouped_variance_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis,
    ) -> GemmReductionConfig | None:
        candidate = cls._n_axis_grouped_pointwise_candidate(
            gemm_node, scheduler_node, analysis
        )
        if candidate is None:
            return None
        parameters = variance_parameters_ir(
            candidate.store, gemm_node.get_name(), candidate.geometry.group
        )
        if parameters is None:
            return None
        reduce_type = GemmReductionDescriptor("variance_affine", parameters).serialize()
        return candidate.config(reduce_type)

    @classmethod
    def _direct_bool_mask_config(
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
        store = analysis.store(buffer.get_name())
        if store is None or not is_direct_bool_gt_zero_ir(store, gemm_node.get_name()):
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
        return GemmReductionConfig(
            output_name=buffer.get_name(),
            group=2,
            axis=1,
            reduction_type="direct_bool_gt_zero",
            source_type="identity",
        )

    @classmethod
    def _grouped_logsumexp_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis,
    ) -> GemmReductionConfig | None:
        candidate = cls._n_axis_grouped_pointwise_candidate(
            gemm_node, scheduler_node, analysis
        )
        if candidate is None:
            return None
        if not is_logsumexp_ir(
            candidate.store, gemm_node.get_name(), candidate.geometry.group
        ):
            return None
        return candidate.config("logsumexp")

    @classmethod
    def _online_softmax_reduction_config(
        cls, context: NVGemmEpilogueCapture
    ) -> GemmReductionConfig | None:
        analysis = context.analysis
        if len(context.nodes) != 1 or analysis is None:
            return None
        buffer = analysis.buffers[0]
        if not isinstance(buffer.data, MultiOutputReduction):
            return None
        group = _online_softmax_group(buffer.data)
        if group is None:
            return None
        return GemmReductionConfig(
            output_name=buffer.get_name(),
            group=group,
            axis=1,
            reduction_type="online_softmax",
            source_type="identity",
        )

    @classmethod
    def _pointwise_softmax_config(
        cls, context: NVGemmEpilogueCapture
    ) -> GemmReductionConfig | None:
        analysis = context.analysis
        if len(context.nodes) != 1 or analysis is None:
            return None
        buffer = analysis.buffers[0]
        if not isinstance(buffer.data, Pointwise):
            return None
        layout = GemmReductionGeometry.from_output_shape(
            buffer.get_size(), context.gemm.get_size()
        )
        if (
            layout is None
            or layout.axis != 1
            or layout.group > NVGEMM_SOFTMAX_GROUP_LIMIT
        ):
            return None
        store = analysis.store(buffer.get_name())
        if (
            store is None
            or not is_softmax_ir(store, context.gemm.get_name(), layout.group)
            or not context.nodes[0].read_writes.reads
            or any(
                read.name != context.gemm.get_name()
                for read in context.nodes[0].read_writes.reads
            )
        ):
            return None
        return GemmReductionConfig(
            output_name=buffer.get_name(),
            group=layout.group,
            axis=1,
            reduction_type="online_softmax",
            source_type="identity",
        )

    @classmethod
    def _chained_softmax_config(
        cls, context: NVGemmEpilogueCapture
    ) -> GemmReductionConfig | None:
        analysis = context.analysis
        if len(context.nodes) != 3 or analysis is None:
            return None
        buffers = analysis.buffers
        if not (
            all(isinstance(buffer.data, MultiOutputReduction) for buffer in buffers[:2])
            and isinstance(buffers[2].data, Pointwise)
        ):
            return None
        reductions: list[MultiOutputReduction] = [
            cast(MultiOutputReduction, buffer.data) for buffer in buffers[:2]
        ]
        groups = tuple(_online_softmax_group(reduction) for reduction in reductions)
        group = groups[0]
        if group is None or any(candidate != group for candidate in groups[1:]):
            return None
        if not V.graph.sizevars.statically_known_equals(
            reductions[0].reduction_ranges[0], group
        ):
            return None
        try:
            m, n = map(V.graph.sizevars.optimization_hint, context.gemm.get_size())
            reduction_sizes = [
                tuple(map(V.graph.sizevars.optimization_hint, reduction.ranges))
                for reduction in reductions
            ]
            output_size = tuple(
                map(
                    V.graph.sizevars.optimization_hint,
                    buffers[2].get_size(),
                )
            )
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        if (
            n % group != 0
            or any(size != (m, n // group, 1) for size in reduction_sizes)
            or output_size != (m, n // group, group)
            or reductions[0].reduction_ranges != reductions[1].reduction_ranges
        ):
            return None
        finalizer_store = analysis.store(buffers[2].get_name())
        if finalizer_store is None or not is_softmax_ir(
            finalizer_store,
            context.gemm.get_name(),
            group,
            frozenset(buffer.get_name() for buffer in buffers[:2]),
        ):
            return None
        return GemmReductionConfig(
            output_name=buffers[2].get_name(),
            group=group,
            axis=1,
            reduction_type="online_softmax",
            source_type="identity",
        )

    @classmethod
    def _feed_main_config(
        cls, context: NVGemmEpilogueCapture, *, allow_softmax: bool = True
    ) -> GemmReductionConfig | None:
        matched = cls._generic_feed_main_config(context)
        if matched is not None:
            return matched
        return cls._softmax_config(context) if allow_softmax else None

    @classmethod
    def _generic_feed_main_config(
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
                    and reduction.source_type == config.source_type
                ):
                    return dataclasses.replace(config, output_name=buffer.get_name())
            if reads != OrderedSet((gemm_name,)):
                continue
            try:
                m, n = map(V.graph.sizevars.optimization_hint, context.gemm.get_size())
            except (GuardOnDataDependentSymNode, TypeError, ValueError):
                continue
            inferred = analysis.synthetic_reduction_region(
                buffer.get_name(),
                gemm_name,
                V.graph.get_dtype(gemm_name),
                n,
            )
            if inferred is None:
                continue
            geometry, region = inferred.geometry, inferred.region
            if (m, n)[geometry.axis] % geometry.group != 0:
                continue
            if not geometry.matches_output_shape(
                buffer.get_size(), context.gemm.get_size()
            ):
                continue
            reduction = region.reductions[0]
            if reduction.source_type is None:
                continue
            return GemmReductionConfig(
                output_name=buffer.get_name(),
                group=geometry.group,
                axis=geometry.axis,
                reduction_type=reduction.reduction_type,
                source_type=reduction.source_type,
            )
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
        return (
            cls._grouped_reduce_config(gemm, node, analysis)
            or cls._grouped_variance_config(gemm, node, analysis)
            or cls._grouped_logsumexp_config(gemm, node, analysis)
            or cls._direct_bool_mask_config(gemm, node, analysis)
        )

    @classmethod
    def _reduction_region(
        cls,
        source: BaseSchedulerNode,
        config: GemmReductionConfig,
        candidates: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis,
    ) -> NVGemmReductionRegion:
        matches = []
        for candidate in candidates:
            if candidate is source:
                continue
            store = cls._pointwise_finalizer_match(source, candidate, analysis=analysis)
            if store is None:
                continue
            buffer = cast(ComputedBuffer, candidate.get_nodes()[0].node)
            finalizer = analysis.reduction_finalizer(
                buffer.get_name(), config.output_name, config.group
            )
            if finalizer is not None:
                matches.append((candidate, buffer, finalizer))
        if len(matches) != 1:
            return NVGemmReductionRegion(config=config, nodes=(source,))
        source_name = config.output_name
        candidate, buffer, finalizer = matches[0]
        materialize = finalizer.kind != "identity"
        config = dataclasses.replace(config, output_name=buffer.get_name())
        generated_finalizer = (
            NVGemmReductionFinalizer(source_name=source_name, buffer=buffer)
            if materialize
            else None
        )
        return NVGemmReductionRegion(
            config=config,
            nodes=(source, candidate),
            finalizer=generated_finalizer,
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
        if (
            feed_plan is not None
            and feed_plan.plan.reduction_type not in ("online_softmax", "logsumexp")
            and feed_plan.plan.consumer_fn is None
        ):
            feed_plan = None
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
    def _feed_plan(
        context: NVGemmEpilogueCapture,
        feed_main: GemmReductionConfig,
    ) -> NVGemmFeedPlan | None:
        gemm_node = context.gemm
        nodes = context.nodes
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

        def consumer_source(buffer: ComputedBuffer) -> str | None:
            if len(feed_reads) > 1:
                return None
            reduction_name = next(iter(feed_reads)) if feed_reads else None
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
        equivalent = []
        secondary = None
        secondary_type = None
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
                secondary_type = feed_main.reduction_type
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
                secondary_feed_type=secondary_type,
                consumer_fn=matched_source,
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
            is_secondary_feed = config.reduction_type == "direct_bool_gt_zero"
            local_reduce = GemmReductionPlan.from_config(
                config,
                reduction_output=None if is_secondary_feed else config.output_name,
                primary_output=gemm_name,
                secondary_feed_output=(
                    config.output_name if is_secondary_feed else None
                ),
                secondary_feed_type=(
                    config.reduction_type if is_secondary_feed else None
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
                secondary_feed_type=local_reduce.secondary_feed_type,
            )
        if (
            local_reduce.geometry != plan.geometry
            or local_reduce.reduction_type != plan.reduction_type
            or local_reduce.source_type != plan.source_type
        ):
            return None
        return dataclasses.replace(plan, reduction_output=local_reduce.reduction_output)


def _computed_buffers(
    nodes: Sequence[BaseSchedulerNode],
) -> tuple[ComputedBuffer, ...] | None:
    buffers = tuple(node.node for node in nodes)
    if not buffers or not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
        return None
    return cast(tuple[ComputedBuffer, ...], buffers)


def _single_computed_buffer(node: BaseSchedulerNode) -> ComputedBuffer | None:
    nodes = node.get_nodes()
    if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
        return None
    return nodes[0].node


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


def _online_softmax_group(reduction: MultiOutputReduction) -> int | None:
    if (
        reduction.reduction_type != "online_softmax_reduce"
        or len(reduction.reduction_ranges) != 1
    ):
        return None
    extent = reduction.reduction_ranges[0]
    try:
        group = V.graph.sizevars.optimization_hint(extent)
    except (GuardOnDataDependentSymNode, TypeError, ValueError):
        return None
    if group > NVGEMM_SOFTMAX_GROUP_LIMIT:
        return None
    return group
