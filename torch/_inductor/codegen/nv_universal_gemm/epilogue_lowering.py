# mypy: allow-untyped-defs
"""Recognize NVGEMM epilogues and lower them to shared contracts."""

import dataclasses
import enum
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._ordered_set import OrderedSet

from ...ir import Buffer, ComputedBuffer, MultiOutputReduction, Pointwise, Reduction
from ...kernel.gemm_epilogue import (
    GemmReductionConfig,
    GemmReductionGeometry,
    GemmReductionPlan,
)
from ...kernel.loop_ir_epilogue_lowering import (
    centered_mean_consumer_type_ir,
    centered_mean_consumer_type_unrolled_ir,
    GemmEpilogueIRAnalysis,
    GemmEpilogueIRStore,
    is_absmax_normalize_ir,
    is_absmax_scale_finalizer_ir,
    is_direct_bool_gt_zero_ir,
    is_logsumexp_ir,
    is_softmax_ir,
    operation_names_ir,
    sum_multiply_consumer_type_ir,
    sum_normalize_consumer_type_ir,
    variance_parameters_ir,
)
from ...scheduler import BaseSchedulerNode
from ...virtualized import V


@dataclasses.dataclass(frozen=True)
class NVGemmFeedOutputs:
    matched: str
    typed: str | None
    secondary: str | None
    secondary_type: str | None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (self.matched, self.typed, self.secondary)
            if name is not None
        )


@dataclasses.dataclass(frozen=True)
class NVGemmEpilogueProgram:
    """Semantic NVGEMM epilogue IR produced before scheduling policy."""

    nodes: tuple[BaseSchedulerNode, ...]
    reductions: tuple[GemmReductionConfig, ...]
    reduction_nodes: tuple[BaseSchedulerNode, ...]
    feed_main: GemmReductionConfig | None
    feed_outputs: NVGemmFeedOutputs | None
    reduction_plan: GemmReductionPlan | None
    min_tile_n: int
    finalizers: tuple["NVGemmReductionFinalizer", ...] = ()

    @property
    def owned_nodes(self) -> tuple[BaseSchedulerNode, ...]:
        owned = OrderedSet(self.reduction_nodes)
        feed_names = self.feed_outputs.names if self.feed_outputs is not None else ()
        owned.update(
            node
            for node in self.nodes
            if (isinstance(node.node, Buffer) and node.node.get_name() in feed_names)
            or (
                self.feed_main is not None
                and self.feed_main.reduction_type == "online_softmax"
                and isinstance(node.node, ComputedBuffer)
                and isinstance(node.node.data, MultiOutputReduction)
            )
        )
        return tuple(owned)

    @property
    def evt_nodes(self) -> tuple[BaseSchedulerNode, ...]:
        owned = OrderedSet(self.owned_nodes)
        return tuple(node for node in self.nodes if node not in owned)


@dataclasses.dataclass(frozen=True)
class NVGemmEpiloguePatternContext:
    """Normalized candidate view shared by declarative epilogue patterns."""

    gemm: Buffer
    nodes: tuple[BaseSchedulerNode, ...]
    buffers: tuple[ComputedBuffer, ...] | None
    analysis: GemmEpilogueIRAnalysis | None

    @classmethod
    def from_nodes(
        cls,
        gemm: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> "NVGemmEpiloguePatternContext":
        normalized_nodes = tuple(nodes)
        raw_buffers = tuple(node.node for node in normalized_nodes)
        buffers = (
            cast(tuple[ComputedBuffer, ...], raw_buffers)
            if raw_buffers
            and all(isinstance(buffer, ComputedBuffer) for buffer in raw_buffers)
            else None
        )
        if analysis is None and buffers is not None:
            analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        return cls(gemm, normalized_nodes, buffers, analysis)

    @property
    def has_gemm_read(self) -> bool:
        name = self.gemm.get_name()
        return any(
            read.name == name for node in self.nodes for read in node.read_writes.reads
        )


PatternMatcher = Callable[[NVGemmEpiloguePatternContext], GemmReductionConfig | None]


@dataclasses.dataclass(frozen=True)
class NVGemmEpiloguePattern:
    """Declarative constraints and semantic matcher for one epilogue form."""

    kind: "NVGemmEpiloguePatternKind"
    variant: str
    matcher: PatternMatcher
    constraints: tuple[Callable[[NVGemmEpiloguePatternContext], bool], ...]

    def match(
        self, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        if not all(constraint(context) for constraint in self.constraints):
            return None
        return self.matcher(context)


FinalizerPredicate = Callable[
    [
        type["NVGemmEpilogueLowering"],
        Buffer,
        BaseSchedulerNode,
        BaseSchedulerNode,
        GemmEpilogueIRAnalysis | None,
    ],
    bool,
]
FinalizerRewrite = Callable[[GemmReductionConfig, ComputedBuffer], GemmReductionConfig]


@dataclasses.dataclass(frozen=True)
class NVGemmFinalizerPattern:
    kind: "NVGemmEpiloguePatternKind"
    accepts: Callable[[GemmReductionConfig], bool]
    predicate: FinalizerPredicate
    rewrite: FinalizerRewrite

    def match(
        self,
        lowering: type["NVGemmEpilogueLowering"],
        gemm: Buffer,
        source: BaseSchedulerNode,
        config: GemmReductionConfig,
        candidates: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None,
    ) -> tuple[GemmReductionConfig, BaseSchedulerNode] | None:
        if not self.accepts(config):
            return None
        matches = [
            candidate
            for candidate in candidates
            if candidate is not source
            and self.predicate(lowering, gemm, source, candidate, analysis)
        ]
        if len(matches) != 1:
            return None
        finalizer = matches[0]
        buffer = _single_computed_buffer(finalizer)
        if buffer is None:
            return None
        return self.rewrite(config, buffer), finalizer


@dataclasses.dataclass(frozen=True)
class NVGemmReductionFinalizer:
    source_name: str
    output_name: str
    buffer: ComputedBuffer


class NVGemmEpilogueLowering:
    """Lower scheduler nodes to NVGEMM epilogue semantic plans."""

    # TODO: Share the declarative pattern framework with FlexGEMM once its FX
    # matcher can consume the same normalized capture protocol.

    @staticmethod
    def _match_patterns(
        context: NVGemmEpiloguePatternContext,
        patterns: Sequence[NVGemmEpiloguePattern],
    ) -> GemmReductionConfig | None:
        for pattern in patterns:
            if (result := pattern.match(context)) is not None:
                return result
        return None

    @classmethod
    def _match_pattern(
        cls,
        kind: "NVGemmEpiloguePatternKind",
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        patterns = FEED_MAIN_PATTERN_REGISTRY.get(kind)
        if patterns is None:
            raise RuntimeError(f"unknown NVGEMM epilogue pattern: {kind.name}")
        context = NVGemmEpiloguePatternContext.from_nodes(gemm_node, nodes, analysis)
        return cls._match_patterns(context, patterns)

    @staticmethod
    def _computed_buffers(
        nodes: Sequence[BaseSchedulerNode],
    ) -> tuple[ComputedBuffer, ...] | None:
        buffers = tuple(node.node for node in nodes)
        if not buffers or not all(
            isinstance(buffer, ComputedBuffer) for buffer in buffers
        ):
            return None
        return cast(tuple[ComputedBuffer, ...], buffers)

    @classmethod
    def _grouped_reduce_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) not in (1, 2):
            return None
        buffers = cls._computed_buffers(nodes)
        if buffers is None:
            return None
        node = buffers[0]
        finalizer_match = (
            cls._pointwise_finalizer_match(nodes[0], nodes[1])
            if len(buffers) == 2
            else None
        )
        mean_finalizer = finalizer_match is not None and isinstance(
            node.data, Reduction
        )
        layout_finalizer = finalizer_match is not None and operation_names_ir(
            finalizer_match[2]
        ).issubset(("load", "to_dtype", "to_dtype_bitcast", "identity"))
        absmax_finalizer = finalizer_match is not None and is_absmax_scale_finalizer_ir(
            finalizer_match[2], node.get_name()
        )
        reduction_type = None
        if isinstance(node.data, Reduction):
            reduction_type = node.data.reduction_type
            if mean_finalizer:
                reduction_type = "mean"
        elif not isinstance(node.data, Pointwise):
            return None
        access_node = scheduler_node
        output_name = node.get_name()
        if len(buffers) == 2:
            finalizer = buffers[1]
            if not (
                (reduction_type == "mean" and mean_finalizer)
                or absmax_finalizer
                or layout_finalizer
            ):
                return None
            access_node = nodes[0]
            output_name = finalizer.get_name()
        if len(node.data.ranges) not in (2, 3) or len(gemm_node.get_size()) != 2:
            return None
        m, n = gemm_node.get_size()
        out_size = tuple(node.data.ranges)
        if len(out_size) == 2:
            out_m, out_n = out_size
        elif V.graph.sizevars.statically_known_equals(out_size[-1], 1):
            out_m, out_n, _ = out_size
        elif V.graph.sizevars.statically_known_equals(out_size[1], 1):
            out_m, _, out_n = out_size
        else:
            return None

        def known_equals(left, right) -> bool:
            return (
                V.graph.sizevars.statically_known_equals(left, right)
                or V.graph.sizevars.simplify(left - right) == 0
            )

        if isinstance(node.data, Reduction):
            reduction = node.data
            if (
                reduction.reduction_type
                != ("sum" if reduction_type == "mean" else reduction_type)
                or len(reduction.reduction_ranges) != 1
            ):
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
            try:
                m = V.graph.sizevars.optimization_hint(m)
                n = V.graph.sizevars.optimization_hint(n)
                out_m = V.graph.sizevars.optimization_hint(out_m)
                out_n = V.graph.sizevars.optimization_hint(out_n)
            except (GuardOnDataDependentSymNode, TypeError, ValueError):
                return None
            if m == out_m and out_n > 0 and n % out_n == 0:
                group, axis = n // out_n, 1
                expected_strides = [n, group]
            elif n == out_n and out_m > 0 and m % out_m == 0:
                group, axis = m // out_m, 0
                expected_strides = [group * n, 1]
            else:
                return None
        if group <= 1 or (axis == 0 and group > 128):
            return None

        analysis = analysis or GemmEpilogueIRAnalysis.from_buffers((node,))
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
        if absmax_finalizer:
            if (physical_type, source_type) != ("max", "abs"):
                return None
            source_type = "abs_scale"
        if isinstance(node.data, Pointwise):
            reduction_type = physical_type
        elif physical_type != ("sum" if reduction_type == "mean" else reduction_type):
            return None
        if reduction_type is None:
            return None

        reads = list(access_node.read_writes.reads)
        if not reads or any(read.name != gemm_node.get_name() for read in reads):
            return None
        range_vars = access_node.read_writes.range_vars
        if range_vars is None:
            return None
        if not range_vars:
            return GemmReductionConfig(
                output_name, group, axis, reduction_type, source_type
            )
        if isinstance(node.data, Reduction):
            if len(reads) != 1:
                return None
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
        else:
            if len(reads) != group:
                return None
            expected_base = (
                n * range_vars[0] + group * range_vars[1]
                if axis == 1
                else group * n * range_vars[0] + range_vars[1]
            )
            offsets = []
            for read in reads:
                strides = V.graph.sizevars.stride_vars(read.index, range_vars)
                if list(strides) != expected_strides:
                    return None
                offsets.append(V.graph.sizevars.simplify(read.index - expected_base))
            expected_offsets = OrderedSet(
                offset if axis == 1 else offset * n for offset in range(group)
            )
            if OrderedSet(offsets) != expected_offsets:
                return None
        return GemmReductionConfig(
            output_name, group, axis, reduction_type, source_type
        )

    @classmethod
    def _n_axis_grouped_pointwise_reads_match(
        cls,
        gemm_node: Buffer,
        buffer: ComputedBuffer,
        scheduler_node: BaseSchedulerNode,
        group: int,
    ) -> bool:
        """Validate an unrolled N-axis grouped output against scheduler indexing."""
        try:
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
            out_m, out_n = map(V.graph.sizevars.optimization_hint, buffer.get_size())
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return False
        if out_m != m or out_n <= 0 or n != out_n * group:
            return False
        reads = list(scheduler_node.read_writes.reads)
        range_vars = scheduler_node.read_writes.range_vars
        if not reads or range_vars is None:
            return False
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
            base = n * range_vars[0] + group * range_vars[1]
            expected_strides = [n, group]
        elif len(range_vars) == 1:
            base = group * range_vars[0]
            expected_strides = [group]
        else:
            return False
        offsets = OrderedSet()
        for read in reads:
            if read.name != gemm_node.get_name():
                return False
            strides = V.graph.sizevars.stride_vars(read.index, range_vars)
            if list(strides) != expected_strides:
                return False
            offsets.add(V.graph.sizevars.simplify(read.index - base))
        return offsets == OrderedSet(range(group))

    @classmethod
    def _n_axis_grouped_pointwise_candidate(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> tuple[ComputedBuffer, int, GemmEpilogueIRStore] | None:
        buffers = cls._computed_buffers(scheduler_node.get_nodes())
        if (
            buffers is None
            or len(buffers) != 1
            or not isinstance(buffers[0].data, Pointwise)
        ):
            return None
        buffer = buffers[0]
        try:
            _, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
            _, out_n = map(V.graph.sizevars.optimization_hint, buffer.get_size())
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        if out_n <= 0 or n % out_n != 0:
            return None
        group = n // out_n
        if group > 32 or not cls._n_axis_grouped_pointwise_reads_match(
            gemm_node, buffer, scheduler_node, group
        ):
            return None
        store = GemmEpilogueIRAnalysis.store_from_buffer(buffer)
        return (buffer, group, store) if store is not None else None

    @classmethod
    def _grouped_variance_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        candidate = cls._n_axis_grouped_pointwise_candidate(gemm_node, scheduler_node)
        if candidate is None:
            return None
        buffer, group, store = candidate
        parameters = variance_parameters_ir(store, gemm_node.get_name(), group)
        if parameters is None:
            return None
        scale, bias = parameters
        reduce_type = "variance_affine:" + ":".join(
            format(value, ".17g") for value in (scale, bias)
        )
        return GemmReductionConfig(buffer.get_name(), group, 1, reduce_type, "identity")

    @classmethod
    def _direct_bool_mask_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis | None = None,
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
        store = GemmEpilogueIRAnalysis.store_from_buffer(buffer)
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
            buffer.get_name(), 2, 1, "direct_bool_gt_zero", "identity"
        )

    @classmethod
    def _grouped_logsumexp_config(
        cls,
        gemm_node: Buffer,
        scheduler_node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        candidate = cls._n_axis_grouped_pointwise_candidate(gemm_node, scheduler_node)
        if candidate is None:
            return None
        buffer, group, store = candidate
        if not is_logsumexp_ir(store, gemm_node.get_name(), group):
            return None
        return GemmReductionConfig(buffer.get_name(), group, 1, "logsumexp", "identity")

    @classmethod
    def _chained_centered_reduction_config(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        gemm_node, nodes = context.gemm, context.nodes
        if len(nodes) <= 1:
            return None
        candidate_nodes = [
            node for node in nodes if isinstance(node.node, ComputedBuffer)
        ]
        buffers = tuple(cast(ComputedBuffer, node.node) for node in candidate_nodes)
        analysis = context.analysis or GemmEpilogueIRAnalysis.from_buffers(buffers)
        reductions = [
            config
            for node in candidate_nodes
            if (config := cls._grouped_reduce_config(gemm_node, node, analysis))
            is not None
            and config.axis == 0
            and config.reduction_type in ("sum", "mean")
            and config.source_type == "identity"
        ]
        if len(reductions) != 1:
            return None
        group, axis = reductions[0].group, reductions[0].axis
        if axis != 0 or group > 64:
            return None
        layout = GemmReductionGeometry(group=group, axis=axis)
        finalizers = [
            buffer
            for buffer in buffers
            if isinstance(buffer.data, Pointwise)
            and layout.matches_output_shape(buffer.get_size(), gemm_node.get_size())
        ]
        if not finalizers:
            return None
        finalizer = finalizers[-1]
        store = analysis.store(finalizer.get_name())
        role = analysis.output_role(finalizer.get_name())
        reduction_names = frozenset(reduction.output_name for reduction in reductions)
        if (
            role is None
            or gemm_node.get_name() not in role.transitive_inputs
            or not reduction_names.issubset(role.reduction_inputs)
        ):
            return None
        consumer_type = (
            centered_mean_consumer_type_ir(
                store, gemm_node.get_name(), reduction_names, group
            )
            if store is not None
            else None
        )
        if consumer_type is None:
            return None
        return GemmReductionConfig(
            finalizer.get_name(), group, axis, consumer_type, "identity"
        )

    @classmethod
    def _external_centered_reduction_config(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        if len(context.nodes) != 1 or context.buffers is None:
            return None
        scheduler_node, node = context.nodes[0], context.buffers[0]
        if not isinstance(node.data, Pointwise):
            return None
        reads = list(scheduler_node.read_writes.reads)
        if len(reads) != 2 or scheduler_node.read_writes.range_vars is None:
            return None
        reduction_reads = [
            read for read in reads if read.name != context.gemm.get_name()
        ]
        if len(reduction_reads) != 1:
            return None
        reduction = V.graph.get_buffer(reduction_reads[0].name)
        if not (
            isinstance(reduction, ComputedBuffer)
            and isinstance(reduction.data, Reduction)
            and reduction.data.reduction_type == "sum"
            and len(reduction.data.reduction_ranges) == 1
            and len(reduction.data.ranges) == 3
        ):
            return None
        try:
            group = V.graph.sizevars.optimization_hint(
                reduction.data.reduction_ranges[0]
            )
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        assert context.analysis is not None  # noqa: S101
        store = context.analysis.store(node.get_name())
        consumer_type = (
            centered_mean_consumer_type_ir(
                store,
                context.gemm.get_name(),
                frozenset((reduction.get_name(),)),
                group,
            )
            if store is not None
            else None
        )
        m, n = context.gemm.get_size()
        out_m, singleton, out_n = reduction.data.ranges
        if (
            consumer_type is None
            or not 1 < group <= 64
            or not V.graph.sizevars.statically_known_equals(
                reduction.data.reduction_ranges[0], group
            )
            or not V.graph.sizevars.statically_known_list_equals(
                (m, n, singleton), (out_m * group, out_n, 1)
            )
            or not V.graph.sizevars.statically_known_list_equals(
                node.get_size(), context.gemm.get_size()
            )
        ):
            return None
        return GemmReductionConfig(node.get_name(), group, 0, consumer_type, "identity")

    @classmethod
    def _unrolled_centered_reduction_config(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        if len(context.nodes) != 1 or context.buffers is None:
            return None
        scheduler_node, node = context.nodes[0], context.buffers[0]
        if not isinstance(node.data, Pointwise):
            return None
        reads = list(scheduler_node.read_writes.reads)
        if len(reads) <= 2 or scheduler_node.read_writes.range_vars is None:
            return None
        try:
            _, n = map(V.graph.sizevars.optimization_hint, context.gemm.get_size())
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        group = len(reads) - 1
        assert context.analysis is not None  # noqa: S101
        store = context.analysis.store(node.get_name())
        consumer_type = (
            centered_mean_consumer_type_unrolled_ir(
                store, context.gemm.get_name(), group
            )
            if store is not None
            else None
        )
        if consumer_type is None or not 1 < group <= 4:
            return None
        if any(read.name != context.gemm.get_name() for read in reads):
            return None
        grouped_base = reads[1].index
        if any(
            V.graph.sizevars.simplify(read.index - grouped_base) != i * n
            for i, read in enumerate(reads[1:])
        ):
            return None
        return GemmReductionConfig(node.get_name(), group, 0, consumer_type, "identity")

    @classmethod
    def _online_softmax_reduction_config(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        if len(context.nodes) != 1 or context.buffers is None:
            return None
        buffer = context.buffers[0]
        if not (
            isinstance(buffer.data, MultiOutputReduction)
            and buffer.data.reduction_type == "online_softmax_reduce"
            and len(buffer.data.reduction_ranges) == 1
        ):
            return None
        try:
            group = V.graph.sizevars.optimization_hint(buffer.data.reduction_ranges[0])
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        if group > 32:
            return None
        return GemmReductionConfig(
            buffer.get_name(), group, 1, "online_softmax", "identity"
        )

    @classmethod
    def _pointwise_softmax_config(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        if len(context.nodes) != 1 or context.buffers is None:
            return None
        buffer = context.buffers[0]
        if not isinstance(buffer.data, Pointwise):
            return None
        layout = GemmReductionGeometry.from_output_shape(
            buffer.get_size(), context.gemm.get_size()
        )
        if layout is None or layout.axis != 1 or layout.group_size > 32:
            return None
        store = context.analysis.store(buffer.get_name()) if context.analysis else None
        if (
            store is None
            or not is_softmax_ir(store, context.gemm.get_name(), layout.group_size)
            or not context.nodes[0].read_writes.reads
            or any(
                read.name != context.gemm.get_name()
                for read in context.nodes[0].read_writes.reads
            )
        ):
            return None
        return GemmReductionConfig(
            buffer.get_name(),
            layout.group_size,
            1,
            "online_softmax",
            "identity",
        )

    @classmethod
    def _chained_softmax_config(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        if len(context.nodes) != 3 or context.buffers is None:
            return None
        buffers = context.buffers
        if not (
            all(isinstance(buffer.data, MultiOutputReduction) for buffer in buffers[:2])
            and isinstance(buffers[2].data, Pointwise)
        ):
            return None
        reductions: list[MultiOutputReduction] = [
            cast(MultiOutputReduction, buffer.data) for buffer in buffers[:2]
        ]
        if any(
            reduction.reduction_type != "online_softmax_reduce"
            or len(reduction.reduction_ranges) != 1
            for reduction in reductions
        ):
            return None
        try:
            group = V.graph.sizevars.optimization_hint(
                reductions[0].reduction_ranges[0]
            )
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        if group > 32:
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
        finalizer_store = (
            context.analysis.store(buffers[2].get_name())
            if context.analysis is not None
            else None
        )
        if finalizer_store is None or not is_softmax_ir(
            finalizer_store,
            context.gemm.get_name(),
            group,
            frozenset(buffer.get_name() for buffer in buffers[:2]),
        ):
            return None
        return GemmReductionConfig(
            buffers[2].get_name(),
            group,
            1,
            "online_softmax",
            "identity",
        )

    @classmethod
    def _grouped_sum_normalize_config_from_nodes(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        gemm_node, nodes = context.gemm, context.nodes
        assert context.buffers is not None and context.analysis is not None  # noqa: S101
        buffers, analysis = context.buffers, context.analysis
        reductions = [
            config
            for node in nodes
            if (config := cls._grouped_reduce_config(gemm_node, node, analysis))
            is not None
        ]
        finalizers = []
        for buffer in buffers:
            if not isinstance(buffer.data, Pointwise):
                continue
            if len(reductions) == 1:
                group, axis = reductions[0].group, reductions[0].axis
            else:
                scheduler_node = next(
                    (node for node in nodes if node.node is buffer), None
                )
                reads = (
                    list(scheduler_node.read_writes.reads)
                    if scheduler_node is not None
                    else []
                )
                layout = GemmReductionGeometry.from_output_shape(
                    buffer.get_size(), gemm_node.get_size()
                )
                if layout is not None:
                    group, axis = layout.group_size, layout.axis
                else:
                    try:
                        _, n = map(
                            V.graph.sizevars.optimization_hint, gemm_node.get_size()
                        )
                    except (GuardOnDataDependentSymNode, TypeError, ValueError):
                        continue
                    group = len(reads) - 1
                    if group <= 1 or group > 4:
                        continue
                    offsets = sorted(
                        V.graph.sizevars.simplify(read.index - reads[0].index)
                        for read in reads[1:]
                    )
                    axis = 0 if offsets == [i * n for i in range(group)] else 1
            store = analysis.store(buffer.get_name())
            consumer_type = (
                sum_normalize_consumer_type_ir(
                    store,
                    gemm_node.get_name(),
                    frozenset(reduction.output_name for reduction in reductions),
                    group,
                )
                if store is not None
                else None
            )
            if consumer_type is None:
                continue
            max_group = 32 if axis == 1 else 64
            layout = GemmReductionGeometry(group=group, axis=axis)
            if group <= max_group and layout.matches_output_shape(
                buffer.get_size(), gemm_node.get_size()
            ):
                finalizers.append((buffer, group, axis, consumer_type))
        if not finalizers or not any(
            read.name == gemm_node.get_name()
            for node in nodes
            for read in node.read_writes.reads
        ):
            return None
        contracts = OrderedSet(
            (group, axis, consumer_type) for _, group, axis, consumer_type in finalizers
        )
        if len(contracts) != 1:
            return None
        group, axis, consumer_type = next(iter(contracts))
        if reductions:
            if len(reductions) != 1 or reductions[0].contract != (
                group,
                axis,
                "sum",
                "identity",
            ):
                return None
        else:
            reads = list(nodes[0].read_writes.reads) if len(nodes) == 1 else []
            if (
                group > 4
                or len(reads) not in (group, group + 1)
                or any(read.name != gemm_node.get_name() for read in reads)
            ):
                return None
        return GemmReductionConfig(
            finalizers[0][0].get_name(), group, axis, consumer_type, "identity"
        )

    @classmethod
    def _grouped_absmax_normalize_config_from_nodes(
        cls, context: NVGemmEpiloguePatternContext
    ) -> GemmReductionConfig | None:
        gemm_node, nodes = context.gemm, context.nodes
        assert context.buffers is not None and context.analysis is not None  # noqa: S101
        buffers, analysis = context.buffers, context.analysis
        candidates = []
        for buffer in buffers:
            if not isinstance(buffer.data, Pointwise):
                continue
            layout = GemmReductionGeometry.from_output_shape(
                buffer.get_size(), gemm_node.get_size()
            )
            if layout is None or layout.axis != 1:
                continue
            group = layout.group_size
            store = analysis.store(buffer.get_name())
            if 1 < group <= 32 and store is not None:
                candidates.append((buffer, group, store))
        reductions, reduction_nodes, _ = cls._partition_local_reductions(
            gemm_node, nodes, analysis
        )
        if len(reductions) != 1:
            return None
        scale_names = frozenset(
            [reductions[0].output_name]
            + [
                node.node.get_name()
                for node in reduction_nodes
                if isinstance(node.node, ComputedBuffer)
            ]
        )
        finalizers = [
            (buffer, group)
            for buffer, group, store in candidates
            if is_absmax_normalize_ir(store, gemm_node.get_name(), scale_names)
        ]
        if len(finalizers) != 1:
            return None
        finalizer, group = finalizers[0]
        if len(reductions) != 1 or reductions[0].contract != (
            group,
            1,
            "max",
            "abs_scale",
        ):
            return None
        return GemmReductionConfig(
            finalizer.get_name(), group, 1, "normalize_absmax", "abs_scale"
        )

    @classmethod
    def _feed_main_config_from_nodes(
        cls,
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        *,
        allow_softmax: bool = True,
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        context = NVGemmEpiloguePatternContext.from_nodes(gemm_node, nodes, analysis)
        patterns = (
            FEED_MAIN_PATTERNS
            if allow_softmax
            else tuple(
                pattern
                for pattern in FEED_MAIN_PATTERNS
                if pattern.kind is not NVGemmEpiloguePatternKind.SOFTMAX
            )
        )
        return cls._match_patterns(context, patterns)

    @staticmethod
    def _pointwise_finalizer_match(
        source_node: BaseSchedulerNode,
        finalizer_node: BaseSchedulerNode,
        *,
        require_reduction: bool = False,
    ) -> tuple[ComputedBuffer, ComputedBuffer, GemmEpilogueIRStore] | None:
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
            and (not require_reduction or isinstance(source.data, Reduction))
        ):
            return None
        reads = list(finalizer_nodes[0].read_writes.reads)
        store = GemmEpilogueIRAnalysis.store_from_buffer(finalizer)
        if (
            store is not None
            and bool(reads)
            and all(read.name == source.get_name() for read in reads)
            and V.graph.sizevars.statically_known_list_equals(
                source.get_size(), finalizer.get_size()
            )
        ):
            return source, finalizer, store
        return None

    @staticmethod
    def _is_layout_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        match = NVGemmEpilogueLowering._pointwise_finalizer_match(
            reduction_node, finalizer_node
        )
        if match is None:
            return False
        source, finalizer, _ = match
        finalizer_ir = GemmEpilogueIRAnalysis.from_buffers(
            (source, finalizer)
        ).reduction_finalizer(finalizer.get_name(), source.get_name())
        return finalizer_ir is not None and finalizer_ir.kind == "identity"

    @staticmethod
    def _is_absmax_scale_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        match = NVGemmEpilogueLowering._pointwise_finalizer_match(
            reduction_node, finalizer_node
        )
        if match is None:
            return False
        source, finalizer, _ = match
        finalizer_ir = GemmEpilogueIRAnalysis.from_buffers(
            (source, finalizer)
        ).reduction_finalizer(finalizer.get_name(), source.get_name())
        return finalizer_ir is not None and finalizer_ir.kind == "absmax_scale"

    @classmethod
    def _local_reduction_config(
        cls,
        gemm_node: Buffer,
        node: BaseSchedulerNode,
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        for _, matcher in LOCAL_REDUCTION_PATTERNS:
            config = matcher(gemm_node, node, analysis)
            if config is not None:
                return config
        return None

    @classmethod
    def _partition_local_reductions(
        cls,
        gemm_node: Buffer,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> tuple[
        list[GemmReductionConfig],
        OrderedSet[BaseSchedulerNode],
        list[NVGemmReductionFinalizer],
    ]:
        reductions: list[GemmReductionConfig] = []
        reduction_nodes: OrderedSet[BaseSchedulerNode] = OrderedSet()
        reduction_finalizers: list[NVGemmReductionFinalizer] = []
        claimed: OrderedSet[BaseSchedulerNode] = OrderedSet()

        for node in epilogue_nodes:
            if node in claimed:
                continue
            config = cls._local_reduction_config(gemm_node, node, analysis)
            if config is None:
                continue
            finalizers: tuple[BaseSchedulerNode, ...] = ()
            candidates = tuple(
                candidate for candidate in epilogue_nodes if candidate not in claimed
            )
            for pattern in FINALIZER_PATTERNS:
                source_name = config.output_name
                finalizer_match = pattern.match(
                    cls, gemm_node, node, config, candidates, analysis
                )
                if finalizer_match is not None:
                    config, finalizer = finalizer_match
                    finalizers = (finalizer,)
                    if pattern.kind is NVGemmEpiloguePatternKind.POINTWISE_FINALIZER:
                        buffer = _single_computed_buffer(finalizer)
                        assert buffer is not None  # noqa: S101
                        reduction_finalizers.append(
                            NVGemmReductionFinalizer(
                                source_name, config.output_name, buffer
                            )
                        )
                    break
            for finalizer in finalizers:
                claimed.add(finalizer)
                reduction_nodes.add(finalizer)
            claimed.add(node)
            reduction_nodes.add(node)
            reductions.append(config)
        return reductions, reduction_nodes, reduction_finalizers

    @classmethod
    def _epilogue_plan(
        cls,
        gemm_node: Buffer,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> NVGemmEpilogueProgram:
        nodes = tuple(
            child
            for epilogue_node in epilogue_nodes
            for child in epilogue_node.get_nodes()
        )
        reductions, reduction_nodes, finalizers = cls._partition_local_reductions(
            gemm_node, nodes, analysis
        )
        feed_main = cls._feed_main_config_from_nodes(
            gemm_node, nodes, analysis=analysis
        )
        min_tile_n = max(
            (config.group for config in reductions if config.axis == 1), default=0
        )
        if feed_main is not None and feed_main.axis == 1:
            min_tile_n = max(min_tile_n, feed_main.group)
        feed_outputs = (
            cls._feed_outputs(gemm_node, nodes, feed_main)
            if feed_main is not None
            else None
        )
        reduction_plan = cls._static_reduction_plan(
            gemm_node, tuple(reductions), feed_main, feed_outputs
        )
        return NVGemmEpilogueProgram(
            nodes,
            tuple(reductions),
            tuple(reduction_nodes),
            feed_main,
            feed_outputs,
            reduction_plan,
            min_tile_n,
            tuple(finalizers),
        )

    @staticmethod
    def _feed_outputs(
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        feed_main: GemmReductionConfig,
    ) -> NVGemmFeedOutputs:
        output_name = feed_main.output_name
        typed = (
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
        equivalent = []
        secondary = None
        secondary_type = None
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
            if not feed_reads or candidate_reads != feed_reads:
                continue
            store = GemmEpilogueIRAnalysis.store_from_buffer(buffer)
            reduction_names = frozenset(feed_reads)
            consumer_type = (
                sum_normalize_consumer_type_ir(
                    store,
                    gemm_node.get_name(),
                    reduction_names,
                    feed_main.group,
                )
                or centered_mean_consumer_type_ir(
                    store,
                    gemm_node.get_name(),
                    reduction_names,
                    feed_main.group,
                )
                if store is not None
                else None
            )
            if consumer_type == feed_main.reduction_type:
                equivalent.append(buffer.get_name())
            elif secondary is None:
                candidate_type = (
                    sum_multiply_consumer_type_ir(
                        store,
                        gemm_node.get_name(),
                        reduction_names,
                        feed_main.group,
                    )
                    if store is not None
                    else None
                )
                if candidate_type is not None:
                    secondary = buffer.get_name()
                    secondary_type = candidate_type
        for output in equivalent:
            if typed is None:
                typed = output
            elif output != typed and secondary is None:
                secondary = output
                secondary_type = feed_main.reduction_type
        return NVGemmFeedOutputs(output_name, typed, secondary, secondary_type)

    @staticmethod
    def _static_reduction_plan(
        gemm_node: Buffer,
        reductions: tuple[GemmReductionConfig, ...],
        feed_main: GemmReductionConfig | None,
        feed_outputs: NVGemmFeedOutputs | None,
    ) -> GemmReductionPlan | None:
        if len(reductions) > 1:
            return None
        local_reduce = None
        if reductions:
            config = reductions[0]
            if config.reduction_type == "direct_bool_gt_zero":
                local_reduce = GemmReductionPlan(
                    reduction_output=None,
                    group=config.group,
                    axis=config.axis,
                    reduction_type=config.reduction_type,
                    source_type=config.source_type,
                    primary_output=gemm_node.get_name(),
                    secondary_feed_output=config.output_name,
                    secondary_feed_type=config.reduction_type,
                )
            else:
                local_reduce = GemmReductionPlan(
                    reduction_output=config.output_name,
                    group=config.group,
                    axis=config.axis,
                    reduction_type=config.reduction_type,
                    source_type=config.source_type,
                    primary_output=gemm_node.get_name(),
                )
        if feed_main is None:
            return local_reduce
        assert feed_outputs is not None  # noqa: S101
        reduce_output = (
            local_reduce.reduction_output
            if local_reduce is not None and local_reduce.geometry == feed_main.geometry
            else None
        )
        primary_output = (
            gemm_node.get_name()
            if feed_outputs.typed is not None
            else feed_main.output_name
        )
        return GemmReductionPlan(
            reduction_output=reduce_output,
            group=feed_main.group,
            axis=feed_main.axis,
            reduction_type=feed_main.reduction_type,
            source_type=feed_main.source_type,
            primary_output=primary_output,
            feeds_main=True,
            feed_output=feed_outputs.typed,
            secondary_feed_output=feed_outputs.secondary,
            secondary_feed_type=feed_outputs.secondary_type,
        )

    @staticmethod
    def _finalize_reduction_plan(
        gemm_node: Buffer, plan: NVGemmEpilogueProgram
    ) -> GemmReductionPlan | None:
        if len(plan.reductions) > 1:
            raise NotImplementedError("NVGEMM supports one grouped local reduction")
        reduction_plan = plan.reduction_plan
        if (
            reduction_plan is None
            or not reduction_plan.feeds_main
            or reduction_plan.reduction_output is None
        ):
            return reduction_plan
        fused_names = OrderedSet(node.get_name() for node in plan.nodes)
        fused_names.add(gemm_node.get_name())
        if V.graph.scheduler.can_buffer_be_removed_through_fusion(
            reduction_plan.reduction_output, fused_names
        ):
            return dataclasses.replace(reduction_plan, reduction_output=None)
        return reduction_plan


def _is_2d_gemm(context: NVGemmEpiloguePatternContext) -> bool:
    return len(context.gemm.get_size()) == 2


def _has_computed_buffers(context: NVGemmEpiloguePatternContext) -> bool:
    return context.buffers is not None


def _reads_gemm(context: NVGemmEpiloguePatternContext) -> bool:
    return context.has_gemm_read


STANDARD_PATTERN_CONSTRAINTS = (_is_2d_gemm, _has_computed_buffers)


class NVGemmEpiloguePatternKind(enum.Enum):
    CENTERED_REDUCTION = enum.auto()
    SOFTMAX = enum.auto()
    SUM_NORMALIZE = enum.auto()
    ABSMAX_NORMALIZE = enum.auto()
    GROUPED_REDUCTION = enum.auto()
    VARIANCE = enum.auto()
    LOGSUMEXP = enum.auto()
    BOOL_MASK = enum.auto()
    MEAN_FINALIZER = enum.auto()
    LAYOUT_FINALIZER = enum.auto()
    ABSMAX_FINALIZER = enum.auto()
    POINTWISE_FINALIZER = enum.auto()


FEED_MAIN_PATTERNS = (
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.CENTERED_REDUCTION,
        "chained",
        NVGemmEpilogueLowering._chained_centered_reduction_config,
        (_is_2d_gemm,),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.CENTERED_REDUCTION,
        "external_reduction",
        NVGemmEpilogueLowering._external_centered_reduction_config,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.CENTERED_REDUCTION,
        "unrolled",
        NVGemmEpilogueLowering._unrolled_centered_reduction_config,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.SOFTMAX,
        "online_reduction",
        NVGemmEpilogueLowering._online_softmax_reduction_config,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.SOFTMAX,
        "pointwise",
        NVGemmEpilogueLowering._pointwise_softmax_config,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.SOFTMAX,
        "chained",
        NVGemmEpilogueLowering._chained_softmax_config,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.SUM_NORMALIZE,
        "sum_normalize",
        NVGemmEpilogueLowering._grouped_sum_normalize_config_from_nodes,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
    NVGemmEpiloguePattern(
        NVGemmEpiloguePatternKind.ABSMAX_NORMALIZE,
        "absmax_normalize",
        NVGemmEpilogueLowering._grouped_absmax_normalize_config_from_nodes,
        (*STANDARD_PATTERN_CONSTRAINTS, _reads_gemm),
    ),
)
FEED_MAIN_PATTERN_REGISTRY = {
    kind: tuple(pattern for pattern in FEED_MAIN_PATTERNS if pattern.kind is kind)
    for kind in (
        NVGemmEpiloguePatternKind.CENTERED_REDUCTION,
        NVGemmEpiloguePatternKind.SOFTMAX,
        NVGemmEpiloguePatternKind.SUM_NORMALIZE,
        NVGemmEpiloguePatternKind.ABSMAX_NORMALIZE,
    )
}


LOCAL_REDUCTION_PATTERNS = (
    (
        NVGemmEpiloguePatternKind.GROUPED_REDUCTION,
        NVGemmEpilogueLowering._grouped_reduce_config,
    ),
    (
        NVGemmEpiloguePatternKind.VARIANCE,
        NVGemmEpilogueLowering._grouped_variance_config,
    ),
    (
        NVGemmEpiloguePatternKind.LOGSUMEXP,
        NVGemmEpilogueLowering._grouped_logsumexp_config,
    ),
    (
        NVGemmEpiloguePatternKind.BOOL_MASK,
        NVGemmEpilogueLowering._direct_bool_mask_config,
    ),
)


def _single_computed_buffer(node: BaseSchedulerNode) -> ComputedBuffer | None:
    nodes = node.get_nodes()
    if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
        return None
    return nodes[0].node


def _match_mean_finalizer(
    lowering: type[NVGemmEpilogueLowering],
    gemm: Buffer,
    source: BaseSchedulerNode,
    candidate: BaseSchedulerNode,
    analysis: GemmEpilogueIRAnalysis | None,
) -> bool:
    match = lowering._pointwise_finalizer_match(
        source, candidate, require_reduction=True
    )
    if match is None:
        return False
    source_config = lowering._local_reduction_config(gemm, source, analysis)
    source_buffer, finalizer_buffer, _ = match
    if source_config is None:
        return False
    finalizer_analysis = analysis or GemmEpilogueIRAnalysis.from_buffers(
        (source_buffer, finalizer_buffer)
    )
    finalizer_ir = finalizer_analysis.reduction_finalizer(
        finalizer_buffer.get_name(), source_config.output_name, source_config.group
    )
    return finalizer_ir is not None and finalizer_ir.kind == "mean"


def _match_layout_finalizer(
    lowering: type[NVGemmEpilogueLowering],
    gemm: Buffer,
    source: BaseSchedulerNode,
    candidate: BaseSchedulerNode,
    analysis: GemmEpilogueIRAnalysis | None,
) -> bool:
    return lowering._is_layout_finalizer(source, candidate)


def _match_absmax_finalizer(
    lowering: type[NVGemmEpilogueLowering],
    gemm: Buffer,
    source: BaseSchedulerNode,
    candidate: BaseSchedulerNode,
    analysis: GemmEpilogueIRAnalysis | None,
) -> bool:
    return lowering._is_absmax_scale_finalizer(source, candidate)


def _match_pointwise_finalizer(
    lowering: type[NVGemmEpilogueLowering],
    gemm: Buffer,
    source: BaseSchedulerNode,
    candidate: BaseSchedulerNode,
    analysis: GemmEpilogueIRAnalysis | None,
) -> bool:
    return lowering._pointwise_finalizer_match(source, candidate) is not None


def _rewrite_mean_finalizer(
    config: GemmReductionConfig, buffer: ComputedBuffer
) -> GemmReductionConfig:
    return dataclasses.replace(
        config, output_name=buffer.get_name(), reduction_type="mean"
    )


def _rewrite_layout_finalizer(
    config: GemmReductionConfig, buffer: ComputedBuffer
) -> GemmReductionConfig:
    return dataclasses.replace(config, output_name=buffer.get_name())


def _rewrite_absmax_finalizer(
    config: GemmReductionConfig, buffer: ComputedBuffer
) -> GemmReductionConfig:
    return dataclasses.replace(
        config, output_name=buffer.get_name(), source_type="abs_scale"
    )


FINALIZER_PATTERNS = (
    NVGemmFinalizerPattern(
        NVGemmEpiloguePatternKind.MEAN_FINALIZER,
        lambda config: config.reduction_type == "sum",
        _match_mean_finalizer,
        _rewrite_mean_finalizer,
    ),
    NVGemmFinalizerPattern(
        NVGemmEpiloguePatternKind.LAYOUT_FINALIZER,
        lambda config: True,
        _match_layout_finalizer,
        _rewrite_layout_finalizer,
    ),
    NVGemmFinalizerPattern(
        NVGemmEpiloguePatternKind.ABSMAX_FINALIZER,
        lambda config: (config.reduction_type, config.source_type) == ("max", "abs"),
        _match_absmax_finalizer,
        _rewrite_absmax_finalizer,
    ),
    NVGemmFinalizerPattern(
        NVGemmEpiloguePatternKind.POINTWISE_FINALIZER,
        lambda config: True,
        _match_pointwise_finalizer,
        _rewrite_layout_finalizer,
    ),
)
