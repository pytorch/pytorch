# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

import dataclasses
import hashlib
import logging
from collections.abc import Sequence
from typing import Any, cast, Literal, overload

import torch
from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
)
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import (
    Buffer,
    ComputedBuffer,
    Layout,
    MultiOutputReduction,
    MultiTemplateBuffer,
    NVUniversalGemmBuffer,
    Pointwise,
    Reduction,
)
from ...kernel.gemm_epilogue import (
    GemmReductionConfig,
    GemmReductionGeometry,
    GemmReductionPlan,
)
from ...kernel.gemm_epilogue_codegen import GemmEpilogueIRCodegen
from ...kernel.gemm_epilogue_ir import (
    centered_mean_consumer_type_ir,
    centered_mean_consumer_type_unrolled_ir,
    GemmEpilogueIRAnalysis,
    GemmEpilogueIRStore,
    grouped_reduction_ir,
    is_absmax_normalize_ir,
    is_absmax_scale_finalizer_ir,
    is_direct_bool_gt_zero_ir,
    is_logsumexp_ir,
    is_softmax_ir,
    operation_names_ir,
    single_source_affine_ir,
    sum_multiply_consumer_type_ir,
    sum_normalize_consumer_type_ir,
    variance_parameters_ir,
)
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from ..cutlass.python_evt import _ACCUMULATOR_ARG_NAME, CutlassEVTCodegen
from .nv_universal_gemm import NVUniversalGemmCaller


log = logging.getLogger(__name__)

MAIN_SUFFIX = "main"
_BENCHMARK_KERNEL_PREFIX = "nv_gemm_"
EPILOGUE_FN_NAME = "_epilogue_fn"


@dataclasses.dataclass(frozen=True)
class NVGemmGeneratedSource:
    source: str
    epilogue_reads: tuple[str, ...]
    output_buffers: tuple[str, ...]


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
class NVGemmEpilogueAnalysis:
    """Scheduler-owned fusion analysis before producing a shared reduction plan."""

    nodes: tuple[BaseSchedulerNode, ...]
    reductions: tuple[GemmReductionConfig, ...]
    reduction_nodes: tuple[BaseSchedulerNode, ...]
    feed_main: GemmReductionConfig | None
    feed_outputs: NVGemmFeedOutputs | None
    reduction_plan: GemmReductionPlan | None
    min_tile_n: int

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


class NVUniversalGemmScheduling(BaseScheduling):
    """
    Scheduling implementation for NVIDIA Universal GEMM kernels.

    This class is intended to be used in combination with other schedulers,
    and delegated to by CUDACombinedScheduling.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def _is_nvgemm_ir_buffer(ir_node: Any) -> bool:
        """Return True if `ir_node` is an NVGEMM buffer or an MTB resolving to one.

        Honors finalize_as_*_caller's swap (a MultiTemplateBuffer whose render
        kind is "triton" is no longer NVGEMM, even if the autotune winner was).
        Falls back to the autotune winner only when no swap has happened.
        """
        if isinstance(ir_node, NVUniversalGemmBuffer):
            return True
        if not isinstance(ir_node, MultiTemplateBuffer):
            return False
        if ir_node._render_kind == "triton":
            return False
        if ir_node._render_kind == "nvgemm":
            return True
        # Fast path: avoid forcing autotune just to answer this query.
        if not any(isinstance(c, NVUniversalGemmCaller) for c in ir_node._choices):
            return False
        try:
            min_choice, _ = ir_node.get_min_choice()
            return isinstance(min_choice, NVUniversalGemmCaller)
        except (RuntimeError, ValueError):
            return False

    @staticmethod
    def _has_nvgemm_choice(ir_node: Any) -> bool:
        return isinstance(ir_node, NVUniversalGemmBuffer) or (
            isinstance(ir_node, MultiTemplateBuffer)
            and any(
                isinstance(choice, NVUniversalGemmCaller) for choice in ir_node._choices
            )
        )

    @staticmethod
    def is_nv_universal_gemm_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is an NVGEMM template SchedulerNode."""
        if not isinstance(node, SchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.node)

    @staticmethod
    def _best_nvgemm_choice(
        ir_node: MultiTemplateBuffer,
        require_epilogue_fusion: bool = False,
        min_tile_n: int = 0,
    ) -> NVUniversalGemmCaller:
        """Find the best NVUniversalGemmCaller from an MTB's choice timings."""
        choice_timings = ir_node.choice_timings()
        best: NVUniversalGemmCaller | None = None
        best_time = float("inf")
        for choice in ir_node._choices:
            if not isinstance(choice, NVUniversalGemmCaller):
                continue
            if require_epilogue_fusion and not choice.supports_epilogue_fusion:
                continue
            if choice.kernel.metadata.design.tile_shape[1] < min_tile_n:
                continue
            timing = choice_timings.get(choice, float("inf"))
            if best is None or timing < best_time:
                best_time = timing
                best = choice
        if best is None:
            kind = "EFC kernel" if require_epilogue_fusion else "NVUniversalGemmCaller"
            raise RuntimeError(f"No {kind} found in choices")
        return best

    @staticmethod
    def get_nv_gemm_buffer_from_node(
        node: BaseSchedulerNode,
        require_epilogue_fusion: bool = False,
        min_tile_n: int = 0,
    ) -> NVUniversalGemmBuffer:
        """Extract NVUniversalGemmBuffer from node (direct or via MultiTemplateBuffer)."""
        assert isinstance(node, SchedulerNode)  # noqa: S101
        ir_node = node.node

        if isinstance(ir_node, NVUniversalGemmBuffer):
            return ir_node
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Honor an explicit swap/finalize -- the fusion benchmark loop swaps
            # in each EFC choice one at a time and must not re-select from timings.
            if (
                isinstance(ir_node._render_caller, NVUniversalGemmCaller)
                and (
                    not require_epilogue_fusion
                    or ir_node._render_caller.supports_epilogue_fusion
                )
                and ir_node._render_caller.kernel.metadata.design.tile_shape[1]
                >= min_tile_n
            ):
                selected_choice = ir_node._render_caller
            elif require_epilogue_fusion:
                selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                    ir_node,
                    require_epilogue_fusion=True,
                    min_tile_n=min_tile_n,
                )
            else:
                min_choice, _ = ir_node.get_min_choice()
                if isinstance(min_choice, NVUniversalGemmCaller):
                    selected_choice = min_choice
                else:
                    selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                        ir_node
                    )
            tensor_box = selected_choice.output_node()
            # pyrefly: ignore [missing-attribute]
            return cast(NVUniversalGemmBuffer, tensor_box.data.data)

        raise TypeError(
            f"Expected NVUniversalGemmBuffer or MultiTemplateBuffer, got {type(ir_node).__name__}"
        )

    @staticmethod
    def is_nv_universal_gemm_fused_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused NVIDIA Universal GEMM template."""
        if not isinstance(node, FusedSchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.get_template_node())

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_nv_universal_gemm_template(node1):
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, node1),
                [],
                node2,
            )
        elif self.is_nv_universal_gemm_fused_template(node1):
            fnode1 = cast(FusedSchedulerNode, node1)
            template_snode = next(
                (n for n in fnode1.snodes if self.is_nv_universal_gemm_template(n)),
                None,
            )
            if template_snode is None:
                return False
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, template_snode),
                self._unwrap_epilogue_nodes(fnode1),
                node2,
            )
        return False

    def _unwrap_epilogue_nodes(
        self, fused_node: FusedSchedulerNode
    ) -> list[BaseSchedulerNode]:
        """Extract epilogue nodes from a fused node."""
        epilogue_nodes = []
        for node in fused_node.snodes:
            if not self.is_nv_universal_gemm_template(node):
                epilogue_nodes.append(node)
        return epilogue_nodes

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
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
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

        store = GemmEpilogueIRAnalysis.store_from_buffer(node)
        classified = (
            grouped_reduction_ir(
                store,
                gemm_node.get_name(),
                group,
                gemm_node.get_dtype(),
            )
            if store is not None
            else None
        )
        if classified is None:
            return None
        physical_type, source_type = classified
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
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
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
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
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
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> GemmReductionConfig | None:
        candidate = cls._n_axis_grouped_pointwise_candidate(gemm_node, scheduler_node)
        if candidate is None:
            return None
        buffer, group, store = candidate
        if not is_logsumexp_ir(store, gemm_node.get_name(), group):
            return None
        return GemmReductionConfig(buffer.get_name(), group, 1, "logsumexp", "identity")

    @classmethod
    def _grouped_reduce_feeds_main_config_from_nodes(
        cls,
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        if len(gemm_node.get_size()) != 2:
            return None
        candidate_nodes = [
            node for node in nodes if isinstance(node.node, ComputedBuffer)
        ]
        buffers = tuple(cast(ComputedBuffer, node.node) for node in candidate_nodes)
        analysis = analysis or GemmEpilogueIRAnalysis.from_buffers(buffers)
        if len(nodes) > 1:
            reductions = [
                config
                for node in candidate_nodes
                if (config := cls._grouped_reduce_config(gemm_node, node)) is not None
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
            reduction_names = frozenset(
                reduction.output_name for reduction in reductions
            )
            if (
                role is None
                or gemm_node.get_name() not in role.transitive_inputs
                or not reduction_names.issubset(role.reduction_inputs)
            ):
                return None
            consumer_type = (
                centered_mean_consumer_type_ir(
                    store,
                    gemm_node.get_name(),
                    reduction_names,
                    group,
                )
                if store is not None
                else None
            )
            if consumer_type is None:
                return None
            return GemmReductionConfig(
                finalizer.get_name(),
                group,
                axis,
                consumer_type,
                "identity",
            )
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        node = cast(ComputedBuffer, nodes[0].node)
        if not isinstance(node.data, Pointwise):
            return None
        reads = list(nodes[0].read_writes.reads)
        range_vars = nodes[0].read_writes.range_vars
        if range_vars is None:
            return None
        if len(reads) == 2:
            reduction_reads = [
                read for read in reads if read.name != gemm_node.get_name()
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
            axis = 0
            store = analysis.store(node.get_name())
            consumer_type = (
                centered_mean_consumer_type_ir(
                    store,
                    gemm_node.get_name(),
                    frozenset((reduction.get_name(),)),
                    group,
                )
                if store is not None
                else None
            )
            m, n = gemm_node.get_size()
            out_m, singleton, out_n = reduction.data.ranges
            if (
                consumer_type is None
                or group <= 1
                or group > 64
                or not V.graph.sizevars.statically_known_equals(
                    reduction.data.reduction_ranges[0], group
                )
                or not V.graph.sizevars.statically_known_list_equals(
                    (m, n, singleton), (out_m * group, out_n, 1)
                )
                or not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), gemm_node.get_size()
                )
            ):
                return None
            return GemmReductionConfig(
                node.get_name(), group, axis, consumer_type, "identity"
            )
        if len(reads) <= 2:
            return None
        try:
            _, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
        except (GuardOnDataDependentSymNode, TypeError, ValueError):
            return None
        group, axis = len(reads) - 1, 0
        store = analysis.store(node.get_name())
        consumer_type = (
            centered_mean_consumer_type_unrolled_ir(store, gemm_node.get_name(), group)
            if store is not None
            else None
        )
        if consumer_type is None or group <= 1 or group > 4:
            return None
        if any(read.name != gemm_node.get_name() for read in reads):
            return None
        grouped_base = reads[1].index
        if any(
            V.graph.sizevars.simplify(read.index - grouped_base) != i * n
            for i, read in enumerate(reads[1:])
        ):
            return None
        return GemmReductionConfig(
            node.get_name(), group, axis, consumer_type, "identity"
        )

    @classmethod
    def _grouped_softmax_config_from_nodes(
        cls,
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        if len(gemm_node.get_size()) != 2:
            return None
        if len(nodes) == 1 and isinstance(nodes[0].node, ComputedBuffer):
            buffer = cast(ComputedBuffer, nodes[0].node)
            if (
                isinstance(buffer.data, MultiOutputReduction)
                and buffer.data.reduction_type == "online_softmax_reduce"
                and len(buffer.data.reduction_ranges) == 1
            ):
                try:
                    group = V.graph.sizevars.optimization_hint(
                        buffer.data.reduction_ranges[0]
                    )
                except (GuardOnDataDependentSymNode, TypeError, ValueError):
                    return None
                if group <= 32:
                    return GemmReductionConfig(
                        buffer.get_name(), group, 1, "online_softmax", "identity"
                    )
            if isinstance(buffer.data, Pointwise):
                layout = GemmReductionGeometry.from_output_shape(
                    buffer.get_size(), gemm_node.get_size()
                )
                if layout is None or layout.axis != 1:
                    return None
                group = layout.group_size
                store = (
                    analysis.store(buffer.get_name())
                    if analysis is not None
                    else GemmEpilogueIRAnalysis.store_from_buffer(buffer)
                )
                if (
                    group <= 32
                    and store is not None
                    and is_softmax_ir(store, gemm_node.get_name(), group)
                    and nodes[0].read_writes.reads
                    and all(
                        read.name == gemm_node.get_name()
                        for read in nodes[0].read_writes.reads
                    )
                ):
                    return GemmReductionConfig(
                        buffer.get_name(), group, 1, "online_softmax", "identity"
                    )
            return None
        if len(nodes) != 3:
            return None
        buffers = cls._computed_buffers(nodes)
        if not (
            buffers is not None
            and all(
                isinstance(buffer.data, MultiOutputReduction) for buffer in buffers[:2]
            )
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
            m, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
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
            analysis.store(buffers[2].get_name())
            if analysis is not None
            else GemmEpilogueIRAnalysis.store_from_buffer(buffers[2])
        )
        if finalizer_store is None or not is_softmax_ir(
            finalizer_store,
            gemm_node.get_name(),
            group,
            frozenset(buffer.get_name() for buffer in buffers[:2]),
        ):
            return None
        if not any(
            read.name == gemm_node.get_name()
            for node in nodes
            for read in node.read_writes.reads
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
        cls,
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        if len(gemm_node.get_size()) != 2:
            return None
        buffers = cls._computed_buffers(nodes)
        if buffers is None:
            return None
        reductions = [
            config
            for node in nodes
            if (config := cls._grouped_reduce_config(gemm_node, node)) is not None
        ]
        finalizers = []
        analysis = analysis or GemmEpilogueIRAnalysis.from_buffers(buffers)
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
        cls,
        gemm_node: Buffer,
        nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> GemmReductionConfig | None:
        if len(gemm_node.get_size()) != 2:
            return None
        buffers = cls._computed_buffers(nodes)
        if buffers is None:
            return None
        analysis = analysis or GemmEpilogueIRAnalysis.from_buffers(buffers)
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
        reductions, reduction_nodes = cls._partition_local_reductions(gemm_node, nodes)
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
        buffers = cls._computed_buffers(nodes)
        analysis = analysis or (
            GemmEpilogueIRAnalysis.from_buffers(buffers)
            if buffers is not None
            else None
        )
        matchers = [
            cls._grouped_reduce_feeds_main_config_from_nodes,
            cls._grouped_sum_normalize_config_from_nodes,
            cls._grouped_absmax_normalize_config_from_nodes,
        ]
        if allow_softmax:
            matchers.insert(1, cls._grouped_softmax_config_from_nodes)
        for matcher in matchers:
            if (config := matcher(gemm_node, nodes, analysis)) is not None:
                return config
        return None

    @staticmethod
    def _is_mean_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        return bool(
            NVUniversalGemmScheduling._pointwise_finalizer_match(
                reduction_node, finalizer_node, require_reduction=True
            )
        )

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
        match = NVUniversalGemmScheduling._pointwise_finalizer_match(
            reduction_node, finalizer_node
        )
        if match is None:
            return False
        return operation_names_ir(match[2]).issubset(
            ("load", "to_dtype", "to_dtype_bitcast", "identity")
        )

    @staticmethod
    def _is_absmax_scale_finalizer(
        reduction_node: BaseSchedulerNode, finalizer_node: BaseSchedulerNode
    ) -> bool:
        match = NVUniversalGemmScheduling._pointwise_finalizer_match(
            reduction_node, finalizer_node
        )
        if match is None:
            return False
        reduction, _, store = match
        return is_absmax_scale_finalizer_ir(store, reduction.get_name())

    @classmethod
    def _local_reduction_config(
        cls, gemm_node: Buffer, node: BaseSchedulerNode
    ) -> GemmReductionConfig | None:
        for matcher in (
            cls._grouped_reduce_config,
            cls._grouped_variance_config,
            cls._grouped_logsumexp_config,
            cls._direct_bool_mask_config,
        ):
            config = matcher(gemm_node, node)
            if config is not None:
                return config
        return None

    @classmethod
    def _partition_local_reductions(
        cls, gemm_node: Buffer, epilogue_nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[list[GemmReductionConfig], OrderedSet[BaseSchedulerNode]]:
        reductions: list[GemmReductionConfig] = []
        reduction_nodes: OrderedSet[BaseSchedulerNode] = OrderedSet()
        claimed: OrderedSet[BaseSchedulerNode] = OrderedSet()

        def buffer_of(node: BaseSchedulerNode) -> ComputedBuffer | None:
            nodes = node.get_nodes()
            if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
                return None
            return nodes[0].node

        def unique_finalizer(source, predicate):
            matches = [
                candidate
                for candidate in epilogue_nodes
                if candidate not in claimed
                and candidate is not source
                and predicate(source, candidate)
            ]
            return matches[0] if len(matches) == 1 else None

        def mean_affine_finalizer(source, candidate) -> bool:
            if not cls._is_mean_finalizer(source, candidate):
                return False
            source_config = cls._local_reduction_config(gemm_node, source)
            finalizer_buffer = buffer_of(candidate)
            if source_config is None or finalizer_buffer is None:
                return False
            store = GemmEpilogueIRAnalysis.store_from_buffer(finalizer_buffer)
            affine = (
                single_source_affine_ir(store, source_config.output_name)
                if store is not None
                else None
            )
            return affine == (1.0 / source_config.group, 0.0)

        for node in epilogue_nodes:
            if node in claimed:
                continue
            config = cls._local_reduction_config(gemm_node, node)
            if config is None:
                continue
            finalizers: tuple[BaseSchedulerNode, ...] = ()
            if (
                config.reduction_type == "sum"
                and (finalizer := unique_finalizer(node, mean_affine_finalizer))
                is not None
            ):
                finalizer_buffer = buffer_of(finalizer)
                assert finalizer_buffer is not None  # noqa: S101
                config = dataclasses.replace(
                    config,
                    output_name=finalizer_buffer.get_name(),
                    reduction_type="mean",
                )
                finalizers = (finalizer,)
            elif (
                finalizer := unique_finalizer(node, cls._is_layout_finalizer)
            ) is not None:
                finalizer_buffer = buffer_of(finalizer)
                assert finalizer_buffer is not None  # noqa: S101
                config = dataclasses.replace(
                    config, output_name=finalizer_buffer.get_name()
                )
                finalizers = (finalizer,)
            elif (config.reduction_type, config.source_type) == ("max", "abs"):
                finalizer = unique_finalizer(node, cls._is_absmax_scale_finalizer)
                finalizer_buffer = (
                    buffer_of(finalizer) if finalizer is not None else None
                )
                if finalizer_buffer is not None:
                    config = dataclasses.replace(
                        config,
                        output_name=finalizer_buffer.get_name(),
                        source_type="abs_scale",
                    )
                    finalizers = (finalizer,)
            for finalizer in finalizers:
                claimed.add(finalizer)
                reduction_nodes.add(finalizer)
            claimed.add(node)
            reduction_nodes.add(node)
            reductions.append(config)
        return reductions, reduction_nodes

    @classmethod
    def _epilogue_plan(
        cls,
        gemm_node: Buffer,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        analysis: GemmEpilogueIRAnalysis | None = None,
    ) -> NVGemmEpilogueAnalysis:
        nodes = tuple(
            child
            for epilogue_node in epilogue_nodes
            for child in epilogue_node.get_nodes()
        )
        reductions, reduction_nodes = cls._partition_local_reductions(gemm_node, nodes)
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
        return NVGemmEpilogueAnalysis(
            nodes,
            tuple(reductions),
            tuple(reduction_nodes),
            feed_main,
            feed_outputs,
            reduction_plan,
            min_tile_n,
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
        gemm_node: Buffer, plan: NVGemmEpilogueAnalysis
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

    def _can_fuse_epilogue_impl(
        self,
        gemm_template_node: SchedulerNode,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        from .nv_universal_gemm import GemmVariant

        if not config.epilogue_fusion:
            return False

        ir_node = gemm_template_node.node
        if not isinstance(ir_node, (NVUniversalGemmBuffer, MultiTemplateBuffer)):
            return False

        if isinstance(ir_node, NVUniversalGemmBuffer):
            if ir_node.variant not in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM):
                log.debug(
                    "NVGEMM epilogue fusion: not supported for %s variant",
                    ir_node.variant.op_name,
                )
                return False
            if not ir_node.supports_epilogue_fusion:
                log.debug(
                    "NVGEMM epilogue fusion: kernel %s does not support epilogue fusion",
                    ir_node.kernel_metadata.get("kernel_name", "unknown"),
                )
                return False
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Use _choices, not choice_timings() — the latter forces autotune sync.
            has_efc_choice = False
            for choice in ir_node._choices:
                if not (
                    isinstance(choice, NVUniversalGemmCaller)
                    and choice.supports_epilogue_fusion
                ):
                    continue
                has_efc_choice = True
                if choice.variant not in (
                    GemmVariant.GEMM,
                    GemmVariant.SCALED_GEMM,
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: MultiTemplateBuffer has unsupported EFC choices"
                    )
                    return False
            if not has_efc_choice:
                log.debug("NVGEMM epilogue fusion: no EFC kernel available in choices")
                return False

        all_scheduler_nodes = [
            node
            for epilogue_node in (*existing_epilogue_nodes, node_to_fuse)
            for node in epilogue_node.get_nodes()
        ]
        computed_buffers = self._computed_buffers(all_scheduler_nodes)
        ir_analysis = (
            GemmEpilogueIRAnalysis.from_buffers(computed_buffers)
            if computed_buffers is not None
            else None
        )
        softmax = self._grouped_softmax_config_from_nodes(
            ir_node, all_scheduler_nodes, ir_analysis
        )
        if (
            softmax is None
            and sum(
                isinstance(scheduler_node.node.data, Reduction)
                for scheduler_node in all_scheduler_nodes
                if isinstance(scheduler_node.node, ComputedBuffer)
            )
            > 1
        ):
            log.debug("NVGEMM supports one grouped local reduction")
            return False
        epilogue_plan = self._epilogue_plan(ir_node, all_scheduler_nodes, ir_analysis)
        feed_main = epilogue_plan.feed_main
        if feed_main is not None:
            fused_names = OrderedSet(
                scheduler_node.get_name() for scheduler_node in all_scheduler_nodes
            )
            fused_names.add(gemm_template_node.get_name())
            if not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                ir_node.get_name(), fused_names
            ):
                return False
        local_reduce_nodes = epilogue_plan.owned_nodes
        local_reduce = self._local_reduction_config(ir_node, node_to_fuse)
        variants = (
            (ir_node.variant,)
            if isinstance(ir_node, NVUniversalGemmBuffer)
            else tuple(
                choice.variant
                for choice in ir_node._choices
                if isinstance(choice, NVUniversalGemmCaller)
                and choice.supports_epilogue_fusion
            )
        )
        scaled_epilogue = bool(variants) and all(
            variant == GemmVariant.SCALED_GEMM for variant in variants
        )
        if local_reduce is not None:
            standard_aux_sum = (
                bool(variants)
                and all(variant == GemmVariant.GEMM for variant in variants)
                and local_reduce.axis in (0, 1)
                and (
                    local_reduce.reduction_type
                    in (
                        "sum",
                        "mean",
                        "prod",
                        "max",
                        "min",
                    )
                    or local_reduce.reduction_type.startswith("variance_affine:")
                    or local_reduce.reduction_type == "logsumexp"
                    or local_reduce.reduction_type == "direct_bool_gt_zero"
                )
                and local_reduce.source_type in ("identity", "square", "abs")
            )
            if not scaled_epilogue and not standard_aux_sum:
                return False

        for s_node in all_scheduler_nodes:
            node = s_node.node
            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return False
            if (
                feed_main is None
                and not isinstance(node.data, Pointwise)
                and s_node not in local_reduce_nodes
            ):
                log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                return False

            if (
                feed_main is None
                and s_node not in local_reduce_nodes
                and not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), ir_node.get_size()
                )
            ):
                log.debug(
                    "NVGEMM epilogue fusion: size mismatch %s vs %s",
                    node.get_size(),
                    ir_node.get_size(),
                )
                return False
            if (
                feed_main is None
                and s_node not in local_reduce_nodes
                and isinstance(node.data, Pointwise)
                and not V.graph.sizevars.statically_known_list_equals(
                    node.data.ranges, ir_node.get_size()
                )
            ):
                log.debug(
                    "NVGEMM epilogue fusion: iteration-domain mismatch %s vs %s",
                    node.data.ranges,
                    ir_node.get_size(),
                )
                return False
        # cutlass.operators' EVT supports matrix, row, and column loads here.
        # Reject unresolved inputs and shapes outside those broadcast patterns;
        # the trial EVT trace below remains the final capability check.
        gemm_size = ir_node.get_size()
        name_to_buf = V.graph.name_to_buffer | V.graph.graph_inputs
        internal_names = OrderedSet([ir_node.get_name()]) | OrderedSet(
            [s_node.get_name() for s_node in all_scheduler_nodes]
        )
        epilogue_inputs: OrderedSet[str] = OrderedSet()
        for s_node in epilogue_plan.evt_nodes:
            for rd in s_node.read_writes.reads:
                if rd.name in internal_names:
                    continue
                epilogue_inputs.add(rd.name)
                read_buf = name_to_buf.get(rd.name)
                if read_buf is None:
                    log.debug(
                        "NVGEMM epilogue fusion: read %s not in name_to_buffer/graph_inputs, refusing to fuse",
                        rd.name,
                    )
                    return False
                read_size = read_buf.get_size()
                if not read_size or len(read_size) > len(gemm_size):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s has unsupported rank",
                        rd.name,
                    )
                    return False
                padded_size = [1] * (len(gemm_size) - len(read_size)) + list(read_size)
                supported_shapes = (
                    gemm_size,
                    [1, gemm_size[1]],
                    [gemm_size[0], 1],
                )
                if not any(
                    V.graph.sizevars.statically_known_list_equals(padded_size, shape)
                    for shape in supported_shapes
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s size %s is not broadcastable to GEMM size %s",
                        rd.name,
                        read_size,
                        gemm_size,
                    )
                    return False
        if not existing_epilogue_nodes:
            reads = OrderedSet(rd.name for rd in node_to_fuse.read_writes.reads)
            if ir_node.get_name() not in reads:
                log.debug(
                    "NVGEMM epilogue fusion: first epilogue node doesn't read from GEMM output"
                )
                return False

        if node_to_fuse.has_aliasing_or_mutation():
            log.debug("NVGEMM epilogue fusion: node has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction() and local_reduce is None and feed_main is None:
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return False

        fused_buffer_names = OrderedSet(
            n.get_name() for n in [gemm_template_node, *epilogue_plan.nodes]
        )
        preserve_gemm_output = (
            not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                ir_node.get_name(), fused_buffer_names
            )
        )
        # Multi-store epilogues wire each output to its own destination tensor.
        trial_removed_buffers = V.graph.removed_buffers.copy()
        if not preserve_gemm_output:
            trial_removed_buffers.add(ir_node.get_name())
        try:
            trial_reads: list[str] = []
            trial_writes: list[str] = []
            evt_nodes = epilogue_plan.evt_nodes
            if evt_nodes:
                trial_reads, trial_writes, _, _ = (
                    CutlassEVTCodegen.ir_to_evt_python_code(
                        ir_node.get_name(),
                        list(evt_nodes),
                        trial_removed_buffers,
                    )
                )
            if scaled_epilogue:
                for read_name in trial_reads:
                    read_buf = name_to_buf.get(read_name)
                    if read_buf is None:
                        log.debug(
                            "NVGEMM scaled EVT input %s cannot be resolved",
                            read_name,
                        )
                        return False
        except (NotImplementedError, AssertionError) as e:
            log.debug("NVGEMM epilogue fusion: trial EVT codegen failed: %s", e)
            return False

        return True

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support horizontal fusion yet
        return False

    @staticmethod
    def has_bool_output(node: BaseSchedulerNode) -> bool:
        return any(
            isinstance(scheduler_node.node, ComputedBuffer)
            and scheduler_node.node.get_dtype() == torch.bool
            for scheduler_node in node.get_nodes()
        )

    def can_fuse_reduction_chain(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if not any(
            isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, Reduction)
            for node in node1.get_nodes()
        ):
            return False
        for read in node1.read_writes.reads:
            producer = V.graph.try_get_buffer(read.name)
            if not isinstance(producer, Buffer) or not self._has_nvgemm_choice(
                producer
            ):
                continue
            combined_nodes = [*node1.get_nodes(), *node2.get_nodes()]
            return bool(
                self._feed_main_config_from_nodes(
                    producer, combined_nodes, allow_softmax=False
                )
            )
        return False

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        has_reduction = any(
            isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, Reduction)
            for node in node1.get_nodes()
        )
        if has_reduction and self.can_fuse_reduction_chain(node1, node2):
            return 0
        if not has_reduction:
            node1_ir = node1.node if isinstance(node1, SchedulerNode) else None
            if self._has_nvgemm_choice(node1_ir) and any(
                isinstance(node.node, ComputedBuffer)
                and isinstance(node.node.data, Reduction)
                for node in node2.get_nodes()
            ):
                return 1
            return 2
        node1_outputs = node1.get_buffer_names()
        if not any(read.name in node1_outputs for read in node2.read_writes.reads):
            return 2
        scheduler = self.scheduler
        if scheduler is None:
            return 2
        for read in node1.read_writes.reads:
            producer = scheduler.name_to_fused_node.get(read.name)
            try:
                producer_buffer = V.graph.get_buffer(read.name)
            except RuntimeError:
                producer_buffer = None
            has_nvgemm_producer = (
                producer is not None
                and (
                    self.is_nv_universal_gemm_template(producer)
                    or self.is_nv_universal_gemm_fused_template(producer)
                )
            ) or self._has_nvgemm_choice(producer_buffer)
            if has_nvgemm_producer and isinstance(producer_buffer, Buffer):
                combined_nodes = [*node1.get_nodes(), *node2.get_nodes()]
                if self._feed_main_config_from_nodes(
                    producer_buffer, combined_nodes, allow_softmax=False
                ):
                    return 0
                combined_reductions, combined_reduction_nodes = (
                    self._partition_local_reductions(producer_buffer, combined_nodes)
                )
                if combined_reductions and all(
                    node in combined_reduction_nodes for node in node2.get_nodes()
                ):
                    log.debug(
                        "prioritizing NVGEMM reduction chain %s -> %s",
                        node1.get_name(),
                        node2.get_name(),
                    )
                    return 0
                reductions, reduction_nodes = self._partition_local_reductions(
                    producer_buffer, node1.get_nodes()
                )
                if reductions and all(
                    node in reduction_nodes for node in node1.get_nodes()
                ):
                    return 2
                return 0
        return 2

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        template = node1.get_template_node()
        if not isinstance(template, Buffer):
            return False
        epilogue_nodes = [
            node
            for node in node1.get_nodes()
            if not self.is_nv_universal_gemm_template(node)
        ]
        combined_nodes = [*epilogue_nodes, *node2.get_nodes()]
        combined_softmax = self._grouped_softmax_config_from_nodes(
            template, combined_nodes
        )
        if (
            combined_softmax is None
            and sum(
                isinstance(node.node.data, Reduction)
                for node in combined_nodes
                if isinstance(node.node, ComputedBuffer)
            )
            > 1
        ):
            return False
        combined_feed_main = self._grouped_reduce_feeds_main_config_from_nodes(
            template, combined_nodes
        )
        feed_main_ordered = combined_feed_main is not None and (
            bool(epilogue_nodes)
            or all(
                read.name == template.get_name()
                for node in node2.get_nodes()
                for read in node.read_writes.reads
            )
        )
        candidate_reductions, candidate_reduction_nodes = (
            self._partition_local_reductions(template, node2.get_nodes())
            if isinstance(template, Buffer)
            else ([], OrderedSet())
        )
        exact_reduction_candidate = bool(candidate_reductions) and all(
            node in candidate_reduction_nodes for node in node2.get_nodes()
        )
        eligible = isinstance(template, Buffer) and (
            bool(candidate_reductions)
            or feed_main_ordered
            or combined_softmax is not None
            or self._grouped_sum_normalize_config_from_nodes(template, combined_nodes)
            is not None
            or self._grouped_absmax_normalize_config_from_nodes(
                template, combined_nodes
            )
            is not None
        )
        if not eligible:
            return False
        if exact_reduction_candidate:
            return True
        template_snode = next(
            node
            for node in node1.get_nodes()
            if self.is_nv_universal_gemm_template(node)
        )
        return self._can_fuse_epilogue_impl(
            cast(SchedulerNode, template_snode), epilogue_nodes, node2
        )

    def define_kernel(
        self, src_code: str, node_schedule, precompile_metadata=None
    ) -> str:
        """
        Define a NVIDIA Universal GEMM kernel by writing source code and generating wrapper.

        Based on CuteDSLScheduling.define_kernel.
        """
        wrapper = V.graph.wrapper_code

        # Use the string as the key for caching
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )

        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"nv_universal_gemm_{kernel_hash}"
        else:
            kernel_name = f"nv_universal_gemm_{fused_name}_{kernel_hash}"

        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, _, kernel_path = get_path(code_hash(src_code), "py")

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(
            f"async_compile.nv_universal_gemm({kernel_name!r}, r'''"
        )
        compile_wrapper.splice(src_code, strip=True)
        if precompile_metadata is not None:
            compile_wrapper.writeline(
                f"''', precompile_metadata={precompile_metadata!r})"
            )
        else:
            compile_wrapper.writeline("''')")

        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name

    @overload
    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        *,
        only_gen_src_code: Literal[False] = False,
    ) -> str | None: ...

    @overload
    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        *,
        only_gen_src_code: Literal[True],
    ) -> NVGemmGeneratedSource: ...

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        *,
        only_gen_src_code: bool = False,
    ) -> str | NVGemmGeneratedSource | None:
        """
        Codegen a NVIDIA Universal GEMM template with optional epilogue fusion.

        If `only_gen_src_code=True` the src code will be returned instead of being
        codegenned into the wrapper (used for benchmarking).
        """
        log.debug(
            "NVGEMM codegen_template: template_node=%s, epilogue_nodes=%s, prologue_nodes=%s",
            template_node,
            [n.get_name() for n in epilogue_nodes] if epilogue_nodes else [],
            [n.get_name() for n in prologue_nodes] if prologue_nodes else [],
        )
        assert self.is_nv_universal_gemm_template(template_node), (  # noqa: S101
            "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
            "SchedulerNode that wraps a NVUniversalGemmBuffer or MultiTemplateBuffer with NVGEMM choice"
        )
        assert not prologue_nodes, (  # noqa: S101
            "NVIDIA Universal GEMM doesn't support prologue fusion yet"
        )

        template_node = cast(SchedulerNode, template_node)

        original_ir_node = template_node.node
        assert isinstance(original_ir_node, Buffer)  # noqa: S101
        original_buffer_name = original_ir_node.get_name()

        epilogue_plan = self._epilogue_plan(original_ir_node, epilogue_nodes)
        epilogue_nodes = epilogue_plan.nodes
        feed_main = epilogue_plan.feed_main
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node,
            require_epilogue_fusion=bool(epilogue_nodes),
            min_tile_n=epilogue_plan.min_tile_n,
        )

        epilogue_fn_code: str | None = None
        epilogue_is_cutedsl = False
        epilogue_reads: list[str] = []
        epilogue_writes: list[str] = []
        epilogue_var_renames: dict[str, Any] = {}
        local_reduce: GemmReductionPlan | None = None

        if epilogue_nodes:
            scheduler = V.graph.scheduler
            try:
                local_reduce = self._finalize_reduction_plan(
                    original_ir_node, epilogue_plan
                )
                if feed_main is not None:
                    assert local_reduce is not None  # noqa: S101
                    primary_output = local_reduce.primary_output
                    epilogue_fn_code = (
                        f"def {EPILOGUE_FN_NAME}(accum):\n    D = accum\n    return D"
                    )
                    epilogue_writes = [primary_output]
                    epilogue_var_renames = {
                        _ACCUMULATOR_ARG_NAME: original_buffer_name,
                        "D": primary_output,
                    }
                evt_nodes = epilogue_plan.evt_nodes
                fused_buffer_names: OrderedSet[str] = OrderedSet(
                    n.get_name() for n in epilogue_nodes
                )
                fused_buffer_names.add(original_buffer_name)
                removed_buffers_with_gemm = V.graph.removed_buffers.copy()
                if scheduler.can_buffer_be_removed_through_fusion(
                    original_buffer_name, fused_buffer_names
                ):
                    removed_buffers_with_gemm.add(original_buffer_name)

                if evt_nodes:
                    evt_buffers = [
                        node.node
                        for node in evt_nodes
                        if isinstance(node.node, ComputedBuffer)
                    ]
                    try:
                        reads, writes, var_renames, evt_code = (
                            GemmEpilogueIRCodegen.from_buffers(
                                original_buffer_name,
                                evt_buffers,
                                removed_buffers_with_gemm,
                                EPILOGUE_FN_NAME,
                            )
                        )
                        epilogue_is_cutedsl = True
                    except NotImplementedError:
                        reads, writes, var_renames, evt_code = (
                            CutlassEVTCodegen.ir_to_evt_python_code(
                                original_buffer_name,
                                list(evt_nodes),
                                removed_buffers_with_gemm,
                                fn_name=EPILOGUE_FN_NAME,
                                as_standalone_function=True,
                            )
                        )
                    epilogue_fn_code = evt_code
                    epilogue_reads = reads
                    epilogue_writes = writes
                    epilogue_var_renames = var_renames
                    if feed_main is not None and local_reduce is not None:
                        d_buf = var_renames.get("D")
                        if isinstance(d_buf, str):
                            local_reduce = dataclasses.replace(
                                local_reduce, primary_output=d_buf
                            )

                if not only_gen_src_code:
                    write_bufs = OrderedSet(epilogue_writes)
                    if local_reduce is not None:
                        write_bufs.update(local_reduce.auxiliary_outputs)
                    for node in epilogue_nodes:
                        node_name = node.get_name()
                        if node_name in write_bufs:
                            continue
                        if scheduler.can_buffer_be_removed_through_fusion(
                            node_name, fused_buffer_names
                        ):
                            V.graph.removed_buffers.add(node_name)
                    if (
                        original_buffer_name not in write_bufs
                        and scheduler.can_buffer_be_removed_through_fusion(
                            original_buffer_name, fused_buffer_names
                        )
                    ):
                        V.graph.removed_buffers.add(original_buffer_name)
                    for node in epilogue_nodes:
                        node.mark_run()

                log.debug(
                    "NVGEMM epilogue fusion: %d nodes, reads=%s, writes=%s, local_reduce=%s",
                    len(epilogue_nodes),
                    epilogue_reads,
                    epilogue_writes,
                    local_reduce,
                )
            except (NotImplementedError, AssertionError) as e:
                log_fn = log.debug if only_gen_src_code else log.warning
                log_fn("NVGEMM epilogue codegen failed unexpectedly: %s", e)
                raise

        assert ctb.make_kernel_render is not None  # noqa: S101 # noqa: S101
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_is_cutedsl=epilogue_is_cutedsl,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
            local_reduce=local_reduce,
        )

        if not only_gen_src_code:
            template_node.mark_run()

        src_code = render()

        if only_gen_src_code:
            return NVGemmGeneratedSource(
                src_code,
                tuple(epilogue_reads),
                tuple(kernel.ordered_output_buffers()),
            )

        # Precompile only base (non-EFC) kernels. EFC kernels produce
        # closure-wrapped artifacts that can't be serialized to disk cache.
        if epilogue_nodes:
            precompile_metadata = None
        else:
            precompile_metadata = self._build_precompile_metadata(kernel, ctb)

        with V.set_kernel_handler(kernel):
            node_schedule: list[BaseSchedulerNode] = [template_node]
            if epilogue_nodes:
                node_schedule.extend(epilogue_nodes)
            kernel_name = self.define_kernel(
                src_code, node_schedule, precompile_metadata
            )

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.free_buffers_in_scheduler()
        return None

    def _build_precompile_metadata(self, kernel, ctb):
        """Extract shapes and dtypes from kernel inputs/output for subprocess precompilation.

        Only called for base (non-epilogue) kernels. Returns None if shapes are
        symbolic (dynamic shapes), in which case the subprocess will skip
        precompilation and the kernel compiles lazily on first call.
        """
        if not hasattr(kernel, "_template_input_args"):
            return None

        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}

        try:
            for param_name, input_node in kernel._template_input_args:
                size = input_node.get_size()
                precompile_shapes[param_name] = [int(s) for s in size]
                stride = input_node.get_stride()
                precompile_strides[param_name] = [int(s) for s in stride]
                precompile_dtypes[param_name] = str(
                    input_node.get_dtype()
                ).removeprefix("torch.")

            out_layout = cast(Layout, ctb.layout)
            precompile_shapes["output"] = [int(s) for s in out_layout.size]
            precompile_strides["output"] = [int(s) for s in out_layout.stride]
            precompile_dtypes["output"] = str(out_layout.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            log.debug(
                "Skipping NV Universal GEMM precompile metadata: symbolic sizes "
                "cannot be resolved to concrete values"
            )
            return None

        device = ctb.layout.device
        device_index = device.index if device.index is not None else 0

        import torch

        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)

        max_active_clusters = None
        kernel_name = ctb.kernel_metadata.get("kernel_name")
        try:
            if kernel_name and torch.cuda.is_available():
                from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
                    get_kernel_by_name,
                )

                k = get_kernel_by_name(kernel_name)
                if k is not None and hasattr(k, "impl"):
                    from cutlass.operators.providers.cutedsl.integration_utils.mma import (
                        get_max_active_clusters,
                    )

                    max_active_clusters = get_max_active_clusters(
                        k.impl.cluster_shape_mn
                    )
        except Exception:
            log.debug(
                "Failed to resolve max_active_clusters for precompile", exc_info=True
            )

        return {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
            "max_active_clusters": max_active_clusters,
        }

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[BaseSchedulerNode],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
        """Generate benchmark source for an NVGEMM template and its epilogue."""
        prologue, template, epilogue = nodes[0].get_prologue_template_epilogue(
            list(nodes)
        )

        with config.patch("benchmark_kernel", benchmark_kernel):
            try:
                generated = self.codegen_template(
                    template,
                    epilogue,
                    prologue,
                    only_gen_src_code=True,
                )
            except (NotImplementedError, AssertionError) as exc:
                from ..simd import CantSplit

                raise CantSplit("NVGEMM epilogue", "supported EVT") from exc

        assert isinstance(generated, NVGemmGeneratedSource)  # noqa: S101
        src_code = generated.source.replace(
            str(Placeholder.KERNEL_NAME), _BENCHMARK_KERNEL_PREFIX
        )

        if benchmark_kernel:
            src_code = self._add_benchmark_helpers(
                src_code,
                template,
                epilogue,
                list(generated.epilogue_reads),
                list(generated.output_buffers),
            )

        return src_code

    def _add_benchmark_helpers(
        self,
        src_code: str,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        epilogue_reads: list[str],
        output_bufs: list[str] | None = None,
    ) -> str:
        template_node = cast(SchedulerNode, template_node)
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        input_nodes = cast(list[Buffer], ctb.inputs)
        # Output store layouts in out_ptr order. A multi-store epilogue has one
        # per graph output; otherwise a single output (the fused final node, or
        # the plain GEMM layout).
        output_layouts: list[Layout] = []
        if output_bufs:
            for b in output_bufs:
                buf = V.graph.get_buffer(b)
                # pyrefly: ignore [missing-attribute]
                output_layouts.append(cast(Layout, buf.get_layout()))
        elif epilogue_nodes:
            final_node = cast(SchedulerNode, epilogue_nodes[-1])
            # pyrefly: ignore [missing-attribute]
            output_layouts.append(cast(Layout, final_node.node.get_layout()))
        else:
            output_layouts.append(cast(Layout, ctb.layout))

        args_code = IndentedBuffer()
        args_code.writeline("")
        args_code.writeline("is_nvgemm = True")
        args_code.writeline("")
        args_code.writeline("def get_args():")
        with args_code.indent():
            args_code.writeline("import torch")
            args_code.writeline("from torch._dynamo.testing import rand_strided")
            args_code.writeline("args = []")

            for inp in input_nodes:
                size = V.graph.sizevars.optimization_hints(inp.get_size())
                stride = V.graph.sizevars.optimization_hints(inp.get_stride())
                dtype = inp.get_dtype()
                device = inp.get_device()
                args_code.writeline(
                    f"args.append(rand_strided({size}, {stride}, device='{device}', dtype={dtype}))"
                )

            for ol in output_layouts:
                out_size = V.graph.sizevars.optimization_hints(ol.size)
                out_stride = V.graph.sizevars.optimization_hints(ol.stride)
                args_code.writeline(
                    f"args.append(rand_strided({out_size}, {out_stride}, device='{ol.device}', dtype={ol.dtype}))"
                )

            for read_name in epilogue_reads:
                buf = V.graph.get_buffer(read_name)
                size = V.graph.sizevars.optimization_hints(buf.get_size())
                stride = V.graph.sizevars.optimization_hints(buf.get_stride())
                dtype = buf.get_dtype()
                device = buf.get_device()
                args_code.writeline(
                    f"args.append(rand_strided({size}, {stride}, device='{device}', dtype={dtype}))"
                )

            if ctb.workspace_size > 0:
                args_code.writeline(
                    f"args.append(torch.empty({ctb.workspace_size}, "
                    f"device='{output_layouts[0].device}', dtype=torch.int8))"
                )

            args_code.writeline("return args")

        args_code.writeline("")
        args_code.writeline("def call(args):")
        with args_code.indent():
            args_code.writeline("import torch")
            num_inputs = len(input_nodes)
            n_fixed = num_inputs + len(output_layouts)
            param_list = [f"args[{i}]" for i in range(n_fixed)]

            for j in range(len(epilogue_reads)):
                param_list.append(f"args[{n_fixed + j}]")

            if ctb.workspace_size > 0:
                param_list.append(f"args[{n_fixed + len(epilogue_reads)}]")

            params_str = ", ".join(param_list)
            args_code.writeline("stream = torch.cuda.current_stream().cuda_stream")
            bench_fn_name = f"{_BENCHMARK_KERNEL_PREFIX}_{MAIN_SUFFIX}"
            args_code.writeline(f"{bench_fn_name}({params_str}, stream=stream)")

        return src_code + args_code.getvalue()
