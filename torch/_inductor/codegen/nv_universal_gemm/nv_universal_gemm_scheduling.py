# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import logging
from typing import Any, cast, Literal, overload, TYPE_CHECKING

import torch
from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
)
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import (
    Buffer,
    ComputedBuffer,
    Layout,
    MultiTemplateBuffer,
    NVUniversalGemmBuffer,
    Pointwise,
    Reduction,
)
from ...kernel.gemm_epilogue import (
    GEMM_ACCUMULATOR_ARG_NAME,
    GemmEpiloguePlan,
    GemmReductionGeometry,
)
from ...kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from ..cutlass.python_evt import CutlassEVTCodegen
from .epilogue_lowering import NVGemmEpilogueCapture, NVGemmEpilogueLowering
from .nv_universal_gemm import GemmVariant, NVUniversalGemmCaller


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...kernel.gemm_epilogue import GemmReductionPlan
    from .epilogue_lowering import NVGemmEpilogueProgram


log = logging.getLogger(__name__)

MAIN_SUFFIX = "main"
_BENCHMARK_KERNEL_PREFIX = "nv_gemm_"
EPILOGUE_FN_NAME = "_epilogue_fn"


@dataclasses.dataclass(frozen=True, kw_only=True)
class NVGemmGeneratedSource:
    source: str
    epilogue_reads: tuple[str, ...]
    output_buffers: tuple[str, ...]


class NVGemmVerticalFusionDecision(enum.Enum):
    FUSE = enum.auto()
    DEFER = enum.auto()
    REJECT = enum.auto()


class NVUniversalGemmScheduling(NVGemmEpilogueLowering, BaseScheduling):
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
    def _uses_swap_ab(ir_node: Any) -> bool:
        if isinstance(ir_node, NVUniversalGemmBuffer):
            return ir_node.swap_ab
        return (
            isinstance(ir_node, MultiTemplateBuffer)
            and isinstance(ir_node._render_caller, NVUniversalGemmCaller)
            and ir_node._render_caller.swap_ab
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
        if not isinstance(node, SchedulerNode):
            raise AssertionError(f"expected SchedulerNode, got {type(node)}")
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
        return (
            self.vertical_fusion_decision(node1, node2)
            is NVGemmVerticalFusionDecision.FUSE
        )

    def vertical_fusion_decision(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> NVGemmVerticalFusionDecision:
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
                return NVGemmVerticalFusionDecision.REJECT
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, template_snode),
                [
                    node
                    for node in fnode1.snodes
                    if not self.is_nv_universal_gemm_template(node)
                ],
                node2,
            )
        return NVGemmVerticalFusionDecision.REJECT

    @staticmethod
    def _lower_pointwise_epilogue(
        gemm_name: str,
        nodes: Sequence[BaseSchedulerNode],
        removed_buffers: OrderedSet[str],
        reduction_geometries: dict[str, GemmReductionGeometry] | None = None,
        suppressed_outputs: OrderedSet[str] | None = None,
    ) -> GemmEpiloguePlan:
        buffers = [node.node for node in nodes if isinstance(node.node, ComputedBuffer)]
        if len(buffers) != len(nodes):
            raise NotImplementedError("NVGEMM epilogue nodes must be computed buffers")
        try:
            return LoopIRCuteDSLCodegen.from_buffers(
                gemm_name,
                buffers,
                removed_buffers,
                EPILOGUE_FN_NAME,
                reduction_geometries,
                suppressed_outputs,
            )
        except NotImplementedError:
            return CutlassEVTCodegen.ir_to_evt_python_code(
                gemm_name,
                list(nodes),
                removed_buffers,
                fn_name=EPILOGUE_FN_NAME,
                as_standalone_function=True,
            )

    def _can_fuse_epilogue_impl(
        self,
        gemm_template_node: SchedulerNode,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> NVGemmVerticalFusionDecision:
        if not config.epilogue_fusion:
            return NVGemmVerticalFusionDecision.DEFER

        ir_node = gemm_template_node.node
        if not isinstance(ir_node, (NVUniversalGemmBuffer, MultiTemplateBuffer)):
            return NVGemmVerticalFusionDecision.DEFER

        if isinstance(ir_node, NVUniversalGemmBuffer):
            if ir_node.variant not in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM):
                log.debug(
                    "NVGEMM epilogue fusion: not supported for %s variant",
                    ir_node.variant.op_name,
                )
                return NVGemmVerticalFusionDecision.DEFER
            if not ir_node.supports_epilogue_fusion:
                log.debug(
                    "NVGEMM epilogue fusion: kernel %s does not support epilogue fusion",
                    ir_node.kernel_metadata.get("kernel_name", "unknown"),
                )
                return NVGemmVerticalFusionDecision.DEFER
            variants = (ir_node.variant,)
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Use _choices, not choice_timings() — the latter forces autotune sync.
            variants = tuple(
                choice.variant
                for choice in ir_node._choices
                if isinstance(choice, NVUniversalGemmCaller)
                and choice.supports_epilogue_fusion
            )
            if not variants:
                log.debug("NVGEMM epilogue fusion: no EFC kernel available in choices")
                return NVGemmVerticalFusionDecision.DEFER
            if any(
                variant not in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM)
                for variant in variants
            ):
                log.debug(
                    "NVGEMM epilogue fusion: MultiTemplateBuffer has unsupported EFC choices"
                )
                return NVGemmVerticalFusionDecision.DEFER

        epilogue_program = self._lower_epilogue(
            ir_node, (*existing_epilogue_nodes, node_to_fuse)
        )
        all_scheduler_nodes = epilogue_program.capture.nodes
        if not epilogue_program.supported:
            log.debug("NVGEMM could not lower every captured reduction region")
            return NVGemmVerticalFusionDecision.DEFER
        feeds_main = epilogue_program.feeds_main
        if feeds_main:
            fused_names = OrderedSet(
                scheduler_node.get_name() for scheduler_node in all_scheduler_nodes
            )
            fused_names.add(gemm_template_node.get_name())
            if not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                ir_node.get_name(), fused_names
            ):
                return NVGemmVerticalFusionDecision.DEFER
        reduction_region = epilogue_program.reduction_partition.region_for(
            node_to_fuse.get_nodes()
        )
        scaled_epilogue = all(
            variant == GemmVariant.SCALED_GEMM for variant in variants
        )
        reduction_plan = epilogue_program.reduction_plan
        if reduction_plan is not None and self._uses_swap_ab(ir_node):
            log.debug("NVGEMM swap_ab does not support fused local reductions")
            return NVGemmVerticalFusionDecision.DEFER
        if reduction_plan is not None and not all(
            variant.supports_reduction(reduction_plan) for variant in variants
        ):
            return NVGemmVerticalFusionDecision.DEFER

        for s_node in all_scheduler_nodes:
            node = s_node.node
            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return NVGemmVerticalFusionDecision.DEFER

        generated_reduction_nodes = OrderedSet(
            node
            for region in epilogue_program.generated_reduction_regions
            for node in region.nodes
        )
        generated_reduction_outputs = epilogue_program.generated_reduction_geometries
        if not feeds_main:
            for s_node in epilogue_program.pointwise_nodes:
                node = cast(ComputedBuffer, s_node.node)
                if (
                    s_node in generated_reduction_nodes
                    or node.get_name() in generated_reduction_outputs
                ):
                    continue
                if not isinstance(node.data, Pointwise):
                    log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                    return NVGemmVerticalFusionDecision.DEFER
                if not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), ir_node.get_size()
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: size mismatch %s vs %s",
                        node.get_size(),
                        ir_node.get_size(),
                    )
                    return NVGemmVerticalFusionDecision.DEFER
                if not V.graph.sizevars.statically_known_list_equals(
                    node.data.ranges, ir_node.get_size()
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: iteration-domain mismatch %s vs %s",
                        node.data.ranges,
                        ir_node.get_size(),
                    )
                    return NVGemmVerticalFusionDecision.DEFER
        # Epilogue inputs support matrix, row, and column loads here. Reject
        # other broadcast patterns before the pointwise lowerer's capability check.
        gemm_size = ir_node.get_size()
        name_to_buf = V.graph.name_to_buffer | V.graph.graph_inputs
        internal_names = OrderedSet([ir_node.get_name()]) | OrderedSet(
            node.node.get_name()
            for node in all_scheduler_nodes
            if isinstance(node.node, Buffer)
        )
        for s_node in epilogue_program.pointwise_nodes:
            for rd in s_node.read_writes.reads:
                if rd.name in internal_names:
                    continue
                read_buf = name_to_buf.get(rd.name)
                if read_buf is None:
                    log.debug(
                        "NVGEMM epilogue fusion: read %s not in name_to_buffer/graph_inputs, refusing to fuse",
                        rd.name,
                    )
                    return NVGemmVerticalFusionDecision.DEFER
                read_size = read_buf.get_size()
                if len(read_size) > len(gemm_size):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s has unsupported rank",
                        rd.name,
                    )
                    return NVGemmVerticalFusionDecision.DEFER
                padded_size = [1] * (len(gemm_size) - len(read_size)) + list(read_size)
                supported_shapes = (
                    gemm_size,
                    [1, gemm_size[1]],
                    [gemm_size[0], 1],
                    [1] * len(gemm_size),
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
                    return NVGemmVerticalFusionDecision.DEFER
        if not existing_epilogue_nodes:
            reads = OrderedSet(rd.name for rd in node_to_fuse.read_writes.reads)
            if ir_node.get_name() not in reads:
                log.debug(
                    "NVGEMM epilogue fusion: first epilogue node doesn't read from GEMM output"
                )
                return NVGemmVerticalFusionDecision.DEFER

        if node_to_fuse.has_aliasing_or_mutation():
            log.debug("NVGEMM epilogue fusion: node has aliasing or mutation")
            return NVGemmVerticalFusionDecision.DEFER
        elif (
            node_to_fuse.is_reduction() and reduction_region is None and not feeds_main
        ):
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return NVGemmVerticalFusionDecision.DEFER

        fused_buffer_names = OrderedSet(
            n.get_name() for n in [gemm_template_node, *epilogue_program.capture.nodes]
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
            trial_reads: tuple[str, ...] = ()
            epilogue_dtype = ir_node.get_dtype()
            pointwise_nodes = epilogue_program.pointwise_nodes
            codegen_nodes = pointwise_nodes
            if (
                reduction_plan is not None
                and reduction_plan.tensor_epilogue_returns_local_reduce
            ):
                codegen_nodes = epilogue_program.capture.nodes
                generated_nodes = OrderedSet(epilogue_program.reduction_partition.nodes)
                trial_removed_buffers.update(
                    node.node.get_name()
                    for node in generated_nodes
                    if isinstance(node.node, Buffer)
                    and node.node.get_name() != reduction_plan.reduction_output
                )
            if codegen_nodes:
                suppressed_outputs = OrderedSet()
                if (
                    reduction_plan is not None
                    and reduction_plan.tensor_epilogue_returns_local_reduce
                    and reduction_plan.reduction_output is not None
                ):
                    suppressed_outputs.add(reduction_plan.reduction_output)
                lowered_epilogue = self._lower_pointwise_epilogue(
                    ir_node.get_name(),
                    codegen_nodes,
                    trial_removed_buffers,
                    epilogue_program.generated_reduction_geometries,
                    suppressed_outputs,
                )
                trial_reads = lowered_epilogue.reads
                epilogue_dtype = V.graph.get_dtype(lowered_epilogue.writes[0])
            if GemmVariant.GEMM in variants:
                for read_name in trial_reads:
                    read_buf = name_to_buf.get(read_name)
                    read_dtype = None if read_buf is None else read_buf.get_dtype()
                    try:
                        dtype_supported = read_dtype is None or (
                            read_dtype.is_floating_point
                            and torch.promote_types(read_dtype, epilogue_dtype)
                            == epilogue_dtype
                        )
                    except (RuntimeError, TypeError):
                        dtype_supported = False
                    if not dtype_supported:
                        log.debug(
                            "NVGEMM dense epilogue input %s has dtype %s, which cannot be represented by %s",
                            read_name,
                            read_dtype,
                            epilogue_dtype,
                        )
                        return NVGemmVerticalFusionDecision.REJECT
            if scaled_epilogue:
                for read_name in trial_reads:
                    read_buf = name_to_buf.get(read_name)
                    if read_buf is None:
                        log.debug(
                            "NVGEMM scaled epilogue input %s cannot be resolved",
                            read_name,
                        )
                        return NVGemmVerticalFusionDecision.DEFER
        except NotImplementedError as e:
            log.debug("NVGEMM epilogue fusion: trial pointwise codegen failed: %s", e)
            return NVGemmVerticalFusionDecision.DEFER

        return NVGemmVerticalFusionDecision.FUSE

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support horizontal fusion yet
        return False

    def has_conflicting_epilogue_reductions(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        read_names = OrderedSet(read.name for read in node1.read_writes.reads)
        read_names.intersection_update(read.name for read in node2.read_writes.reads)
        for name in read_names:
            producer = V.graph.try_get_buffer(name)
            if not isinstance(producer, Buffer) or not self._has_nvgemm_choice(
                producer
            ):
                continue
            program = self._lower_epilogue(producer, (node1, node2))
            if program.has_unclaimed_reduction:
                return True
        return False

    @staticmethod
    def has_bool_output(node: BaseSchedulerNode) -> bool:
        return any(
            isinstance(scheduler_node.node, ComputedBuffer)
            and scheduler_node.node.get_dtype() == torch.bool
            for scheduler_node in node.get_nodes()
        )

    def has_nvgemm_bool_output(self, node: BaseSchedulerNode) -> bool:
        if not self.has_bool_output(node):
            return False
        is_template = self.is_nv_universal_gemm_template(node)
        if is_template or self.is_nv_universal_gemm_fused_template(node):
            return True
        return any(
            isinstance(producer := V.graph.try_get_buffer(read.name), Buffer)
            and self._has_nvgemm_choice(producer)
            for read in node.read_writes.reads
        )

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        has_reduction = any(
            isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, Reduction)
            for node in node1.get_nodes()
        )
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
                capture = NVGemmEpilogueCapture.from_nodes(
                    producer_buffer, (node1, node2)
                )
                if self._feed_main_config(capture):
                    return 0
                combined_partition = self._partition_local_reductions(capture)
                if combined_partition.owns(node2.get_nodes()):
                    log.debug(
                        "prioritizing NVGEMM reduction chain %s -> %s",
                        node1.get_name(),
                        node2.get_name(),
                    )
                    return 0
                if combined_partition.owns(node1.get_nodes()):
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
        combined_program = self._lower_epilogue(template, (*epilogue_nodes, node2))
        if not combined_program.supported:
            return False
        feed_main_ordered = combined_program.feeds_main and (
            bool(epilogue_nodes)
            or all(
                read.name == template.get_name()
                for node in node2.get_nodes()
                for read in node.read_writes.reads
            )
        )
        candidate_nodes = node2.get_nodes()
        reduction_partition = combined_program.reduction_partition
        generated_reduction_outputs = combined_program.generated_reduction_geometries
        generated_reduction = any(
            isinstance(node.node, ComputedBuffer)
            and node.node.get_name() in generated_reduction_outputs
            for node in candidate_nodes
        )
        if not (
            reduction_partition.intersects(candidate_nodes)
            or feed_main_ordered
            or generated_reduction
        ):
            return False
        template_snode = next(
            node
            for node in node1.get_nodes()
            if self.is_nv_universal_gemm_template(node)
        )
        return (
            self._can_fuse_epilogue_impl(
                cast(SchedulerNode, template_snode), epilogue_nodes, node2
            )
            is NVGemmVerticalFusionDecision.FUSE
        )

    @staticmethod
    def _schedule_reduction_plan(
        program: NVGemmEpilogueProgram,
    ) -> GemmReductionPlan | None:
        if not program.supported:
            return None
        reduction_plan = program.reduction_plan
        if (
            reduction_plan is None
            or not reduction_plan.feeds_main
            or reduction_plan.reduction_output is None
        ):
            return reduction_plan
        fused_names = OrderedSet(node.get_name() for node in program.capture.nodes)
        fused_names.add(program.capture.gemm.get_name())
        if V.graph.scheduler.can_buffer_be_removed_through_fusion(
            reduction_plan.reduction_output, fused_names
        ):
            return dataclasses.replace(reduction_plan, reduction_output=None)
        return reduction_plan

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
        if not self.is_nv_universal_gemm_template(template_node):
            raise AssertionError(
                "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
                "SchedulerNode that wraps a NVUniversalGemmBuffer or MultiTemplateBuffer with NVGEMM choice"
            )
        if prologue_nodes:
            raise AssertionError(
                "NVIDIA Universal GEMM doesn't support prologue fusion yet"
            )

        template_node = cast(SchedulerNode, template_node)

        original_ir_node = template_node.node
        if not isinstance(original_ir_node, Buffer):
            raise AssertionError(f"expected Buffer, got {type(original_ir_node)}")
        original_buffer_name = original_ir_node.get_name()

        epilogue_program = self._lower_epilogue(original_ir_node, epilogue_nodes)
        epilogue_nodes = epilogue_program.capture.nodes
        feeds_main = epilogue_program.feeds_main
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node,
            require_epilogue_fusion=bool(epilogue_nodes),
            min_tile_n=epilogue_program.min_tile_n,
        )

        lowered_epilogue = GemmEpiloguePlan()
        reduction_plan: GemmReductionPlan | None = None

        if epilogue_nodes:
            scheduler = V.graph.scheduler
            try:
                reduction_plan = self._schedule_reduction_plan(epilogue_program)
                if feeds_main:
                    if reduction_plan is None:
                        raise AssertionError("expected feed-main reduction plan")
                    primary_output = reduction_plan.primary_output
                    lowered_epilogue = GemmEpiloguePlan(
                        writes=(primary_output,),
                        renames={
                            GEMM_ACCUMULATOR_ARG_NAME: original_buffer_name,
                            "D": primary_output,
                        },
                        source=(
                            f"def {EPILOGUE_FN_NAME}(accum):\n"
                            "    D = accum\n    return D"
                        ),
                        is_evt_fallback=True,
                    )
                pointwise_nodes = epilogue_program.pointwise_nodes
                fused_buffer_names: OrderedSet[str] = OrderedSet(
                    n.get_name() for n in epilogue_nodes
                )
                fused_buffer_names.add(original_buffer_name)
                removed_buffers_with_gemm = V.graph.removed_buffers.copy()
                if scheduler.can_buffer_be_removed_through_fusion(
                    original_buffer_name, fused_buffer_names
                ):
                    removed_buffers_with_gemm.add(original_buffer_name)

                codegen_nodes = pointwise_nodes
                if (
                    reduction_plan is not None
                    and reduction_plan.tensor_epilogue_returns_local_reduce
                ):
                    codegen_nodes = epilogue_program.capture.nodes
                    generated_nodes = OrderedSet(
                        epilogue_program.reduction_partition.nodes
                    )
                    removed_buffers_with_gemm.update(
                        node.node.get_name()
                        for node in generated_nodes
                        if isinstance(node.node, Buffer)
                        and node.node.get_name() != reduction_plan.reduction_output
                    )

                if codegen_nodes:
                    suppressed_outputs = OrderedSet()
                    if (
                        reduction_plan is not None
                        and reduction_plan.tensor_epilogue_returns_local_reduce
                        and reduction_plan.reduction_output is not None
                    ):
                        suppressed_outputs.add(reduction_plan.reduction_output)
                    lowered_epilogue = self._lower_pointwise_epilogue(
                        original_buffer_name,
                        codegen_nodes,
                        removed_buffers_with_gemm,
                        epilogue_program.generated_reduction_geometries,
                        suppressed_outputs,
                    )
                    if feeds_main and reduction_plan is not None:
                        d_buf = lowered_epilogue.renames.get("D")
                        if isinstance(d_buf, str):
                            reduction_plan = dataclasses.replace(
                                reduction_plan, primary_output=d_buf
                            )

                if not only_gen_src_code:
                    write_bufs = OrderedSet(lowered_epilogue.writes)
                    if reduction_plan is not None:
                        write_bufs.update(reduction_plan.auxiliary_outputs)
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
                    "NVGEMM epilogue fusion: %d nodes, reads=%s, writes=%s, reduction_plan=%s",
                    len(epilogue_nodes),
                    lowered_epilogue.reads,
                    lowered_epilogue.writes,
                    reduction_plan,
                )
            except NotImplementedError as e:
                log_fn = log.debug if only_gen_src_code else log.warning
                log_fn("NVGEMM epilogue codegen failed unexpectedly: %s", e)
                raise

        if ctb.make_kernel_render is None:
            raise AssertionError("expected ctb.make_kernel_render to be not None")
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue=lowered_epilogue,
            local_reduce=reduction_plan,
        )

        if not only_gen_src_code:
            template_node.mark_run()

        src_code = render()

        if only_gen_src_code:
            return NVGemmGeneratedSource(
                source=src_code,
                epilogue_reads=lowered_epilogue.reads,
                output_buffers=tuple(kernel.ordered_output_buffers()),
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
            except NotImplementedError as exc:
                from ..simd import CantSplit

                raise CantSplit(
                    "NVGEMM epilogue", "supported pointwise lowering"
                ) from exc

        if not isinstance(generated, NVGemmGeneratedSource):
            raise AssertionError(
                f"expected NVGemmGeneratedSource, got {type(generated)}"
            )
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
