# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

import dataclasses
import hashlib
import logging
from collections.abc import Sequence
from typing import Any, cast

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
    GemmReductionConfig,
    GemmReductionPlan,
)
from ...kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen
from ...kernel.loop_ir_epilogue_lowering import (
    centered_mean_consumer_type_unrolled_ir,
    GemmEpilogueIRAnalysis,
    grouped_reduction_ir,
    single_source_affine_ir,
)
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from ..cutlass.python_evt import CutlassEVTCodegen
from .nv_universal_gemm import NVUniversalGemmCaller


log = logging.getLogger(__name__)

MAIN_SUFFIX = "main"
_BENCHMARK_KERNEL_PREFIX = "nv_gemm_"
EPILOGUE_FN_NAME = "_epilogue_fn"


@dataclasses.dataclass(frozen=True)
class NVGemmEpiloguePlan:
    nodes: tuple[BaseSchedulerNode, ...]
    reductions: tuple[GemmReductionConfig, ...]
    reduction_nodes: tuple[BaseSchedulerNode, ...]
    candidate_reduction: GemmReductionConfig | None = None
    feed_main: GemmReductionConfig | None = None

    @property
    def evt_nodes(self) -> list[BaseSchedulerNode]:
        owned = OrderedSet(self.reduction_nodes)
        return [node for node in self.nodes if node not in owned]


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
    def is_nv_universal_gemm_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is an NVGEMM template SchedulerNode."""
        if not isinstance(node, SchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.node)

    @staticmethod
    def _best_nvgemm_choice(
        ir_node: MultiTemplateBuffer,
        require_epilogue_fusion: bool = False,
    ) -> NVUniversalGemmCaller:
        """Find the best NVUniversalGemmCaller from an MTB's choice timings."""
        choice_timings = ir_node.choice_timings()
        best: NVUniversalGemmCaller | None = None
        best_time = float("inf")
        for choice, timing in choice_timings.items():
            if not isinstance(choice, NVUniversalGemmCaller):
                continue
            if require_epilogue_fusion and not choice.supports_epilogue_fusion:
                continue
            if best is None or timing < best_time:
                best_time = timing
                best = choice
        if best is None:
            kind = "EFC kernel" if require_epilogue_fusion else "NVUniversalGemmCaller"
            raise RuntimeError(f"No {kind} found in choices")
        return best

    @staticmethod
    def get_nv_gemm_buffer_from_node(
        node: BaseSchedulerNode, require_epilogue_fusion: bool = False
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
            if isinstance(ir_node._render_caller, NVUniversalGemmCaller) and (
                not require_epilogue_fusion
                or ir_node._render_caller.supports_epilogue_fusion
            ):
                selected_choice = ir_node._render_caller
            elif require_epilogue_fusion:
                selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                    ir_node, require_epilogue_fusion=True
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

    @classmethod
    def _grouped_reduce_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) not in (1, 2):
            return None
        buffers = [snode.node for snode in nodes]
        if not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
            return None
        buffers = [cast(ComputedBuffer, buffer) for buffer in buffers]
        node = buffers[0]
        access_node = scheduler_node
        output_name = node.get_name()
        if len(node.data.ranges) != 2 or len(gemm_node.get_size()) != 2:
            return None
        try:
            m, n = (V.graph.sizevars.optimization_hint(v) for v in gemm_node.get_size())
            out_m, out_n = (
                V.graph.sizevars.optimization_hint(v) for v in node.data.ranges
            )
        except Exception:
            return None

        if isinstance(node.data, Reduction):
            reduction = node.data
            if len(reduction.reduction_ranges) != 1:
                return None
            group = V.graph.sizevars.optimization_hint(reduction.reduction_ranges[0])
            if m == out_m and n % group == 0 and out_n == n // group:
                axis = 1
                expected_strides = [n, group, 1]
                max_group = n
            elif n == out_n and m % group == 0 and out_m == m // group:
                axis = 0
                expected_strides = [group * n, 1, n]
                max_group = 16
            else:
                return None
        elif isinstance(node.data, Pointwise):
            if n != out_n or out_m <= 0 or m % out_m != 0:
                return None
            group = m // out_m
            axis = 0
            max_group = 16
            expected_strides = [group * n, 1]
        else:
            return None
        if group <= 1 or group > max_group:
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
        reduction_type, source_type = classified
        if reduction_type not in (
            "sum",
            "mean",
            "prod",
            "max",
            "min",
        ) or source_type not in ("identity", "square"):
            return None
        if (
            isinstance(node.data, Reduction)
            and node.data.reduction_type != reduction_type
        ):
            return None
        if len(buffers) == 2:
            finalizer = buffers[1]
            finalizer_store = GemmEpilogueIRAnalysis.store_from_buffer(finalizer)
            finalizer_reads = list(nodes[1].read_writes.reads)
            if (
                reduction_type != "sum"
                or not isinstance(node.data, Reduction)
                or not isinstance(finalizer.data, Pointwise)
                or not finalizer_reads
                or any(read.name != node.get_name() for read in finalizer_reads)
                or not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), finalizer.get_size()
                )
                or finalizer_store is None
                or single_source_affine_ir(finalizer_store, node.get_name())
                != (1.0 / group, 0.0)
            ):
                return None
            reduction_type = "mean"
            access_node = nodes[0]
            output_name = finalizer.get_name()

        reads = list(access_node.read_writes.reads)
        if not reads or any(read.name != gemm_node.get_name() for read in reads):
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
        if isinstance(node.data, Reduction):
            if len(reads) != 1:
                return None
            strides = V.graph.sizevars.stride_vars(reads[0].index, range_vars)
            if list(strides) != expected_strides:
                return None
        else:
            if len(reads) != group:
                return None
            expected_base = group * n * range_vars[0] + range_vars[1]
            offsets = []
            for read in reads:
                strides = V.graph.sizevars.stride_vars(read.index, range_vars)
                if list(strides) != expected_strides:
                    return None
                offsets.append(V.graph.sizevars.simplify(read.index - expected_base))
            expected_offsets = OrderedSet(offset * n for offset in range(group))
            if OrderedSet(offsets) != expected_offsets:
                return None
        return GemmReductionConfig(
            output_name=output_name,
            group=group,
            axis=axis,
            reduction_type=reduction_type,
            source_type=source_type,
        )

    @classmethod
    def _grouped_reduce_feeds_main_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        node = cast(ComputedBuffer, nodes[0].node)
        if not isinstance(node.data, Pointwise):
            return None
        reads = list(scheduler_node.read_writes.reads)
        range_vars = scheduler_node.read_writes.range_vars
        if range_vars is None or len(reads) <= 2:
            return None
        try:
            _, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
        except Exception:
            return None
        group = len(reads) - 1
        if group <= 1 or group > 4:
            return None
        store = GemmEpilogueIRAnalysis.store_from_buffer(node)
        if (
            store is None
            or centered_mean_consumer_type_unrolled_ir(
                store, gemm_node.get_name(), group
            )
            != "mean_linear:1:-1:0"
        ):
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
            output_name=node.get_name(),
            group=group,
            axis=0,
            reduction_type="mean",
            source_type="identity",
        )

    @classmethod
    def _epilogue_plan(
        cls,
        gemm_node: Buffer,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        candidate: BaseSchedulerNode | None = None,
    ) -> NVGemmEpiloguePlan:
        reductions: list[GemmReductionConfig] = []
        reduction_nodes: list[BaseSchedulerNode] = []
        candidate_reduction = None
        for node in epilogue_nodes:
            config = cls._grouped_reduce_config(gemm_node, node)
            if config is None:
                continue
            reductions.append(config)
            reduction_nodes.extend(node.get_nodes())
            if node is candidate:
                candidate_reduction = config
        nodes = tuple(
            scheduler_node
            for node in epilogue_nodes
            for scheduler_node in node.get_nodes()
        )
        feed_main = (
            cls._grouped_reduce_feeds_main_config(gemm_node, epilogue_nodes[0])
            if len(epilogue_nodes) == 1
            else None
        )
        return NVGemmEpiloguePlan(
            nodes,
            tuple(reductions),
            tuple(reduction_nodes),
            candidate_reduction,
            feed_main,
        )

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

        plan = self._epilogue_plan(
            ir_node, (*existing_epilogue_nodes, node_to_fuse), node_to_fuse
        )
        reduction_nodes = OrderedSet(plan.reduction_nodes)
        local_reduce = plan.candidate_reduction
        feed_main = plan.feed_main
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
            if not scaled_epilogue:
                return False

        for s_node in plan.nodes:
            node = s_node.node
            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return False
            if (
                feed_main is None
                and not isinstance(node.data, Pointwise)
                and s_node not in reduction_nodes
            ):
                log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                return False

            if (
                feed_main is None
                and s_node not in reduction_nodes
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
        # cutlass.operators' EVT supports matrix, row, and column loads here.
        # Reject unresolved inputs and shapes outside those broadcast patterns;
        # the trial EVT trace below remains the final capability check.
        gemm_size = ir_node.get_size()
        name_to_buf = V.graph.name_to_buffer | V.graph.graph_inputs
        internal_names = OrderedSet([ir_node.get_name()]) | OrderedSet(
            [s_node.get_name() for s_node in plan.nodes]
        )
        epilogue_inputs: OrderedSet[str] = OrderedSet()
        for s_node in plan.nodes:
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
        if scaled_epilogue and len(epilogue_inputs) > 4:
            log.debug(
                "NVGEMM scaled epilogue has %d captured tensors; at most four are supported",
                len(epilogue_inputs),
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
        elif node_to_fuse.is_reduction() and local_reduce is None:
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return False

        all_epilogue_nodes = list(plan.nodes)
        fused_buffer_names = OrderedSet(
            n.get_name() for n in [gemm_template_node, *all_epilogue_nodes]
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
            evt_nodes = plan.evt_nodes
            trial_reads: list[str] = []
            trial_writes: list[str] = []
            if feed_main is not None:
                evt_nodes = []
            if evt_nodes:
                trial_epilogue = CutlassEVTCodegen.ir_to_evt_python_code(
                    ir_node.get_name(),
                    evt_nodes,
                    trial_removed_buffers,
                )
                trial_reads = list(trial_epilogue.reads)
                trial_writes = list(trial_epilogue.writes)
            if scaled_epilogue:
                if len(trial_reads) > 4 or len(trial_writes) > 4:
                    return False
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

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        template = node1.get_template_node()
        return isinstance(template, Buffer) and (
            self._grouped_reduce_config(template, node2) is not None
            or self._grouped_reduce_feeds_main_config(template, node2) is not None
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

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        *,
        only_gen_src_code: bool = False,
    ) -> str | None:
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

        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        epilogue_fn_code: str | None = None
        epilogue_reads: list[str] = []
        epilogue_writes: list[str] = []
        epilogue_var_renames: dict[str, Any] = {}
        local_reduce: GemmReductionPlan | None = None

        if epilogue_nodes:
            scheduler = V.graph.scheduler
            try:
                plan = self._epilogue_plan(original_ir_node, epilogue_nodes)
                if len(plan.reductions) > 1:
                    raise NotImplementedError(
                        "NVGEMM supports one grouped local reduction"
                    )
                if plan.reductions:
                    reduction = plan.reductions[0]
                    local_reduce = GemmReductionPlan(
                        reduction_output=reduction.output_name,
                        group=reduction.group,
                        axis=reduction.axis,
                        reduction_type=reduction.reduction_type,
                        source_type=reduction.source_type,
                        primary_output=original_buffer_name,
                    )
                feed_main = plan.feed_main
                if feed_main is not None:
                    local_reduce = GemmReductionPlan(
                        reduction_output=None,
                        group=feed_main.group,
                        axis=feed_main.axis,
                        reduction_type=feed_main.reduction_type,
                        source_type=feed_main.source_type,
                        primary_output=feed_main.output_name,
                        feeds_main=True,
                    )
                    epilogue_fn_code = (
                        f"def {EPILOGUE_FN_NAME}(accum):\n    D = accum\n    return D"
                    )
                    epilogue_writes = [feed_main.output_name]
                    epilogue_var_renames = {
                        GEMM_ACCUMULATOR_ARG_NAME: original_buffer_name,
                        "D": feed_main.output_name,
                    }
                evt_nodes = [] if feed_main is not None else plan.evt_nodes
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
                        lowered_epilogue = LoopIRCuteDSLCodegen.from_buffers(
                            original_buffer_name,
                            evt_buffers,
                            removed_buffers_with_gemm,
                            EPILOGUE_FN_NAME,
                        )
                    except NotImplementedError:
                        lowered_epilogue = CutlassEVTCodegen.ir_to_evt_python_code(
                            original_buffer_name,
                            list(evt_nodes),
                            removed_buffers_with_gemm,
                            fn_name=EPILOGUE_FN_NAME,
                            as_standalone_function=True,
                        )
                    epilogue_fn_code = lowered_epilogue.source
                    epilogue_reads = list(lowered_epilogue.reads)
                    epilogue_writes = list(lowered_epilogue.writes)
                    epilogue_var_renames = lowered_epilogue.renames

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
                log.warning("NVGEMM epilogue codegen failed unexpectedly: %s", e)
                raise

        if ctb.make_kernel_render is None:
            raise AssertionError("expected ctb.make_kernel_render to be not None")
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
            local_reduce=local_reduce,
        )

        if not only_gen_src_code:
            template_node.mark_run()

        src_code = render()

        if only_gen_src_code:
            return src_code

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
        prologue, template, epilogue = nodes[0].get_prologue_template_epilogue(
            list(nodes)
        )

        epilogue_reads: list[str] = []
        output_bufs: list[str] = []
        if epilogue:
            template_sn = cast(SchedulerNode, template)
            if not isinstance(template_sn.node, Buffer):
                raise AssertionError(f"expected Buffer, got {type(template_sn.node)}")
            original_buffer_name = template_sn.node.get_name()
            plan = self._epilogue_plan(template_sn.node, epilogue)
            evt_nodes = [] if plan.feed_main is not None else plan.evt_nodes
            removed_buffers_with_gemm = V.graph.removed_buffers.copy()
            if not plan.reductions:
                removed_buffers_with_gemm.add(original_buffer_name)
            try:
                if evt_nodes:
                    lowered_epilogue = CutlassEVTCodegen.ir_to_evt_python_code(
                        original_buffer_name,
                        evt_nodes,
                        removed_buffers_with_gemm,
                    )
                    epilogue_reads = list(lowered_epilogue.reads)
                    d_buf = lowered_epilogue.renames.get("D")
                    output_bufs = ([d_buf] if d_buf else []) + [
                        w for w in lowered_epilogue.writes if w != d_buf
                    ]
                if plan.reductions:
                    output_bufs = [
                        original_buffer_name,
                        *output_bufs,
                        plan.reductions[0].output_name,
                    ]
                elif plan.feed_main is not None:
                    output_bufs = [plan.feed_main.output_name]
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM benchmark epilogue codegen failed: %s", e)

        with config.patch("benchmark_kernel", benchmark_kernel):
            src_code = self.codegen_template(
                template,
                epilogue,
                prologue,
                only_gen_src_code=True,
            )

        if src_code is None:
            raise AssertionError("expected src_code to be not None")
        src_code = src_code.replace(
            str(Placeholder.KERNEL_NAME), _BENCHMARK_KERNEL_PREFIX
        )

        if benchmark_kernel:
            src_code = self._add_benchmark_helpers(
                src_code, template, epilogue, epilogue_reads, output_bufs
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
