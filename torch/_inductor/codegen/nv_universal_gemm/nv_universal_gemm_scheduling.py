# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

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
    def get_nv_gemm_buffer_from_node(
        node: BaseSchedulerNode, require_epilogue_fusion: bool = False
    ) -> NVUniversalGemmBuffer:
        """Extract NVUniversalGemmBuffer from node (direct or via MultiTemplateBuffer)."""
        assert isinstance(node, SchedulerNode)
        ir_node = node.node

        if isinstance(ir_node, NVUniversalGemmBuffer):
            return ir_node
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Honor an explicit swap/finalize — the fusion benchmark loop swaps
            # in each EFC choice one at a time and must not re-select from timings.
            if isinstance(ir_node._render_caller, NVUniversalGemmCaller) and (
                not require_epilogue_fusion
                or ir_node._render_caller.supports_epilogue_fusion
            ):
                selected_choice = ir_node._render_caller
            elif require_epilogue_fusion:
                # `best is None or timing < best_time` so a choice with timing=inf
                # (benchmark failed) can still win when it's the only EFC option.
                choice_timings = ir_node.choice_timings()
                best_efc_choice = None
                best_efc_time = float("inf")
                for choice, timing in choice_timings.items():
                    if (
                        isinstance(choice, NVUniversalGemmCaller)
                        and choice.supports_epilogue_fusion
                    ):
                        if best_efc_choice is None or timing < best_efc_time:
                            best_efc_time = timing
                            best_efc_choice = choice
                if best_efc_choice is None:
                    raise RuntimeError("No EFC kernel found for epilogue fusion")
                selected_choice = best_efc_choice
            else:
                min_choice, _ = ir_node.get_min_choice()
                if isinstance(min_choice, NVUniversalGemmCaller):
                    selected_choice = min_choice
                else:
                    choice_timings = ir_node.choice_timings()
                    best_nvgemm = None
                    best_time = float("inf")
                    for choice, timing in choice_timings.items():
                        if isinstance(choice, NVUniversalGemmCaller) and (
                            best_nvgemm is None or timing < best_time
                        ):
                            best_time = timing
                            best_nvgemm = choice
                    if best_nvgemm is None:
                        raise RuntimeError("No NVUniversalGemmCaller found in choices")
                    selected_choice = best_nvgemm
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
            if ir_node.variant != GemmVariant.GEMM:
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
                if choice.variant != GemmVariant.GEMM:
                    log.debug(
                        "NVGEMM epilogue fusion: MultiTemplateBuffer has non-GEMM EFC choices"
                    )
                    return False
            if not has_efc_choice:
                log.debug("NVGEMM epilogue fusion: no EFC kernel available in choices")
                return False

        scheduler_nodes_to_fuse = node_to_fuse.get_nodes()

        for s_node in scheduler_nodes_to_fuse:
            node = s_node.node
            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return False
            if not isinstance(node.data, Pointwise):
                log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                return False

            if not V.graph.sizevars.statically_known_list_equals(
                node.get_size(), ir_node.get_size()
            ):
                log.debug(
                    "NVGEMM epilogue fusion: size mismatch %s vs %s",
                    node.get_size(),
                    ir_node.get_size(),
                )
                return False

        # EFC kernels don't support broadcast. Reject conservatively when a read
        # can't be resolved — folded constants (weights/biases) aren't in
        # name_to_buf, and silently skipping them would let a stride-0 bias through.
        gemm_size = ir_node.get_size()
        name_to_buf = V.graph.name_to_buffer | V.graph.graph_inputs
        for s_node in scheduler_nodes_to_fuse:
            for rd in s_node.read_writes.reads:
                if rd.name == ir_node.get_name():
                    continue
                read_buf = name_to_buf.get(rd.name)
                if read_buf is None:
                    log.debug(
                        "NVGEMM epilogue fusion: read %s not in name_to_buffer/graph_inputs, refusing to fuse",
                        rd.name,
                    )
                    return False
                read_size = read_buf.get_size()
                if not V.graph.sizevars.statically_known_list_equals(
                    read_size, gemm_size
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s size %s != GEMM size %s (broadcast not supported)",
                        rd.name,
                        read_size,
                        gemm_size,
                    )
                    return False
                if hasattr(read_buf, "get_stride"):
                    for s in read_buf.get_stride():
                        if s == 0:
                            log.debug(
                                "NVGEMM epilogue fusion: read buffer %s has zero stride (broadcast not supported)",
                                rd.name,
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
        elif node_to_fuse.is_reduction():
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return False

        all_epilogue_nodes = list(existing_epilogue_nodes) + list(
            node_to_fuse.get_nodes()
        )

        # Multi-store chains (>1 node) not yet supported: _render_epilogue_kwargs
        # skips intermediate stores, so EpilogueArguments construction would fail.
        # Normally pre-fusion collapses chains, but reject explicitly to avoid
        # silent miscompile if that changes.
        if len(all_epilogue_nodes) > 1:
            log.debug(
                "NVGEMM epilogue fusion: multi-stage chains (%d nodes) not yet supported",
                len(all_epilogue_nodes),
            )
            return False

        trial_removed_buffers = V.graph.removed_buffers | OrderedSet(
            [ir_node.get_name()]
        )
        try:
            CutlassEVTCodegen.ir_to_evt_python_code(
                ir_node.get_name(),
                all_epilogue_nodes,
                trial_removed_buffers,
            )
        except (NotImplementedError, AssertionError) as e:
            log.debug("NVGEMM epilogue fusion: trial EVT codegen failed: %s", e)
            return False

        return True

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support horizontal fusion yet
        return False

    def define_kernel(self, src_code: str, node_schedule) -> str:
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
        assert self.is_nv_universal_gemm_template(template_node), (
            "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
            "SchedulerNode that wraps a NVUniversalGemmBuffer or MultiTemplateBuffer with NVGEMM choice"
        )
        assert not prologue_nodes, (
            "NVIDIA Universal GEMM doesn't support prologue fusion yet"
        )

        template_node = cast(SchedulerNode, template_node)

        original_ir_node = template_node.node
        assert isinstance(original_ir_node, Buffer)
        original_buffer_name = original_ir_node.get_name()

        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        epilogue_fn_code: str | None = None
        epilogue_reads: list[str] = []
        epilogue_writes: list[str] = []
        epilogue_var_renames: dict[str, Any] = {}

        if epilogue_nodes:
            try:
                removed_buffers_with_gemm = V.graph.removed_buffers | OrderedSet(
                    [original_buffer_name]
                )

                reads, writes, var_renames, evt_code = (
                    CutlassEVTCodegen.ir_to_evt_python_code(
                        original_buffer_name,
                        list(epilogue_nodes),
                        removed_buffers_with_gemm,
                        fn_name=EPILOGUE_FN_NAME,
                        as_standalone_function=True,
                    )
                )
                epilogue_fn_code = evt_code
                epilogue_reads = reads
                epilogue_writes = writes
                epilogue_var_renames = var_renames

                if not only_gen_src_code:
                    fused_buffer_names: OrderedSet[str] = OrderedSet(
                        n.get_name() for n in epilogue_nodes
                    )
                    fused_buffer_names.add(original_buffer_name)
                    scheduler = V.graph.scheduler
                    # Must add to removed_buffers BEFORE mark_run: mark_run emits
                    # AllocateLine eagerly, and codegen_allocation only skips it
                    # when the name is already in removed_buffers.
                    for node in epilogue_nodes:
                        node_name = node.get_name()
                        if epilogue_writes and node_name == epilogue_writes[-1]:
                            continue
                        if scheduler.can_buffer_be_removed_through_fusion(
                            node_name, fused_buffer_names
                        ):
                            V.graph.removed_buffers.add(node_name)
                    if (
                        epilogue_writes
                        and original_buffer_name != epilogue_writes[-1]
                        and scheduler.can_buffer_be_removed_through_fusion(
                            original_buffer_name, fused_buffer_names
                        )
                    ):
                        V.graph.removed_buffers.add(original_buffer_name)
                    for node in epilogue_nodes:
                        node.mark_run()

                log.debug(
                    "NVGEMM epilogue fusion: %d nodes, reads=%s, writes=%s",
                    len(epilogue_nodes),
                    epilogue_reads,
                    epilogue_writes,
                )
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM epilogue codegen failed unexpectedly: %s", e)
                raise

        assert ctb.make_kernel_render is not None
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
        )

        if not only_gen_src_code:
            template_node.mark_run()

        src_code = render()

        if only_gen_src_code:
            return src_code

        with V.set_kernel_handler(kernel):
            node_schedule: list[BaseSchedulerNode] = [template_node]
            if epilogue_fn_code and epilogue_nodes:
                node_schedule.extend(epilogue_nodes)
            kernel_name = self.define_kernel(src_code, node_schedule)

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.free_buffers_in_scheduler()
        return None

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
        if epilogue:
            template_sn = cast(SchedulerNode, template)
            assert isinstance(template_sn.node, Buffer)
            original_buffer_name = template_sn.node.get_name()
            removed_buffers_with_gemm = V.graph.removed_buffers | OrderedSet(
                [original_buffer_name]
            )
            try:
                reads, _, _, _ = CutlassEVTCodegen.ir_to_evt_python_code(
                    original_buffer_name,
                    list(epilogue),
                    removed_buffers_with_gemm,
                )
                epilogue_reads = reads
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM benchmark epilogue codegen failed: %s", e)

        with config.patch("benchmark_kernel", benchmark_kernel):
            src_code = self.codegen_template(
                template,
                epilogue,
                prologue,
                only_gen_src_code=True,
            )

        assert src_code is not None
        src_code = src_code.replace(
            str(Placeholder.KERNEL_NAME), _BENCHMARK_KERNEL_PREFIX
        )

        if benchmark_kernel:
            src_code = self._add_benchmark_helpers(
                src_code, template, epilogue, epilogue_reads
            )

        return src_code

    def _add_benchmark_helpers(
        self,
        src_code: str,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        epilogue_reads: list[str],
    ) -> str:
        template_node = cast(SchedulerNode, template_node)
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        input_nodes = cast(list[Buffer], ctb.inputs)
        output_layout = cast(Layout, ctb.layout)

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

            out_size = V.graph.sizevars.optimization_hints(output_layout.size)
            out_stride = V.graph.sizevars.optimization_hints(output_layout.stride)
            out_dtype = output_layout.dtype
            out_device = output_layout.device
            args_code.writeline(
                f"args.append(rand_strided({out_size}, {out_stride}, device='{out_device}', dtype={out_dtype}))"
            )

            for read_name in epilogue_reads:
                buf = V.graph.get_buffer(read_name)
                if buf is not None:
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
                    f"device='{out_device}', dtype=torch.int8))"
                )

            args_code.writeline("return args")

        args_code.writeline("")
        args_code.writeline("def call(args):")
        with args_code.indent():
            args_code.writeline("import torch")
            num_inputs = len(input_nodes)
            param_list = [f"args[{i}]" for i in range(num_inputs)]
            param_list.append(f"args[{num_inputs}]")

            for j in range(len(epilogue_reads)):
                param_list.append(f"args[{num_inputs + 1 + j}]")

            if ctb.workspace_size > 0:
                param_list.append(f"args[{num_inputs + 1 + len(epilogue_reads)}]")

            params_str = ", ".join(param_list)
            args_code.writeline("stream = torch.cuda.current_stream().cuda_stream")
            bench_fn_name = f"{_BENCHMARK_KERNEL_PREFIX}_{MAIN_SUFFIX}"
            args_code.writeline(f"{bench_fn_name}({params_str}, stream=stream)")

        return src_code + args_code.getvalue()
