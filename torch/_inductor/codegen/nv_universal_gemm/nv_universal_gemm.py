# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM (NVGEMM) backend for PyTorch Inductor.

This module provides integration with the cutlass.operators library to enable
high-performance GEMM kernels for NVIDIA GPUs.
"""

import itertools
import re
from enum import auto, Enum
from typing import Any, Literal

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
    _compile_nvgemm,
    _create_gemm_arguments,
    _create_gemm_cache_key,
    _current_target_sm,
    _get_scaled_gemm_modes,
    _make_disk_config_key,
    _nvgemm_bias_add_epilogue,
    _rewrap_efc_compiled_obj,
    _unwrap_efc_compiled_obj,
)
from torch._inductor.heuristics.template.nv_universal_gemm import get_nvgemm_heuristics
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    FixedLayout,
    Layout,
    PermuteView,
    TensorBox,
)
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.utils import ensure_nv_universal_gemm_available
from torch._inductor.virtualized import V
from torch._logging import getArtifactLogger


log = getArtifactLogger(__name__, "output_code")


class GemmVariant(Enum):
    """
    Enum for different GEMM operation types supported by NVIDIA Universal GEMM.
    """

    GEMM = auto()

    GROUPED_GEMM = auto()

    SCALED_GEMM = auto()

    @property
    def op_name(self) -> str:
        """Return the operation name for logging and naming."""
        if self == GemmVariant.GROUPED_GEMM:
            return "nv_universal_grouped_gemm"
        if self == GemmVariant.SCALED_GEMM:
            return "nv_universal_scaled_gemm"
        return "nv_universal_gemm"


class NVUniversalGemmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for NVIDIA Universal GEMM kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        kernel,  # cutlass.operators.Operator object
        accumulator_type: torch.dtype,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        swap_ab: bool = False,
        has_bias_epilogue: bool = False,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.has_bias_epilogue = has_bias_epilogue
        self._workspace: torch.Tensor | None = None
        self._disk_fn_cache: dict = {}
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.swap_ab = swap_ab

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        """Benchmark the NVIDIA Universal GEMM kernel.

        Override the base class to always create tensors from input_tensor_meta.
        This is necessary because input_nodes may be ReinterpretViews that share
        the same underlying buffer name. The autotuning framework deduplicates
        inputs by name (in AlgorithmSelectorCache.get_inputs()), resulting in
        fewer tensors than expected. By always creating from input_tensor_meta,
        we ensure each input gets its own tensor with the correct size/stride/offset
        from the view's layout.

        """
        from torch._inductor.runtime.benchmarking import benchmarker

        input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
        if out is None:
            out = self.output_tensor_meta.to_tensor()

        fn = self.make_run_fn(*input_tensors, out=out)
        try:
            if self.benchmark_with_cudagraphs:
                res = benchmarker.benchmark_gpu_with_cuda_graph(fn)
            else:
                res = self.do_bench(fn, *input_tensors, out=out)
        finally:
            self.cleanup_run_fn()
        return res

    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor):
        """Create a function to run the NVIDIA Universal GEMM kernel."""
        from cutlass.operators.artifact import CompiledArtifact

        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set
        from torch._inductor.utils import _ensure_fp4_dtype_registered

        _ensure_fp4_dtype_registered()

        if self.has_bias_epilogue:
            # Benchmark the real fused kernel (GEMM + bias-add epilogue). The
            # bias is the last input; the rest are the GEMM operands.
            return self._make_bias_run_fn(input_tensors, out)

        # swap_ab: transpose operands and write into a transposed view of `out`
        # (zero-copy). See _nvgemm_run for the full explanation.
        if self.swap_ab and len(input_tensors) >= 2:
            a, b = input_tensors[0], input_tensors[1]
            if len(input_tensors) >= 4:
                sa, sb = input_tensors[2], input_tensors[3]
                input_tensors = (b.t(), a.t(), sb, sa) + input_tensors[4:]
            else:
                input_tensors = (b.t(), a.t()) + input_tensors[2:]
            out = out.t()

        helper_kwargs: dict[str, Any] = {}
        if self.variant == GemmVariant.SCALED_GEMM:
            scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
                _get_scaled_gemm_modes(
                    self.scale_type_a,
                    self.swizzle_type_a,
                    self.scale_type_b,
                    self.swizzle_type_b,
                )
            )
            helper_kwargs = {
                "scale_mode_a": scale_mode_a,
                "swizzle_mode_a": swizzle_mode_a,
                "scale_mode_b": scale_mode_b,
                "swizzle_mode_b": swizzle_mode_b,
            }

        cache_key = _create_gemm_cache_key(input_tensors, out)
        dev_idx = input_tensors[0].device.index or 0
        kernel_name = self.kernel.metadata.operator_name
        disk_config_key = _make_disk_config_key(
            kernel_name,
            self.variant.name,
            self.accumulator_type,
            self.scale_type_a,
            self.scale_type_b,
            self.swizzle_type_a,
            self.swizzle_type_b,
        )

        def disk_fallback(kernel):
            compiled_fn = disk_cache_get(
                self._disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                dev_idx,
            )
            if compiled_fn is not None:
                compiled_fn = _rewrap_efc_compiled_obj(compiled_fn, kernel)
            if compiled_fn is not None:
                return CompiledArtifact(
                    compiled_fn, kernel, _current_target_sm(dev_idx)
                )
            return None

        artifact, args, kernel, was_compiled = _compile_nvgemm(
            self.variant.name,
            input_tensors,
            out,
            self.accumulator_type,
            kernel_obj=self.kernel,
            args_kwargs=helper_kwargs,
            fallback_fn=disk_fallback,
        )

        if was_compiled:
            obj = _unwrap_efc_compiled_obj(artifact.compiled_obj)
            disk_cache_set(
                self._disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                obj,
                dev_idx,
            )

        # Allocate workspace if needed
        if self.workspace_size > 0:
            self._workspace = torch.empty(
                self.workspace_size, device=out.device, dtype=torch.int8
            )
        else:
            self._workspace = None

        workspace = self._workspace

        def run_kernel():
            stream = torch.cuda.current_stream()
            kernel.run(
                args,
                artifact,
                stream=stream,
                workspace=workspace,
                assume_supported_args=True,
            )

        return run_kernel

    def _make_bias_run_fn(self, input_tensors, out):
        """Build a run fn for the fused GEMM + bias-add epilogue (addmm).

        Mirrors make_run_fn's disk caching but with the bias epilogue: the
        unwrap/rewrap machinery serializes the EFC artifact, so subprocess
        precompile (which writes the same key) hands off to the benchmark
        instead of recompiling serially.
        """
        from cutlass.operators.arguments import EpilogueArguments
        from cutlass.operators.artifact import CompiledArtifact

        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

        *gemm_tensors, bias = input_tensors
        gemm_tensors = tuple(gemm_tensors)
        epilogue_args = EpilogueArguments(_nvgemm_bias_add_epilogue, bias=bias, D=out)

        kernel_name = self.kernel.metadata.operator_name
        cache_key = _create_gemm_cache_key(
            gemm_tensors, out, has_epilogue=True, aux_tensors=(bias,)
        )
        dev_idx = gemm_tensors[0].device.index or 0
        disk_config_key = _make_disk_config_key(
            kernel_name,
            self.variant.name,
            self.accumulator_type,
            self.scale_type_a,
            self.scale_type_b,
            self.swizzle_type_a,
            self.swizzle_type_b,
        )

        def disk_fallback(kernel):
            compiled_fn = disk_cache_get(
                self._disk_fn_cache, kernel_name, disk_config_key, cache_key, dev_idx
            )
            if compiled_fn is not None:
                compiled_fn = _rewrap_efc_compiled_obj(
                    compiled_fn, kernel, epilogue_args=epilogue_args
                )
            if compiled_fn is not None:
                return CompiledArtifact(
                    compiled_fn, kernel, _current_target_sm(dev_idx)
                )
            return None

        artifact, args, kernel, was_compiled = _compile_nvgemm(
            self.variant.name,
            gemm_tensors,
            out,
            self.accumulator_type,
            kernel_name=kernel_name,
            epilogue_args=epilogue_args,
            epilogue_source="nvgemm_addmm_bias_v1",
            fallback_fn=disk_fallback,
            base_kernel=self.kernel,
        )

        if was_compiled:
            disk_cache_set(
                self._disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                _unwrap_efc_compiled_obj(artifact.compiled_obj),
                dev_idx,
            )

        if self.workspace_size > 0:
            self._workspace = torch.empty(
                self.workspace_size, device=out.device, dtype=torch.int8
            )
        else:
            self._workspace = None
        workspace = self._workspace

        def run_kernel():
            kernel.run(
                args,
                artifact,
                stream=torch.cuda.current_stream(),
                workspace=workspace,
                assume_supported_args=True,
            )

        return run_kernel

    def precompile(self):
        input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
        out = self.output_tensor_meta.to_tensor()
        self.make_run_fn(*input_tensors, out=out)
        self.cleanup_run_fn()

    def cleanup_run_fn(self) -> None:
        self._workspace = None


class NVUniversalGemmCaller(ChoiceCaller):
    """
    ChoiceCaller for NVIDIA Universal GEMM kernels.

    Wraps a cutlass.operators kernel and integrates with Inductor's autotuning.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        kernel,  # cutlass.operators.Operator object
        accumulator_type: torch.dtype,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        supports_epilogue_fusion: bool = False,
        swap_ab: bool = False,
        kernel_layout: Layout | None = None,
        bias_node: Buffer | None = None,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"{variant.op_name} {kernel.metadata.operator_name}"
            + (" swap_ab" if swap_ab else "")
            + (" bias" if bias_node is not None else ""),
        )
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.supports_epilogue_fusion = supports_epilogue_fusion
        self.swap_ab = swap_ab
        self.bias_node = bias_node
        self._kernel_layout = kernel_layout or layout
        self._cached_output_node: TensorBox | None = None

        output_buffer = Buffer(name=f"{variant.op_name}_out", layout=layout)

        self.bmreq = NVUniversalGemmBenchmarkRequest(
            kernel_name=name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_buffer),
            kernel=kernel,
            accumulator_type=accumulator_type,
            workspace_size=workspace_size,
            variant=variant,
            scale_type_a=scale_type_a,
            scale_type_b=scale_type_b,
            swizzle_type_a=swizzle_type_a,
            swizzle_type_b=swizzle_type_b,
            swap_ab=swap_ab,
            has_bias_epilogue=bias_node is not None,
        )

    def __str__(self) -> str:
        return f"NVUniversalGemmCaller({self.kernel.metadata.operator_name})"

    def precompile(self):
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        self.bmreq.benchmark_with_cudagraphs = self._benchmark_with_cudagraphs
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        from torch._inductor.ir import NVUniversalGemmBuffer

        # Without memoization, each call registers a new buffer (via
        # TemplateBuffer.__init__ → V.graph.register_buffer), leaking orphan
        # buffers into the graph's name tables during EFC benchmarking.
        #
        # But a buffer registered during a benchmark sub-context can have its
        # name freed when that context rolls back `V.graph.buffers` (names are
        # `buf{len(buffers)}`, so a freed slot is reused). A stale cached buffer
        # then collides with a fresh buffer that reused its name, corrupting
        # finalize_multi_template_buffers. Drop the cache entry in that case and
        # rebuild a properly-registered buffer.
        if self._cached_output_node is not None:
            # pyrefly: ignore [missing-attribute]
            cached_name = self._cached_output_node.data.data.get_name()
            if cached_name in V.graph.name_to_buffer:
                return self._cached_output_node
            self._cached_output_node = None

        # For swap_ab, use the original (M, N) layout so the scheduler and
        # downstream IR see the expected shape. The runtime handles the
        # transpose internally via a temp buffer in _nvgemm_run.
        buffer = NVUniversalGemmBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            kernel=self.kernel,
            accumulator_type=self.accumulator_type,
            workspace_size=self.workspace_size,
            variant=self.variant,
            scale_type_a=self.scale_type_a,
            scale_type_b=self.scale_type_b,
            swizzle_type_a=self.swizzle_type_a,
            swizzle_type_b=self.swizzle_type_b,
            supports_epilogue_fusion=self.supports_epilogue_fusion,
            swap_ab=self.swap_ab,
            bias_node=self.bias_node,
        )
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]

        self._cached_output_node = TensorBox.create(buffer)
        return self._cached_output_node

    def call_name(self) -> str:
        return self.name

    def to_callable(self):
        return self.bmreq.make_run_fn

    def hash_key(self) -> str:
        # `select_algorithm` uses this as a precompile dedup key. Two callers
        # wrapping the same physical kernel name but with different accumulator/
        # scale/swizzle types produce distinct compiled artifacts; collapsing
        # them here would silently drop the second from autotuning.
        return "_".join(
            str(x)
            for x in (
                self.variant.op_name,
                self.kernel.metadata.operator_name,
                self.accumulator_type,
                self.scale_type_a,
                self.scale_type_b,
                self.swizzle_type_a,
                self.swizzle_type_b,
                "swap_ab" if self.swap_ab else "",
            )
        )

    def info_dict(self) -> dict[str, Any]:
        info = {
            "name": self.name,
            "backend": self.variant.op_name,
            "kernel_name": self.kernel.metadata.operator_name,
        }
        if self.swap_ab:
            info["swap_ab"] = True
        return info

    def get_make_kernel_render(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernel,
        )
        from torch._inductor.utils import Placeholder

        kernel_metadata = {
            "kernel_name": self.kernel.metadata.operator_name,
            "min_cc": self.kernel.designed_for_min_cc,
        }
        accumulator_type = self.accumulator_type
        workspace_size = self.workspace_size
        variant = self.variant
        scale_type_a = self.scale_type_a
        scale_type_b = self.scale_type_b
        swizzle_type_a = self.swizzle_type_a
        swizzle_type_b = self.swizzle_type_b
        input_nodes = self.input_nodes
        swap_ab = self.swap_ab
        has_bias = self.bias_node is not None

        def make_kernel_render(
            out_node,
            hint_override=None,
            epilogue_fn_code=None,
            epilogue_reads=None,
            epilogue_writes=None,
            epilogue_var_renames=None,
        ):
            from torch._inductor.ir import StorageBox, TensorBox

            processed_inputs = []
            for inp in input_nodes:
                if isinstance(inp, TensorBox):
                    inp = inp.data
                if isinstance(inp, StorageBox):
                    inp = inp.data
                processed_inputs.append(inp)

            # A baked bias is the last input, consumed by the epilogue rather
            # than as a GEMM operand.
            bias_node = None
            if has_bias:
                bias_node = processed_inputs[-1]
                processed_inputs = processed_inputs[:-1]

            kernel_name = str(Placeholder.KERNEL_NAME)

            render_kernel = NVUniversalGemmKernel(
                kernel_name=kernel_name,
                # pyrefly: ignore [bad-argument-type]
                input_nodes=processed_inputs,
                output_node=out_node,
                kernel_metadata=kernel_metadata,
                accumulator_type=accumulator_type,
                workspace_size=workspace_size,
                variant=variant,
                scale_type_a=scale_type_a,
                scale_type_b=scale_type_b,
                swizzle_type_a=swizzle_type_a,
                swizzle_type_b=swizzle_type_b,
                epilogue_fn_code=epilogue_fn_code,
                epilogue_reads=epilogue_reads,
                epilogue_writes=epilogue_writes,
                epilogue_var_renames=epilogue_var_renames,
                swap_ab=swap_ab,
                # pyrefly: ignore [bad-argument-type]
                bias_node=bias_node,
            )

            def render():
                return render_kernel.render()

            return render_kernel, render

        return make_kernel_render


def _create_dummy_tensor_from_layout(
    layout: Layout, dtype_override: torch.dtype | None = None
) -> torch.Tensor | None:
    """
    Create a FakeTensor from a Layout for kernel filtering.

    Uses Layout.get_example() which creates FakeTensors within V.fake_mode,
    avoiding real CUDA memory allocation. cutlass.operators only needs shape/stride/dtype
    metadata for its supports() checks.
    """
    try:
        result = layout.get_example()
        if dtype_override is not None and result.dtype != dtype_override:
            result = result.view(dtype_override)
        return result
    except Exception:
        # Broad: layout.get_example()/torch.empty_strided under fake mode can
        # raise a variety of unexpected errors (TypeError, AssertionError, etc.)
        # depending on stride/symint state. Failing to materialize a dummy
        # should never abort autotune — just skip this candidate.
        return None


_TILE_RE = re.compile(r"tile(\d+)x\d+x\d+")


def _include_efc_kernels_only(metadata) -> bool:
    """Filter to include only EFC (Epilogue Fusion Compatible) kernels.

    Excludes tile_M=64 EFC kernels: cutlass.operators has a broadcast bug in the
    epilogue thread operation for aux-tensor inputs with tile_M=64, and we
    don't yet know at autotune time whether fusion will consume aux tensors.
    Non-EFC kernels still cover tile_M=64, so plain GEMM autotune is unaffected.

    Strictly requires the kernel name to encode tile dims; if cutlass.operators ever
    changes the naming scheme, this raises rather than silently letting the
    broken tile_M=64 kernels through.
    """
    if "EFC" not in metadata.operator_class.__name__:
        return False
    match = _TILE_RE.search(metadata.operator_name)
    if match is None:
        raise RuntimeError(
            f"NVGEMM EFC kernel name does not match expected tile pattern "
            f"'tileMxNxK': {metadata.operator_name}. The tile_M=64 broadcast "
            f"workaround in _include_efc_kernels_only depends on this naming "
            f"convention; update the regex or move to metadata-based filtering."
        )
    return int(match.group(1)) >= 128


def _exclude_efc_kernels(metadata) -> bool:
    """Filter to exclude EFC kernels (for non-epilogue cases)."""
    return "EFC" not in metadata.operator_class.__name__


def _add_nv_gemm_choices_impl(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    variant: GemmVariant,
    accumulator_type: torch.dtype,
    mm_inputs: MMKernelInputs | None = None,
    scale_type_a: Any | None = None,
    scale_type_b: Any | None = None,
    swizzle_type_a: Any | None = None,
    swizzle_type_b: Any | None = None,
    swap_ab: bool = False,
    kernel_layout: Layout | None = None,
    bias_node: Buffer | None = None,
) -> None:
    """
    Unified implementation for adding NVIDIA Universal GEMM choices.

    Args:
        choices: List to append ChoiceCaller objects to
        layout: Output layout
        input_nodes: GEMM operand nodes (A, B)
        variant: The GEMM variant (determines behavior)
        accumulator_type: Accumulator dtype
        mm_inputs: Optional MMKernelInputs for heuristics
        scale_type_a: ScalingType for A (required for SCALED_GEMM)
        scale_type_b: ScalingType for B (required for SCALED_GEMM)
        swizzle_type_a: SwizzleType for A (required for SCALED_GEMM)
        swizzle_type_b: SwizzleType for B (required for SCALED_GEMM)
        bias_node: Optional addmm bias, consumed as a fixed bias-add epilogue.
            Requires EFC kernels; only these are considered when set.
    """
    from torch._inductor.utils import _ensure_fp4_dtype_registered

    _ensure_fp4_dtype_registered()

    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        partition_compatible_kernels,
    )

    # Create dummy tensors for cutlass.operators's supports() checks.
    # Pass node dtype to handle FP4 ReinterpretView (uint8 storage viewed as float4_e2m1fn_x2).
    dummy_tensors = [
        _create_dummy_tensor_from_layout(
            node.get_layout(), dtype_override=node.get_dtype()
        )
        for node in input_nodes
    ]

    if swap_ab and len(dummy_tensors) >= 2:
        # swap_ab: transpose mat_a/mat_b dummies (and swap scales when scaled).
        # Original: dummy[0]=mat_a(M,K) row, dummy[1]=mat_b(K,N) col
        # Swap: new_A=mat_b.t()=(N,K) row, new_B=mat_a.t()=(K,M) col
        d_a, d_b = dummy_tensors[0], dummy_tensors[1]
        if d_a is not None and d_b is not None:
            if len(dummy_tensors) >= 4:
                # scaled: new_scale_a=scale_b, new_scale_b=scale_a
                d_sa, d_sb = dummy_tensors[2], dummy_tensors[3]
                dummy_tensors = [d_b.t(), d_a.t(), d_sb, d_sa] + dummy_tensors[4:]
            else:
                dummy_tensors = [d_b.t(), d_a.t()] + dummy_tensors[2:]

    effective_layout = kernel_layout if kernel_layout is not None else layout
    out_tensor = _create_dummy_tensor_from_layout(effective_layout)

    if any(t is None for t in dummy_tensors) or out_tensor is None:
        log.debug("Failed to create dummy tensors for %s", variant.op_name)
        return

    helper_kwargs: dict[str, Any] = {}
    if variant == GemmVariant.SCALED_GEMM:
        try:
            scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
                _get_scaled_gemm_modes(
                    scale_type_a,
                    swizzle_type_a,
                    scale_type_b,
                    swizzle_type_b,
                )
            )
        except NotImplementedError:
            return
        helper_kwargs = {
            "scale_mode_a": scale_mode_a,
            "swizzle_mode_a": swizzle_mode_a,
            "scale_mode_b": scale_mode_b,
            "swizzle_mode_b": swizzle_mode_b,
        }

    args = _create_gemm_arguments(
        variant.name,
        tuple(dummy_tensors),
        out_tensor,
        accumulator_type,
        **helper_kwargs,
    )

    cc = get_cuda_arch()
    if cc is None:
        log.debug("Failed to get CUDA arch")
        return
    cc_int = int(cc)

    # Single-pass partition over the ~390K-entry kernel cache. The two-pass
    # form below called `kernel.supports(args)` once per bucket -- i.e. twice
    # per non-EFC-class kernel -- across the full cache.
    def _classify(metadata) -> int:
        if _include_efc_kernels_only(metadata):
            return 1  # efc bucket (with tile_M >= 128)
        if _exclude_efc_kernels(metadata):
            return 0  # non-efc bucket
        # NOTE: tile_M < 128 EFC kernels are dropped due to a cutlass.operators
        # broadcast bug. Tracking: https://github.com/pytorch/pytorch/issues/181901
        return -1

    # A baked bias-add requires an epilogue-fusion-capable (EFC) kernel, so
    # scan only EFC kernels -- skips supports() on the far larger non-EFC set.
    # The args-filtered fast query is complete only for dense GEMM; scaled uses
    # direct block-scaled sub-provider enumeration (get_operators under-generates
    # operands from scaled args); anything else falls back to the full manifest.
    candidate_source: Literal["args", "scaled", "manifest"]
    if variant == GemmVariant.GEMM:
        candidate_source = "args"
    elif variant == GemmVariant.SCALED_GEMM:
        candidate_source = "scaled"
    else:
        candidate_source = "manifest"
    non_efc_kernels, efc_kernels = partition_compatible_kernels(
        args,
        cc_int,
        _classify,
        num_buckets=2,
        efc_only=bias_node is not None,
        candidate_source=candidate_source,
        classifier_key="nvgemm_efc_partition_v1",
    )
    if not config.epilogue_fusion or swap_ab:
        efc_kernels = []
    if bias_node is not None:
        non_efc_kernels = []
    if not non_efc_kernels and not efc_kernels:
        log.debug("No compatible %s kernels found", variant.op_name)
        return

    max_configs = config.nvgemm_max_profiling_configs or max(
        len(non_efc_kernels), len(efc_kernels)
    )
    if variant in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM) and mm_inputs is not None:
        heuristics = get_nvgemm_heuristics()
        non_efc_kernels = heuristics.filter_kernels(
            non_efc_kernels, mm_inputs, max_configs, accumulator_type
        )
        efc_kernels = heuristics.filter_kernels(
            efc_kernels, mm_inputs, max_configs, accumulator_type
        )
    else:
        # TODO(nikhilap): Enable heuristics for grouped GEMM
        # when nvMatmulHeuristics adds support
        non_efc_kernels = non_efc_kernels[:max_configs]
        efc_kernels = efc_kernels[:max_configs]

    all_kernels = [(kernel, False) for kernel in non_efc_kernels] + [
        (kernel, True) for kernel in efc_kernels
    ]

    # When a bias is baked in, it is appended as the last caller input so the
    # scheduler tracks it as a read; the caller/kernel split it back off.
    caller_input_nodes = (
        [*input_nodes, bias_node] if bias_node is not None else input_nodes
    )

    num_added = 0
    for kernel, supports_epilogue_fusion in all_kernels:
        name = f"{variant.op_name}_{next(NVUniversalGemmCaller.index_counter)}"
        workspace_size = kernel.get_workspace_size(args).size_bytes
        try:
            caller = NVUniversalGemmCaller(
                name=name,
                input_nodes=caller_input_nodes,
                layout=layout,
                kernel=kernel,
                accumulator_type=accumulator_type,
                workspace_size=workspace_size,
                variant=variant,
                scale_type_a=scale_type_a,
                scale_type_b=scale_type_b,
                swizzle_type_a=swizzle_type_a,
                swizzle_type_b=swizzle_type_b,
                supports_epilogue_fusion=supports_epilogue_fusion,
                swap_ab=swap_ab,
                kernel_layout=kernel_layout,
                bias_node=bias_node,
            )
            choices.append(caller)
            num_added += 1
        except Exception:
            # Broad: caller construction touches cutlass.operators / fake-mode tensors
            # which can raise types other than RuntimeError/ValueError. A single
            # bad choice should never abort the rest of autotune choice population.
            log.debug("Failed to create %s choice", variant.op_name, exc_info=True)

    log.debug("Added %d %s choices", num_added, variant.op_name)


def add_nv_universal_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    inputs: MMKernelInputs,
    accumulator_type: torch.dtype | None = None,
) -> None:
    """
    Add NVIDIA Universal GEMM kernels to the autotune choices.

    Thin wrapper around _add_nv_gemm_choices_impl for regular GEMM.
    """
    if not ensure_nv_universal_gemm_available():
        log.debug(
            "cutlass.operators not available, skipping NVIDIA Universal GEMM choices"
        )
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=inputs.nodes(),
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=inputs,
    )

    # swap_ab: swap A/B so the large N goes on the M-axis, improving tile
    # utilization for small-M / tall-skinny shapes (M << N). The kernel computes
    # (N, M) and the runtime writes it into a transposed view of the (M, N)
    # output. Mirrors the scaled path (see add_nv_universal_scaled_gemm_choices)
    # but without scale operands. Only valid for 2D mm: this entry point also
    # serves BMM (layout [B, M, N]), where the (M, N) / 2D-PermuteView swap logic
    # would be wrong, so skip it there.
    if not config.nvgemm_swap_ab or len(layout.size) != 2:
        return
    m, n = layout.size[0], layout.size[1]
    swap_kernel_layout = FixedLayout(layout.device, layout.dtype, [n, m])
    # Rank swap configs with the heuristic on the SWAPPED (N, M, K) problem:
    # new mat1 = mat_b.t() (N, K) row, new mat2 = mat_a.t() (K, M) col. Without
    # this, the swap variant would fall back to an unranked prefix and miss the
    # small-tile configs that make swap_ab win.
    mat_a, mat_b = inputs.mat1mat2()
    swap_inputs = MMKernelInputs(
        [PermuteView.create(mat_b, [1, 0]), PermuteView.create(mat_a, [1, 0])]
    )
    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=inputs.nodes(),
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=swap_inputs,
        swap_ab=True,
        kernel_layout=swap_kernel_layout,
    )


def add_nv_universal_addmm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    inputs: MMKernelInputs,
    bias_node: Buffer,
    alpha: float | int = 1,
    beta: float | int = 1,
    accumulator_type: torch.dtype | None = None,
) -> None:
    """Add NVGEMM addmm choices: ``bias + mat1 @ mat2`` via an EFC bias-add
    epilogue.

    Only the plain ``alpha == beta == 1`` case is handled; other scalings fall
    back to the remaining backends. ``bias_node`` should be the original
    (un-expanded) bias so a 1D row vector routes to the row-broadcast epilogue
    impl -- passing the (M, N) broadcast *view* is rejected by cutlass.operators.
    """
    if not ensure_nv_universal_gemm_available():
        return
    # Bias-add requires an epilogue-fusion-capable (EFC) kernel.
    if not config.epilogue_fusion:
        return
    if alpha != 1 or beta != 1:
        return

    # Only broadcastable row-vector or full biases are supported. The bias last
    # dim must match N; higher-rank or column-only biases are left to fallbacks.
    n = layout.size[-1]
    bias_size = bias_node.get_size()
    # Only rank-1 (row vector) or rank-2 (full) biases are supported. The
    # epilogue indexes bias_size[-1], so a rank-0 scalar bias would IndexError;
    # reject it (and higher-rank biases) so they fall through to other backends.
    if not 1 <= len(bias_size) <= 2:
        return
    if not V.graph.sizevars.statically_known_equals(bias_size[-1], n):
        return

    mat1, mat2 = inputs.mat1mat2()
    gemm_inputs = MMKernelInputs([mat1, mat2])
    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=[mat1, mat2],
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=gemm_inputs,
        bias_node=bias_node,
    )


def add_nv_universal_grouped_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: torch.dtype | None = None,
) -> None:
    """
    Add NVIDIA Universal Grouped GEMM kernels to the autotune choices.

    Thin wrapper around _add_nv_gemm_choices_impl for grouped GEMM.

    For grouped GEMM (contiguous offset variant):
    - A is (TotalM, K) with problems stacked along M
    - B is (G, K, N) where B[i] is the weight for problem i
    - offsets is (G,) marking where each problem ends in A
    - Output is (TotalM, N)
    """
    if not ensure_nv_universal_gemm_available():
        log.debug(
            "cutlass.operators not available, skipping NVIDIA Universal Grouped GEMM choices"
        )
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=input_nodes,
        variant=GemmVariant.GROUPED_GEMM,
        accumulator_type=accumulator_type or torch.float32,
    )


def add_nv_universal_scaled_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: torch.dtype | None = None,
    kernel_inputs: MMKernelInputs | None = None,
) -> None:
    """
    Add NVIDIA Universal Scaled GEMM (FP8) kernels to the autotune choices.

    The scaling type is inferred from the input shapes/dtypes.
    If the scaling mode is unsupported by NVGEMM, this function returns without
    adding any choices.
    """
    if not ensure_nv_universal_gemm_available():
        return

    from torch._inductor.utils import infer_scale_swizzle_ir

    if len(input_nodes) < 4:
        return

    mat_a, mat_b, scale_a, scale_b = input_nodes[:4]

    scale_type_a, swizzle_type_a = infer_scale_swizzle_ir(mat_a, scale_a)
    scale_type_b, swizzle_type_b = infer_scale_swizzle_ir(
        mat_b, scale_b, transpose=True
    )

    if scale_type_a is None or scale_type_b is None:
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=input_nodes,
        variant=GemmVariant.SCALED_GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=kernel_inputs,
        scale_type_a=scale_type_a,
        scale_type_b=scale_type_b,
        swizzle_type_a=swizzle_type_a,
        swizzle_type_b=swizzle_type_b,
    )

    # swap_ab: see add_nv_universal_gemm_choices for the rationale (swap A/B so
    # the large N lands on the well-tiled M-axis for small-M shapes).
    if not config.nvgemm_swap_ab:
        return

    # In the IR, mat_a=(M, K/2) row-major, mat_b=(K/2, N) column-major (already .t()).
    # swap_ab computes: un_transpose(mat_b) @ transpose(mat_a) = (N,K/2)@(K/2,M) = (N,M).
    # Scales: new A uses scale_b (weight scale), new B uses scale_a (activation scale).
    # The un-transposed mat_b is (N, K/2) row-major — same as the original weight.
    # The transposed mat_a is (K/2, M) column-major.
    # Both have the same layout pattern as the original (row-major A, column-major B).

    # Scale inference: new A is the original weight (N, K/2), no transpose.
    # For infer_scale_swizzle_ir, we need to pass the un-transposed shape.
    # mat_b.t() gives (N, K/2) — but we use transpose=True on mat_b itself
    # since mat_b is (K/2, N), so transpose=True flips it to (N, K/2).
    swap_scale_type_a, swap_swizzle_type_a = infer_scale_swizzle_ir(
        mat_b, scale_b, transpose=True
    )
    # New B is the original activation (M, K/2), used transposed as (K/2, M).
    # infer_scale_swizzle_ir with transpose=True on mat_a flips (M, K/2) to (K/2, M).
    swap_scale_type_b, swap_swizzle_type_b = infer_scale_swizzle_ir(
        mat_a, scale_a, transpose=True
    )

    if swap_scale_type_a is None or swap_scale_type_b is None:
        return

    # Kernel output shape is (N, M) — the transpose of the original (M, N)
    m, n = layout.size[0], layout.size[1]
    swap_kernel_layout = FixedLayout(layout.device, layout.dtype, [n, m])

    # Skip heuristic filtering for swap_ab: mm_inputs has original (M, N, K) but
    # the swapped kernel sees (N, M, K). Let the benchmark pick the best kernel.
    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=input_nodes,
        variant=GemmVariant.SCALED_GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=None,
        scale_type_a=swap_scale_type_a,
        scale_type_b=swap_scale_type_b,
        swizzle_type_a=swap_swizzle_type_a,
        swizzle_type_b=swap_swizzle_type_b,
        swap_ab=True,
        kernel_layout=swap_kernel_layout,
    )
