# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM (NVGEMM) backend for PyTorch Inductor.

This module provides integration with the cutlass_api library to enable
high-performance GEMM kernels for NVIDIA GPUs.
"""

import itertools
import re
from enum import auto, Enum
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
    _create_gemm_arguments,
    _get_scaled_gemm_modes,
)
from torch._inductor.ir import Buffer, ChoiceCaller, Layout, TensorBox
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.template_heuristics.nv_universal_gemm import get_nvgemm_heuristics
from torch._inductor.utils import ensure_nv_universal_gemm_available
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

    @property
    def arguments_class_name(self) -> str:
        """Return the cutlass_api arguments class name."""
        if self == GemmVariant.GROUPED_GEMM:
            return "GroupedGemmArguments"
        return "GemmArguments"


class NVUniversalGemmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for NVIDIA Universal GEMM kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        kernel,  # cutlass_api.Kernel object
        accumulator_type: torch.dtype,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self._compiled_artifact = None
        self._workspace: torch.Tensor | None = None
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b

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

        from torch._inductor.utils import _ensure_fp4_dtype_registered

        _ensure_fp4_dtype_registered()

        args = _create_gemm_arguments(
            self.variant.name,
            input_tensors,
            out,
            self.accumulator_type,
            **helper_kwargs,
        )

        if self._compiled_artifact is None:
            self._compiled_artifact = self.kernel.compile(args)
        artifact = self._compiled_artifact
        kernel = self.kernel

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

    def cleanup_run_fn(self) -> None:
        self._workspace = None


class NVUniversalGemmCaller(ChoiceCaller):
    """
    ChoiceCaller for NVIDIA Universal GEMM kernels.

    Wraps a cutlass_api kernel and integrates with Inductor's autotuning.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        kernel,  # cutlass_api.Kernel object
        accumulator_type: torch.dtype,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        supports_epilogue_fusion: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"{variant.op_name} {kernel.metadata.kernel_name}",
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
        )

    def __str__(self) -> str:
        return f"NVUniversalGemmCaller({self.kernel.metadata.kernel_name})"

    def benchmark(self, *args, out) -> float:
        self.bmreq.benchmark_with_cudagraphs = self._benchmark_with_cudagraphs
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        from torch._inductor.ir import NVUniversalGemmBuffer

        # Without memoization, each call registers a new buffer (via
        # TemplateBuffer.__init__ → V.graph.register_buffer), leaking orphan
        # buffers into the graph's name tables during EFC benchmarking.
        if self._cached_output_node is not None:
            return self._cached_output_node

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
                self.kernel.metadata.kernel_name,
                self.accumulator_type,
                self.scale_type_a,
                self.scale_type_b,
                self.swizzle_type_a,
                self.swizzle_type_b,
            )
        )

    def info_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.variant.op_name,
            "kernel_name": self.kernel.metadata.kernel_name,
        }

    def get_make_kernel_render(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernel,
        )
        from torch._inductor.utils import Placeholder

        kernel_metadata = {
            "kernel_name": self.kernel.metadata.kernel_name,
            "min_cc": self.kernel.metadata.min_cc,
        }
        accumulator_type = self.accumulator_type
        workspace_size = self.workspace_size
        variant = self.variant
        scale_type_a = self.scale_type_a
        scale_type_b = self.scale_type_b
        swizzle_type_a = self.swizzle_type_a
        swizzle_type_b = self.swizzle_type_b
        input_nodes = self.input_nodes

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
    avoiding real CUDA memory allocation. cutlass_api only needs shape/stride/dtype
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

    Excludes tile_M=64 EFC kernels: cutlass_api has a broadcast bug in the
    epilogue thread operation for aux-tensor inputs with tile_M=64, and we
    don't yet know at autotune time whether fusion will consume aux tensors.
    Non-EFC kernels still cover tile_M=64, so plain GEMM autotune is unaffected.

    Strictly requires the kernel name to encode tile dims; if cutlass_api ever
    changes the naming scheme, this raises rather than silently letting the
    broken tile_M=64 kernels through.
    """
    if "EFC" not in metadata.kernel_class.__name__:
        return False
    match = _TILE_RE.search(metadata.kernel_name)
    if match is None:
        raise RuntimeError(
            f"NVGEMM EFC kernel name does not match expected tile pattern "
            f"'tileMxNxK': {metadata.kernel_name}. The tile_M=64 broadcast "
            f"workaround in _include_efc_kernels_only depends on this naming "
            f"convention; update the regex or move to metadata-based filtering."
        )
    return int(match.group(1)) >= 128


def _exclude_efc_kernels(metadata) -> bool:
    """Filter to exclude EFC kernels (for non-epilogue cases)."""
    return "EFC" not in metadata.kernel_class.__name__


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
) -> None:
    """
    Unified implementation for adding NVIDIA Universal GEMM choices.

    Args:
        choices: List to append ChoiceCaller objects to
        layout: Output layout
        input_nodes: Input tensor nodes
        variant: The GEMM variant (determines behavior)
        accumulator_type: Accumulator dtype
        mm_inputs: Optional MMKernelInputs for heuristics
        scale_type_a: ScalingType for A (required for SCALED_GEMM)
        scale_type_b: ScalingType for B (required for SCALED_GEMM)
        swizzle_type_a: SwizzleType for A (required for SCALED_GEMM)
        swizzle_type_b: SwizzleType for B (required for SCALED_GEMM)
    """
    from torch._inductor.utils import _ensure_fp4_dtype_registered

    _ensure_fp4_dtype_registered()

    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        get_compatible_kernels,
    )

    # Create dummy tensors for cutlass_api's supports() checks.
    # Pass node dtype to handle FP4 ReinterpretView (uint8 storage viewed as float4_e2m1fn_x2).
    dummy_tensors = [
        _create_dummy_tensor_from_layout(
            node.get_layout(), dtype_override=node.get_dtype()
        )
        for node in input_nodes
    ]
    out_tensor = _create_dummy_tensor_from_layout(layout)

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

    non_efc_kernels = get_compatible_kernels(
        args, cc_int, metadata_filter=_exclude_efc_kernels
    )
    efc_kernels = get_compatible_kernels(
        args, cc_int, metadata_filter=_include_efc_kernels_only
    )
    if not non_efc_kernels and not efc_kernels:
        log.debug("No compatible %s kernels found", variant.op_name)
        return

    max_configs = config.nvgemm_max_profiling_configs or max(
        len(non_efc_kernels), len(efc_kernels)
    )
    if variant == GemmVariant.GEMM and mm_inputs is not None:
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

    num_added = 0
    for kernel, supports_epilogue_fusion in all_kernels:
        name = f"{variant.op_name}_{next(NVUniversalGemmCaller.index_counter)}"
        workspace_size = kernel.get_workspace_size(args)
        try:
            caller = NVUniversalGemmCaller(
                name=name,
                input_nodes=input_nodes,
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
            )
            choices.append(caller)
            num_added += 1
        except Exception:
            # Broad: caller construction touches cutlass_api / fake-mode tensors
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
        log.debug("cutlass_api not available, skipping NVIDIA Universal GEMM choices")
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=inputs.nodes(),
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=inputs,
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
            "cutlass_api not available, skipping NVIDIA Universal Grouped GEMM choices"
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
