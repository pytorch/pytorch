# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM (NVGEMM) backend for PyTorch Inductor.

This module provides integration with the cutlass_api library to enable
high-performance GEMM kernels for NVIDIA GPUs.
"""

import itertools
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
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_utils import (
    to_cutlass_scale_mode,
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
        # Always create tensors from input_tensor_meta, ignoring passed-in tensors
        input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
        if out is None:
            out = self.output_tensor_meta.to_tensor()

        fn = self.make_run_fn(*input_tensors, out=out)
        return self.do_bench(fn, *input_tensors, out=out)

    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor):
        """Create a function to run the NVIDIA Universal GEMM kernel."""
        import cutlass_api

        args = self._create_gemm_arguments(cutlass_api, input_tensors, out)

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

    def _create_gemm_arguments(self, cutlass_api, input_tensors, out):
        """Create the appropriate GemmArguments based on variant."""
        if self.variant == GemmVariant.GROUPED_GEMM:
            a, b, offsets = input_tensors
            return cutlass_api.arguments.GroupedGemmArguments(
                a,
                b,
                out,
                accumulator_type=self.accumulator_type,
                offsets=offsets,
            )
        elif self.variant == GemmVariant.SCALED_GEMM:
            from cutlass_api.arguments import ScaledTensor

            scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(
                self.scale_type_a, self.swizzle_type_a
            )
            scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(
                self.scale_type_b, self.swizzle_type_b
            )

            a, b, scale_a, scale_b = input_tensors
            scaled_a = ScaledTensor(a, scale_a, scale_mode_a, swizzle_mode_a)
            scaled_b = ScaledTensor(b, scale_b, scale_mode_b, swizzle_mode_b)
            return cutlass_api.arguments.GemmArguments(
                scaled_a,
                scaled_b,
                out,
                accumulator_type=self.accumulator_type,
            )
        else:
            a, b = input_tensors
            return cutlass_api.arguments.GemmArguments(
                a,
                b,
                out,
                accumulator_type=self.accumulator_type,
            )

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
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        from torch._inductor.ir import NVUniversalGemmBuffer

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
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)

    def call_name(self) -> str:
        return self.name

    def to_callable(self):
        return self.bmreq.make_run_fn

    def hash_key(self) -> str:
        return f"{self.variant.op_name}_{self.kernel.metadata.kernel_name}"

    def info_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.variant.op_name,
            "kernel_name": self.kernel.metadata.kernel_name,
        }


def _create_dummy_tensor_from_layout(layout: Layout) -> torch.Tensor | None:
    """
    Create a FakeTensor from a Layout for kernel filtering.

    Uses Layout.get_example() which creates FakeTensors within V.fake_mode,
    avoiding real CUDA memory allocation. cutlass_api only needs shape/stride/dtype
    metadata for its supports() checks.
    """
    try:
        return layout.get_example()
    except Exception:
        return None


def _exclude_efc_kernels(metadata) -> bool:
    """
    Filter out EFC kernels.

    EFC kernels support custom epilogue operations but have additional overhead.
    Since NVGEMM doesn't support epilogue fusion yet (see nv_universal_gemm_scheduling.py),
    we use non-EFC kernels which are equivalent for identity epilogue (out = acc).
    TODO(nikhilap): Remove this filter once NVGEMM supports epilogue fusion.
    """
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
    import cutlass_api

    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        get_compatible_kernels,
    )

    # Create dummy tensors for cutlass_api's supports() checks
    dummy_tensors = [
        _create_dummy_tensor_from_layout(node.get_layout()) for node in input_nodes
    ]
    out_tensor = _create_dummy_tensor_from_layout(layout)

    if any(t is None for t in dummy_tensors) or out_tensor is None:
        log.debug("Failed to create dummy tensors for %s", variant.op_name)
        return

    if variant == GemmVariant.GROUPED_GEMM:
        a_tensor, b_tensor, offs_tensor = dummy_tensors
        assert b_tensor is not None
        args = cutlass_api.arguments.GroupedGemmArguments(
            a_tensor,
            b_tensor,
            out_tensor,
            accumulator_type=accumulator_type,
            offsets=offs_tensor,
        )
    elif variant == GemmVariant.SCALED_GEMM:
        from cutlass_api.arguments import ScaledTensor

        scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(
            scale_type_a, swizzle_type_a
        )
        scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(
            scale_type_b, swizzle_type_b
        )
        if scale_mode_a is None or scale_mode_b is None:
            return

        a_tensor, b_tensor, scale_a_tensor, scale_b_tensor = dummy_tensors
        scaled_a = ScaledTensor(
            a_tensor,
            scale_a_tensor,
            scale_mode_a,
            swizzle_mode_a,
        )
        scaled_b = ScaledTensor(
            b_tensor,
            scale_b_tensor,
            scale_mode_b,
            swizzle_mode_b,
        )
        args = cutlass_api.arguments.GemmArguments(
            scaled_a,
            scaled_b,
            out_tensor,
            accumulator_type=accumulator_type,
        )
    else:
        a_tensor, b_tensor = dummy_tensors
        args = cutlass_api.arguments.GemmArguments(
            a_tensor,
            b_tensor,
            out_tensor,
            accumulator_type=accumulator_type,
        )

    cc = get_cuda_arch()
    if cc is None:
        log.debug("Failed to get CUDA arch")
        return
    cc_int = int(cc)

    kernels = get_compatible_kernels(args, cc_int, metadata_filter=_exclude_efc_kernels)
    if not kernels:
        log.debug("No compatible %s kernels found", variant.op_name)
        return

    max_configs = config.nvgemm_max_profiling_configs or len(kernels)
    if variant in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM) and mm_inputs is not None:
        heuristics = get_nvgemm_heuristics()
        kernels = heuristics.filter_kernels(
            kernels, mm_inputs, max_configs, accumulator_type
        )
    else:
        # TODO(nikhilap): Enable heuristics for grouped GEMM
        # when nvMatmulHeuristics adds support
        kernels = kernels[:max_configs]

    # Add callers for each kernel
    num_added = 0
    for kernel in kernels:
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
            )
            choices.append(caller)
            num_added += 1
        except Exception:
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
