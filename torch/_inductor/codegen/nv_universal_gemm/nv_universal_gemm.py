# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM (NVGEMM) backend for PyTorch Inductor.

This module provides integration with the cutlass_api library to enable
high-performance GEMM kernels for NVIDIA GPUs.
"""

import itertools
from typing import Any, Optional, Union

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch
from torch._inductor.ir import Buffer, ChoiceCaller, Layout, TensorBox
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.template_heuristics.nv_universal_gemm import get_nvgemm_heuristics
from torch._inductor.utils import ensure_nv_universal_gemm_available
from torch._logging import getArtifactLogger


log = getArtifactLogger(__name__, "output_code")


class NVUniversalGemmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for NVIDIA Universal GEMM kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        kernel,  # cutlass_api.Kernel object
        accumulator_type: torch.dtype,
        workspace_size: int = 0,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self._compiled_artifact = None
        self._workspace: Optional[torch.Tensor] = None
        self.workspace_size = workspace_size

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
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

        a, b = input_tensors
        args = cutlass_api.arguments.GemmArguments(
            a,
            b,
            out,
            accumulator_type=self.accumulator_type,
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
        workspace_size: int = 0,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"nv_universal_gemm {kernel.metadata.kernel_name}",
        )
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size

        output_buffer = Buffer(name="nv_universal_gemm_out", layout=layout)

        self.bmreq = NVUniversalGemmBenchmarkRequest(
            kernel_name=name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_buffer),
            kernel=kernel,
            accumulator_type=accumulator_type,
            workspace_size=workspace_size,
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
        return f"nv_universal_gemm_{self.kernel.metadata.kernel_name}"

    def info_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": "nv_universal_gemm",
            "kernel_name": self.kernel.metadata.kernel_name,
        }


def _create_dummy_tensor_from_layout(layout: Layout) -> Optional[torch.Tensor]:
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


def add_nv_universal_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    inputs: MMKernelInputs,
    accumulator_type: Optional[torch.dtype] = None,
) -> None:
    """
    Add NVIDIA Universal GEMM kernels to the autotune choices.

    Queries cutlass_api for compatible kernels and adds them as autotune choices.
    """
    if ensure_nv_universal_gemm_available():
        import cutlass_api
    else:
        log.debug("cutlass_api not available, skipping NVIDIA Universal GEMM choices")
        return

    if accumulator_type is None:
        accumulator_type = torch.float32

    input_nodes = inputs.nodes()
    a_node, b_node = input_nodes

    # Create dummy tensors for cutlass_api's supports() checks
    a_tensor = _create_dummy_tensor_from_layout(a_node.get_layout())
    b_tensor = _create_dummy_tensor_from_layout(b_node.get_layout())
    out_tensor = _create_dummy_tensor_from_layout(layout)

    if a_tensor is None or b_tensor is None or out_tensor is None:
        log.debug("Failed to create dummy tensors")
        return

    try:
        args = cutlass_api.arguments.GemmArguments(
            a_tensor,
            b_tensor,
            out_tensor,
            accumulator_type=accumulator_type,
        )
    except Exception:
        log.debug("Failed to create GemmArguments", exc_info=True)
        return

    cc = get_cuda_arch()
    if cc is None:
        log.debug("Failed to get CUDA arch")
        return
    cc_int = int(cc)

    # EFC kernels support custom epilogue operations but have additional overhead.
    # Since NVGEMM doesn't support epilogue fusion yet (see nv_universal_gemm_scheduling.py),
    # we use non-EFC kernels which are equivalent for identity epilogue (out = acc).
    # TODO(nikhilap): Remove this filter once NVGEMM supports epilogue fusion.
    def _exclude_efc_kernels(metadata) -> bool:
        return "EFC" not in metadata.kernel_class.__name__

    kernels = cutlass_api.get_kernels(
        args=args, cc=cc_int, metadata_filter=_exclude_efc_kernels
    )
    if not kernels:
        log.debug("No compatible NVIDIA Universal GEMM kernels found")
        return

    max_configs = config.cuda.nvgemm_max_profiling_configs or len(kernels)

    heuristics = get_nvgemm_heuristics()
    kernels = heuristics.filter_kernels(kernels, inputs, max_configs, accumulator_type)
    num_added = 0
    for kernel in kernels:
        name = f"nv_universal_gemm_{next(NVUniversalGemmCaller.index_counter)}"
        workspace_size = kernel.get_workspace_size(args)
        try:
            caller = NVUniversalGemmCaller(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                kernel=kernel,
                accumulator_type=accumulator_type,
                workspace_size=workspace_size,
            )
            choices.append(caller)
            num_added += 1
        except Exception:
            log.debug("Failed to create NVIDIA Universal GEMM choice", exc_info=True)
    log.debug("Added %d NVIDIA Universal GEMM choices", num_added)
