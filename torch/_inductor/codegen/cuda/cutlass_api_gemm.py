# mypy: allow-untyped-defs
import itertools
import random
from typing import Any, Optional, Union

import torch
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    Layout,
    ShapeAsConstantBuffer,
    TensorBox,
)
from torch._inductor.utils import ensure_cutlass_api_available
from torch._logging import getArtifactLogger


log = getArtifactLogger(__name__, "output_code")

MAX_CUTLASS_API_PROFILING_CONFIGS = 5


class CutlassAPIBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """
    Benchmark request for cutlass_api kernels.

    This class handles benchmarking of cutlass_api kernels in potentially
    separate processes. Since kernel objects are not serializable, we store
    the kernel metadata and recreate the kernel when needed.
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        kernel_metadata: dict[str, Any],
        accumulator_type: torch.dtype,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type
        self._kernel = None
        self._compiled_artifact = None

    def _get_kernel(self):
        """
        Lazily get the kernel object.

        Kernel objects are not serializable, so we recreate them using the
        stored metadata when needed (e.g., in a subprocess).
        """
        if self._kernel is None:
            import cutlass_api

            kernels = cutlass_api.get_kernels(
                metadata_filter=lambda m: m.kernel_name
                == self.kernel_metadata["kernel_name"]
            )
            if not kernels:
                raise RuntimeError(
                    f"Could not find cutlass_api kernel: {self.kernel_metadata['kernel_name']}"
                )
            self._kernel = kernels[0]
        return self._kernel

    def _get_accumulator_type(self):
        """Convert torch dtype to cutlass accumulator type."""
        import cutlass

        dtype_map = {
            torch.float32: cutlass.Float32,
            torch.float16: cutlass.Float16,
            torch.bfloat16: cutlass.BFloat16,
        }
        return dtype_map.get(self.accumulator_type, cutlass.Float32)

    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor):
        """
        Create a function to run the cutlass_api kernel with the given tensors.
        """
        import cutlass_api

        kernel = self._get_kernel()
        a, b = input_tensors

        args = cutlass_api.arguments.GemmArguments(
            a,
            b,
            out,
            accumulator_type=self._get_accumulator_type(),
        )

        if self._compiled_artifact is None:
            self._compiled_artifact = kernel.compile(args)

        artifact = self._compiled_artifact

        def run_kernel():
            kernel.run(args, artifact, assume_supported_args=True)

        return run_kernel

    def cleanup_run_fn(self) -> None:
        """Clean up any resources used by the kernel."""


class CutlassAPIGemmCaller(ChoiceCaller):
    """
    ChoiceCaller for cutlass_api GEMM kernels.

    This class wraps a cutlass_api kernel object and integrates it with
    Inductor's autotuning system.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        kernel,  # cutlass_api.Kernel object
        accumulator_type: torch.dtype,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"cutlass_api {kernel.metadata.kernel_name}",
        )
        self.kernel = kernel
        self.accumulator_type = accumulator_type

        # Create a fake output buffer for TensorMeta
        output_buffer = Buffer(name="cutlass_api_out", layout=layout)

        # Create benchmark request with serializable metadata
        self.bmreq = CutlassAPIBenchmarkRequest(
            kernel_name=name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_buffer),
            kernel_metadata=self._serialize_metadata(kernel.metadata),
            accumulator_type=accumulator_type,
        )

    def _serialize_metadata(self, metadata) -> dict[str, Any]:
        """
        Convert KernelMetadata to a serializable dict.

        Only include fields needed to recreate the kernel.
        """
        return {
            "kernel_name": metadata.kernel_name,
            "min_cc": metadata.min_cc,
        }

    def __str__(self) -> str:
        return f"CutlassAPIGemmCaller({self.kernel.metadata.kernel_name})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> Union[TensorBox, ShapeAsConstantBuffer]:
        """Create the output node for this kernel choice."""
        # Import here to avoid circular imports
        from torch._inductor.ir import CutlassAPIGemmBuffer

        return TensorBox.create(
            CutlassAPIGemmBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                kernel=self.kernel,
                accumulator_type=self.accumulator_type,
            )
        )

    def call_name(self) -> str:
        """Return the kernel call name."""
        return self.name

    def to_callable(self):
        """Return callable that can execute this kernel."""
        return self.bmreq.make_run_fn

    def hash_key(self) -> str:
        """Return unique hash key for this choice."""
        return f"cutlass_api_{self.kernel.metadata.kernel_name}"

    def info_dict(self) -> dict[str, Any]:
        """Return information about this kernel."""
        return {
            "name": self.name,
            "backend": "cutlass_api",
            "kernel_name": self.kernel.metadata.kernel_name,
        }


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    """Convert torch dtype to cutlass dtype for comparison."""
    import cutlass

    dtype_map = {
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.int8: cutlass.Int8,
        torch.int32: cutlass.Int32,
    }
    return dtype_map.get(dtype)


def _get_stride_pattern(layout: Layout) -> tuple[int, ...]:
    """
    Convert Inductor layout strides to normalized stride pattern for comparison.

    Returns tuple like (0, 1) for row-major or (1, 0) for col-major.
    Normalizes strides: if size is 1, stride becomes 0 (matches cutlass_api convention).
    Handles symbolic strides using V.graph.sizevars.
    """
    from torch._inductor.virtualized import V

    stride = layout.stride
    size = layout.size
    result = []

    for s, sz in zip(stride, size):
        if V.graph.sizevars.statically_known_equals(sz, 1):
            result.append(0)
        elif V.graph.sizevars.statically_known_equals(s, 1):
            result.append(1)
        else:
            result.append(V.graph.sizevars.size_hint(s))

    return tuple(result)


def _get_alignment_bytes(layout: Layout) -> int:
    """
    Compute max alignment in BYTES (to match cutlass_api convention).

    cutlass_api uses byte alignments: 128, 64, 32, 16, 8, 4, 2, 1
    Inductor's get_max_alignment returns elements, so we convert.
    """
    from torch._inductor.codegen.cuda import cutlass_utils

    elem_alignment = cutlass_utils.get_max_alignment(layout)
    bytes_per_elem = layout.dtype.itemsize
    return elem_alignment * bytes_per_elem


def _stride_compatible(kernel_stride: tuple, operand_stride: tuple) -> bool:
    """
    Check if operand stride is compatible with kernel's expected stride.

    Matches TensorAttributes.supports() logic from cutlass_api/metadata.py:138-165.
    The key check is that the leading mode (stride=1) position must match.
    """
    if len(kernel_stride) == len(operand_stride):
        expected = kernel_stride
    elif len(kernel_stride) - 1 == len(operand_stride):
        expected = kernel_stride[1:]
    else:
        return False

    if all(x == 0 for x in expected) and all(x == 0 for x in operand_stride):
        return True

    if 1 not in expected:
        return True  # No constraint from kernel

    leading_idx = expected.index(1)
    if leading_idx >= len(operand_stride):
        return False

    return operand_stride[leading_idx] == 1


def _create_full_metadata_filter(
    a_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    accumulator_type: torch.dtype,
):
    """
    Create a metadata filter that replicates kernel.supports(args) logic.

    This should return exactly the same kernels as if we passed actual GemmArguments.
    Checks: dtype, accumulator_type, stride/layout compatibility, and alignment.
    """
    import cutlass

    # Convert dtypes
    dtype_map = {
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.int8: cutlass.Int8,
        torch.int32: cutlass.Int32,
    }

    a_dtype = dtype_map.get(a_layout.dtype)
    b_dtype = dtype_map.get(b_layout.dtype)
    out_dtype = dtype_map.get(out_layout.dtype)
    acc_dtype = dtype_map.get(accumulator_type)

    # Get stride patterns (normalized)
    a_stride = _get_stride_pattern(a_layout)
    b_stride = _get_stride_pattern(b_layout)
    out_stride = _get_stride_pattern(out_layout)

    # Get alignments in bytes
    a_align = _get_alignment_bytes(a_layout)
    b_align = _get_alignment_bytes(b_layout)
    out_align = _get_alignment_bytes(out_layout)

    def metadata_filter(meta) -> bool:
        """
        Filter kernels to match kernel.supports(args) logic.

        Checks performed (matching cutlass_api/metadata.py):
        1. dtype match (TensorAttributes.supports)
        2. accumulator_type match (GemmOperandsMetadata.supports)
        3. stride compatibility (TensorAttributes.supports)
        4. alignment divisibility (TensorAttributes.supports)
        """
        try:
            ops = meta.operands

            # 1. Check dtypes (TensorAttributes.supports)
            if a_dtype is not None and ops.A.dtype != a_dtype:
                return False
            if b_dtype is not None and ops.B.dtype != b_dtype:
                return False
            if out_dtype is not None and ops.out.dtype != out_dtype:
                return False

            # 2. Check accumulator type (GemmOperandsMetadata.supports)
            if acc_dtype is not None and ops.accumulator_type != acc_dtype:
                return False

            # 3. Check stride compatibility (TensorAttributes.supports)
            if not _stride_compatible(ops.A.stride, a_stride):
                return False
            if not _stride_compatible(ops.B.stride, b_stride):
                return False
            if not _stride_compatible(ops.out.stride, out_stride):
                return False

            # 4. Check alignment (TensorAttributes.supports)
            # operand_alignment % kernel_alignment == 0
            if a_align % ops.A.alignment != 0:
                return False
            if b_align % ops.B.alignment != 0:
                return False
            if out_align % ops.out.alignment != 0:
                return False

            return True

        except Exception:
            return False

    return metadata_filter


def add_cutlass_api_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: Optional[torch.dtype] = None,
) -> None:
    """
    Add cutlass_api GEMM kernels to the autotune choices.

    This function queries cutlass_api for compatible GEMM kernels and adds
    them as choices for autotuning. Uses full metadata filtering to match
    kernel.supports(args) logic, including dtype, layout, alignment, and
    compute capability checks.

    Args:
        choices: List of ChoiceCaller objects to append to
        layout: Output layout
        input_nodes: Input buffer nodes [A, B]
        accumulator_type: Data type for accumulation (default: float32)
    """
    if ensure_cutlass_api_available():
        import cutlass_api
    else:
        log.debug("cutlass_api not available, skipping cutlass_api choices")
        return

    if accumulator_type is None:
        accumulator_type = torch.float32

    a_node, b_node = input_nodes
    a_layout = a_node.get_layout()
    b_layout = b_node.get_layout()
    out_layout = layout

    cc = torch.cuda.get_device_capability()
    cc_int = cc[0] * 10 + cc[1]

    metadata_filter = _create_full_metadata_filter(
        a_layout, b_layout, out_layout, accumulator_type
    )

    try:
        kernels = cutlass_api.get_kernels(
            metadata_filter=metadata_filter,
            cc=cc_int,
        )
    except Exception:
        log.debug("Failed to get cutlass_api kernels", exc_info=True)
        return

    if not kernels:
        log.debug(
            "No compatible cutlass_api kernels found for dtypes: A=%s, B=%s, out=%s",
            a_layout.dtype,
            b_layout.dtype,
            out_layout.dtype,
        )
        return

    if len(kernels) > MAX_CUTLASS_API_PROFILING_CONFIGS:
        kernels = random.sample(kernels, MAX_CUTLASS_API_PROFILING_CONFIGS)

    num_added = 0
    for kernel in kernels:
        name = f"cutlass_api_gemm_{next(CutlassAPIGemmCaller.index_counter)}"
        try:
            caller = CutlassAPIGemmCaller(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                kernel=kernel,
                accumulator_type=accumulator_type,
            )
            choices.append(caller)
            num_added += 1
        except Exception:
            log.debug("Failed to create cutlass_api choice for %s", name, exc_info=True)

    log.debug("Added %d cutlass_api GEMM choices", num_added)
