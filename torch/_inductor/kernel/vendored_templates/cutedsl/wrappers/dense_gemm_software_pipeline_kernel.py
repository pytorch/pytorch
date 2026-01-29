"""Kernel wrapper for dense_gemm_software_pipeline vendored template."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Generator  # noqa: TC003

import cutlass_api
from cutlass_api.arguments import GemmArguments  # noqa: TC002
from cutlass_api.artifact import CompiledArtifact
from cutlass_api.metadata import (
    DenseTensorAttributes,
    GemmOperandsMetadata,
    KernelMetadata,
    Sm100DesignMetadata,
)
from cutlass_api.providers.cutedsl.kernel import CuteDslKernel
from cutlass_api.status import Status
from cutlass_api.utils import strides_to_layout_string, to_cuda_stream, tuple_to_string

from torch.utils._ordered_set import OrderedSet

from .utils import ensure_3d_tensor_wrapper, get_3d_runtime_tensor


log = logging.getLogger(__name__)


try:
    from ..kernels.dense_gemm_software_pipeline import (  # pyrefly: ignore[missing-import]
        DenseGemmKernel as DenseGemmKernelSWPipeImpl,
    )
except ImportError:
    DenseGemmKernelSWPipeImpl = None  # type: ignore[misc, assignment]


class VendoredDenseGemmKernelSWPipe(CuteDslKernel):
    """Wrapper for vendored dense GEMM template with software pipelining for SM100."""

    def __init__(self, metadata: KernelMetadata):
        """Initialize the kernel wrapper."""
        super().__init__(metadata)

        mma_tiler_mn = (metadata.design.tile_shape[0], metadata.design.tile_shape[1])
        cluster_shape_mn = (
            metadata.design.cluster_shape[0],
            metadata.design.cluster_shape[1],
        )

        self.impl = DenseGemmKernelSWPipeImpl(  # pyrefly: ignore[not-callable]
            metadata.operands.accumulator_type,
            metadata.design.use_2cta_mma,
            mma_tiler_mn,
            cluster_shape_mn,
            metadata.design.use_tma_store,
        )

    def compile(self, args: GemmArguments, cc: int | None = None) -> CompiledArtifact:
        """Compile the kernel for the given arguments."""
        import cutlass.cute as cute

        stream = cute.runtime.make_fake_stream()

        a_tw = ensure_3d_tensor_wrapper(args.A.tensor)
        b_tw = ensure_3d_tensor_wrapper(args.B.tensor)
        out_tw = ensure_3d_tensor_wrapper(args.out.tensor)

        # pyrefly: ignore[missing-attribute]
        compiled_kernel = self.cute_compile(self.impl, a_tw, b_tw, out_tw, stream)

        return CompiledArtifact(compiled_kernel, self)

    def _run(
        self,
        args: GemmArguments,
        compiled_artifact: CompiledArtifact,
        stream,
        workspace=None,
    ) -> None:
        """Run the compiled kernel."""
        stream = to_cuda_stream(stream)
        compiled_gemm = compiled_artifact.compiled_obj

        a_rt = get_3d_runtime_tensor(args.A.tensor)
        b_rt = get_3d_runtime_tensor(args.B.tensor)
        out_rt = get_3d_runtime_tensor(args.out.tensor)

        compiled_gemm(a_rt, b_rt, out_rt, stream)

    def get_workspace_size(self, args: GemmArguments) -> int:
        """Get the workspace size required for this kernel."""
        return 0

    def _supports(self, args: GemmArguments) -> Status:
        """Check if this kernel supports the given arguments."""
        return Status.success()

    @staticmethod
    def _valid_operands(operands: GemmOperandsMetadata) -> bool:
        """Validate dtype combinations."""
        import cutlass

        if operands.A.dtype != operands.B.dtype:
            return False

        ab_dtype = operands.A.dtype
        acc_dtype = operands.accumulator_type
        out_dtype = operands.out.dtype

        valid_ab = OrderedSet(
            [
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Int8,
                cutlass.Uint8,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            ]
        )
        if ab_dtype not in valid_ab:
            return False

        acc_ab_compat = {
            cutlass.Float32: OrderedSet(
                [
                    cutlass.Float16,
                    cutlass.BFloat16,
                    cutlass.TFloat32,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E5M2,
                ]
            ),
            cutlass.Float16: OrderedSet(
                [
                    cutlass.Float16,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E5M2,
                ]
            ),
            cutlass.Int32: OrderedSet([cutlass.Uint8, cutlass.Int8]),
        }
        if acc_dtype not in acc_ab_compat or ab_dtype not in acc_ab_compat[acc_dtype]:
            return False

        acc_c_compat = {
            cutlass.Float32: OrderedSet(
                [
                    cutlass.Float32,
                    cutlass.Float16,
                    cutlass.BFloat16,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E5M2,
                    cutlass.Int32,
                    cutlass.Int8,
                    cutlass.Uint8,
                ]
            ),
            cutlass.Float16: OrderedSet([cutlass.BFloat16, cutlass.Float16]),
            cutlass.Int32: OrderedSet(
                [
                    cutlass.BFloat16,
                    cutlass.Float16,
                    cutlass.Float32,
                    cutlass.Int32,
                    cutlass.Int8,
                    cutlass.Uint8,
                ]
            ),
        }
        if out_dtype not in acc_c_compat.get(acc_dtype, OrderedSet()):
            return False

        return True

    @staticmethod
    def _metadata_operand_combinations() -> Generator[GemmOperandsMetadata, None, None]:
        """Generate all valid operand combinations."""
        import cutlass

        ab_dtypes = [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.TFloat32,
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ]

        row_major = (0, 0, 1)
        col_major = (0, 1, 0)
        alignment_bytes = 16

        for ab_dtype in ab_dtypes:
            if ab_dtype in {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.TFloat32,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            }:
                valid_acc = [cutlass.Float32]
                if ab_dtype in {
                    cutlass.Float16,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E5M2,
                }:
                    valid_acc.append(cutlass.Float16)
            else:  # Int8, Uint8
                valid_acc = [cutlass.Int32]

            for acc_dtype in valid_acc:
                if acc_dtype == cutlass.Float32:
                    valid_out = [
                        cutlass.Float32,
                        cutlass.Float16,
                        cutlass.BFloat16,
                        cutlass.Float8E4M3FN,
                        cutlass.Float8E5M2,
                        cutlass.Int32,
                        cutlass.Int8,
                        cutlass.Uint8,
                    ]
                elif acc_dtype == cutlass.Float16:
                    valid_out = [cutlass.Float16, cutlass.BFloat16]
                else:  # Int32
                    valid_out = [
                        cutlass.Float32,
                        cutlass.Int32,
                        cutlass.Int8,
                        cutlass.Uint8,
                        cutlass.Float16,
                        cutlass.BFloat16,
                    ]

                for out_dtype in valid_out:
                    for stride_A, stride_B, stride_out in itertools.product(
                        [row_major, col_major], repeat=3
                    ):
                        ab_div = alignment_bytes * 8 // ab_dtype.width
                        out_div = alignment_bytes * 8 // out_dtype.width

                        yield GemmOperandsMetadata(
                            A=DenseTensorAttributes(
                                dtype=ab_dtype, stride=stride_A, divisibility=ab_div
                            ),
                            B=DenseTensorAttributes(
                                dtype=ab_dtype, stride=stride_B, divisibility=ab_div
                            ),
                            out=DenseTensorAttributes(
                                dtype=out_dtype, stride=stride_out, divisibility=out_div
                            ),
                            accumulator_type=acc_dtype,
                        )

    @staticmethod
    def _valid_metadata(metadata: KernelMetadata) -> bool:
        """Validate full metadata."""
        if not VendoredDenseGemmKernelSWPipe._valid_operands(metadata.operands):
            return False

        design = metadata.design
        if not isinstance(design, Sm100DesignMetadata):
            return False

        cm, cn, _ = design.cluster_shape
        if cm <= 0 or cn <= 0:
            return False
        if cm * cn > 16:
            return False
        # Cluster dimensions must be powers of 2 for GPU hardware multicast/reduce.
        # Bit trick: n & (n-1) == 0 iff n is a power of 2 (for n > 0)
        if cm & (cm - 1) != 0 or cn & (cn - 1) != 0:
            return False

        tile_m, tile_n, _ = design.tile_shape
        if design.use_2cta_mma:
            if cm % 2 != 0:
                return False
            if tile_m not in [128, 256]:
                return False
        else:
            if tile_m not in [64, 128]:
                return False
        if tile_n not in range(32, 257, 32):
            return False

        if metadata.epilogue is not None:
            return False

        return True

    @staticmethod
    def generate_kernels(
        metadata_filter: Callable[[KernelMetadata], bool],
        epilogue_args=None,
        cc: int | None = None,
    ) -> list[VendoredDenseGemmKernelSWPipe]:
        """Generate all valid kernel configurations for the given compute capability."""
        if cc is not None and cc not in [100, 101, 103]:
            return []
        if epilogue_args is not None:
            return []

        use_2cta_values = [True, False]
        tile_m_values_1cta = [64, 128]
        tile_m_values_2cta = [128, 256]
        tile_n_values = list(range(32, 257, 32))
        cluster_m_values = [1, 2, 4, 8, 16]
        cluster_n_values = [1, 2, 4, 8, 16]

        kernel_list = []

        for operands in VendoredDenseGemmKernelSWPipe._metadata_operand_combinations():
            for use_2cta in use_2cta_values:
                tile_m_values = tile_m_values_2cta if use_2cta else tile_m_values_1cta

                for tile_m in tile_m_values:
                    for tile_n in tile_n_values:
                        for cluster_m in cluster_m_values:
                            for cluster_n in cluster_n_values:
                                if cluster_m * cluster_n > 16:
                                    continue
                                if use_2cta and cluster_m % 2 != 0:
                                    continue

                                design = Sm100DesignMetadata(
                                    use_2cta_mma=use_2cta,
                                    tile_shape=(tile_m, tile_n, 64),
                                    cluster_shape=(cluster_m, cluster_n, 1),
                                    use_tma_store=True,
                                )

                                kernel_name = (
                                    f"inductor_vendored.DenseGemmKernelSWPipe_sm100_"
                                    f"{strides_to_layout_string(operands.A.stride, operands.B.stride, operands.out.stride)}_"
                                    f"A{operands.A.dtype}_B{operands.B.dtype}_"
                                    f"out{operands.out.dtype}_"
                                    f"acc{operands.accumulator_type}_"
                                    f"{'2cta' if use_2cta else '1cta'}_"
                                    f"cluster{tuple_to_string(design.cluster_shape)}_"
                                    f"tile{tuple_to_string(design.tile_shape)}"
                                )

                                metadata = KernelMetadata(
                                    operands=operands,
                                    design=design,
                                    kernel_name=kernel_name,
                                    kernel_class=VendoredDenseGemmKernelSWPipe,
                                    min_cc=100,
                                    epilogue=None,
                                )

                                if VendoredDenseGemmKernelSWPipe._valid_metadata(
                                    metadata
                                ):
                                    if metadata_filter is None or metadata_filter(
                                        metadata
                                    ):
                                        kernel_list.append(
                                            VendoredDenseGemmKernelSWPipe(metadata)
                                        )

        log.debug("Generated %d DenseGemmKernelSWPipe configurations", len(kernel_list))
        return kernel_list


# Only register if kernel implementation is available
if DenseGemmKernelSWPipeImpl is not None:
    cutlass_api.providers.cutedsl.CuTeDSLProvider.register(
        VendoredDenseGemmKernelSWPipe
    )
