"""Kernel wrapper for dense_blockscaled_gemm_persistent vendored template."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Generator  # noqa: TC003

import cutlass_api
from cutlass_api.arguments import GemmArguments  # noqa: TC002
from cutlass_api.artifact import CompiledArtifact
from cutlass_api.library import ScaleMode, ScaleSwizzleMode
from cutlass_api.metadata import (
    DenseTensorAttributes,
    GemmOperandsMetadata,
    KernelMetadata,
    ScaledTensorAttributes,
    Sm100DesignMetadata,
)
from cutlass_api.providers.cutedsl.kernel import CuteDslKernel
from cutlass_api.providers.cutedsl.utils import get_max_active_clusters
from cutlass_api.status import Status
from cutlass_api.utils import (
    ceil_div,
    round_up,
    strides_to_layout_string,
    to_cuda_stream,
    tuple_to_string,
)


log = logging.getLogger(__name__)


try:
    from ..dense_blockscaled_gemm_persistent import (  # pyrefly: ignore[missing-import]
        Sm100BlockScaledPersistentDenseGemmKernel as BlockScaledGemmKernelImpl,
    )
except ImportError:
    BlockScaledGemmKernelImpl = None  # type: ignore[misc, assignment]


class VendoredDenseBlockScaledGemmKernel(CuteDslKernel):
    """Wrapper for vendored dense blockscaled GEMM template for SM100 GPUs."""

    def __init__(self, metadata: KernelMetadata):
        super().__init__(metadata)

        self.sf_vec_size = metadata.operands.A.mode[-1]
        mma_tiler_mn = (metadata.design.tile_shape[0], metadata.design.tile_shape[1])
        cluster_shape_mn = (
            metadata.design.cluster_shape[0],
            metadata.design.cluster_shape[1],
        )

        self.impl = BlockScaledGemmKernelImpl(  # pyrefly: ignore[not-callable]
            self.sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
        )
        self.cluster_shape_mn = cluster_shape_mn

    @staticmethod
    def _major_modes(args):
        """Extract major modes from arguments or operand metadata."""
        import cutlass.utils as utils
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        if args.A.stride[-2:].index(1) == 1:
            a_major_mode = (OperandMajorMode.K, "k")
        else:
            a_major_mode = (OperandMajorMode.MN, "m")

        if args.B.stride[-2:].index(1) == 0:
            b_major_mode = (OperandMajorMode.K, "k")
        else:
            b_major_mode = (OperandMajorMode.MN, "n")

        if args.out.stride[-2:].index(1) == 1:
            out_layout = (utils.LayoutEnum.ROW_MAJOR, "n")
        else:
            out_layout = (utils.LayoutEnum.COL_MAJOR, "m")

        return a_major_mode, b_major_mode, out_layout

    def compile(self, args: GemmArguments, cc: int | None = None) -> CompiledArtifact:
        import cutlass.cute as cute

        stream = cute.runtime.make_fake_stream()
        max_active_clusters = get_max_active_clusters(self.cluster_shape_mn)

        # pyrefly: ignore[missing-attribute]
        compiled_kernel = self.cute_compile(
            self.impl,
            args.A.tensor,
            args.B.tensor,
            args.A.scale.tensor,
            args.B.scale.tensor,
            args.out.tensor,
            max_active_clusters,
            stream,
        )

        return CompiledArtifact(compiled_kernel, self)

    def _run(
        self,
        args: GemmArguments,
        compiled_artifact: CompiledArtifact,
        stream,
        workspace=None,
    ) -> None:
        import torch

        stream = to_cuda_stream(stream)
        compiled_gemm = compiled_artifact.compiled_obj

        # TVM FFI needs a torch.cuda.Stream, not a raw int handle
        if isinstance(stream, int):
            stream = torch.cuda.ExternalStream(stream)

        self.cute_run(  # pyrefly: ignore[missing-attribute]
            compiled_gemm,
            args.A.tensor,
            args.B.tensor,
            args.A.scale.tensor,
            args.B.scale.tensor,
            args.out.tensor,
            stream,
        )

    def get_workspace_size(self, args: GemmArguments) -> int:
        return 0

    def _supports(self, args: GemmArguments) -> Status:
        import cutlass

        m, n = args.out.shape[-2:]
        k = args.A.shape[-1]
        if args.A.element_type == cutlass.Float4E2M1FN:
            k = k * 2
        l = args.A.shape[0] if len(args.A.shape) == 3 else 1

        expected_sf_k = round_up(ceil_div(k, self.sf_vec_size), 4)
        expected_sfa_elements = l * m * expected_sf_k
        expected_sfb_elements = l * n * expected_sf_k

        if args.A.scale.numel() != expected_sfa_elements:
            return Status.fail(
                f"Scale factor A for tensor A of shape {args.A.shape} must have "
                f"{expected_sfa_elements} elements, got {args.A.scale.numel()}."
            )
        if args.B.scale.numel() != expected_sfb_elements:
            return Status.fail(
                f"Scale factor B for tensor B of shape {args.B.shape} must have "
                f"{expected_sfb_elements} elements, got {args.B.scale.numel()}."
            )

        return Status.success()

    @staticmethod
    def _is_valid_dtype_combo(ab_dtype, sf_dtype, sf_vec_size, out_dtype) -> bool:
        """Validate dtype/scale-factor/vec-size combinations.

        Matches constraints from the upstream kernel:
          MXF8: Float8E5M2/Float8E4M3FN + Float8E8M0FNU, sf_vec_size=32
          MXF4: Float4E2M1FN + Float8E8M0FNU, sf_vec_size=32
          NVF4: Float4E2M1FN + Float8E8M0FNU/Float8E4M3FN, sf_vec_size=16
        """
        import cutlass

        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            return False
        if sf_vec_size not in {16, 32}:
            return False
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            return False
        # Float8E4M3FN as SF only valid with sf_vec_size=16 (NVF4)
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            return False
        # Float8 AB types require sf_vec_size=32 (MXF8)
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            return False
        if out_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            return False
        return True

    @staticmethod
    def _is_valid_layouts(ab_dtype, a_major, b_major) -> bool:
        """Float4E2M1FN (MXF4/NVF4) requires row-major (K-major) A and B."""
        import cutlass

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            return False
        return True

    @staticmethod
    def _valid_operands(operands: GemmOperandsMetadata, sf_vec_size: int) -> bool:
        import cutlass

        if operands.A.dtype != operands.B.dtype:
            return False
        if operands.A.scale.dtype != operands.B.scale.dtype:
            return False
        if operands.accumulator_type != cutlass.Float32:
            return False

        ab_dtype = operands.A.dtype
        sf_dtype = operands.A.scale.dtype
        out_dtype = operands.out.dtype

        if not VendoredDenseBlockScaledGemmKernel._is_valid_dtype_combo(
            ab_dtype, sf_dtype, sf_vec_size, out_dtype
        ):
            return False

        (_, a_major), (_, b_major), _ = VendoredDenseBlockScaledGemmKernel._major_modes(
            operands
        )
        if not VendoredDenseBlockScaledGemmKernel._is_valid_layouts(
            ab_dtype, a_major, b_major
        ):
            return False

        return True

    @staticmethod
    def _metadata_operand_combinations() -> Generator[GemmOperandsMetadata, None, None]:
        import cutlass

        ab_dtypes = [cutlass.Float8E5M2, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN]
        out_dtypes = [
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ]
        sf_dtypes = [cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN]
        scale_modes = [ScaleMode.Blockwise1x16, ScaleMode.Blockwise1x32]

        row_major = (0, 0, 1)
        col_major = (0, 1, 0)
        alignment_bytes = 16

        def major_str_a(stride):
            return "k" if stride == row_major else "m"

        def major_str_b(stride):
            return "n" if stride == row_major else "k"

        for ab_dtype, sf_dtype, scale_mode, out_dtype in itertools.product(
            ab_dtypes, sf_dtypes, scale_modes, out_dtypes
        ):
            sf_vec_size = scale_mode[-1]
            if not VendoredDenseBlockScaledGemmKernel._is_valid_dtype_combo(
                ab_dtype, sf_dtype, sf_vec_size, out_dtype
            ):
                continue

            for stride_A, stride_B, stride_out in itertools.product(
                [row_major, col_major], repeat=3
            ):
                a_major = major_str_a(stride_A)
                b_major = major_str_b(stride_B)

                if not VendoredDenseBlockScaledGemmKernel._is_valid_layouts(
                    ab_dtype, a_major, b_major
                ):
                    continue

                ab_div = alignment_bytes * 8 // ab_dtype.width
                out_div = alignment_bytes * 8 // out_dtype.width
                sf_div = alignment_bytes * 8 // sf_dtype.width

                yield GemmOperandsMetadata(
                    A=ScaledTensorAttributes(
                        base=DenseTensorAttributes(
                            dtype=ab_dtype,
                            stride=stride_A,
                            divisibility=ab_div,
                        ),
                        scale=DenseTensorAttributes(
                            dtype=sf_dtype,
                            stride=None,
                            divisibility=sf_div,
                        ),
                        mode=scale_mode,
                        swizzle=ScaleSwizzleMode.Swizzle32x4x4,
                    ),
                    B=ScaledTensorAttributes(
                        base=DenseTensorAttributes(
                            dtype=ab_dtype,
                            stride=stride_B,
                            divisibility=ab_div,
                        ),
                        scale=DenseTensorAttributes(
                            dtype=sf_dtype,
                            stride=None,
                            divisibility=sf_div,
                        ),
                        mode=scale_mode,
                        swizzle=ScaleSwizzleMode.Swizzle32x4x4,
                    ),
                    out=DenseTensorAttributes(
                        dtype=out_dtype,
                        stride=stride_out,
                        divisibility=out_div,
                    ),
                    accumulator_type=cutlass.Float32,
                )

    @staticmethod
    def _valid_metadata(metadata: KernelMetadata) -> bool:
        scale_vec = metadata.operands.A.mode

        if len(scale_vec) > 1:
            for i in range(len(scale_vec) - 1):
                if scale_vec[i] != 1:
                    return False

        sf_vec_size = scale_vec[-1]
        if not VendoredDenseBlockScaledGemmKernel._valid_operands(
            metadata.operands, sf_vec_size
        ):
            return False

        design = metadata.design
        if not isinstance(design, Sm100DesignMetadata):
            return False

        cm, cn, _ = design.cluster_shape
        if cm <= 0 or cn <= 0:
            return False
        if cm * cn > 16:
            return False
        if cm & (cm - 1) != 0 or cn & (cn - 1) != 0:
            return False
        # SF multicast constraint: cluster dims <=4
        if cm > 4 or cn > 4:
            return False

        tile_m, tile_n, _ = design.tile_shape
        if tile_m not in [128, 256]:
            return False
        if tile_n not in [64, 128, 192, 256]:
            return False
        use_2cta = tile_m == 256
        if use_2cta and cm % 2 != 0:
            return False

        if metadata.epilogue is not None:
            return False

        return True

    @staticmethod
    def generate_kernels(
        metadata_filter: Callable[[KernelMetadata], bool],
        epilogue_args=None,
        cc: int | None = None,
    ) -> list[VendoredDenseBlockScaledGemmKernel]:
        if cc is not None and cc not in [100, 101, 103]:
            return []
        if epilogue_args is not None:
            return []

        design_params = {
            "use_2cta_mma": [True],
            "tile_shape": [
                (M, N, 256) for M in [128, 256] for N in [64, 128, 192, 256]
            ],
            "cluster_shape": [(M, N, 1) for M in [1, 2, 4] for N in [1, 2, 4]],
            "use_tma_store": [True],
        }

        param_names = list(design_params.keys())
        param_values = [design_params[name] for name in param_names]

        kernel_list = []

        for (
            operands
        ) in VendoredDenseBlockScaledGemmKernel._metadata_operand_combinations():
            # pyrefly: ignore[no-matching-overload]
            for values in itertools.product(*param_values):
                design = Sm100DesignMetadata(**dict(zip(param_names, values)))

                kernel_name = (
                    "inductor_vendored.DenseBlockScaledGemmKernel_sm100_"
                    "{layout}_A{A}_B{B}_out{out}_SFA{SFA}_SFB{SFB}_"
                    "acc{acc}_scale{scale_mode}_swizzle{scale_swizzle}_"
                    "{num_cta}cta_cluster{cluster}_tile{tile}"
                    "{_tma_store}"
                ).format(
                    layout=strides_to_layout_string(
                        operands.A.stride,
                        operands.B.stride,
                        operands.out.stride,
                    ),
                    A=operands.A.dtype,
                    B=operands.B.dtype,
                    out=operands.out.dtype,
                    SFA=operands.A.scale.dtype,
                    SFB=operands.B.scale.dtype,
                    acc=operands.accumulator_type,
                    scale_mode=operands.A.mode,
                    scale_swizzle=operands.A.swizzle,
                    num_cta="2" if design.use_2cta_mma else "1",
                    cluster=tuple_to_string(design.cluster_shape),
                    tile=tuple_to_string(design.tile_shape),
                    _tma_store="_tma_store" if design.use_tma_store else "",
                )

                metadata = KernelMetadata(
                    operands=operands,
                    design=design,
                    kernel_name=kernel_name,
                    kernel_class=VendoredDenseBlockScaledGemmKernel,
                    min_cc=100,
                    epilogue=None,
                )

                if VendoredDenseBlockScaledGemmKernel._valid_metadata(metadata):
                    if metadata_filter is None or metadata_filter(metadata):
                        kernel_list.append(VendoredDenseBlockScaledGemmKernel(metadata))

        log.debug(
            "Generated %d DenseBlockScaledGemmKernel configurations",
            len(kernel_list),
        )
        return kernel_list


# Only register if kernel implementation is available
if BlockScaledGemmKernelImpl is not None:
    cutlass_api.providers.cutedsl.CuTeDSLProvider.register(
        VendoredDenseBlockScaledGemmKernel
    )
