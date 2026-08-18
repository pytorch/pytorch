"""Kernel wrapper for the vendored SM100 dense EFC GEMM template."""

from __future__ import annotations

from cutlass.operators.arch import TargetSm  # noqa: TC002
from cutlass.operators.arguments import GemmArguments
from cutlass.operators.artifact import CompiledArtifact  # noqa: TC002
from cutlass.operators.fusion.library import ActivationOp
from cutlass.operators.providers.cutedsl.evt.converter import (
    EFCConverter,
    OpToCuteImpl,
    OpToCuteImplStr,
)
from cutlass.operators.providers.cutedsl.gemm.sm100_static_persistent_efc import (
    PersistentDenseGemmEFCOperator,
)
from cutlass.operators.status import Status

from torch._inductor.codegen.nv_universal_gemm.epilogue_capabilities import (
    DENSE_GEMM_REDUCTION_CAPABILITIES,
)
from torch._inductor.kernel.gemm_epilogue import (
    GEMM_REDUCTION_FRAGMENT_WIDTH,
    GemmReductionDescriptor,
)
from torch._inductor.kernel.gemm_epilogue_codegen import GemmReductionCompileConfig

from ..dense_gemm_efc import PersistentDenseGemmEFCKernel


class VendoredDenseGemmEFCOperator(PersistentDenseGemmEFCOperator):
    """Dense EFC operator backed by Inductor's vendored SM100 implementation."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100

    def __init__(self, metadata):
        super().__init__(metadata)
        OpToCuteImpl.setdefault(ActivationOp.Identity, lambda efc_config, value: value)
        OpToCuteImplStr.setdefault(ActivationOp.Identity, lambda value: value)
        epilogue_op = (
            EFCConverter.convert(
                metadata.epilogue.traced_epilogue.dag_ir,
                metadata.epilogue.parameter_names,
                parameter_tensors=metadata.epilogue.tensors,
            )
            if metadata.epilogue is not None
            else EFCConverter.identity_efc
        )
        self.impl = PersistentDenseGemmEFCKernel(
            metadata.operands.accumulator_type,
            metadata.operands.out.dtype,
            metadata.design.use_2cta_mma,
            metadata.design.tile_shape[:2],
            metadata.design.cluster_shape[:2],
            epilogue_op,
        )
        self.cta_tile_m = metadata.design.tile_shape[0] // (
            2 if metadata.design.use_2cta_mma else 1
        )
        self.cta_tile_n = metadata.design.tile_shape[1]

    def _supports(
        self, args: GemmArguments, target_sm: TargetSm | None = None
    ) -> Status:
        status = super()._supports(args, target_sm)
        if not status:
            return status
        reduction = args.local_reduce
        if not reduction.enabled:
            return status
        axis = reduction.axis
        if axis not in (0, 1):
            return Status.fail("Dense EFC local reductions require an M or N axis")
        group = reduction.group
        max_group = self.cta_tile_m if axis == 0 else self.cta_tile_n
        if group <= 1 or group > max_group:
            return Status.fail("Dense EFC local reduction group exceeds its tile")
        descriptor = GemmReductionDescriptor.parse(reduction.reduction_type)
        if (
            axis == 1
            and group > GEMM_REDUCTION_FRAGMENT_WIDTH
            and descriptor.kind == "mean"
        ):
            return Status.fail("Dense EFC cross-fragment mean is unsupported")
        if (
            reduction.feeds_main
            and axis == 0
            and self.cta_tile_n > GEMM_REDUCTION_FRAGMENT_WIDTH
        ):
            return Status.fail(
                "Dense M-axis feed-main requires a "
                f"{GEMM_REDUCTION_FRAGMENT_WIDTH}-column tile"
            )
        if not DENSE_GEMM_REDUCTION_CAPABILITIES.supports_contract(reduction):
            return Status.fail("Unsupported dense EFC local reduction contract")
        return status

    def _compile(
        self, args: GemmArguments, target_sm: TargetSm | None = None
    ) -> CompiledArtifact:
        import cutlass
        from cutlass.operators.providers.cutedsl.integration_utils.mma import (
            get_max_active_clusters,
        )
        from cutlass.operators.utils.tensor import TensorWrapper

        stream = cutlass.cute.runtime.make_fake_stream()
        max_active_clusters = get_max_active_clusters(self.impl.cluster_shape_mn)
        epilogue_params = (
            [
                value.compile_time_tensor if isinstance(value, TensorWrapper) else value
                for value in args.epilogue.parameters
            ]
            if args.epilogue is not None
            else [args.out.tensor.compile_time_tensor]
        )
        self.impl.efc.compile(*epilogue_params)
        reduction_args = args.local_reduce
        reduction_tensors = reduction_args.map_tensors(
            lambda value: value.compile_time_tensor
        )
        reduction_config = GemmReductionCompileConfig.from_args(
            reduction_args, cutlass.cute
        )
        return self.cute_compile(
            self.impl,
            args.A.tensor,
            args.B.tensor,
            max_active_clusters,
            stream,
            reduction_tensors.output,
            reduction_tensors.feed_output,
            reduction_tensors.secondary_feed_output,
            *reduction_config.constexprs(include_consumers=False),
            *epilogue_params,
            target_sm=target_sm,
        )

    def _run(
        self,
        args: GemmArguments,
        compiled_artifact: CompiledArtifact,
        stream,
        workspace=None,
    ) -> None:
        from cutlass.operators.arguments import Operand
        from cutlass.operators.utils.device import to_cuda_stream

        epilogue_params = (
            args.epilogue.parameters if args.epilogue is not None else [args.out]
        )
        for index, value in enumerate(epilogue_params):
            if isinstance(value, Operand):
                value = value.tensor
            epilogue_params[index] = getattr(value, "runtime_tensor", value)
        reduction_tensors = args.local_reduce.map_tensors(
            lambda value: value.runtime_tensor
        )
        self.cute_run(
            compiled_artifact.compiled_obj,
            args.A.tensor,
            args.B.tensor,
            to_cuda_stream(stream),
            reduction_tensors.output,
            reduction_tensors.feed_output,
            reduction_tensors.secondary_feed_output,
            *epilogue_params,
        )
