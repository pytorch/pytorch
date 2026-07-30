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

from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    canonical_tensorssa_reduction_type,
    materialize_tensorssa_reduction,
)
from torch._inductor.kernel.gemm_epilogue import GemmReductionDescriptor
from torch.utils._ordered_set import OrderedSet

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
        if reduction.feeds_main and axis == 0 and self.cta_tile_n > 32:
            return Status.fail("Dense M-axis feed-main requires a 32-column tile")
        secondary_feed = reduction.secondary_feed_output
        secondary_type = reduction.secondary_feed_type
        try:
            expression = reduction.descriptor
            secondary_expression = (
                GemmReductionDescriptor.parse(secondary_type)
                if secondary_type is not None
                else None
            )
        except ValueError:
            return Status.fail("Malformed dense EFC reduction expression")
        secondary_kinds = (
            OrderedSet(
                [
                    "direct_bool_gt_zero",
                    "normalize_sum_affine",
                    "normalize_sum_reverse_affine",
                    "sum_mul_affine",
                ]
            )
            if reduction.feeds_main and axis == 0
            else OrderedSet(["direct_bool_gt_zero"])
        )
        if secondary_feed is not None and (
            secondary_expression is None
            or secondary_expression.kind not in secondary_kinds
        ):
            return Status.fail("Unsupported dense EFC secondary feed expression")
        reduction_kinds = OrderedSet(
            [
                "sum",
                "mean",
                "prod",
                "max",
                "min",
                "logsumexp",
                "direct_bool_gt_zero",
                "variance_affine",
            ]
        )
        if reduction.feeds_main:
            reduction_kinds.update(
                OrderedSet(
                    [
                        "mean_linear",
                        "normalize_sum_affine",
                        "normalize_sum_reverse_affine",
                    ]
                )
            )
        if expression.kind not in reduction_kinds:
            return Status.fail("Unsupported dense EFC local reduction type")
        if reduction.source_type not in (
            "identity",
            "square",
            "abs",
        ):
            return Status.fail("Unsupported dense EFC local reduction source")
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
        local_reduce, local_reduce_feed, secondary_feed = reduction_args.tensors(
            "compile_time_tensor"
        )
        reduction = materialize_tensorssa_reduction(
            canonical_tensorssa_reduction_type(reduction_args.reduction_type),
            reduction_args.source_type,
            reduction_args.reduction_type,
        )
        return self.cute_compile(
            self.impl,
            args.A.tensor,
            args.B.tensor,
            max_active_clusters,
            stream,
            local_reduce,
            local_reduce_feed,
            secondary_feed,
            reduction_args.group,
            reduction_args.axis,
            reduction_args.reduction_type,
            reduction_args.source_type,
            reduction_args.feeds_main,
            reduction_args.secondary_feed_type,
            reduction.reduce_op,
            reduction.init_val,
            reduction.combine,
            reduction.source,
            reduction.finalize,
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
        reduction = args.local_reduce
        local_reduce, local_reduce_feed, secondary_feed = reduction.tensors(
            "runtime_tensor"
        )
        self.cute_run(
            compiled_artifact.compiled_obj,
            args.A.tensor,
            args.B.tensor,
            to_cuda_stream(stream),
            local_reduce,
            local_reduce_feed,
            secondary_feed,
            *epilogue_params,
        )
