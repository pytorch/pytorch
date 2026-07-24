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
        local_reduce = getattr(args, "local_reduce_out", None)
        feeds_main = getattr(args, "local_reduce_feeds_main", False)
        if local_reduce is None and not feeds_main:
            return status
        axis = getattr(args, "local_reduce_axis", None)
        if axis not in (0, 1):
            return Status.fail("Dense EFC local reductions require an M or N axis")
        group = getattr(args, "local_reduce_group", 0)
        max_group = self.cta_tile_m if axis == 0 else self.cta_tile_n
        if group <= 1 or group > max_group:
            return Status.fail("Dense EFC local reduction group exceeds its tile")
        if feeds_main and axis == 0 and self.cta_tile_n > 32:
            return Status.fail("Dense M-axis feed-main requires a 32-column tile")
        reduce_type = getattr(args, "local_reduce_type", None)
        secondary_feed = getattr(args, "local_reduce_secondary_feed_out", None)
        secondary_type = getattr(args, "local_reduce_secondary_feed_type", None)
        if secondary_feed is not None and not (
            isinstance(secondary_type, str)
            and (
                secondary_type == "direct_bool_gt_zero"
                or (
                    feeds_main
                    and axis == 0
                    and secondary_type.startswith(
                        (
                            "sum_mul_affine:",
                            "normalize_sum_affine:",
                            "normalize_sum_reverse_affine:",
                        )
                    )
                )
            )
        ):
            return Status.fail("Unsupported dense EFC secondary feed expression")
        if reduce_type not in (
            "sum",
            "mean",
            "prod",
            "max",
            "min",
            "logsumexp",
            "direct_bool_gt_zero",
        ) and not (
            isinstance(reduce_type, str)
            and (
                reduce_type.startswith("variance_affine:")
                or (
                    feeds_main
                    and reduce_type.startswith(
                        (
                            "mean_linear:",
                            "normalize_sum_affine:",
                            "normalize_sum_reverse_affine:",
                        )
                    )
                )
            )
        ):
            return Status.fail("Unsupported dense EFC local reduction type")
        if getattr(args, "local_reduce_source", None) not in (
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
        local_reduce = getattr(args, "local_reduce_out", None)
        local_reduce_feed = getattr(args, "local_reduce_feed_out", None)
        secondary_feed = getattr(args, "local_reduce_secondary_feed_out", None)
        reduction = materialize_tensorssa_reduction(
            canonical_tensorssa_reduction_type(
                getattr(args, "local_reduce_type", "sum")
            ),
            getattr(args, "local_reduce_source", "identity"),
            getattr(args, "local_reduce_type", "sum"),
        )
        return self.cute_compile(
            self.impl,
            args.A.tensor,
            args.B.tensor,
            max_active_clusters,
            stream,
            local_reduce.compile_time_tensor if local_reduce is not None else None,
            (
                local_reduce_feed.compile_time_tensor
                if local_reduce_feed is not None
                else None
            ),
            secondary_feed.compile_time_tensor if secondary_feed is not None else None,
            getattr(args, "local_reduce_group", 0),
            getattr(args, "local_reduce_axis", 1),
            getattr(args, "local_reduce_type", "sum"),
            getattr(args, "local_reduce_source", "identity"),
            getattr(args, "local_reduce_feeds_main", False),
            getattr(args, "local_reduce_secondary_feed_type", None),
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
        local_reduce = getattr(args, "local_reduce_out", None)
        local_reduce_feed = getattr(args, "local_reduce_feed_out", None)
        secondary_feed = getattr(args, "local_reduce_secondary_feed_out", None)
        self.cute_run(
            compiled_artifact.compiled_obj,
            args.A.tensor,
            args.B.tensor,
            to_cuda_stream(stream),
            local_reduce.runtime_tensor if local_reduce is not None else None,
            (
                local_reduce_feed.runtime_tensor
                if local_reduce_feed is not None
                else None
            ),
            secondary_feed.runtime_tensor if secondary_feed is not None else None,
            *epilogue_params,
        )
