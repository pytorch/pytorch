"""Kernel wrapper for the vendored SM100 dense EFC GEMM template."""

from __future__ import annotations

from collections.abc import Iterable
from types import FunctionType
from typing import Any

from cutlass.operators.arch import TargetSm  # noqa: TC002
from cutlass.operators.arguments import GemmArguments
from cutlass.operators.artifact import CompiledArtifact  # noqa: TC002
from cutlass.operators.fusion import trace_in_out
from cutlass.operators.fusion.library import ActivationOp
from cutlass.operators.providers.cutedsl.evt import common_efc
from cutlass.operators.providers.cutedsl.evt.converter import (
    _build_source_mode_map,
    EFCConverter,
    OpToCuteImpl,
    OpToCuteImplStr,
)
from cutlass.operators.providers.cutedsl.gemm.sm100_static_persistent_efc import (
    PersistentDenseGemmEFCOperator,
)
from cutlass.operators.providers.cutedsl.operator import CuteDslOperator
from cutlass.operators.status import Status

from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    canonical_tensorssa_reduction_type,
    materialize_tensorssa_reduction,
)
from torch._inductor.kernel.gemm_epilogue import GemmReductionDescriptor
from torch._inductor.kernel.gemm_epilogue_codegen import gemm_epilogue_op_scope
from torch.utils._ordered_set import OrderedSet

from ..dense_gemm_efc import PersistentDenseGemmEFCKernel


class _EFCOpScope:
    def __init__(self, efc_config):
        self.efc_config = efc_config
        self.math = self

    def __getattr__(self, name):
        return getattr(self.efc_config, name)


def _direct_cutedsl_epilogue(metadata):
    import cutlass.cute as cute

    scope = gemm_epilogue_op_scope(cute)
    original_names = OrderedSet(scope)
    exec(metadata.epilogue.epilogue_fn, scope)
    direct_fn = next(
        value
        for name, value in scope.items()
        if name not in original_names and callable(value)
    )
    inputs, outputs = trace_in_out(metadata.epilogue.epilogue_fn)
    inputs = ["accum", *inputs] if "accum" not in inputs else inputs
    parameter_names = metadata.epilogue.parameter_names
    tensors = metadata.epilogue.tensors
    output_shape = tuple(tensors[outputs[-1]].shape)
    broadcast_names = OrderedSet()

    def load(name, parameter):
        shape = tuple(tensors[name].shape)
        if shape == output_shape:
            return parameter.load()
        if not shape or len(shape) > 3:
            raise NotImplementedError(f"unsupported dense EFC broadcast shape: {shape}")
        stride: Any = tensors[name].stride
        if callable(stride):
            stride = stride()
        assert isinstance(stride, Iterable)  # noqa: S101
        stride = tuple(stride)
        padded_shape = (1,) * (3 - len(shape)) + shape
        padded_stride = (0,) * (3 - len(stride)) + stride
        propagated_stride = tuple(
            0 if size == 1 else step for size, step in zip(padded_shape, padded_stride)
        )
        source_mode_map = _build_source_mode_map(propagated_stride, len(shape))
        broadcast_names.add(name)
        return parameter.remap_modes[source_mode_map].load()

    def epilogue(efc_config, *parameters):
        by_name = dict(zip(parameter_names, parameters))
        values = [
            efc_config.accum() if name == "accum" else load(name, by_name[name])
            for name in inputs
        ]
        op_scope = _EFCOpScope(efc_config)
        call_scope = direct_fn.__globals__.copy()
        call_scope.update(gemm_epilogue_op_scope(op_scope))
        call_scope["mlir_math"] = op_scope
        # Rebind globals because the traced function targets CuTeDSL operations,
        # while EFC exposes the same operations through its configuration object.
        call_fn = FunctionType(
            direct_fn.__code__,
            call_scope,
            direct_fn.__name__,
            direct_fn.__defaults__,
            direct_fn.__closure__,
        )
        call_fn.__kwdefaults__ = direct_fn.__kwdefaults__
        results = call_fn(*values)
        if len(outputs) == 1:
            result_values = (results,)
        else:
            assert isinstance(results, tuple)  # noqa: S101
            result_values = results
        for name, value in zip(outputs, result_values):
            by_name[name].store(value)

    named_epilogue = common_efc.create_named_epilogue(
        ["efc_config", *parameter_names], epilogue
    )
    named_epilogue._broadcast_source_names = broadcast_names
    return named_epilogue


class VendoredDenseGemmEFCOperator(PersistentDenseGemmEFCOperator):
    """Dense EFC operator backed by Inductor's vendored SM100 implementation."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100

    def __init__(self, metadata):
        CuteDslOperator.__init__(self, metadata)
        OpToCuteImpl.setdefault(ActivationOp.Identity, lambda efc_config, value: value)
        OpToCuteImplStr.setdefault(ActivationOp.Identity, lambda value: value)
        epilogue_op = (
            _direct_cutedsl_epilogue(metadata)
            if metadata.epilogue is not None
            and metadata.epilogue.traced_epilogue is None
            else EFCConverter.convert(
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
        epilogue = args.epilogue
        status = (
            Status.success()
            if epilogue is not None and getattr(epilogue, "_is_direct_cutedsl", False)
            else super()._supports(args, target_sm)
        )
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
