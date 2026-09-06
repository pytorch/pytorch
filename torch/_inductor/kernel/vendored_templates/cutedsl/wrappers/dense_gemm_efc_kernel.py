"""Kernel wrapper for the vendored SM100 dense EFC GEMM template."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cutlass.cute as cute
from cutlass.operators.arch import TargetSm  # noqa: TC002
from cutlass.operators.arguments import GemmArguments
from cutlass.operators.artifact import CompiledArtifact  # noqa: TC002
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

from torch._inductor.codegen.nv_universal_gemm.epilogue_capabilities import (
    DENSE_GEMM_REDUCTION_CAPABILITIES,
)
from torch._inductor.kernel.gemm_epilogue import GEMM_REDUCTION_FRAGMENT_WIDTH
from torch._inductor.kernel.gemm_epilogue_codegen import (
    GemmReductionCompileConfig,
    get_cutedsl_epilogue_schema,
    materialize_epilogue_function,
)

from ..dense_gemm_efc import PersistentDenseGemmEFCKernel


class _EfcCuteNamespace:
    """Provide the module-shaped namespaces expected by generated epilogues.

    Generated source calls both ``cute.math`` and ``mlir_math`` operations. EFC
    exposes those operations on its configuration object, so this adapter serves
    as the ``cute`` namespace and its own ``math`` child while forwarding every
    operation to that configuration.
    """

    def __init__(self, efc_config):
        self.efc_config = efc_config
        self.math = self

    def __getattr__(self, name):
        if name == "ReductionOp":
            return cute.ReductionOp
        try:
            return getattr(self.efc_config, name)
        except AttributeError:
            return getattr(cute, name)


def _direct_cutedsl_epilogue(metadata):
    """Adapt generated CuTeDSL source to EFC's accumulator and parameter API."""

    epilogue_source = metadata.epilogue.epilogue_fn
    schema = get_cutedsl_epilogue_schema(epilogue_source)
    if schema is None:
        raise AssertionError("expected a direct CuTeDSL epilogue schema")
    inputs = schema.inputs
    outputs = schema.outputs
    parameter_names = metadata.epilogue.parameter_names
    tensors = metadata.epilogue.tensors
    output_shape = tuple(tensors[outputs[-1]].shape)
    scalar_broadcast_names = schema.scalar_broadcast_names

    def source_mode_map(name):
        shape = tuple(tensors[name].shape)
        if not shape or len(shape) > 3:
            raise NotImplementedError(f"unsupported dense EFC broadcast shape: {shape}")
        if name in scalar_broadcast_names:
            return _build_source_mode_map((0, 0, 0), len(shape))
        stride: Any = tensors[name].stride
        if callable(stride):
            stride = stride()
        if not isinstance(stride, Iterable):
            raise AssertionError(f"expected iterable stride, got {type(stride)}")
        stride = tuple(stride)
        padded_shape = (1,) * (3 - len(shape)) + shape
        padded_stride = (0,) * (3 - len(stride)) + stride
        propagated_stride = tuple(
            0 if size == 1 else step
            for size, step in zip(padded_shape, padded_stride, strict=True)
        )
        if shape == output_shape and all(propagated_stride):
            return None
        return _build_source_mode_map(propagated_stride, len(shape))

    input_mode_maps = {
        name: source_mode_map(name) for name in inputs if name != "accum"
    }

    def load(name, parameter):
        mode_map = input_mode_maps[name]
        return (
            parameter.load()
            if mode_map is None
            else parameter.remap_modes[mode_map].load()
        )

    def epilogue(efc_config, *parameters):
        by_name = dict(zip(parameter_names, parameters, strict=True))
        if efc_config.phase == common_efc.EFC.Phase.ParameterAnalysis:
            efc_config.accum()
            for name in inputs:
                if name != "accum":
                    load(name, by_name[name])
            for name in outputs:
                by_name[name].store(1)
            return

        values = [
            efc_config.accum() if name == "accum" else load(name, by_name[name])
            for name in inputs
        ]
        op_scope = _EfcCuteNamespace(efc_config)
        call_fn = materialize_epilogue_function(
            epilogue_source, op_scope, mlir_math=op_scope
        )
        results = call_fn(*values)
        if schema.returns_local_reduce:
            if not isinstance(results, tuple) or len(results) != len(outputs) + 1:
                raise AssertionError(
                    "expected generated epilogue outputs followed by local reduction"
                )
            result_values = results[:-1]
            if efc_config.phase == common_efc.EFC.Phase.ThreadOperation:
                efc_config.epilogue_context.local_reduce = results[-1]
        elif len(outputs) == 1:
            result_values = (results,)
        else:
            if not isinstance(results, tuple):
                raise AssertionError(f"expected tuple results, got {type(results)}")
            result_values = results
        for name, value in zip(outputs, result_values, strict=True):
            by_name[name].store(value)

    named_epilogue = common_efc.create_named_epilogue(
        ["efc_config", *parameter_names], epilogue
    )
    return named_epilogue


class VendoredDenseGemmEFCOperator(PersistentDenseGemmEFCOperator):
    """Dense EFC operator backed by Inductor's vendored SM100 implementation."""

    supported_args_type = GemmArguments
    designed_for_min_cc = 100

    def __init__(self, metadata):
        # The parent requires a traced EVT DAG, which direct CuTeDSL epilogues do
        # not have. Initialize the common operator metadata before choosing the
        # direct or EVT-specific epilogue adapter below.
        CuteDslOperator.__init__(self, metadata)
        OpToCuteImpl.setdefault(ActivationOp.Identity, lambda efc_config, value: value)
        OpToCuteImplStr.setdefault(ActivationOp.Identity, lambda value: value)
        if metadata.epilogue is None:
            epilogue_op = EFCConverter.identity_efc
        elif get_cutedsl_epilogue_schema(metadata.epilogue.epilogue_fn) is not None:
            epilogue_op = _direct_cutedsl_epilogue(metadata)
        else:
            epilogue_op = EFCConverter.convert(
                metadata.epilogue.traced_epilogue.dag_ir,
                metadata.epilogue.parameter_names,
                parameter_tensors=metadata.epilogue.tensors,
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
        # ``supports`` has already checked the operator metadata. Direct source
        # has no EVT DAG, so bypass only the parent's EVT-specific validation.
        status = (
            CuteDslOperator._supports(self, args, target_sm)
            if epilogue is not None
            and get_cutedsl_epilogue_schema(epilogue.epilogue_fn) is not None
            else super()._supports(args, target_sm)
        )
        if not status:
            return status
        reduction = args.local_reduce
        if not reduction.enabled:
            return status
        schema = (
            None
            if epilogue is None
            else get_cutedsl_epilogue_schema(epilogue.epilogue_fn)
        )
        if reduction.tensor_epilogue_returns_local_reduce != (
            schema is not None and schema.returns_local_reduce
        ):
            return Status.fail(
                "Dense EFC local reduction contract must match the tensor epilogue return"
            )
        axis = reduction.axis
        if axis not in (0, 1):
            return Status.fail("Dense EFC local reductions require an M or N axis")
        group = reduction.group
        max_group = self.cta_tile_m if axis == 0 else self.cta_tile_n
        if group <= 1 or group > max_group:
            return Status.fail("Dense EFC local reduction group exceeds its tile")
        if (
            reduction.feeds_main
            and axis == 0
            and self.cta_tile_n > GEMM_REDUCTION_FRAGMENT_WIDTH
        ):
            return Status.fail(
                "Dense M-axis feed-main requires a "
                f"{GEMM_REDUCTION_FRAGMENT_WIDTH}-column tile"
            )
        if (
            reduction.feeds_main
            and axis == 1
            and (self.impl.use_2cta_instrs or self.cta_tile_m != 128)
        ):
            return Status.fail(
                "Dense N-axis feed-main requires a single 128-row CTA tile"
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
            *reduction_config.constexprs(),
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
