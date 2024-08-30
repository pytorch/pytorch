# mypy: allow-untyped-defs
import logging

import torch
from torch._inductor.kernel.mm_common import mm_args

# Makes sure that quantized_decomposed ops are registered
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401

from . import config as inductor_config, lowering
from .codegen.cpp_gemm_template import CppPackedGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .lowering import expand, register_lowering
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
)
from .utils import use_aten_gemm_kernels, use_cpp_packed_gemm_template


log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
)

aten__weight_int4pack_mm = ExternKernelChoice(
    torch.ops.quantized_decomposed.int4mm_packed, None, has_out_variant=False
)

quantized = torch.ops.quantized
_quantized = torch.ops._quantized
aten = torch.ops.aten


def register_quantized_ops():
    lowering.add_needs_realized_inputs(
        [
            quantized.max_pool2d,
            _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16,
            _quantized.wrapped_fbgemm_linear_fp16_weight,
        ]
    )

    lowering.make_fallback(quantized.max_pool2d)
    lowering.make_fallback(_quantized.wrapped_fbgemm_pack_gemm_matrix_fp16)
    lowering.make_fallback(_quantized.wrapped_fbgemm_linear_fp16_weight)


def register_woq_mm_ops():
    @register_lowering(aten._weight_int8pack_mm, type_promotion_kind=None)
    def int8pack_mm(input, weight, scale, *, layout=None):
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, mat2_transposed=True
        )
        assert (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.int8
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [aten__weight_int8pack_mm.bind((mat1, mat2, scale), aten_layout)]
            if use_aten_gemm_kernels()
            else []
        )

        # scale is applied as an epilogue, and the scale tensor is expanded (with a view op)
        # for broadcasting, as it's 1D.
        def _mul_epilogue(buf):
            return create_epilogue_with_attr(
                buf, "mul", other=realize_inputs(expand(scale, layout.size))
            )

        if use_cpp_packed_gemm_template(aten_layout, mat1, mat2, mat2_transposed=True):
            CppPackedGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, scale],
                trans_w=True,
                epilogue_creator=_mul_epilogue,
            )

        if (
            len(choices) == 0
            and inductor_config.autotune_fallback_to_aten
            and not use_aten_gemm_kernels()
        ):
            log.warning("No choices for GEMM, using ATen backend as fallback")
            return aten__weight_int8pack_mm.bind(
                (mat1, mat2, scale), aten_layout
            ).output_node()

        return autotune_select_algorithm(
            "_weight_int8pack_mm", choices, [mat1, mat2, scale], aten_layout
        )

    @register_lowering(
        torch.ops.quantized_decomposed.int4mm_packed, type_promotion_kind=None
    )
    def int4pack_mm(input, weight, qScaleAndZeros, *, layout=None):
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, packed_int4_weights=True
        )
        assert (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.int32
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [aten__weight_int4pack_mm.bind((mat1, mat2, qScaleAndZeros), aten_layout)]
            if use_aten_gemm_kernels()
            else []
        )
        qGroupSize = (weight.get_numel() * 8) // (qScaleAndZeros.get_numel() / 2)
        if use_cpp_packed_gemm_template(
            aten_layout, mat1, mat2, q_group_size=qGroupSize
        ):
            can_work = CppPackedGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, qScaleAndZeros],
                is_int4_woq_gemm=True,
            )
            if can_work is False:
                choices = []
            else:
                return autotune_select_algorithm(
                    "_weight_int4pack_mm",
                    choices,
                    [mat1, mat2, qScaleAndZeros],
                    aten_layout,
                )

        if len(choices) == 0:
            log.warning("No choices for GEMM, using ATen backend as fallback")
            return aten__weight_int4pack_mm.bind(
                (mat1, mat2, qScaleAndZeros), aten_layout
            ).output_node()
