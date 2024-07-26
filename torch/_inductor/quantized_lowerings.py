# mypy: allow-untyped-defs
import logging

import torch
from torch._inductor.kernel.mm_common import mm_args
from . import config as inductor_config, lowering
from .codegen.cpp_gemm_template import CppPackedGemmTemplate
from .lowering import register_lowering
from .mkldnn_lowerings import create_epilogue_with_attr
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    NoValidChoicesError,
)
from .utils import use_aten_gemm_kernels, use_cpp_packed_gemm_template

log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
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
        _, _, _, layout, mat1, mat2, vec1 = mm_args(
            input, weight, scale, layout=layout, is_woq_gemm=True
        )
        assert (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.int8
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [aten__weight_int8pack_mm.bind((mat1, mat2, vec1), aten_layout)]
            if use_aten_gemm_kernels()
            else []
        )

        def epilogue_creator(buf):
            return create_epilogue_with_attr(buf, "mul", other=vec1)

        if use_cpp_packed_gemm_template(aten_layout, mat1, mat2, is_woq_gemm=True):
            CppPackedGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, vec1],
                is_woq_gemm=True,
                epilogue_creator=epilogue_creator,
            )

        if (
            len(choices) == 0
            and inductor_config.autotune_fallback_to_aten
            and not use_aten_gemm_kernels()
        ):
            log.warning("No choices for GEMM, using ATen backend as fallback")
            return aten__weight_int8pack_mm.bind(
                (mat1, mat2, vec1), aten_layout
            ).output_node()

        try:
            return autotune_select_algorithm(
                "_weight_int8pack_mm", choices, [mat1, mat2, vec1], aten_layout
            )
        except NoValidChoicesError:
            if not inductor_config.autotune_fallback_to_aten:
                raise
            log.warning(
                "All choices for GEMM were invalid, using ATen backend as fallback"
            )
            return aten__weight_int8pack_mm.bind(
                (mat1, mat2, vec1), aten_layout
            ).output_node()
