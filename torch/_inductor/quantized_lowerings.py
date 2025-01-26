import logging
from typing import Any, List, Optional

import sympy

import torch
from torch._inductor.kernel.mm_common import mm_args

from . import config as inductor_config, lowering
from .codegen.cpp_gemm_template import CppGemmTemplate
from .codegen.cpp_int8_sdpa_template import CppInt8SdpaTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .ir import FixedLayout, get_fill_order, TensorBox
from .kernel.flex_attention import construct_strides, maybe_realize
from .lowering import expand, register_lowering
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template


log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
)

quantized = torch.ops.quantized
_quantized = torch.ops._quantized
aten = torch.ops.aten


def register_quantized_ops() -> None:
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


def register_woq_mm_ops() -> None:
    @register_lowering(aten._weight_int8pack_mm, type_promotion_kind=None)  # type: ignore[misc]
    def int8pack_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
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
        def _mul_epilogue(buf: torch.Tensor) -> Any:
            return create_epilogue_with_attr(
                buf, "mul", other=realize_inputs(expand(scale, layout.size))
            )

        if use_cpp_gemm_template(aten_layout, mat1, mat2, mat2_transposed=True):
            CppGemmTemplate.add_choices(
                choices,
                aten_layout,
                [mat1, mat2, scale],
                trans_w=True,
                epilogue_creator=_mul_epilogue,  # type: ignore[arg-type]
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

    lowering.make_fallback(aten._dyn_quant_matmul_4bit)
    lowering.make_fallback(aten._dyn_quant_pack_4bit_weight)


def int8_sdpa_lowering(
    query: TensorBox,
    key: TensorBox,
    value: TensorBox,
    inv_scale: float,
    attn_mask: Optional[TensorBox],
    q_zp: Optional[int] = 0,
    q_scale: Optional[float] = 1.0,
    k_zp: Optional[int] = 0,
    k_scale: Optional[float] = 1.0,
    v_zp: Optional[int] = 0,
    v_scale: Optional[float] = 1.0,
    a_zp: Optional[int] = 0,
    a_scale: Optional[float] = 1.0,
    o_zp: Optional[int] = 0,
    o_scale: Optional[float] = 1.0,
) -> TensorBox:
    (
        query,
        key,
        value,
        attn_mask,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            attn_mask,
        ]
    )

    if (
        query.get_dtype() is not torch.uint8
        or key.get_dtype() is not torch.uint8
        or value.get_dtype() is not torch.uint8
    ):
        raise NotImplementedError(
            "Only `torch.uint8` is supported in Int8 SDPA template for CPU device. "
            f"Found input tensors are `{query.get_dtype()}`,`{key.get_dtype()}`,`{value.get_dtype()}`."
        )

    # Construct output layout with strides matching the query.
    out_size = query.get_size()
    fill_order = get_fill_order(query.get_stride())
    out_strides = construct_strides(out_size, fill_order)

    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        out_size,
        stride=[sympy.sympify(s) for s in out_strides],
    )
    _choices: List[Any] = []
    input_nodes = [query, key, value]
    if attn_mask is not None:
        input_nodes.append(attn_mask)

    CppInt8SdpaTemplate.add_choices(
        choices=_choices,
        input_nodes=input_nodes,
        layout=layout,
        scale=1.0 / inv_scale,
        q_zp=q_zp,
        q_scale=q_scale,
        k_zp=k_zp,
        k_scale=k_scale,
        v_zp=v_zp,
        v_scale=v_scale,
        a_zp=a_zp,
        a_scale=a_scale,
        o_zp=o_zp,
        o_scale=o_scale,
    )
    inputs_for_autotuning = [
        query,
        key,
        value,
    ]
    res = autotune_select_algorithm(
        "int8_sdpa",
        _choices,
        inputs_for_autotuning,
        layout,
    )
    return res


int8_sdpa_lowering._inductor_lowering_function = True  # type: ignore[attr-defined]
