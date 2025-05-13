import logging
from typing import Any

import torch
from torch._inductor.kernel.mm_common import mm_args

from . import config as inductor_config, lowering
from .codegen.cpp_gemm_template import CppGemmTemplate, CppWoqInt4GemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .lowering import expand, register_lowering
from .mkldnn_ir import WeightInt4PackMatmul
from .select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template, use_max_autotune
from .virtualized import V


log = logging.getLogger(__name__)

aten__weight_int8pack_mm = ExternKernelChoice(
    torch._weight_int8pack_mm, "at::_weight_int8pack_mm", has_out_variant=False
)

aten__weight_int4pack_mm_cpu = ExternKernelChoice(
    torch.ops.quantized.int4mm_packed_weight_cpu,
    "at::native::_weight_int4pack_mm_cpu_tensor",
    has_out_variant=False,
    kernel_creator=WeightInt4PackMatmul.create,
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

    @register_lowering(aten._weight_int4pack_mm_for_cpu, type_promotion_kind=None)  # type: ignore[misc]
    def int4pack_mm_cpu(
        input: torch.Tensor,
        weight: torch.Tensor,
        qGroupSize: int,
        qScaleAndZeros: torch.Tensor,
        *,
        layout: Any = None,
    ) -> Any:
        _, _, _, layout, mat1, mat2 = mm_args(
            input, weight, layout=layout, use_4x2_dim=True, mat2_transposed=True
        )
        assert (
            mat1.get_dtype() in [torch.bfloat16, torch.float16, torch.float]
            and mat2.get_dtype() == torch.uint8
        )
        group_size = V.graph.add_tensor_constant(
            torch.tensor(qGroupSize, dtype=torch.int64), name=None
        )
        aten_layout = layout

        # options to tune from
        choices = (
            [
                aten__weight_int4pack_mm_cpu.bind(
                    (mat1, mat2, group_size, qScaleAndZeros), aten_layout
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        if (
            use_max_autotune()
            and use_cpp_gemm_template(
                aten_layout,
                mat1,
                mat2,
                mat2_transposed=True,
                is_woq_int4=True,
                q_group_size=qGroupSize,
            )
            and mat2.get_layout().is_contiguous()
        ):
            CppWoqInt4GemmTemplate[qGroupSize].add_choices(
                choices,
                aten_layout,
                [mat1, mat2, group_size, qScaleAndZeros],
            )

        if (
            len(choices) == 0
            and inductor_config.autotune_fallback_to_aten
            and not use_aten_gemm_kernels()
        ):
            log.warning("No choices for GEMM, using ATen backend as fallback")
            return aten__weight_int4pack_mm_cpu.bind(
                (mat1, mat2, group_size, qScaleAndZeros), aten_layout
            ).output_node()

        # define functions to generate example inputs for weight and group size
        # otherwise, autotuner generates example inputs of all zeros for them
        def get_example_weight(x: torch._inductor.ir.IRNode) -> torch.Tensor:
            assert x.get_layout().is_contiguous()
            shape = x.get_size()
            device = x.get_device()
            return torch.randint(0, 255, shape, dtype=torch.uint8, device=device)

        input_gen_fns = {
            1: get_example_weight,  # packed weight
            2: lambda x: V.graph.constants[x.get_name()],  # group size
        }

        return autotune_select_algorithm(
            "_weight_int4pack_mm_for_cpu",
            choices,
            [mat1, mat2, group_size, qScaleAndZeros],
            aten_layout,
            input_gen_fns=input_gen_fns,
        )

    lowering.make_fallback(aten._dyn_quant_matmul_4bit)
    lowering.make_fallback(aten._dyn_quant_pack_4bit_weight)
