from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor import config as inductor_config

from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..ir import Layout
    from ..kernel_inputs import KernelInputs

from .registry import register_template_heuristic


@register_template_heuristic(torch._inductor.kernel.mm.aten_mm.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.mm.aten_mm.uid, "cpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten_mm.uid, "xpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten_mm.uid, "mtia")
@register_template_heuristic(torch._inductor.kernel.mm.aten__fp8_mm.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.mm.aten__fp8_mm.uid, "cpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten__fp8_mm.uid, "xpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten__fp8_mm.uid, "mtia")
@register_template_heuristic(torch._inductor.kernel.mm.aten__int_mm.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.mm.aten__int_mm.uid, "cpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten__int_mm.uid, "xpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten__int_mm.uid, "mtia")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_bmm_dtype.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_bmm.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_bmm.uid, "cpu")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_bmm.uid, "xpu")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_bmm.uid, "mtia")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_baddbmm.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_baddbmm.uid, "cpu")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_baddbmm.uid, "xpu")
@register_template_heuristic(torch._inductor.kernel.bmm.aten_baddbmm.uid, "mtia")
@register_template_heuristic(
    torch._inductor.kernel.mm_plus_mm.aten_mm_plus_mm.uid, "cuda"
)
@register_template_heuristic(
    torch._inductor.kernel.mm_plus_mm.aten_mm_plus_mm.uid, "cpu"
)
@register_template_heuristic(
    torch._inductor.kernel.mm_plus_mm.aten_mm_plus_mm.uid, "xpu"
)
@register_template_heuristic(
    torch._inductor.kernel.mm_plus_mm.aten_mm_plus_mm.uid, "mtia"
)
class ATenConfigHeuristics(TemplateConfigHeuristics):
    """
    Pseudo heuristic to make ATen choices go through the same flow as other templates

    This is a single choice without kwargs

    If you want to use this with an ATen choice that has kwargs, just subclass
    """

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
        max_autotune: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        yield dict()


@register_template_heuristic(torch._inductor.kernel.mm.aten_addmm.uid, "cuda")
@register_template_heuristic(torch._inductor.kernel.mm.aten_addmm.uid, "cpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten_addmm.uid, "xpu")
@register_template_heuristic(torch._inductor.kernel.mm.aten_addmm.uid, "mtia")
class ATenAddMMConfigHeuristics(ATenConfigHeuristics):
    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, layout, op_name)
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        kwargs["alpha"] = alpha
        kwargs["beta"] = beta
        return kwargs


@register_template_heuristic(torch._inductor.kernel.mm.aten_bias_addmm.uid, "cuda")
class ATenBiasAddMMConfigHeuristics(ATenAddMMConfigHeuristics):
    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
        max_autotune: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        nodes = kernel_inputs.nodes()
        # for addmm, bias is the first input
        bias = nodes[0]
        if bias.get_stride()[0] == 0 and inductor_config.triton.autotune_cublasLt:
            yield dict()
        else:
            yield from []
