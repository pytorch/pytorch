from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor import config as inductor_config
from torch._inductor.kernel_inputs import MMKernelInputs

from ..kernel.bmm import aten_baddbmm, aten_bmm, aten_bmm_dtype
from ..kernel.mm import aten__fp8_mm, aten__int_mm, aten_addmm, aten_bias_addmm, aten_mm
from ..kernel.mm_plus_mm import aten_mm_plus_mm
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..ir import Layout
    from ..kernel_inputs import KernelInputs

from .registry import register_template_heuristic


@register_template_heuristic(aten_mm.uid, "cuda")
@register_template_heuristic(aten_mm.uid, "cpu")
@register_template_heuristic(aten_mm.uid, "xpu")
@register_template_heuristic(aten_mm.uid, "mtia")
@register_template_heuristic(aten__fp8_mm.uid, "cuda")
@register_template_heuristic(aten__fp8_mm.uid, "cpu")
@register_template_heuristic(aten__fp8_mm.uid, "xpu")
@register_template_heuristic(aten__fp8_mm.uid, "mtia")
@register_template_heuristic(aten__int_mm.uid, "cuda")
@register_template_heuristic(aten__int_mm.uid, "cpu")
@register_template_heuristic(aten__int_mm.uid, "xpu")
@register_template_heuristic(aten__int_mm.uid, "mtia")
@register_template_heuristic(aten_bmm_dtype.uid, "cuda")
@register_template_heuristic(aten_bmm.uid, "cuda")
@register_template_heuristic(aten_bmm.uid, "cpu")
@register_template_heuristic(aten_bmm.uid, "xpu")
@register_template_heuristic(aten_bmm.uid, "mtia")
@register_template_heuristic(aten_mm_plus_mm.uid, "cuda")
@register_template_heuristic(aten_mm_plus_mm.uid, "cpu")
@register_template_heuristic(aten_mm_plus_mm.uid, "xpu")
@register_template_heuristic(aten_mm_plus_mm.uid, "mtia")
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


@register_template_heuristic(aten_addmm.uid, "cuda", op_name="addmm")
@register_template_heuristic(aten_addmm.uid, "cpu", op_name="addmm")
@register_template_heuristic(aten_addmm.uid, "xpu", op_name="addmm")
@register_template_heuristic(aten_addmm.uid, "mtia", op_name="addmm")
@register_template_heuristic(aten_baddbmm.uid, "cuda", op_name="baddbmm")
@register_template_heuristic(aten_baddbmm.uid, "cpu", op_name="baddbmm")
@register_template_heuristic(aten_baddbmm.uid, "xpu", op_name="baddbmm")
@register_template_heuristic(aten_baddbmm.uid, "mtia", op_name="baddbmm")
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
        return {
            **kwargs,
            "alpha": alpha,
            "beta": beta,
        }

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        # This is a compatibility layer, as the previous implementation relied on this
        # and it yields sometimes slightly different numerics
        # TODO: figure out if this can be handled cleaner e.g. through a subgraph or
        # through a different decomposition
        assert isinstance(kernel_inputs, MMKernelInputs), (
            f"MMKernelInputs expected for {op_name}"
        )
        nodes = kernel_inputs.nodes()
        max_autotune = inductor_config.max_autotune or inductor_config.max_autotune_gemm
        if op_name == "addmm" and not max_autotune:
            inp_unexpanded = kernel_inputs.views().get("inp_unexpanded")
            assert inp_unexpanded is not None, (
                f"inp_unexpanded needs to be available for {op_name}"
            )
            nodes = [inp_unexpanded, *nodes[1:]]
        return MMKernelInputs(
            nodes,
            scalars=kernel_inputs.scalars(),
            views=kernel_inputs.views(),
            mat1_idx=kernel_inputs._mat1_idx,
            mat2_idx=kernel_inputs._mat2_idx,
        )


@register_template_heuristic(aten_bias_addmm.uid, "cuda", op_name="addmm")
@register_template_heuristic(aten_bias_addmm.uid, "cpu", op_name="addmm")
@register_template_heuristic(aten_bias_addmm.uid, "xpu", op_name="addmm")
@register_template_heuristic(aten_bias_addmm.uid, "mtia", op_name="addmm")
class ATenBiasAddMMConfigHeuristics(ATenAddMMConfigHeuristics):
    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
        max_autotune: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        if not (inductor_config.max_autotune or inductor_config.max_autotune_gemm):
            # NOTE: this preserves the original logic that if there is not max-autotune
            # then we skip bias_addmm
            return
        nodes = kernel_inputs.nodes()
        # for addmm, bias is the first input
        bias = nodes[0]
        if bias.get_stride()[0] == 0 and inductor_config.triton.autotune_cublasLt:
            yield dict()
