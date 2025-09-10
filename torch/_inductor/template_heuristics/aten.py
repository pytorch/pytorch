from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch._inductor import config as inductor_config

from ..kernel.bmm import aten_baddbmm, aten_bmm, aten_bmm_dtype
from ..kernel.mm import aten__fp8_mm, aten__int_mm, aten_addmm, aten_bias_addmm, aten_mm
from ..kernel.mm_plus_mm import aten_mm_plus_mm
from ..kernel_inputs import MMKernelInputs
from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..kernel_inputs import KernelInputs


# These are all labeled as device type None to indicate that they
# are valid for all device types
@register_template_heuristic(aten_mm.uid, None)
@register_template_heuristic(aten__fp8_mm.uid, None)
@register_template_heuristic(aten__int_mm.uid, None)
@register_template_heuristic(aten_bmm.uid, None)
@register_template_heuristic(aten_mm_plus_mm.uid, None)
# bmm dtype is only valid on cuda
@register_template_heuristic(aten_bmm_dtype.uid, "cuda")
class ATenConfigHeuristics(TemplateConfigHeuristics):
    """
    Pseudo heuristic to make ATen choices go through the same flow as other templates

    This is a single choice without kwargs

    If you want to use this with an ATen choice that has kwargs, just subclass
    """

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        yield dict()


# None here indicates that this is valid for all device types on that op
# Note (None, op) takes precedence over (device_type, None)
@register_template_heuristic(aten_baddbmm.uid, None, op_name="baddbmm")
class BaseATenAddMMConfigHeuristics(ATenConfigHeuristics):
    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_kwargs(kernel_inputs, op_name)
        alpha = kernel_inputs.get_scalar("alpha")
        beta = kernel_inputs.get_scalar("beta")
        return {
            **kwargs,
            "alpha": alpha,
            "beta": beta,
        }


@register_template_heuristic(aten_addmm.uid, None, op_name="addmm")
class ATenAddMMConfigHeuristics(
    BaseATenAddMMConfigHeuristics, TemplateConfigHeuristics
):
    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        # This is a compatibility layer, as the previous implementation relied on this
        # and it yields sometimes slightly different numerics
        # In the original implementation, addmm, when running in not max-autotune mode,
        # would take unexpanded bias
        # TODO: figure out if this can be handled cleaner e.g. through a subgraph or
        # through a different decomposition
        max_autotune = inductor_config.max_autotune or inductor_config.max_autotune_gemm
        assert isinstance(kernel_inputs, MMKernelInputs)
        if not max_autotune:
            nodes = kernel_inputs.nodes()
            bias = nodes[0]
            from ..ir import as_storage_and_layout, ReinterpretView

            # remove the expansion from the bias
            bias, old_layout = as_storage_and_layout(bias)
            filtered_data = [
                (old_layout.size[idx], old_layout.stride[idx])
                for idx, s in enumerate(old_layout.stride)
                if s != 0
            ]
            new_size, new_stride = zip(*filtered_data) if filtered_data else ([], [])
            new_size, new_stride = list(new_size), list(new_stride)
            layout = type(old_layout)(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            bias = ReinterpretView(data=bias, layout=layout)
            return MMKernelInputs(
                [bias, *nodes[1:]],
                scalars=kernel_inputs.scalars(),
                mat1_idx=kernel_inputs._mat1_idx,
                mat2_idx=kernel_inputs._mat2_idx,
            )
        # do the regular bias expansion
        return super().adjust_kernel_inputs(kernel_inputs, op_name)


@register_template_heuristic(aten_bias_addmm.uid, None, op_name="addmm")
class ATenBiasAddMMConfigHeuristics(
    ATenAddMMConfigHeuristics,
    BaseATenAddMMConfigHeuristics,
    GemmMaxAutotuneTemplateConfigHeuristics,
):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        nodes = kernel_inputs.nodes()
        # for addmm, bias is the first input
        bias = nodes[0]
        if bias.get_stride()[0] == 0 and inductor_config.triton.autotune_cublasLt:
            yield dict()
