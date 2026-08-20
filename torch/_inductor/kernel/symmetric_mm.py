import torch
from torch import Tensor

from ..ir import FixedLayout, FlexibleLayout
from ..lowering import register_lowering
from ..select_algorithm import ExternKernelChoice, realize_inputs


def _quack_symmetric_mm(x: Tensor, *, out: Tensor) -> None:
    from torch._vendor.quack.gemm_symmetric import gemm_symmetric

    gemm_symmetric(x, out, autotune=True)


quack_symmetric_mm_extern = ExternKernelChoice(_quack_symmetric_mm)


@torch.library.custom_op("inductor::quack_symmetric_mm", mutates_args=())
def quack_symmetric_mm(x: Tensor) -> Tensor:
    out = torch.empty(
        (*x.shape[:-2], x.shape[-2], x.shape[-2]),
        dtype=x.dtype,
        device=x.device,
    )
    _quack_symmetric_mm(x, out=out)
    return out


@quack_symmetric_mm.register_fake
def _(x: Tensor) -> Tensor:
    return x.new_empty((*x.shape[:-2], x.shape[-2], x.shape[-2]))


@register_lowering(
    torch.ops.inductor.quack_symmetric_mm.default, type_promotion_kind=None
)
def quack_symmetric_mm_lowering(x):
    x = realize_inputs(x)
    size = x.get_size()
    output_size = [*size[:-2], size[-2], size[-2]]
    layout = FixedLayout(
        x.get_device(),
        x.get_dtype(),
        output_size,
        FlexibleLayout.contiguous_strides(output_size),
    )
    return quack_symmetric_mm_extern.bind([x], layout).output_node()
