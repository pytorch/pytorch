import torch
from torch import Tensor

from ..ir import ExternKernel, FixedLayout, FlexibleLayout
from ..lowering import register_lowering
from ..select_algorithm import ExternKernelChoice, realize_inputs


def _quack_symmetric_mm(x: Tensor, *, out: Tensor) -> None:
    from ..utils import ensure_cute_available

    if (
        torch.version.hip is not None
        or not ensure_cute_available()
        or torch.cuda.get_device_capability(x.device)[0] != 10
        or not x.is_contiguous()
        or x.data_ptr() % 16
    ):
        torch.matmul(x, x.mT, out=out)
        return

    from torch._inductor.kernel.flex_gemm.runtime import inductor_quack_cache_dir
    from torch._vendor.quack.cache import cache_dir_override
    from torch._vendor.quack.gemm_interface import gemm_symmetric

    with cache_dir_override(inductor_quack_cache_dir()):
        gemm_symmetric(x, x.mT, out=out)


quack_symmetric_mm_extern = ExternKernelChoice(_quack_symmetric_mm)


@torch.library.custom_op(
    "inductor::quack_symmetric_mm", mutates_args=(), device_types="cuda"
)
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
    x = ExternKernel.require_contiguous(realize_inputs(x))
    size = x.get_size()
    output_size = [*size[:-2], size[-2], size[-2]]
    layout = FixedLayout(
        x.get_device_or_error(),
        x.get_dtype(),
        output_size,
        FlexibleLayout.contiguous_strides(output_size),
    )
    return quack_symmetric_mm_extern.bind([x], layout).output_node()
