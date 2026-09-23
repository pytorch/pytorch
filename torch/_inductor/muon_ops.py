import torch
from torch import Tensor


@torch.library.custom_op("inductor::muon_reference", mutates_args=())
def muon_reference(
    inputs: list[Tensor],
    a: float,
    b: float,
    c: float,
    steps: int,
    eps: float,
) -> list[Tensor]:
    from .kernel.muon import _reference

    return [_reference(x, (a, b, c), steps, eps) for x in inputs]


def _empty_muon_output(input: Tensor) -> Tensor:
    if input.shape[0] > input.shape[1]:
        return input.new_empty((input.shape[1], input.shape[0]), dtype=torch.bfloat16).T
    return input.new_empty(input.shape, dtype=torch.bfloat16)


@muon_reference.register_fake
def _(inputs, a, b, c, steps, eps):
    return [_empty_muon_output(input) for input in inputs]


@torch.library.custom_op(
    "inductor::grouped_muon",
    mutates_args=(),
    tags=(torch._C.Tag.cudagraph_unsafe,),
)
def grouped_muon(
    inputs: list[Tensor],
    a: float,
    b: float,
    c: float,
    steps: int,
    eps: float,
) -> list[Tensor]:
    from .kernel.muon import grouped_muon_impl

    return grouped_muon_impl(inputs, a, b, c, steps, eps)


@grouped_muon.register_fake
def _(inputs, a, b, c, steps, eps):
    return [_empty_muon_output(input) for input in inputs]
