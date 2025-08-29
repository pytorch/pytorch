import torch
from torch._library.custom_ops import custom_op


Tensor = torch.Tensor


@custom_op("streams::fork", mutates_args=())
def fork_stream(index: int, args: list[Tensor]) -> list[Tensor]:
    return [arg.clone() for arg in args]


@fork_stream.register_fake
def _(index: int, args: list[Tensor]) -> list[Tensor]:
    return [arg.clone() for arg in args]


@custom_op("streams::join", mutates_args=())
def join_stream(index: int, args: list[Tensor]) -> list[Tensor]:
    return [arg.clone() for arg in args]


@join_stream.register_fake
def _(index: int, args: list[Tensor]) -> list[Tensor]:
    return [arg.clone() for arg in args]
