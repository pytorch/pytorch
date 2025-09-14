import torch
from torch._library.custom_ops import custom_op


Tensor = torch.Tensor


@custom_op("streams::fork", mutates_args={"args"})
def fork_stream_(
    index: int, device: torch.device, device_index: int, args: list[Tensor]
) -> None:
    pass


@fork_stream_.register_fake
def _(index: int, device: torch.device, device_index: int, args: list[Tensor]) -> None:
    pass


@custom_op("streams::join", mutates_args={"args"})
def join_stream_(
    index: int, device: torch.device, device_index: int, args: list[Tensor]
) -> None:
    pass


@join_stream_.register_fake
def _(index: int, device: torch.device, device_index: int, args: list[Tensor]) -> None:
    pass
