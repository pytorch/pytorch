import torch


@torch.library.custom_op("aten_cudagraphs::exclude_from_cudagraphs", mutates_args={})
def exclude_from_cudagraphs(
    inp: torch.Tensor,
    clone: bool = False,
) -> None:
    return None


def exclude_from_cudagraphs_fake(
    inp: torch.Tensor,
    clone: bool = False,
) -> None:
    return None


exclude_from_cudagraphs.register_fake(exclude_from_cudagraphs_fake)
