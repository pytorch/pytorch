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


@torch.library.custom_op("aten_cudagraphs::copy_to_cudagraphs", mutates_args={})
def copy_to_cudagraphs(
    inp: torch.Tensor,
) -> None:
    return None


def copy_to_cudagraphs_fake(
    inp: torch.Tensor,
) -> None:
    return None


copy_to_cudagraphs.register_fake(copy_to_cudagraphs_fake)


@torch.library.custom_op("aten_cudagraphs::disable_cudagraphs_begin", mutates_args={})
def disable_cudagraphs_begin() -> None:
    return None


def disable_cudagraphs_begin_fake() -> None:
    return None


disable_cudagraphs_begin.register_fake(disable_cudagraphs_begin_fake)


@torch.library.custom_op("aten_cudagraphs::disable_cudagraphs_end", mutates_args={})
def disable_cudagraphs_end() -> None:
    return None


def disable_cudagraphs_end_fake() -> None:
    return None


disable_cudagraphs_end.register_fake(disable_cudagraphs_end_fake)
