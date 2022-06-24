import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d


class AllReduceState(object):
    r"""
    Stores state needed to perform the ``all_reduce`` algorithm
    within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used for all-reduce.
        world_size (int): The number of workers in a process group.
            Determined based on a ``process_group``.
        gradient_predivide_factor (float): A factor for gradients' pre-division.
        gradient_postdivide_factor (float): A factor for gradients' post-division.
    """

    __slots__ = [
        "process_group",
        "world_size",
        "gradient_predivide_factor",
        "gradient_postdivide_factor"
    ]

    def __init__(
        self,
        process_group
    ):
        self.process_group = process_group if process_group is not None else distributed_c10d._get_default_group()
        self.world_size = dist.get_world_size(process_group)
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor = self.world_size / self.gradient_predivide_factor

    # setting two factors `self.gradient_predivide_factor`
    # and `self.gradient_postdivide_factor` to avoid underflow and overflow
    def _get_gradient_predivide_factor(self, world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)


def allreduce_hook(state: AllReduceState, grad: torch.Tensor):
    r"""
    This FSDP communication hook implements ``all_reduce`` algorithm
    and a neccessary pre- and post-division of gradients.

    Args:
        state (AllReduceState): State information, configures pre- and post-division factors
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    if state.gradient_postdivide_factor > 1:
        # Average grad by world_size for consistency with PyTorch DDP.
        grad.div_(state.gradient_postdivide_factor)
