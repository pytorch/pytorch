import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d


class DefaultState(object):
    r"""
    Stores state needed to perform the default ``all_reduce`` algorithm
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

class LowPrecisionState(DefaultState):

    __slots__ = [
        "parameter_type",
    ]

    def __init__(
        self,
        process_group,
        parameter_type,
    ):
        super().__init__(process_group)
        self.parameter_type = parameter_type


def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    """
    Casts gradients back to full parameter precision so that
    further computation happens in full precision
    """

    orig_grad_data = grad.data
    grad.data = grad.data.to(state.parameter_type)
    # Don't let this memory get reused until after the transfer.
    orig_grad_data.record_stream(torch.cuda.current_stream())

def allreduce_hook(state: DefaultState, grad: torch.Tensor):
    r"""
    This FSDP communication hook implements ``all_reduce`` algorithm
    and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    if state.gradient_postdivide_factor > 1:
        grad.div_(state.gradient_postdivide_factor)

def fp16_compress_hook(state: LowPrecisionState, grad: torch.Tensor):
    r"""
    This FSDP communication hook implements a simple gradient compression
    approach that casts ``grad`` to half-precision floating-point format (``torch.float16``).
    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.predivide_factor``, and after an allreduce step gradients are averaged by a ``state.postdivide_factor``.
    Onse post-division is done, compressed gradients are casted back to full precision (``torch.float32``).

    Args:
        state (DefaultState): State information, configures pre- and post-division factors
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """

    grad.data = grad.data.to(torch.float16)
    allreduce_hook(state, grad)
    _decompress(state, grad)

def bf16_compress_hook(state: LowPrecisionState, grad: torch.Tensor):
    r"""
    This FSDP communication hook implements a simple gradient compression
    approach that casts ``grad`` to half-precision floating-point format (``torch.float16``).
    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.predivide_factor``, and after an allreduce step gradients are averaged by a ``state.postdivide_factor``.
    Onse post-division is done, compressed gradients are casted back to full precision (``torch.float32``).

    Args:
        state (DefaultState): State information, configures pre- and post-division factors
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """

    grad.data = grad.data.to(torch.bfloat16)
    allreduce_hook(state, grad)
    _decompress(state, grad)
