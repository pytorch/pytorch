from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.optim import create_functional_optim

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"

class _OptimizerHookState(object):
    """
    Holds state for running optimizer in-line after DDP communication hook.
    Currently contains only optimizer class which must have a method `step_param`.
    """

    __slots__ = ["functional_optimizer"]

    def __init__(
        self, functional_optim_cls, *functional_optim_args, **functional_optim_kwargs
    ):
        self.functional_optimizer = create_functional_optim(
            functional_optim_cls,
            *functional_optim_args,
            **functional_optim_kwargs,
        )
        self._check_valid_functional_optim()

    @classmethod
    def from_functional_optim(cls, functional_optim):
        r"""
        Create a `_OptimizerHookState`, which simply
        holds a functional optimizer, directly from a
        functional optimizer given by `functional_optim`.
        Note that the `functional_optim` must implement
        `step_param` to support per-parameter optimization.
        """
        opt_hook_state_inst = cls.__new__(cls)  # Does not call __init__
        opt_hook_state_inst.functional_optimizer = functional_optim
        opt_hook_state_inst._check_valid_functional_optim()
        return opt_hook_state_inst

    def _check_valid_functional_optim(self):
        if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
            raise ValueError(
                f"Class {type(self.functional_optimizer)} must implement method "
                f"{_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}."
            )


# TODO: Add an example to use such a wrapper.
def _hook_then_optimizer(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
    optimizer_state: _OptimizerHookState,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Runs optimizer in a functional fashion after DDP communication hook.

    .. warning ::
        This API is experimental adn subject to change.
    """

    def hook_then_optimizer_wrapper(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Run original hook
        fut = hook(hook_state, bucket)

        def optimizer_step(fut):
            gradient_tensors = bucket.gradients()
            model_params = bucket.parameters()
            for grad_tensor, model_param in zip(gradient_tensors, model_params):
                optimizer_state.functional_optimizer.step_param(
                    model_param,
                    grad_tensor,
                )
            return bucket.buffer()

        return fut.then(optimizer_step)

    return hook_then_optimizer_wrapper
