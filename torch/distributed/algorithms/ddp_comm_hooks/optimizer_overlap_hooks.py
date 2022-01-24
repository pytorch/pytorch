from typing import Any, Callable

import torch
import torch.distributed as dist


class _OptimizerHookState(object):
    """
    Holds state for running optimizer in-line after DDP communication hook.
    Currently contains only optimizer class which must have a method `step_param`.
    """

    __slots__ = ["functional_optimizer"]

    def __init__(
        self, functional_optim_cls, *functional_optim_args, **functional_optim_kwargs
    ):
        self.functional_optimizer = functional_optim_cls(
            [],
            *functional_optim_args,
            **functional_optim_kwargs,
            _allow_empty_param_list=True,
        )
        if not hasattr(self.functional_optimizer, "step_param"):
            raise ValueError(
                f"Class {functional_optim_cls} must implement method step_param."
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
