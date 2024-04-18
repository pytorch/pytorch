from typing import Any, Callable, List, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass

__all__: List[str] = []

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"

class _OptimizerHookState:
    """
    Holds state for running optimizer in-line after DDP communication hook.

    Currently contains only optimizer class which must have a method `step_param`.
    """

    __slots__ = ["functional_optimizer", "params_to_optimize"]

    def __init__(self, functional_optim, params=None):
        self.functional_optimizer = functional_optim
        self._check_valid_functional_optim()
        self._set_params_to_optimize(params)

    def _set_params_to_optimize(self, params):
        if params is not None:
            self.params_to_optimize = set(params)

    def _check_valid_functional_optim(self):
        if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
            raise ValueError(
                f"Class {type(self.functional_optimizer)} must implement method "
                f"{_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}."
            )


@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.cuda.Stream
    wait_for_optim_stream_enqueued: bool

@no_type_check
def _apply_optim_in_backward_hook(
    gradient_is_bucket_view: bool
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
    Register hook to apply the optimizer in backward.

    If torch.distributed.optim._apply_optimizer_in_backward is used to overlap
    optimizer with backward pass, DDP will run the below hook to run optimizer
    step for parameters after gradient communication has taken place.
    """
    optim_in_bwd_state = _OptimInBackwardHookState(
        optim_stream=torch.cuda.Stream(),
        wait_for_optim_stream_enqueued=False,
    )

    def apply_optim_in_backward_hook(
        hook_state: Any, bucket: dist.GradBucket, optim_stream_state,
    ) -> torch.futures.Future[torch.Tensor]:
        # Run original hook
        ddp_weakref = hook_state
        ddp_inst = ddp_weakref()
        reducer, process_group = ddp_inst.reducer, ddp_inst.process_group
        fut = reducer._run_allreduce_hook(bucket)
        optimizer_stream = optim_stream_state.optim_stream
        with torch.cuda.stream(optimizer_stream):
            fut.wait()
            # Apply gradient division since C++ side only allreduces and does
            # not average. TODO: (rohan-varma) the div factor may be different
            # when running with join hook
            bucket.buffer().div_(process_group.size())
            model_params = bucket.parameters()
            grads = bucket.gradients()
            # TODO (rohan-varma): upcast as needed for DDP mixed precision,
            # once optimizer in backward + DDP mixed precision is supported.
            for p, g in zip(model_params, grads):
                if hasattr(p, '_in_backward_optimizers'):
                    # Note: need to set grad to the bucket's grad, because
                    # running allreduce results in the bucket's grad being
                    # reduced, but not grad field.
                    if not gradient_is_bucket_view:
                        p.grad = g
                    for optim in p._in_backward_optimizers:
                        optim.step()

        # Need to return a Future[Tensor] to obey comm hook API contract.
        ret_fut = torch.futures.Future()
        ret_fut.set_result(bucket.buffer())

        # enqueue a callback to wait for this optimizer stream at the end of
        # backward and set all DDP managed grads to None.
        def wait_for_optim_stream_callback():
            torch.cuda.current_stream().wait_stream(
                optim_stream_state.optim_stream
            )
            # Set DDP managed grads to None
            for param in ddp_inst._get_data_parallel_params(ddp_inst.module):
                if hasattr(param, '_in_backward_optimizers'):
                    param.grad = None

            # reset for the next backwards pass
            optim_stream_state.wait_for_optim_stream_enqueued = False

        if not optim_stream_state.wait_for_optim_stream_enqueued:
            Variable._execution_engine.queue_callback(
                wait_for_optim_stream_callback
            )
            # mark that the callback is enqueued
            optim_stream_state.wait_for_optim_stream_enqueued = True

        return ret_fut

    comm_hook = partial(
        apply_optim_in_backward_hook, optim_stream_state=optim_in_bwd_state
    )
    # These are needed for DDP's logging of comm hooks
    comm_hook.__name__ = apply_optim_in_backward_hook.__name__
    comm_hook.__qualname__ = apply_optim_in_backward_hook.__qualname__

    return comm_hook

def _hook_then_optimizer(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
    optimizer_state: _OptimizerHookState,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""Run optimizer in a functional fashion after DDP communication hook."""
    has_set_params = (
        hasattr(optimizer_state, 'params_to_optimize')
        and optimizer_state.params_to_optimize is not None
    )

    def hook_then_optimizer_wrapper(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # Run original hook
        fut = hook(hook_state, bucket)

        def optimizer_step(fut):
            gradient_tensors = bucket.gradients()
            model_params = bucket.parameters()
            for grad_tensor, model_param in zip(gradient_tensors, model_params):
                if not has_set_params or model_param in optimizer_state.params_to_optimize:
                    optimizer_state.functional_optimizer.step_param(
                        model_param,
                        grad_tensor,
                    )
            return bucket.buffer()

        return fut.then(optimizer_step)

    return hook_then_optimizer_wrapper
