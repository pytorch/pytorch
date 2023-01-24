from typing import Any, Callable, List, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable
from functools import partial
from dataclasses import dataclass

__all__: List[str] = []

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"

class _OptimizerHookState(object):
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

# TODO (rohan-varma): Unify the below hooks once DDP MP + optimizer overlap
# is enabled to work together.
@dataclass
class _AllreduceUpcastHookState:
    ddp_inst: Any # TODO (rohan-varma): make weakref
    reducer_weakref: dist.Reducer
    process_group: dist.ProcessGroup
    upcast_stream: torch.cuda.Stream
    wait_for_stream_enqueued: bool
    div: bool
    upcast_params_and_grads: bool

def _reducer_allreduce_and_upcast_hook(
    hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    reducer_weakref = hook_state.reducer_weakref
    process_group = hook_state.process_group
    gradient_is_bucket_view = hook_state.ddp_inst.gradient_as_bucket_view
   # print(f"RV: it {hook_state.ddp_inst.num_iterations}")
   # print("calling allreduce")
   # print(f"RV: gradient buffer dtype {buf.dtype}")
    # Cast bucket if different than param_dtype.
    if (hook_state.ddp_inst.mixed_precision.param_dtype != hook_state.ddp_inst.mixed_precision.reduce_dtype):
        # Cast bucket tensor to reduce_dtype
        bucket.set_buffer(bucket.buffer().to(hook_state.ddp_inst.mixed_precision.reduce_dtype))
    fut = reducer_weakref()._run_allreduce_hook(bucket)
    ret_fut = torch.futures.Future()
    stream = hook_state.upcast_stream
    with torch.cuda.stream(stream):
        #print("in stream")
        fut.wait()
        if hook_state.div:
            bucket.buffer().div_(process_group.size())
        ret_fut.set_result(bucket.buffer())

        if hook_state.upcast_params_and_grads:
            #print("upcasting")
            params, grads = bucket.parameters(), bucket.gradients()
            for p, g in zip(params, grads):
                #print(f"RV: before upcast reduced types: {p.dtype} {g.dtype}")
                # Remove the hook in preparation for next iteration
                p.data = p._fp_param
                if not gradient_is_bucket_view:
                    g.data = g.data.to(p.data.dtype)
                    p.grad = g
                else:
                    p.grad.data = p.grad.to(p.data.dtype)
                #print(f"type after upcast {p.data.dtype} {p.grad.data.dtype}")

    # enqueue a callback to wait for this stream at end of backward
    def wait_for_stream_cb():
        torch.cuda.current_stream().wait_stream(stream)
        # reset for next backward pass
        hook_state.wait_for_stream_enqueued = False
        for p in hook_state.ddp_inst.module.parameters():
            if hasattr(p, '_hook_state'):
                p._hook_state[1].remove()
        # TODO: rohan-varma put this in post-backward
        # for p in self.module.parameters():
        #     if hasattr(p, '_hook_state'):
        #         print("removing hook")
        #         p._hook_state[1].remove()

    if not hook_state.wait_for_stream_enqueued:
        Variable._execution_engine.queue_callback(
            wait_for_stream_cb
        )
        # mark that the callback is enqueued
        hook_state.wait_for_stream_enqueued = True

    return ret_fut

@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.cuda.Stream
    wait_for_optim_stream_enqueued: bool

@no_type_check
def _apply_optim_in_backward_hook(
    gradient_is_bucket_view: bool
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    r"""
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
        reducer_weakref, process_group = hook_state
        fut = reducer_weakref()._run_allreduce_hook(bucket)
        optimizer_stream = optim_stream_state.optim_stream
        with torch.cuda.stream(optimizer_stream):
            fut.wait()
            # Apply gradient division since C++ side only allreduces and does
            # not average. TODO: (rohan-varma) the div factor may be different
            # when running with join hook
            bucket.buffer().div_(process_group.size())
            model_params = bucket.parameters()
            grads = bucket.gradients()
            # TODO (rohan-varma): upcast as needed for DDP mixed precision.
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
        # backward.
        def wait_for_optim_stream_callback():
            torch.cuda.current_stream().wait_stream(
                optim_stream_state.optim_stream
            )
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
    r"""
    Runs optimizer in a functional fashion after DDP communication hook.
    """
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
