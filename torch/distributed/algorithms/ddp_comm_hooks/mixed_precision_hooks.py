import torch
import torch.distributed as dist
from torch.autograd import Variable

from dataclasses import dataclass
from typing import Any, no_type_check
from torch.distributed.utils import _free_storage

@dataclass
class _AllreduceUpcastHookState:
    """
    State to manage DDP mixed precision in backward / gradient communication.

    This contains a weakref to the DDP module for access to reducer and process
    group, and a stream to run parameter and gradient upcasts.
    """

    ddp_weakref: Any
    upcast_stream: torch.cuda.Stream
    wait_for_stream_enqueued: bool = False

@no_type_check
def _reducer_allreduce_and_upcast_hook(
    hook_state: _AllreduceUpcastHookState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Perform allreduce in precision ``reduce_dtype``, upcast to prepare for optimizer.

    Performs allreduce in the reduced precision given by DDP's mixed precision
    reduce_dtype, and upcasts parameters and gradients to fp32 in preparation
    to run the optimizer.
    """
    ddp_weakref = hook_state.ddp_weakref
    reducer, process_group = ddp_weakref().reducer, ddp_weakref().process_group
    gradient_is_bucket_view = ddp_weakref().gradient_as_bucket_view
    # Cast bucket if different than param_dtype.
    if (
        ddp_weakref().mixed_precision.param_dtype != ddp_weakref().mixed_precision.reduce_dtype
    ):
        # Cast bucket tensor to reduce_dtype
        bucket.set_buffer(bucket.buffer().to(ddp_weakref().mixed_precision.reduce_dtype))
    fut = reducer._run_allreduce_hook(bucket)
    ret_fut = torch.futures.Future()
    stream = hook_state.upcast_stream
    with torch.cuda.stream(stream):
        fut.wait()
        bucket.buffer().div_(process_group.size())
        ret_fut.set_result(bucket.buffer())

        # Upcast parameters and gradients so optimizer step can run in fp32.
        params, grads = bucket.parameters(), bucket.gradients()
        for p, g in zip(params, grads):
            p.data = p._fp_param
            # free storage for mp param as it will be allocated again in next
            # forward pass.
            _free_storage(p._mp_param)
            p.grad.data = p.grad.to(p.data.dtype)

    # enqueue a callback to wait for this stream at end of backward
    def wait_for_stream_cb():
        torch.cuda.current_stream().wait_stream(stream)
        # Remove post-backward hooks since they are re-installed in next
        # iteration, similar to FSDP.
        # Parameters that don't require grad still needed to be casted since
        # they may participate in computation. However, they would not be recast
        # by hook above as they don't have a grad hook installed, so cast them
        # back here.
        for n, p in ddp_weakref().module.named_parameters():
            if hasattr(p, '_ddp_mp_hook_state'):
                p._ddp_mp_hook_state[1].remove()
                delattr(p, '_ddp_mp_hook_state')
            if not p.requires_grad and not hasattr(p, '_ddp_ignored'):
                p.data = p._fp_param

        # reset for next backward pass
        hook_state.wait_for_stream_enqueued = False

    if not hook_state.wait_for_stream_enqueued:
        Variable._execution_engine.queue_callback(
            wait_for_stream_cb
        )
        # mark that the callback is enqueued
        hook_state.wait_for_stream_enqueued = True

    return ret_fut
