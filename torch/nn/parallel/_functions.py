import warnings

import torch
import torch.cuda.comm as comm
from torch.autograd import Function
from torch.cuda._utils import _get_device_index


class Broadcast(Function):

    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        if not all(input.is_cuda for input in inputs):
            raise TypeError('Broadcast function not implemented for CPU tensors')
        target_gpus = list(map(lambda x: _get_device_index(x, True), target_gpus))
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return tuple()
        ctx.num_inputs = len(inputs)
        ctx.input_device = inputs[0].get_device()
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)


class ReduceAddCoalesced(Function):

    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

        grads = [grads[i:i + num_inputs]
                 for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None,) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(map(lambda i: i.is_cuda, inputs))
        target_device = _get_device_index(target_device, True)
        ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn('Was asked to gather along dimension 0, but all '
                          'input tensors were scalars; will instead unsqueeze '
                          'and return a vector.')
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs))
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = list(map(lambda x: _get_device_index(x, True), target_gpus))
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.is_cuda else -1
        streams = None
        if ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams = None


def _get_stream(device):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.cuda.device_count()
    if _streams[device] is None:
        _streams[device] = torch.cuda.Stream(device)
    return _streams[device]
