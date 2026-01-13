import warnings
from itertools import chain

import torch
from torch._utils import _get_device_index
from torch.autograd import Function
from torch.nn.parallel import comm


class Broadcast(Function):
    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        if not all(i.device.type != "cpu" for i in inputs):
            raise AssertionError("Broadcast function not implemented for CPU tensors")
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return ()
        ctx.num_inputs = len(inputs)
        ctx.input_device = inputs[0].get_device()

        ctx.complex_mask = [inp.is_complex() for inp in inputs]

        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)

        for device_outputs in outputs:
            for i, is_complex in enumerate(ctx.complex_mask):
                if is_complex:
                    device_outputs[i] = torch.view_as_complex(device_outputs[i])

        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                non_differentiables.extend(output[idx] for output in outputs)
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple(chain.from_iterable(outputs))

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = ReduceAddCoalesced.apply(
            ctx.input_device, ctx.num_inputs, *grad_outputs
        )

        return (None,) + grads


class ReduceAddCoalesced(Function):
    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [
            grads[i].get_device() for i in range(0, len(grads), num_inputs)
        ]

        complex_mask = [grads[i].is_complex() for i in range(num_inputs)]
        ctx.complex_mask = complex_mask

        grads_converted = tuple(
            torch.view_as_real(g) if g.is_complex() else g for g in grads
        )

        grads_ = [
            grads_converted[i : i + num_inputs]
            for i in range(0, len(grads_converted), num_inputs)
        ]
        results = comm.reduce_add_coalesced(grads_, destination)

        results = tuple(
            torch.view_as_complex(r) if is_complex else r
            for r, is_complex in zip(results, complex_mask)
        )

        return results

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (
            None,
            None,
        ) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):
    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        if not all(i.device.type != "cpu" for i in inputs):
            raise AssertionError("Gather function not implemented for CPU tensors")
        if target_device == "cpu":
            ctx.target_device = "cpu"
        else:
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector.",
                stacklevel=2,
            )
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)

        is_complex = len(inputs) > 0 and inputs[0].is_complex()

        output = comm.gather(inputs, ctx.dim, ctx.target_device)

        if is_complex:
            output = torch.view_as_complex(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(
            ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output
        )
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):
    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.accelerator.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(torch.device(device)) for device in target_gpus]

        is_complex = input.is_complex()

        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)

        if is_complex:
            outputs = tuple(torch.view_as_complex(o) for o in outputs)

        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.accelerator.device_index(target_gpus[i]):
                    main_stream = torch.accelerator.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams: list[torch.Stream | None] | None = None


def _get_stream(device: torch.device):
    """Get a background stream for copying between CPU and target device."""
    global _streams
    if device.type == "cpu" or not torch.accelerator.is_available():
        return None
    if torch.accelerator.current_accelerator().type != device.type:
        raise AssertionError(
            f"Expected current accelerator type {torch.accelerator.current_accelerator().type} "
            f"to match device type {device.type}"
        )
    if _streams is None:
        _streams = [None] * torch.accelerator.device_count()
    if _streams[device.index] is None:
        _streams[device.index] = torch.Stream(device.index)
    return _streams[device.index]
