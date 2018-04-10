import torch


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        return (None,) + tuple(inp.grad for inp in detached_inputs)


def checkpoint(run_function, *args):
    r"""Checkpoint a model or part of the model

    Checkpoint works by trading compute for memory. It can be applied on any
    part of the model. In the forward pass, the model activations are not
    stored. The forward pass save the inputs tuple and the run_function
    parameter. In the backwards pass, the saved inputs and run_function is
    retreived, and the forward pass is done on the model again (non-volatile
    this time) since we need to get the activations values for calculating the
    gradient and then the gradients are calculated.

    .. warning::

        checkpointing doesn't work with torch.autograd.grad(), but only with
        torch.autograd.backward()

    Args:
        run_function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, user passes (activation,
            hidden), run_function should correctly use first input as activation
            and second input as hidden
        args: tuple containing inputs to the run_function

    Returns:
        Output of running the run_function on *args
    """
    return CheckpointFunction.apply(run_function, *args)


def checkpoint_sequential(functions, segments, *inputs):
    r"""A helper function for checkpointing Sequential based models.

    For models that are constructed using Sequential, they normally are built
    using various modules/functions. For such models, given a list of modules/functions
    it executes in order (sequentially), we can divide the model in various
    segments and checkpoint the segments. All segments except the last will be
    run in volatile manner i.e. the model activations are not stored. The inputs
    of each checkpointed segment will be saved for re-running the segment in the
    backward pass.

    .. warning::

        checkpointing doesn't work with torch.autograd.grad(), but only with
        torch.autograd.backward()

    Args:
        functions: A sequential or the list of modules or functions (comprising the model) to run in order.
        segments: Number of chunks to create in the model
        inputs: tuple of Tensors that are inputs to run_function

    Returns:
        Output of running the modules/functions on *inputs

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """

    def run_function(start, end, functions):
        def forward(*inputs):
            input = inputs[0]
            for j in range(start, end + 1):
                input = functions[j](input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(functions) - 1, functions)(*inputs)
