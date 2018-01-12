import torch
from torch.autograd import Variable, Function


def repackage_inputs(inputs, requires_grad=False):
    if torch.is_tensor(inputs):
        return Variable(inputs, requires_grad=requires_grad)
    elif isinstance(inputs, tuple):
        return tuple(repackage_inputs(v, requires_grad=requires_grad) for v in inputs)
    else:
        raise RuntimeError("Unknown input type")


def unpack_variables(inputs):
    if type(inputs) == Variable:
        return inputs.data
    elif torch.is_tensor(inputs):
        return inputs
    elif isinstance(inputs, tuple):
        return tuple(unpack_variables(v) for v in inputs)
    else:
        raise RuntimeError("Unknown input type")


class CheckpointFunction(Function):

    # NOTE: *args is the flat inputs list, args is the tuple containing inputs
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        inputs = repackage_inputs(args)
        with torch.no_grad():
            outputs = run_function(*inputs)     # the *inputs* is always a tuple
        return unpack_variables(outputs)

    @staticmethod
    def backward(ctx, *args):
        saved_inputs = ctx.saved_tensors
        inputs = repackage_inputs(saved_inputs, requires_grad=True)
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)

        if isinstance(outputs, tuple):
            output_list = list(outputs)
        elif isinstance(outputs, Variable) or torch.is_tensor(outputs):
            output_list = [outputs]
        out_grads = [grad for grad in args]
        torch.autograd.backward(output_list, out_grads)

        input_grads = None
        if isinstance(inputs, tuple):
            input_grads = tuple(inp.grad for inp in inputs)
            return (None,) + input_grads
        elif isinstance(inputs, Variable) or torch.is_tensor(inputs):
            input_grads = inputs.grad
            return None, input_grads


def checkpoint(run_function, *args):
    ck = CheckpointFunction()
    return ck.apply(run_function, *args)


def checkpoint_sequential(modules, segments, run_function, *inputs):
    segment_size = len(modules) // segments
    # the last chunk has to be non-volatile
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, modules), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(modules) - 1, modules)(*inputs)
