import torch
from torch.autograd import Variable, Function
import pdb, math


def repackage_inputs(inputs, requires_grad=False):
    if type(inputs) == Variable:
        return Variable(inputs.data, requires_grad=requires_grad)
    elif torch.is_tensor(inputs):
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
        inputs = repackage_inputs(args, requires_grad=True)
        with torch.no_grad():
            outputs = run_function(*inputs)     # the *inputs* is always a tuple
        return unpack_variables(outputs)

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.saved_tensors
        inputs_list = repackage_inputs(inputs, requires_grad=True)
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs_list)

        if isinstance(outputs, tuple):
            output_list = list(outputs)
        elif isinstance(outputs, Variable) or torch.is_tensor(outputs):
            output_list = [outputs]
        out_grads = [grad for grad in args]
        torch.autograd.backward(output_list, out_grads)

        input_grads = None
        if isinstance(inputs_list, tuple):
            input_grads = tuple(inp.grad for inp in inputs_list)
            return (None,) + input_grads
        elif isinstance(inputs_list, Variable) or torch.is_tensor(inputs_list):
            input_grads = inputs_list.grad
            return None, input_grads


def checkpoint(run_function, *args):
    ck = CheckpointFunction()
    return ck.apply(run_function, *args)


def checkpoint_sequential(modules, segments, custom_run, *inputs):
    total_modules = len(modules)
    segment_size = int(math.floor(float(total_modules) / segments))
    start, end = 0, -1
    for j in range(segments - 1):     # the last chunk has to be non-volatile
        start = end + 1
        end = start + segment_size - 1
        inputs = checkpoint(custom_run(start, end, modules), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    start = end + 1
    end = len(modules) - 1
    inputs = custom_run(start, end, modules)(*inputs)
    return inputs
