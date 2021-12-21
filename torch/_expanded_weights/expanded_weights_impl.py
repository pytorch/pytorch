import torch
import functools
import warnings

from typing import Callable, Dict, List, Union

HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function] = {}

def implements_per_sample_grads(torch_function):
    @functools.wraps(torch_function)
    def decorator(autograd_func):
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func
    return decorator

# ExpandedWeight represents a weight (parameter) Tensor that has an expanded
# batch dimension. Operations on the ExpandedWeight Tensor take advantage of
# how the batch dimension is expanded by de-expanding the weight before
# computation. A subsequent call to .backward() computes gradients for
# ExpandedWeight. Those gradients are equivalent to per-sample-grads for the
# unexpanded weight Tensors.
#
# ExpandedWeight has a fallback that does the forward + backward computation.
# The backward computation is not optimized: it runs torch.autograd.grad in
# a loop. To optimize the backward computation further, we must register
# overrides for specific operators.
#
# This is a __torch_function__ object but it could have also been a Tensor Extension
# with a dispatch key.
class ExpandedWeight(torch.Tensor):
    handled_functions = HANDLED_FUNCTIONS

    # needed for conv2d default kwargs
    conv_kwarg_options = ['stride', 'padding', 'dilation', 'groups']
    conv_kwarg_defaults = {'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1}

    def __new__(cls, orig_weight, batch_size):
        ret = torch.Tensor._make_subclass(cls, orig_weight.detach(), orig_weight.requires_grad)
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(f"Can only make ExpandedWeights of Tensors, got {type(orig_weight).__name__}")
        cls.batch_size = batch_size
        cls.orig_weight = orig_weight
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in cls.handled_functions:
            warnings.warn(f"don't have custom implementation for function {func.__name__}. Using slow fallback")
            return SlowFallback.apply(func, *(args + tuple(kwargs.values())))
        if func == torch.nn.functional.conv2d:
            remaining_kwargs = 7 - len(args)
            remaining_kwargs_options = cls.conv_kwarg_options[4 - remaining_kwargs:]
            kwargs = {key: cls.conv_kwarg_defaults[key] for key in remaining_kwargs_options} | kwargs
        return cls.handled_functions[func].apply(*(args + tuple(kwargs.values())))

    @property
    def shape(self):
        return self.orig_weight.shape

    @property
    def dtype(self):
        return self.orig_weight.dtype

    @property
    def grad(self):
        return None

    @grad.setter
    def grad(self, value):
        if value is None:
            return
        else:
            raise RuntimeError("ExpandedWeights should never have a grad value set on it.")

    @property
    def requires_grad(self):
        return self.orig_weight.requires_grad

    @property
    def grad_fn(self):
        return None

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "ExpandedWeight for:\n" + self.orig_weight.__repr__() + f" with batch size {self.batch_size}"

# Fallback:
# - forward pass: run operation with de-expanded input
# - backward pass: run autograd.grad in a for-loop to compute per-sample-grads.
#                  This is NOT something the vmap API can handle, something more
#                  low-level is at work here.
class SlowFallback(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *func_and_expanded_args):
        func = func_and_expanded_args[0]
        expanded_args = func_and_expanded_args[1:]
        with torch.enable_grad():
            output = forward_helper(func, ctx, expanded_args, 1)
            true_output = ctx.true_outputs
            if not isinstance(true_output, torch.Tensor):
                raise RuntimeError(f"Fallback only works on Tensor output, got output was {type(true_output).__name__}")
            outputs = tuple(true_output[i] for i in range(true_output.shape[0]))
        ctx.outputs = outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        outputs = ctx.outputs
        diff_args = tuple(arg for arg in ctx.args
                          if isinstance(arg, torch.Tensor) and arg.requires_grad)
        diff_args = tuple(arg.orig_weight if isinstance(arg, ExpandedWeight) else arg for arg in diff_args)
        batch_size = grad_output.shape[0]
        per_sample_grads = tuple(torch.autograd.grad(outputs[i], diff_args, grad_output[i],
                                                     retain_graph=(i != batch_size - 1))
                                 for i in range(batch_size))
        per_sample_grads_iter = zip(*per_sample_grads)
        result: List[Union[torch.Tensor, None]] = []
        result.append(None)  # for function input
        for arg in ctx.args:
            if isinstance(arg, ExpandedWeight):
                cur_per_sample_grad = next(per_sample_grads_iter)
                arg.orig_weight.grad_sample = torch.stack(cur_per_sample_grad)
                result.append(None)
            elif isinstance(arg, torch.Tensor) and arg.requires_grad:
                cur_per_sample_grad = next(per_sample_grads_iter)
                result.append(torch.stack(cur_per_sample_grad).sum(0))
            else:
                result.append(None)
        result.append(None)
        return tuple(result)

def forward_helper(func, ctx, expanded_args, num_true_outs):
    r'''Forward helper computes the forward pass for a function that has expanded weight(s)
    passed to it. It will run the forward pass where all ExpandedWeights are their original
    weight. It runs checks on the given arguments and detaches the outputs.

    .. note:: First argument in :attr:`expanded_args` must be the input with the batch
    dimension as the first element of the shape

    .. note:: :attr:`func` must return a Tensor or tuple of Tensors

    Args:
        func: The function to be called
        ctx: The context from the autograd.Function object. Will be used to save
          computed state from the forward pass
        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments
          that need to be unpacked because they are ExpandedWeights
        num_true_outs: The number of outputs seen by the user since some functions
          return auxillary data that is only used in the backward pass
    '''
    unexpanded_args = _check_and_unexpand_args(func, ctx, expanded_args)
    output = func(*unexpanded_args)
    return _check_and_detach_output(ctx, output, num_true_outs)

def _check_and_unexpand_args(func, ctx, expanded_args):
    ctx.args = expanded_args
    # input must be the first argument passed
    input = expanded_args[0]
    if isinstance(input, ExpandedWeight):
        raise RuntimeError("Expanded Weights do not support inputs that are also ExpandedWeights. "
                           f"Input must be a Tensor, got {type(input).__name__} in function {func.__name__}")
    if not isinstance(input, torch.Tensor):
        raise RuntimeError("Expanded Weights requires a Tensor as the first input to get the batch dimension, "
                           f"got {type(input).__name__} in function {func.__name__}")
    if len(input.shape) == 0:
        raise RuntimeError(f"Expanded Weights requires a batch dimension but got an input of size 0 in function {func.__name__}")
    if input.shape[0] == 0:
        raise RuntimeError("0 is not a valid batch size for Expanded Weights but got input tensor of "
                           f"{input} in function {func.__name__}")
    unexpanded_args = tuple(arg.orig_weight if isinstance(arg, ExpandedWeight) else arg for arg in expanded_args)
    return unexpanded_args

def _check_and_detach_output(ctx, output, num_true_outs):
    ctx.all_outputs = output

    # separates differentiable outputs from outputs only needed for the backwards computation
    if isinstance(output, tuple):
        if len(output) < num_true_outs:
            raise RuntimeError(f"Got fewer outputs ({len(output)}) than expected ({num_true_outs}). "
                               "Issues in ExpandedWeights' autograd.Function")
        if num_true_outs == 1:
            output = output[0]  # removes tuple wrapper
        else:
            output = output[:num_true_outs]
    elif num_true_outs != 1:
        raise RuntimeError(f"Got single output but expected at least {num_true_outs} outputs. "
                           "Issues in ExpandedWeights' autograd.Function")
    ctx.true_outputs = output

    def check_and_detach(output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Can only ")
        return output.detach()

    # NB: currently only works for differentiable, Tensor outputs
    if isinstance(output, tuple):
        return tuple(check_and_detach(o) for o in output)
    else:
        return check_and_detach(output)
