from itertools import repeat
from collections import defaultdict

import torch
from torch._thnn.utils import parse_header, THNN_H_PATH
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend

from . import _all_functions


def _make_function_class_criterion(class_name, update_output, update_grad_input, acc_grad_parameters):
    weight_arg_idx = -1
    for i, arg in enumerate(update_output.arguments):
        if arg.name.startswith('weight'):
            weight_arg_idx = i
            break

    buffers_idx = []
    additional_arg_idx = 0
    for arg in update_output.arguments[4:]:
        if not arg.name.startswith('weight') and arg.type == 'THTensor*':
            buffers_idx.append(additional_arg_idx)
        additional_arg_idx += 1

    def __init__(self, *args, **kwargs):
        Function.__init__(self)
        self.weight = kwargs.get('weight')
        self.additional_args = list(args)

    def forward(self, input, target):
        self._backend = type2backend[type(input)]
        self.save_for_backward(input, target)
        if weight_arg_idx >= 0:
            insert_idx = weight_arg_idx - 4  # state, input, target, output
            self.additional_args.insert(insert_idx, self.weight)
        for idx in buffers_idx:
            self.additional_args.insert(idx, input.new(1))
        output = input.new(1)
        getattr(self._backend, update_output.name)(self._backend.library_state, input, target,
                                                   output, *self.additional_args)
        return output

    def backward(self, grad_output):
        input, target = self.saved_tensors
        grad_input = grad_output.new().resize_as_(input).zero_()
        getattr(self._backend, update_grad_input.name)(self._backend.library_state, input, target,
                                                       grad_input, *self.additional_args)
        grad_output_expanded = grad_output.view(*repeat(1, grad_input.dim()))
        grad_input.mul_(grad_output_expanded.expand_as(grad_input))
        return grad_input, None

    return type(class_name, (Function,), dict(__init__=__init__, forward=forward, backward=backward))


def _find_buffers(args, ignored_args):
    additional_arg_idx = 0
    buffers = []
    for arg in args:
        if arg.name in ignored_args:
            continue
        if arg.type == 'THTensor*':
            buffers.append((additional_arg_idx, arg.name))
        additional_arg_idx += 1
    return buffers


def _make_function_class(class_name, update_output, update_grad_input, acc_grad_parameters):
    def has_argument(fn, name):
        for arg in fn.arguments:
            if arg.name == name:
                return True
        return False
    save_output = has_argument(update_grad_input, 'output')

    param_args = {'weight', 'bias'}
    ignored_args = {'weight', 'bias', 'gradWeight', 'gradBias', 'output'}
    expected_params = [arg for arg in update_output.arguments[3:]
                       if arg.name in param_args]
    buffers = {}
    buffers['update_output'] = _find_buffers(update_output.arguments[3:],
                                             ignored_args)
    buffers['update_grad_input'] = _find_buffers(
        update_grad_input.arguments[4:], ignored_args)
    if acc_grad_parameters is not None:
        buffers['acc_grad_parameters'] = _find_buffers(
            acc_grad_parameters.arguments[3:], ignored_args)

    # This and __init__ assume that only the last argument can be
    # an inplace flag
    is_inplace = update_output.arguments[-1].name == 'inplace'

    def __init__(self, *args):
        if is_inplace:
            InplaceFunction.__init__(self, args[-1])
        else:
            Function.__init__(self)
        self.additional_args = list(args)

    def _initialize_buffers(self, fn_name):
        additional_args = self.additional_args
        for idx, name in buffers[fn_name]:
            # TODO: some buffers are necessary only for update output and can be
            # freed right afterwards
            buffer = self.buffers[name]
            additional_args = additional_args[:idx] + [buffer] + additional_args[idx:]
        return tuple(additional_args)

    def forward(self, input, *params):
        self._backend = type2backend[type(input)]

        for param in params:
            if type(param) != type(input):
                raise RuntimeError("input type ({}) doesn't match the type of "
                                   "a parameter tensor ({})".format(torch.typename(input),
                                                                    torch.typename(param)))

        # Allocate temporary buffers and insert them into additional_args
        self.buffers = defaultdict(type(input))
        additional_args = self._initialize_buffers('update_output')

        # Fill in optional params with None
        args = params
        for i in range(len(params), len(expected_params)):
            param = expected_params[i]
            if param.is_optional:
                args += (None,)
            else:
                raise ValueError("missing required argument '%s'" % param.name)

        args += tuple(additional_args)

        # If the module is working in-place it's output will be set to the
        # same storage as input, but it's variable won't be dirty.
        if is_inplace and self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.new()

        if save_output:
            self.save_for_backward(input, output, *params)
        else:
            self.save_for_backward(input, *params)

        if not self.requires_grad:
            del self.buffers

        getattr(self._backend, update_output.name)(self._backend.library_state, input, output, *args)
        return output

    def backward(self, grad_output):
        t = self.saved_tensors
        if save_output:
            input, output, params = t[0], t[1], t[2:]
        else:
            input, params = t[0], t[1:]
        grad_params = tuple(None for p in params)
        grad_input_tuple = (None,)

        if self.needs_input_grad[0]:
            additional_args = self._initialize_buffers('update_grad_input')
            if save_output:
                additional_args = (output,) + additional_args

            if is_inplace and self.inplace:
                grad_output = grad_output.clone()
                grad_input = grad_output.new()
            else:
                grad_input = input.new().resize_as_(input).zero_()
            params_without_bias = params if len(params) < 2 else params[:1]
            update_grad_input_fn = getattr(self._backend, update_grad_input.name)
            gi_args = params_without_bias + additional_args
            update_grad_input_fn(self._backend.library_state, input, grad_output, grad_input, *gi_args)
            grad_input_tuple = (grad_input,)

        if acc_grad_parameters and any(self.needs_input_grad[1:]):
            additional_args = self._initialize_buffers('acc_grad_parameters')
            grad_params = tuple(p.new().resize_as_(p).zero_() for p in params)
            appended_grads = len(expected_params) - len(grad_params)
            grad_params += (None,) * appended_grads
            acc_grad_parameters_fn = getattr(self._backend, acc_grad_parameters.name)
            param_args = grad_params + additional_args + (1,)
            acc_grad_parameters_fn(self._backend.library_state, input, grad_output, *param_args)
            if appended_grads:
                grad_params = grad_params[:-appended_grads]

        return grad_input_tuple + grad_params

    base_class = Function if not is_inplace else InplaceFunction
    return type(class_name, (base_class,), dict(__init__=__init__, forward=forward, backward=backward,
                                                _initialize_buffers=_initialize_buffers))


def _generate_function_classes(scope_dict):
    global function_list, function_by_name
    function_list = parse_header(THNN_H_PATH)
    function_by_name = {fn.name: fn for fn in function_list}
    classes_to_generate = {fn.name.partition('_')[0] for fn in function_list}
    exceptions = {
        'Linear',
        'SpatialFullConvolution',
        'SpatialConvolutionMM',
        'SparseLinear',
        'TemporalConvolution',
        'SpatialAveragePooling',
        'SpatialMaxPooling',
        'SpatialDilatedMaxPooling',
        'SpatialMaxUnpooling',
        'VolumetricAveragePooling',
        'VolumetricMaxPooling',
        'VolumetricMaxUnpooling',
        'VolumetricConvolution',
        'VolumetricFullConvolution',
        'VolumetricConvolutionMM',
        'TemporalMaxPooling',
        'BatchNormalization',
        'LookupTable',
        'PReLU',
        'RReLU',
        'unfolded',
    }
    name_remap = {
        'TemporalConvolution': 'Conv1d',
        'SpatialDilatedConvolution': 'DilatedConv2d',
        'SpatialMaxUnpooling': 'MaxUnpool2d',
        'SpatialReflectionPadding': 'ReflectionPad2d',
        'SpatialReplicationPadding': 'ReplicationPad2d',
        'VolumetricReplicationPadding': 'ReplicationPad3d',
        'VolumetricMaxUnpooling': 'MaxUnpool3d',
        'SoftMax': 'Softmax',
        'LogSoftMax': 'LogSoftmax',
        'HardTanh': 'Hardtanh',
        'HardShrink': 'Hardshrink',
        'SoftPlus': 'Softplus',
        'SoftShrink': 'Softshrink',
        'MSECriterion': 'MSELoss',
        'AbsCriterion': 'L1Loss',
        'BCECriterion': '_BCELoss',  # TODO: move the glue code into THNN
        'ClassNLLCriterion': 'NLLLoss',
        'DistKLDivCriterion': 'KLDivLoss',
        'SpatialClassNLLCriterion': 'NLLLoss2d',
        'MultiLabelMarginCriterion': 'MultiLabelMarginLoss',
        'MultiMarginCriterion': 'MultiMarginLoss',
        'SmoothL1Criterion': 'SmoothL1Loss',
        'SoftMarginCriterion': 'SoftMarginLoss',
    }
    classes_to_generate -= exceptions
    for fn in classes_to_generate:
        update_output = function_by_name[fn + '_updateOutput']
        update_grad_input = function_by_name[fn + '_updateGradInput']
        acc_grad_parameters = function_by_name.get(fn + '_accGradParameters')
        class_name = name_remap.get(fn, fn)
        # This has to call a function to retain correct references to functions
        if 'Criterion' in fn:
            cls = _make_function_class_criterion(class_name, update_output,
                                                 update_grad_input, acc_grad_parameters)
        else:
            cls = _make_function_class(class_name, update_output,
                                       update_grad_input, acc_grad_parameters)
        scope_dict[class_name] = cls
        if not class_name.startswith('_'):
            _all_functions.append(cls)


_generate_function_classes(locals())
