import torch._thnn.thnn
from torch._thnn.utils import parse_header, THNN_H_PATH
from torch.autograd.function import Function, InplaceFunction
from torch._thnn import type2backend

def _make_function_class_criterion(class_name, update_output, update_grad_input, acc_grad_parameters):
    weight_arg_idx = -1
    for i, arg in enumerate(update_output.arguments):
        if arg.name.startswith('weight'):
            weight_arg_idx = i
            break

    buffers_idx = []
    additional_arg_idx = 0
    for arg in update_output.arguments[4:]:
        # TODO: index tensors, etc.
        if not arg.name.startswith('weight') and arg.type == 'THTensor*':
            buffers_idx.append(additional_arg_idx)
        additional_arg_idx += 1

    def __init__(self, target, *args, **kwargs):
        super(type(self), self).__init__()
        self.target = target
        self.weight = kwargs.get('weight')
        self.additional_args = list(args)

    def forward(self, input):
        self.backend = type2backend[type(input)]
        self.save_for_backward(input)
        if weight_arg_idx >= 0:
            insert_idx = weight_arg_idx - 4 # state, input, target, output
            self.additional_args.insert(insert_idx, self.weight)
        for idx in buffers_idx:
            self.additional_args.insert(idx, input.new(1))
        output = input.new(1)
        getattr(self.backend, update_output.name)(self.backend.library_state, input, self.target,
            output, *self.additional_args)
        return output

    def backward(self, grad_output):
        input = self.saved_tensors[0]
        grad_input = grad_output.new().resizeAs_(input).zero_()
        getattr(self.backend, update_grad_input.name)(self.backend.library_state, input, self.target,
            grad_input, *self.additional_args)
        return grad_input

    return type(class_name, (Function,), dict(__init__=__init__, forward=forward, backward=backward))


def _make_function_class(class_name, update_output, update_grad_input, acc_grad_parameters):
    def has_argument(fn, name):
        for arg in fn.arguments:
            if arg.name == name:
                return True
        return False
    save_output = has_argument(update_grad_input, 'output')

    buffers_idx = []
    additional_arg_idx = 0
    for arg in update_output.arguments[3:]:
        if arg.name in {'weight', 'bias'}:
            continue
        # TODO: index tensors, etc.
        if arg.type == 'THTensor*':
            buffers_idx.append(additional_arg_idx)
        additional_arg_idx += 1

    # This and __init__ assume that only the last argument can be
    # an inplace flag
    is_inplace = update_output.arguments[-1].name == 'inplace'

    def __init__(self, *args):
        if is_inplace:
            super(type(self), self).__init__(args[-1])
        else:
            super(type(self), self).__init__()
        self.additional_args = list(args)

    def forward(self, input, *params):
        self.backend = type2backend[type(input)]

        # Allocate temporary buffers and insert them into additional_args
        for idx in buffers_idx:
            self.additional_args = self.additional_args[:idx] + [input.new()] + self.additional_args[idx:]
        self.additional_args = tuple(self.additional_args)
        # <sigh> python2.7 can't expand multiple tuples as arguments
        args = params + self.additional_args

        output = input.new()
        # If the module is working in-place it's output will be set to the
        # same storage as input, but it's variable won't be dirty.
        if is_inplace and self.inplace:
            self.mark_dirty(input)
            self.save_for_backward(output, *params)
        else:
            self.save_for_backward(input, *params)

        getattr(self.backend, update_output.name)(self.backend.library_state, input, output, *args)
        if save_output:
            self.output = output
        return output

    def backward(self, grad_output):
        t = self.saved_tensors
        input, params = t[0], t[1:]
        grad_params = tuple(None for p in params)
        grad_input_tuple = (None,)
        additional_args = self.additional_args if not save_output else (self.output,) + self.additional_args

        if self.needs_input_grad[0]:
            grad_input = input.new().resizeAs_(input).zero_()
            params_without_bias = params if len(params) < 2 else params[:1]
            update_grad_input_fn = getattr(self.backend, update_grad_input.name)
            gi_args = params_without_bias + additional_args
            update_grad_input_fn(self.backend.library_state, input, grad_output, grad_input, *gi_args)
            grad_input_tuple = (grad_input,)

        if acc_grad_parameters and any(self.needs_input_grad[1:]):
            grad_params = tuple(p.new().resizeAs_(p).zero_() for p in params)
            acc_grad_parameters_fn = getattr(self.backend, acc_grad_parameters.name)
            param_args = grad_params + additional_args + (1,)
            acc_grad_parameters_fn(self.backend.library_state, input, grad_output, *param_args)

        return grad_input_tuple + grad_params

    base_class = Function if not is_inplace else InplaceFunction
    return type(class_name, (base_class,), dict(__init__=__init__, forward=forward, backward=backward))


_function_list = parse_header(THNN_H_PATH)
_function_by_name = {fn.name: fn for fn in _function_list}
def _generate_function_classes(scope_dict):
    classes_to_generate = {fn.name.partition('_')[0] for fn in _function_list}
    exceptions = {
        'SparseLinear',
        'BatchNormalization',
        'LookupTable',
        'unfolded',
    }
    classes_to_generate -= exceptions
    for fn in classes_to_generate:
        update_output = _function_by_name[fn + '_updateOutput']
        update_grad_input = _function_by_name[fn + '_updateGradInput']
        acc_grad_parameters = _function_by_name.get(fn + '_accGradParameters')
        class_name = fn + 'Function'
        # This has to call a function to retain correct references to functions
        if 'Criterion' in fn:
            cls = _make_function_class_criterion(class_name, update_output,
                    update_grad_input, acc_grad_parameters)
        else:
            cls = _make_function_class(class_name, update_output,
                    update_grad_input, acc_grad_parameters)
        scope_dict[class_name] = cls
        _generated_functions.append(cls)


class BatchNormalizationFunction(Function):
    def __init__(self, *args):
        super(BatchNormalizationFunction, self).__init__()
        self.additional_args = args

    def forward(self, input, *params):
        self.backend = type2backend[type(input)]
        self.save_for_backward(input, *params)
        self.num_features = input.size(1)
        # Add save_input and save_std
        self.additional_args = self.additional_args[:2] + \
            (input.new(self.num_features), input.new(self.num_features)) + \
            self.additional_args[2:]
        num_params = len(params)
        if num_params < 2:
            params = params + tuple(None for i in range(2 - num_params))
        additional_args = params + self.additional_args
        output = input.new().resizeAs_(input)
        self.backend.BatchNormalization_updateOutput(self.backend.library_state,
                input, output, *additional_args)
        return output

    def backward(self, grad_output):
        tensors = self.saved_tensors
        input, params = tensors[0], tensors[1:]
        grad_input = (input.new().resizeAs_(input).zero_()
                if self.needs_input_grad[0] else None,)
        grad_param = tuple(p.new().resizeAs_(p).zero_() if self.needs_input_grad[i+1]
                else None for i, p in enumerate(params))
        result_grad = grad_input + grad_param

        num_params = len(params)
        if num_params < 2:
            grad_param = grad_param + tuple(None for i in range(2 - num_params))

        weight_tuple = (params[0],) if len(params) > 0 else (None,)
        # backward takes scale instead of momentum
        additional_args = self.additional_args[:-2] + (1,) + self.additional_args[-1:]
        args = grad_input + grad_param + weight_tuple + additional_args
        self.backend.BatchNormalization_backward(self.backend.library_state,
                input, grad_output, *args)
        return result_grad


_generated_functions = [BatchNormalizationFunction]
_generate_function_classes(locals())
