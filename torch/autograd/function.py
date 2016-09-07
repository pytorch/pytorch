from collections import OrderedDict
from itertools import chain
from .variable import Variable

class Function(object):

    def __init__(self):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.saved_variables = None
        self.to_save = None
        self.backward_hooks = OrderedDict()

    def __call__(self, *input):
        return self._do_forward(*input)

    def save_for_backward(self, *tensors):
        self.to_save = tensors

    def mark_dirty(self, *args):
        dirty_set = set(args)
        for var in self.input:
            if var.data in dirty_set:
                var.mark_dirty()

    @property
    def saved_tensors(self):
        return tuple(arg.data for arg in self.saved_variables)

    def _do_forward(self, *input):
        unpacked_input = tuple(arg.data for arg in input)
        is_volatile = any(arg.volatile for arg in input)
        # Save the input, so _save_for_backward can access it
        self.input = input
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)

        if is_volatile:
            output = tuple(Variable(tensor, volatile=True) for tensor in raw_output)
        else:
            self.needs_input_grad = tuple(arg.creator.requires_grad for arg in input)
            self.requires_grad = any(self.needs_input_grad)
            self.previous_functions = [(arg.creator, id(arg)) for arg in input]
            output = tuple(Variable(tensor, self) for tensor in raw_output)
            self.output_ids = {id(var): i for i, var in enumerate(output)}
            if self.to_save:
                # output has to be chained after input, so if the same tensor
                # appears both in the input and output (happens for in-place
                # function), we save the clean output variable.
                #
                # Some variables might have been changed in-place, so accessing
                # their .data will throw. If they also occur in the output
                # these references will be overwritten by clean variables,
                # if now, they'll raise an error on backward.
                t2var = {var._data: var for var in chain(input, output)}
                self.saved_variables = tuple(t2var[t] for t in self.to_save)
                del self.to_save

        del self.input  # Remove unnecessary references to input
        if len(output) == 1:
            output = output[0]
        return output

    def _do_backward(self, *grad_output):
        grad_input = self.backward(*grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)
        assert len(grad_input) == len(self.previous_functions), \
            self.__class__.__name__ + ' returned an invalid number of gradient tensors'

        self._call_hooks(grad_input, grad_output)
        return grad_input

    def _call_hooks(self, grad_input, grad_output):
        for hook, idx in self.backward_hooks.values():
            if idx is None:
                hook(grad_input, grad_output)
            else:
                hook(grad_output[idx])

    def register_hook(self, name, hook, variable=None):
        assert name not in self.backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        variable_idx = self.output_ids[id(variable)] if variable else None
        self.backward_hooks[name] = (hook, variable_idx)

    def remove_hook(self, name):
        assert name in self.backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self.backward_hooks[name]

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError

class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace
