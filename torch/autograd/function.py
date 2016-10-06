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
        self._shared_pairs = None
        self.non_differentiable = None
        self.backward_hooks = OrderedDict()

    def __call__(self, *input):
        return self._do_forward(*input)

    def save_for_backward(self, *tensors):
        self.to_save = tensors

    def mark_dirty(self, *args):
        dirty_set = set(args)
        for var in self.input:
            if var.data in dirty_set:
                var._version[0] += 1

    def mark_shared_storage(self, *pairs):
        self._shared_pairs = pairs

    def mark_non_differentiable(self, *args):
        self.non_differentiable = set(args)

    @property
    def saved_tensors(self):
        for arg, expected_version in self.saved_variables:
            if arg._version[0] != expected_version:
                raise RuntimeError("one of the variables needed for gradient "
                    "computation has been modified by an inplace operation")
        return tuple(arg.data for arg, _ in self.saved_variables)

    def _do_forward(self, *input):
        for i in input:
            if not isinstance(i, Variable):
                raise RuntimeError("expected a Variable argument, but got " +
                    type(i).__name__)
        unpacked_input = tuple(arg.data for arg in input)
        is_volatile = any(arg.volatile for arg in input)
        # Save the input, so _save_for_backward can access it
        self.input = input
        if not is_volatile:
            self.needs_input_grad = tuple(arg._requires_grad for arg in input)
            self.requires_grad = any(self.needs_input_grad)
            self.previous_functions = [(arg.creator or arg, id(arg)) for arg in input]

        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)

        if is_volatile:
            output = tuple(Variable(tensor, volatile=True)
                           for tensor in raw_output)
        else:
            t2var = {var.data: var for var in input}
            output = tuple(Variable(tensor, self, requires_grad=self.requires_grad)
                            if tensor not in t2var
                            else t2var[tensor]
                            for tensor in raw_output)
            for output_var in output:
                output_var.creator = self
                t2var[output_var.data] = output_var
            self.output_ids = {id(var): i for i, var in enumerate(output)}
            if self.to_save:
                self.saved_variables = tuple((t2var[t], t2var[t]._version[0])
                        for t in self.to_save)
                del self.to_save
            if self._shared_pairs:
                for t1, t2 in self._shared_pairs:
                    v1 = t2var[t1]
                    v2 = t2var[t2]
                    v2._version = v1._version
                del self._shared_pairs
            if self.non_differentiable is not None:
                for var in output:
                    if var.data in self.non_differentiable:
                        var._requires_grad = False

        del self.input  # Remove unnecessary references to input
        del self.non_differentiable  # and output
        if len(output) == 1:
            output = output[0]
        return output

    def _do_backward(self, grad_output, retain_variables):
        if not hasattr(self, 'saved_variables'):
            raise RuntimeError("Trying to backward through the graph second "
                    "time, but the buffers have already been freed. Please "
                    "specify retain_variables=True when calling backward for "
                    "the first time.")
        grad_input = self.backward(*grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)
        assert len(grad_input) == len(self.previous_functions), \
            self.__class__.__name__ + ' returned an invalid number of gradient tensors'

        self._call_hooks(grad_input, grad_output)
        if not retain_variables:
            del self.saved_variables
        return grad_input

    def _call_hooks(self, grad_input, grad_output):
        for hook in self.backward_hooks.values():
            hook(grad_input, grad_output)

    def register_hook(self, name, hook):
        assert name not in self.backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        self.backward_hooks[name] = hook

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
