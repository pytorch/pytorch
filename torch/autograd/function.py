from collections import OrderedDict
from .variable import Variable

class Function(object):

    def __init__(self):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()

    def __call__(self, *input):
        return self._do_forward(*input)

    def _do_forward(self, *input):
        unpacked_input = tuple(arg.data for arg in input)
        is_volatile = any(arg.volatile for arg in input)
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

        return output

    def _do_backward(self, *grad_output):
        grad_input = self.backward(*grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)
        assert len(grad_input) == len(self.previous_functions), \
            self.__class__.__name__ + ' returned an invalid number of gradient tensors'

        for hook, idx in self.backward_hooks.values():
            gi = grad_input if idx is None else grad_input[idx]
            hook(grad_input, grad_output)

        return grad_input

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
