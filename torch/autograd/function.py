from collections import Counter
from .variable import Variable

class Function(object):

    def __init__(self):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None

    def __call__(self, *input):
        return self._do_forward(*input)

    def _do_forward(self, *input):
        unpacked_input = tuple(arg.data for arg in input)
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in input)
        self.requires_grad = any(self.needs_input_grad)
        output = tuple(Variable(tensor, self) for tensor in raw_output)

        self.previous_functions = [(arg.creator, id(arg)) for arg in input]
        self.output_ids = {id(var): i for i, var in enumerate(output)}
        return output

    def _do_backward(self, grad_output, buffers=None):
        grad_input = self.backward(grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)

        assert len(grad_input) == len(self.previous_functions), \
            self.__class__.__name__ + ' returned an invalid number of gradient tensors'
        return grad_input

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError
