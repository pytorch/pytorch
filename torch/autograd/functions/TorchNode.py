import torch
from ..base.Variable import Variable
from ..base.Node import Node

class TorchNode(Node):

    def __init__(self, input_variable, *args):
        self.input_variable = input_variable
        # TODO: check if Tensors aren't mixed with Variables
        self.args = args
        self.unpacked_args = tuple(self._unpack_arg(arg) for arg in args)

    def _forward(self):
        result = getattr(self.input_variable.data, self.fn_name)(*self.unpacked_args)
        # If a function returns a number, we have to wrap it again
        if not torch.isTensor(result):
            result = self.input_variable.new((result,))
        return Variable(result, self)

    def _backward(self, grad_output, *args, **kwargs):
        grad_input = self.backward(grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)
        variables = (self.input_variable,) + tuple(filter(lambda x: isinstance(x, Variable), self.args))
        assert isinstance(grad_input, tuple)
        assert len(variables) == len(grad_input)
        for var, d_var in zip(variables, grad_input):
            var.backward(d_var, *args, **kwargs)

    def _unpack_arg(self, arg):
        if isinstance(arg, Variable):
            return arg.data
        return arg
