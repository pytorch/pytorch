from ..TorchNode import TorchNode
from ...base.Variable import Variable

class Sum(TorchNode):
    fn_name = 'sum'

    def backward(self, grad_output):
        i = self.input_variable.data
        if len(self.args) == 0:
            return i.new(i.size()).fill_(grad_output[0]),
        elif len(self.args) == 1:
            dim = self.args[0]
            repeats = [1 for i in range(i.dim())]
            repeats[dim] = i.size(dim)
            return grad_output.repeatTensor(*repeats),

class Mean(TorchNode):
    fn_name = 'mean'

    def backward(self, grad_output):
        i = inputs[0]
        if len(self.args) == 0:
            return i.new(i.size()).fill_(float(grad_output[0])/i.numel()),
        elif len(self.args) == 1:
            dim = self.args[0]
            repeats = [1 for i in range(i.dim())]
            repeats[dim] = i.size(dim)
            return grad_output.repeatTensor(*repeats).div_(i.size(dim)),

