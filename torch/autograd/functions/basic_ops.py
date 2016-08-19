import torch
from ..variable import Variable
from ..function import Function


class Add(Function):

    def forward(self, a, b):
        return a.add(b)

    def backward(self, grad_output):
        return grad_output, grad_output


class Sub(Function):

    def forward(self, a, b):
        return a.sub(b)

    def backward(self, grad_output):
        return grad_output, grad_output.neg()


class Mul(Function):

    def forward(self, a, b):
        self.input = (a, b)
        return a.mul(b)

    def backward(self, grad_output):
        return grad_output.mul(self.input[1]), grad_output.mul(self.input[0])


class Div(Function):

    def forward(self, a, b):
        self.input = (a, b)
        return a.div(b)

    def backward(self, grad_output):
        a, b = self.input
        return grad_output.div(b), grad_output.neg().mul(a).div_(b).div_(b)

class Pow(Function):

    def forward(self, a, b):
        self.input = (a, b)
        return a.pow(b)

    def backward(self, grad_output):
        a, b = self.input
        return grad_output.mul(b).mul_(a.pow(b-1)), grad_output.mul(a.pow(b)).mul_(a.log())

class AddConstant(Function):

    def __init__(self, constant):
        self.constant = constant

    def forward(self, a):
        return a.add(self.constant)

    def backward(self, grad_output):
        return grad_output


class SubConstant(Function):

    def __init__(self, constant, sub_tensor=False):
        self.constant = constant
        self.sub_tensor = sub_tensor

    def forward(self, a):
        if self.sub_tensor:
            return a.new().resizeAs_(a).fill_(self.constant).sub_(a)
        else:
            return a.sub(self.constant)

    def backward(self, grad_output):
        if self.sub_tensor:
            return grad_output.neg()
        else:
            return grad_output


class MulConstant(Function):

    def __init__(self, constant):
        self.constant = constant

    def forward(self, a):
        return a.mul(self.constant)

    def backward(self, grad_output):
        return grad_output.mul(self.constant)


class DivConstant(Function):

    def __init__(self, constant, div_by_tensor=False):
        self.constant = constant
        self.div_by_tensor = div_by_tensor

    def forward(self, a):
        if self.div_by_tensor:
            self.input = a
            return a.new().resizeAs_(a).fill_(self.constant).div_(a)
        else:
            return a.div(self.constant)

    def backward(self, grad_output):
        if self.div_by_tensor:
            a = self.input
            return grad_output.neg().mul_(self.constant).div_(a).div_(a)
        else:
            return grad_output.div(self.constant)


class PowConstant(Function):

    def __init__(self, constant, tensor_power=False):
        self.constant = constant
        self.tensor_power = tensor_power

    def forward(self, a):
        if self.tensor_power:
            self.fw_result = torch.pow(self.constant, a)
            return result
        else:
            self.input = a
            return a.pow(self.constant)

    def backward(self, grad_output):
        if self.tensor_power:
            return grad_output.mul(self.fw_result).mul_(math.log(self.constant))
        else:
            a = self.input
            return grad_output.mul(self.constant).mul_(a.pow(self.constant-1))

class Negate(Function):

    def forward(self, i):
        return i.neg()

    def backward(self, grad_output):
        return grad_output.neg()
