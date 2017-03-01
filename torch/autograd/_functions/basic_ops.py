import torch
from ..function import Function, InplaceFunction
import math


def maybe_view(tensor, size):
    if tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


class Add(InplaceFunction):

    def forward(self, a, b):
        self.b_size = b.size()
        if self.inplace:
            self.mark_dirty(a)
            return a.add_(b)
        else:
            return a.add(b)

    def backward(self, grad_output):
        return grad_output, maybe_view(grad_output, self.b_size)


class Sub(InplaceFunction):

    def forward(self, a, b):
        self.b_size = b.size()
        if self.inplace:
            self.mark_dirty(a)
            return a.sub_(b)
        else:
            return a.sub(b)

    def backward(self, grad_output):
        return grad_output, maybe_view(grad_output.neg(), self.b_size)


class Mul(Function):

    def forward(self, a, b):
        self.b_size = b.size()
        self.save_for_backward(a, b)
        return a.mul(b)

    def backward(self, grad_output):
        a, b = self.saved_tensors
        return grad_output.mul(b), maybe_view(grad_output.mul(a), self.b_size)


class Div(Function):

    def forward(self, a, b):
        self.b_size = b.size()
        self.save_for_backward(a, b)
        return a.div(b)

    def backward(self, grad_output):
        a, b = self.saved_tensors
        return grad_output.div(b), maybe_view(grad_output.neg().mul(a).div_(b).div_(b), self.b_size)


class Pow(Function):

    def forward(self, a, b):
        self.b_size = b.size()
        self.save_for_backward(a, b)
        return a.pow(b)

    def backward(self, grad_output):
        a, b = self.saved_tensors
        return grad_output.mul(b).mul_(a.pow(b - 1)), maybe_view(grad_output.mul(a.pow(b)).mul_(a.log()), self.b_size)


class AddConstant(InplaceFunction):

    def __init__(self, constant, inplace=False):
        super(AddConstant, self).__init__(inplace)
        self.constant = constant

    def forward(self, a):
        if self.inplace:
            self.mark_dirty(a)
            return a.add_(self.constant)
        else:
            return a.add(self.constant)

    def backward(self, grad_output):
        return grad_output


class SubConstant(InplaceFunction):

    def __init__(self, constant, sub_tensor=False, inplace=False):
        super(SubConstant, self).__init__(inplace)
        self.constant = constant
        self.sub_tensor = sub_tensor

    def forward(self, a):
        if self.sub_tensor:
            if a.is_signed() and self.inplace:
                self.mark_dirty(a)
                return a.neg_().add_(self.constant)
            else:
                assert not self.inplace, "can't perform (constant - tensor) " \
                    "subtraction in-place on an unsigned type"
                return a.new().resize_as_(a).fill_(self.constant).sub_(a)
        else:
            if self.inplace:
                self.mark_dirty(a)
                return a.sub_(self.constant)
            else:
                return a.sub(self.constant)

    def backward(self, grad_output):
        if self.sub_tensor:
            return grad_output.neg()
        else:
            return grad_output


class MulConstant(InplaceFunction):

    def __init__(self, constant, inplace=False):
        super(MulConstant, self).__init__(inplace)
        self.constant = constant

    def forward(self, a):
        if self.inplace:
            self.mark_dirty(a)
            return a.mul_(self.constant)
        else:
            return a.mul(self.constant)

    def backward(self, grad_output):
        return grad_output.mul(self.constant)


class DivConstant(InplaceFunction):

    def __init__(self, constant, div_by_tensor=False, inplace=False):
        super(DivConstant, self).__init__(inplace)
        self.constant = constant
        self.div_by_tensor = div_by_tensor
        if self.inplace and self.div_by_tensor:
            # TODO: actually, as long as the type is floating point, we can
            raise RuntimeError("can't perform (constant / tensor) division in-place")

    def forward(self, a):
        if self.div_by_tensor:
            self.save_for_backward(a)
            return a.new().resize_as_(a).fill_(self.constant).div_(a)
        else:
            if self.inplace:
                return a.div_(self.constant)
            else:
                return a.div(self.constant)

    def backward(self, grad_output):
        if self.div_by_tensor:
            a = self.saved_tensors[0]
            return grad_output.neg().mul_(self.constant).div_(a).div_(a)
        else:
            return grad_output.div(self.constant)


class PowConstant(Function):

    def __init__(self, constant, tensor_power=False):
        super(PowConstant, self).__init__()
        self.constant = constant
        self.tensor_power = tensor_power

    def forward(self, a):
        if self.tensor_power:
            self.fw_result = torch.pow(self.constant, a)
            return self.fw_result
        else:
            self.save_for_backward(a)
            return a.pow(self.constant)

    def backward(self, grad_output):
        if self.tensor_power:
            return grad_output.mul(self.fw_result).mul_(math.log(self.constant))
        else:
            a = self.saved_tensors[0]
            return grad_output.mul(self.constant).mul_(a.pow(self.constant - 1))


class Negate(InplaceFunction):

    def forward(self, i):
        if self.inplace:
            return i.neg_()
        else:
            return i.neg()

    def backward(self, grad_output):
        return grad_output.neg()
