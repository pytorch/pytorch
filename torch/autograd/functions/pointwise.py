from ..variable import Variable
from ..function import Function, InplaceFunction


class Exp(InplaceFunction):

    def forward(self, i):
        if self.inplace:
            self.mark_dirty(i)
            result = i.exp_()
        else:
            result = i.exp()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        return self.saved_tensors[0] * grad_output


class Log(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.log()

    def backward(self, grad_output):
        return grad_output.div(self.saved_tensors[0])


class Log1p(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.log1p()

    def backward(self, grad_output):
        return grad_output.div(self.saved_tensors[0].add(1))


class Tanh(InplaceFunction):

    def forward(self, i):
        if self.inplace:
            self.mark_dirty(i)
            result = i.tanh_()
        else:
            result = i.tanh()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result, = self.saved_tensors
        return grad_output * (1 - result * result)


class Sigmoid(InplaceFunction):

    def forward(self, i):
        if self.inplace:
            self.mark_dirty(i)
            result = i.sigmoid_()
        else:
            result = i.sigmoid()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result, = self.saved_tensors
        return grad_output * ((1 - result) * result)


class Sinh(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.sinh()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * i.cosh()


class Cosh(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.cosh()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * i.sinh()


class Abs(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.abs()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * i.sign()


class Clamp(Function):

    def __init__(self, min_val, max_val):
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, i):
        self.save_for_backward(i)
        return i.clamp(self.min_val, self.max_val)

    def backward(self, grad_output):
        i, = self.saved_tensors
        mask = i.ge(self.min_val) * i.le(self.max_val)
        return grad_output * mask.typeAs(grad_output)


class Sqrt(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.sqrt()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output.mul(i.pow(-0.5)).div(2)


class Sin(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.sin()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * i.cos()


class Cos(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.cos()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output.mul(i.sin()).neg_()


class Tan(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.tan()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output.div(i.cos().pow(2))


class Asin(Function):
    def forward(self, i):
        self.save_for_backward(i)
        return i.asin()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * (1 - i.mul(i)).sqrt_().cinv_()


class Acos(Function):
    def forward(self, i):
        self.save_for_backward(i)
        return i.acos()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output.mul((1 - i.mul(i)).sqrt_().cinv_()).neg_()


class Atan(Function):
    def forward(self, i):
        self.save_for_backward(i)
        return i.atan()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * i.mul(i).add_(1).cinv_()


class Cinv(Function):

    def forward(self, i):
        result = i.cinv()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result, = self.saved_tensors
        return grad_output * result.mul(result).neg_()


class Cmax(Function):

    def forward(self, a, b):
        self._max_buffer = a.gt(b).typeAs(a)
        return a.cmax(b)

    def backward(self, grad_output):
        return (
            grad_output * self._max_buffer,
            grad_output * self._max_buffer.eq(0).typeAs(grad_output)
        )


class CmaxConstant(Function):

    def __init__(self, constant):
        super(CmaxConstant, self).__init__()
        self.constant = constant

    def forward(self, i):
        self._max_buffer = i.gt(self.constant).typeAs(i)
        return i.cmax(self.constant)

    def backward(self, grad_output):
        return grad_output * self._max_buffer


class Cmin(Function):

    def forward(self, a, b):
        self._min_buffer = a.lt(b).typeAs(a)
        return a.cmin(b)

    def backward(self, grad_output):
        return (
            grad_output * self._min_buffer,
            grad_output * self._min_buffer.eq(0).typeAs(grad_output)
        )


class CminConstant(Function):

    def __init__(self, constant):
        super(CminConstant, self).__init__()
        self.constant = constant

    def forward(self, i):
        self._min_buffer = i.lt(self.constant).typeAs(i)
        return i.cmin(self.constant)

    def backward(self, grad_output):
        return grad_output * self._min_buffer


class _ConstantGrad(Function):
    grad_value = 0

    def forward(self, i):
        return getattr(i, type(self).__name__.lower())()

    def backward(self, grad_output):
        return grad_output.new((self.grad_value,)).expandAs(grad_output)


class Floor(_ConstantGrad):
    pass


class Ceil(_ConstantGrad):
    pass


class Frac(_ConstantGrad):
    grad_value = 1


# TODO: addcdiv + inplace
# TODO: addcmul + inplace
# TODO: atan2 + inplace
# TODO: ceil + inplace
# TODO: lerp
# TODO: remainder
# TODO: fmod
# TODO: round
# TODO: rsqrt
# TODO: sign
# TODO: trunc
