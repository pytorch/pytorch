from itertools import repeat

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
        return grad_output * mask.type_as(grad_output)


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
        return grad_output * (1 - i.mul(i)).sqrt_().reciprocal_()


class Acos(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.acos()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output.mul((1 - i.mul(i)).sqrt_().reciprocal_()).neg_()


class Atan(Function):

    def forward(self, i):
        self.save_for_backward(i)
        return i.atan()

    def backward(self, grad_output):
        i, = self.saved_tensors
        return grad_output * i.mul(i).add_(1).reciprocal_()


class Reciprocal(Function):

    def forward(self, i):
        result = i.reciprocal()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result, = self.saved_tensors
        return grad_output * result.mul(result).neg_()


class Cmax(Function):

    def forward(self, a, b):
        self._max_buffer = a.gt(b).type_as(a)
        return a.max(b)

    def backward(self, grad_output):
        return (
            grad_output * self._max_buffer,
            grad_output * self._max_buffer.eq(0).type_as(grad_output)
        )


class CmaxConstant(Function):

    def __init__(self, constant):
        super(CmaxConstant, self).__init__()
        self.constant = constant

    def forward(self, i):
        self._max_buffer = i.gt(self.constant).type_as(i)
        return i.clamp(min=self.constant)

    def backward(self, grad_output):
        return grad_output * self._max_buffer


class Cmin(Function):

    def forward(self, a, b):
        self._min_buffer = a.lt(b).type_as(a)
        return a.min(b)

    def backward(self, grad_output):
        return (
            grad_output * self._min_buffer,
            grad_output * self._min_buffer.eq(0).type_as(grad_output)
        )


class CminConstant(Function):

    def __init__(self, constant):
        super(CminConstant, self).__init__()
        self.constant = constant

    def forward(self, i):
        self._min_buffer = i.lt(self.constant).type_as(i)
        return i.clamp(max=self.constant)

    def backward(self, grad_output):
        return grad_output * self._min_buffer


class _ConstantGrad(Function):
    grad_value = 0

    def __init__(self, *args):
        super(_ConstantGrad, self).__init__()
        self.args = args

    def forward(self, i):
        return getattr(i, type(self).__name__.lower())(*self.args)

    def backward(self, grad_output):
        grad_input = grad_output.new(*repeat(1, grad_output.dim()))
        grad_input = grad_input.fill_(self.grad_value).expand_as(grad_output)
        return grad_input.mul(grad_output)


class Floor(_ConstantGrad):
    pass


class Ceil(_ConstantGrad):
    pass


class Round(_ConstantGrad):
    pass


class Sign(_ConstantGrad):
    pass


class Trunc(_ConstantGrad):
    pass


class Frac(_ConstantGrad):
    grad_value = 1


class Fmod(_ConstantGrad):
    grad_value = 1


class Remainder(_ConstantGrad):
    grad_value = 1


class Lerp(Function):

    def __init__(self, weight):
        super(Lerp, self).__init__()
        self.weight = float(weight)

    def forward(self, a, b):
        return a.lerp(b, self.weight)

    def backward(self, grad_output):
        return grad_output.mul(1 - self.weight), grad_output.mul(self.weight)


class Rsqrt(InplaceFunction):

    def forward(self, input):
        if self.inplace:
            self.mark_dirty(input)
            result = input.rsqrt_()
        else:
            result = input.rsqrt()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result, = self.saved_tensors
        return result.pow(3).div_(-2).mul_(grad_output)


class Addcmul(InplaceFunction):

    def __init__(self, scale=1, inplace=False):
        super(Addcmul, self).__init__(inplace)
        self.scale = scale

    def forward(self, add_tensor, mul_tensor1, mul_tensor2):
        self.save_for_backward(mul_tensor1, mul_tensor2)
        if self.inplace:
            return add_tensor.addcmul_(self.scale, mul_tensor1, mul_tensor2)
        else:
            return add_tensor.addcmul(self.scale, mul_tensor1, mul_tensor2)

    def backward(self, grad_output):
        grad_add = grad_mul1 = grad_mul2 = None
        mul_tensor1, mul_tensor2 = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_add = grad_output

        if self.needs_input_grad[1]:
            grad_mul1 = grad_output.mul(mul_tensor2).mul(self.scale)

        if self.needs_input_grad[2]:
            grad_mul2 = grad_output.mul(mul_tensor1).mul(self.scale)

        return grad_add, grad_mul1, grad_mul2


class Addcdiv(InplaceFunction):

    def __init__(self, scale=1, inplace=False):
        super(Addcdiv, self).__init__(inplace)
        self.scale = scale

    def forward(self, add_tensor, div_tensor1, div_tensor2):
        self.save_for_backward(div_tensor1, div_tensor2)
        if self.inplace:
            return add_tensor.addcdiv_(self.scale, div_tensor1, div_tensor2)
        else:
            return add_tensor.addcdiv(self.scale, div_tensor1, div_tensor2)

    def backward(self, grad_output):
        grad_add = grad_div1 = grad_div2 = None
        div_tensor1, div_tensor2 = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_add = grad_output

        if self.needs_input_grad[1]:
            grad_div1 = grad_output.div(div_tensor2).mul(self.scale)

        if self.needs_input_grad[2]:
            div_tensor2_sq = div_tensor2.mul(div_tensor2)
            grad_div2 = grad_output.mul(div_tensor1).div_(div_tensor2_sq)
            grad_div2.neg_().mul_(self.scale)

        return grad_add, grad_div1, grad_div2


# TODO: atan2 + inplace
