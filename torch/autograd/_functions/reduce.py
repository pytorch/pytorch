from functools import reduce

from ..function import Function


class _DimReduceFunction(Function):

    def __init__(self, dim=None):
        super(_DimReduceFunction, self).__init__()
        self.dim = dim

    def forward(self, input):
        self.input_size = input.size()
        fn = getattr(input, self.fn_name)
        if self.dim is None:
            return input.new((fn(),))
        else:
            return fn(self.dim)


class Sum(_DimReduceFunction):
    fn_name = 'sum'

    def backward(self, grad_output):
        if self.dim is None:
            return grad_output.new(self.input_size).fill_(grad_output[0])
        else:
            repeats = [1 for _ in self.input_size]
            repeats[self.dim] = self.input_size[self.dim]
            return grad_output.repeat(*repeats),


class Prod(_DimReduceFunction):

    def forward(self, input):
        self.input_size = input.size()
        if self.dim is None:
            self.result = input.prod()
            self.save_for_backward(input)
            return input.new((self.result,))
        else:
            output = input.prod(self.dim)
            self.save_for_backward(input, output)
            return output

    def backward(self, grad_output):
        if self.dim is None:
            input, = self.saved_tensors
            grad_input = grad_output.new(self.input_size).fill_(self.result)
            return grad_input.div(input)
        else:
            input, output = self.saved_tensors
            repeats = [1 for _ in self.input_size]
            repeats[self.dim] = self.input_size[self.dim]
            return output.mul(grad_output).repeat(*repeats).div_(input)


class Mean(_DimReduceFunction):
    fn_name = 'mean'

    def backward(self, grad_output):
        if self.dim is None:
            grad_input_val = grad_output[0]
            grad_input_val /= reduce(lambda x, y: x * y, self.input_size, 1)
            return grad_output.new(*self.input_size).fill_(grad_input_val)
        else:
            repeats = [1 for _ in self.input_size]
            dim_size = self.input_size[self.dim]
            repeats[self.dim] = dim_size
            return grad_output.repeat(*repeats).div_(dim_size)


class _SelectionFunction(Function):
    has_all_reduce = True
    # additional_args is prepended before dim when calling the tensor
    # function. It's a no-op for subclasses other than kthvalue.
    # kthvalue not only requires us to pass a dim, but also preceed it with k.
    additional_args = tuple()

    def __init__(self, dim=None):
        super(_SelectionFunction, self).__init__()
        self.dim = dim

    def forward(self, input):
        fn = getattr(input, type(self).__name__.lower())
        self.input_size = input.size()
        if self.dim is None and self.has_all_reduce:
            value = fn(*self.additional_args)
            self.indices = tuple(input.eq(value).nonzero()[0])
            return input.new((value,))
        else:
            if self.dim is None:
                dim = input.dim() - 1
            else:
                dim = self.dim
            args = (dim,)
            if self.additional_args:
                args = self.additional_args + args
            output, indices = fn(*args)
            self.save_for_backward(indices)
            self.mark_non_differentiable(indices)
            return output, indices

    def backward(self, grad_output, grad_indices=None):
        grad_input = grad_output.new(*self.input_size).zero_()
        if self.dim is None and self.has_all_reduce:
            grad_input[self.indices] = grad_output[0]
        else:
            if self.dim is None:
                dim = input.dim() - 1
            else:
                dim = self.dim
            indices, = self.saved_tensors
            grad_input.scatter_(dim, indices, grad_output)
        return grad_input


class Max(_SelectionFunction):
    pass


class Min(_SelectionFunction):
    pass


class Mode(_SelectionFunction):
    has_all_reduce = False


class Median(_SelectionFunction):
    has_all_reduce = False


class Kthvalue(_SelectionFunction):
    has_all_reduce = False

    def __init__(self, k, dim=None):
        super(Kthvalue, self).__init__(dim)
        self.additional_args = (k,)


class Norm(Function):

    def __init__(self, norm_type=2, dim=None):
        super(Norm, self).__init__()
        self.norm_type = norm_type
        self.dim = dim

    def forward(self, input):
        if self.dim is None:
            self.norm = input.norm(self.norm_type)
            self.save_for_backward(input)
            return input.new((self.norm,))
        else:
            output = input.norm(self.norm_type, self.dim)
            self.save_for_backward(input, output)
            return output

    def backward(self, grad_output):
        if self.dim is None:
            input, = self.saved_tensors
            if self.norm_type == 2:
                return input.mul(grad_output[0] / self.norm)
            else:
                pow = input.abs().pow(self.norm_type - 2)
                scale = grad_output[0] / self.norm ** (self.norm_type - 1)
                return input.mul(pow).mul(scale)
        else:
            input, output = self.saved_tensors
            big_grad_output = grad_output.expand_as(input)
            if self.norm_type == 2:
                big_output = output.expand_as(input)
                return input.mul(big_grad_output).div(big_output)
            else:
                pow = input.abs().pow(self.norm_type - 2)
                big_output = output.pow(self.norm_type - 1).expand_as(input)
                return input.mul(pow).mul(big_grad_output).div(big_output)


# TODO: renorm
# TODO: std
# TODO: var
