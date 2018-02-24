import math
import types
import random

import torch

##########################
# base initializer
##########################


class Initializer(object):
    """
    Base class for all initializations.
    """
    def __call__(self, tensor):
        raise NotImplementedError

##########################
# initializers
##########################


class Ones(Initializer):
    def __call__(self, tensor):
        with torch.no_grad():
            return tensor.fill_(1)


class Zeros(Initializer):
    def __call__(self, tensor):
        with torch.no_grad():
            return tensor.fill_(0)


class Constant(Initializer):
    def __init__(self, val):
        self.val = val

    def __call__(self, tensor):
        with torch.no_grad():
            return tensor.fill_(self.val)


class Eye(Initializer):
    def __call__(self, tensor):
        if tensor.ndimension() != 2:
            raise ValueError("Only tensors with 2 dimensions are supported")

        with torch.no_grad():
            torch.eye(*tensor.shape, out=tensor)
        return tensor


class Orthogonal(Initializer):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, tensor):
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")

        rows = tensor.size(0)
        cols = tensor[0].numel()
        flattened = tensor.new(rows, cols).normal_(0, 1)

        if rows < cols:
            flattened.t_()

        q, r = torch.qr(flattened)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()

        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(self.gain)
        return tensor


class Uniform(Initializer):
    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b

    def __call__(self, tensor):
        with torch.no_grad():
            return tensor.uniform_(self.a, self.b)


class Normal(Initializer):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        with torch.no_grad():
            return tensor.normal_(self.mean, self.std)


class Dirac(Initializer):
    def __call__(self, tensor):
        dimensions = tensor.ndimension()
        if dimensions not in [3, 4, 5]:
            raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

        sizes = tensor.size()
        min_dim = min(sizes[0], sizes[1])
        with torch.no_grad():
            tensor.zero_()

            for d in range(min_dim):
                if dimensions == 3:  # Temporal convolution
                    tensor[d, d, tensor.size(2) // 2] = 1
                elif dimensions == 4:  # Spatial convolution
                    tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2] = 1
                else:  # Volumetric convolution
                    tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2, tensor.size(4) // 2] = 1
        return tensor


class Sparse(Initializer):
    def __init__(self, sparsity, std=0.01):
        self.sparsity = sparsity
        self.std = std

    def __call__(self, tensor):
        if tensor.ndimension() != 2:
            raise ValueError("Only tensors with 2 dimensions are supported")

        rows, cols = tensor.shape
        num_zeros = int(math.ceil(rows * self.sparsity))

        with torch.no_grad():
            tensor.normal_(0, self.std)
            for col_idx in range(cols):
                row_indices = list(range(rows))
                random.shuffle(row_indices)
                zero_indices = row_indices[:num_zeros]
                for row_idx in zero_indices:
                    tensor[row_idx, col_idx] = 0
        return tensor


class VarianceScaling(Initializer):
    def __init__(self, scale, gain, mode, distribution):
        if scale < 0.:
            raise ValueError('`scale` must be a positive float.')
        mode = mode.lower()
        valid_modes = ["fan_in", "fan_avg", "fan_out"]
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
        distribution = distribution.lower()
        valid_distributions = ["uniform", "normal"]
        if distribution not in valid_distributions:
            raise ValueError("Distribution {} not supported, please use one of {}".format(
                distribution, valid_distributions)
            )

        self.scale = scale
        self.gain = gain
        self.mode = mode
        self.distribution = distribution

    def __call__(self, tensor):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        scale = self.scale
        scale *= self.gain
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)

        if self.distribution == "uniform":
            bound = math.sqrt(3.0 * scale)
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)
        else:
            std = math.sqrt(scale)
            with torch.no_grad():
                return tensor.normal_(0, std)


class XavierUniform(Initializer):
    def __init__(self, gain=1.):
        self.vs = VarianceScaling(
            scale=1., gain=gain,
            mode="fan_avg",
            distribution="uniform"
        )

    def __call__(self, tensor):
        return self.vs(tensor)


class XavierNormal(Initializer):
    def __init__(self, gain=1.):
        self.vs = VarianceScaling(
            scale=1., gain=gain,
            mode="fan_avg",
            distribution="normal"
        )

    def __call__(self, tensor):
        return self.vs(tensor)


class KaimingUniform(Initializer):
    def __init__(self, a=0, mode="fan_in"):
        gain = calculate_gain('leaky_relu', a)
        self.vs = VarianceScaling(
            scale=2., gain=gain,
            mode=mode,
            distribution="uniform"
        )

    def __call__(self, tensor):
        return self.vs(tensor)


class KaimingNormal(Initializer):
    def __init__(self, a=0, mode="fan_in"):
        gain = calculate_gain('leaky_relu', a)
        self.vs = VarianceScaling(
            scale=2., gain=gain,
            mode=mode,
            distribution="normal"
        )

    def __call__(self, tensor):
        return self.vs(tensor)

##########################
# aliases
##########################


ones = Ones
zeros = Zeros
constant = Constant
eye = Eye
orthogonal = Orthogonal
uniform = Uniform
normal = Normal
dirac = Dirac
sparse = Sparse
xavier_uniform = XavierUniform
xavier_normal = XavierNormal
kaiming_uniform = KaimingUniform
kaiming_normal = KaimingNormal

##########################
# utility functions
##########################


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    selu         :math:`1`
    tanh         :math:`25 / 9`
    relu         :math:`2`
    leaky_relu   :math:`2 / (1 + negative\_slope^2)`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn.functional` name)
        param: optional parameter for the nonlinear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    non_linear_fns = ['sigmoid', 'selu']
    if nonlinearity in linear_fns or nonlinearity in non_linear_fns:
        return 1.
    elif nonlinearity == 'tanh':
        return 25 / 9
    elif nonlinearity == 'relu':
        return 2.0
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return 2.0 / (1 + negative_slope ** 2)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def get(identifier):
    if isinstance(identifier, str):
        func = globals()[identifier]
        return func if isinstance(func, types.FunctionType) else func()
    elif callable(identifier):
        return identifier
