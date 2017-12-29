import math
import random

import torch
from torch.autograd import Variable


def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn.functional` name)
        param: optional parameter for the nonlinear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def uniform(tensor, a=0, b=1):
    """Fills the input Tensor or Variable with values drawn from the uniform
    distribution :math:`U(a, b)`.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor

    return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from the normal
    distribution :math:`N(mean, std)`.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor

    return tensor.normal_(mean, std)


def constant(tensor, val):
    """Fills the input Tensor or Variable with the value `val`.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.constant(w, 0.3)
    """
    if isinstance(tensor, Variable):
        constant(tensor.data, val)
        return tensor

    return tensor.fill_(val)


def eye(tensor):
    """Fills the 2-dimensional input Tensor or Variable with the identity
    matrix. Preserves the identity of the inputs in Linear layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional torch.Tensor or autograd.Variable

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.eye(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    if isinstance(tensor, Variable):
        eye(tensor.data)
        return tensor

    return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))


def dirac(tensor):
    """Fills the {3, 4, 5}-dimensional input Tensor or Variable with the Dirac
    delta function. Preserves the identity of the inputs in Convolutional
    layers, where as many input channels are preserved as possible.

    Args:
        tensor: a {3, 4, 5}-dimensional torch.Tensor or autograd.Variable

    Examples:
        >>> w = torch.Tensor(3, 16, 5, 5)
        >>> nn.init.dirac(w)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    if isinstance(tensor, Variable):
        dirac(tensor.data)
        return tensor

    sizes = tensor.size()
    min_dim = min(sizes[0], sizes[1])
    tensor.zero_()

    for d in range(min_dim):
        if dimensions == 3:  # Temporal convolution
            tensor[d, d, tensor.size(2) // 2] = 1
        elif dimensions == 4:  # Spatial convolution
            tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2] = 1
        else:  # Volumetric convolution
            tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2, tensor.size(4) // 2] = 1
    return tensor


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


def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`U(-a, a)` where
    :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`.
    Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`N(0, std)` where
    :math:`std = gain \\times \sqrt{2 / (fan\_in + fan\_out)}`.
    Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_normal(w)
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform(tensor, a=0, mode='fan_in'):
    """Fills the input Tensor or Variable with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`U(-bound, bound)` where
    :math:`bound = \sqrt{2 / ((1 + a^2) \\times fan\_in)} \\times \sqrt{3}`.
    Also known as He initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_uniform(w, mode='fan_in')
    """
    if isinstance(tensor, Variable):
        kaiming_uniform(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('leaky_relu', a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-bound, bound)


def kaiming_normal(tensor, a=0, mode='fan_in'):
    """Fills the input Tensor or Variable with values according to the method
    described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`N(0, std)` where
    :math:`std = \sqrt{2 / ((1 + a^2) \\times fan\_in)}`. Also known as He
    initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either 'fan_in' (default) or 'fan_out'. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_normal(w, mode='fan_out')
    """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('leaky_relu', a)
    std = gain / math.sqrt(fan)
    return tensor.normal_(0, std)


def orthogonal(tensor, gain=1):
    """Fills the input Tensor or Variable with a (semi) orthogonal matrix, as
    described in "Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks" - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable, where n >= 2
        gain: optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.orthogonal(w)
    """
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


def sparse(tensor, sparsity, std=0.01):
    """Fills the 2D input Tensor or Variable as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`N(0, 0.01)`, as described in "Deep learning via
    Hessian-free optimization" - Martens, J. (2010).

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
        the non-zero values

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.sparse(w, sparsity=0.1)
    """
    if isinstance(tensor, Variable):
        sparse(tensor.data, sparsity, std=std)
        return tensor

    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    tensor.normal_(0, std)
    rows, cols = tensor.size(0), tensor.size(1)
    num_zeros = int(math.ceil(rows * sparsity))

    for col_idx in range(tensor.size(1)):
        row_indices = list(range(rows))
        random.shuffle(row_indices)
        zero_indices = row_indices[:num_zeros]
        for row_idx in zero_indices:
            tensor[row_idx, col_idx] = 0

    return tensor
