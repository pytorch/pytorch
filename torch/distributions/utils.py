from functools import update_wrapper
import math
from numbers import Number
import torch
import torch.nn.functional as F


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    """
    if not all(torch.is_tensor(v) or isinstance(v, Number) for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
    if not all(map(torch.is_tensor, values)):
        options = dict(dtype=torch.get_default_dtype())
        for value in values:
            if torch.is_tensor(value):
                options = dict(dtype=value.dtype, device=value.device)
                break
        values = [v if torch.is_tensor(v) else torch.tensor(v, **options)
                  for v in values]
    return torch.broadcast_tensors(*values)


def _standard_normal(shape, dtype, device):
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
                            torch.ones(shape, dtype=dtype, device=device))
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits, is_binary=False):
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs, is_binary=False):
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


def contfractbeta(a, b, x, niters=200, tol=3.0e-6):
    """ contfractbeta() evaluates the continued fraction form of the incomplete Beta function; incompbeta().  
    (Code translated from: Numerical Recipes in C.)"""
    bm = az = am = 1.0
    qab = a+b
    qap = a+1.0
    qam = a-1.0
    bz = 1.0-qab*x/qap

    for i in range(niters+1):
        em = float(i+1)
        tem = em + em
        d = em*(b-em)*x/((qam+tem)*(a+tem))
        ap = az + d*am
        bp = bz+d*bm
        d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
        app = ap+d*az
        bpp = bp+d*bz
        aold = az
        am = ap/bpp
        bm = bp/bpp
        az = app/bpp
        bz = 1.0
        if (abs(az-aold)<(tol*abs(az))):
            return az

def cont_frac():
    pass

class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value
