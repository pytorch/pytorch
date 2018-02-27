import torch
from operator import mul
from functools import reduce
import math

__all__ = [
    'bartlett_window',
    'btriunpack',
    'hamming_window',
    'hann_window',
    'isnan',
    'unbind',
    'unique',
]


def unbind(tensor, dim=0):
    r"""Removes a tensor dimension.

    Returns a tuple of all slices along a given dimension, already without it.

    Arguments:
        tensor (Tensor): the tensor to unbind
        dim (int): dimension to remove
    """
    return tuple(tensor.select(dim, i) for i in range(tensor.size(dim)))


def btriunpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    r"""Unpacks the data and pivots from a batched LU factorization (btrifact) of a tensor.

    Returns a tuple as ``(the pivots, the L tensor, the U tensor)``.

    Arguments:
        LU_data (Tensor): the packed LU factorization data
        LU_pivots (Tensor): the packed LU factorization pivots
        unpack_data (bool): flag indicating if the data should be unpacked
        unpack_pivots (bool): tlag indicating if the pivots should be unpacked

    Example::

        >>> A = torch.randn(2, 3, 3)
        >>> A_LU, pivots = A.btrifact()
        >>> P, a_L, a_U = torch.btriunpack(A_LU, pivots)
        >>>
        >>> # test that (P, A_L, A_U) gives LU factorization
        >>> A_ = torch.bmm(P, torch.bmm(A_L, A_U))
        >>> assert torch.equal(A_, A) == True  # can recover A
    """

    nBatch, sz, _ = LU_data.size()

    if unpack_data:
        I_U = torch.triu(torch.ones(sz, sz)).type_as(LU_data).byte().unsqueeze(0).expand(nBatch, sz, sz)
        I_L = 1 - I_U
        L = LU_data.new(LU_data.size()).zero_()
        U = LU_data.new(LU_data.size()).zero_()
        I_diag = torch.eye(sz).type_as(LU_data).byte().unsqueeze(0).expand(nBatch, sz, sz)
        L[I_diag] = 1.0
        L[I_L] = LU_data[I_L]
        U[I_U] = LU_data[I_U]
    else:
        L = U = None

    if unpack_pivots:
        P = torch.eye(sz).type_as(LU_data).unsqueeze(0).repeat(nBatch, 1, 1)
        for i in range(nBatch):
            for j in range(sz):
                k = int(LU_pivots[i, j] - 1)
                t = P[i, :, j].clone()
                P[i, :, j] = P[i, :, k]
                P[i, :, k] = t
    else:
        P = None

    return P, L, U


def hann_window(window_length, periodic=True):
    r"""Hann window function.

    This method computes the Hann window function:

    .. math::
        w[n] = \frac{1}{2}\ [1 - \cos \left( \frac{2 \pi n}{N - 1} \right)] = \sin^2 \left( \frac{\pi n}{N - 1} \right)

    , where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`window\_length + 1`. Also, we always have
    ``torch.hann_window(L, periodic=True)`` equal to
    ``torch.hann_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`\leq 2`, the returned window contains a single value 1.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.

    Returns:
        Tensor: A 1-D tensor of size :math:`(window\_length)` containing the window
    """
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    return hamming_window(window_length, periodic=periodic, alpha=0.5, beta=0.5)


def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46):
    r"""Hamming window function.

    This method computes the Hamming window function:

    .. math::
        w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right)

    , where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`window\_length + 1`. Also, we always have
    ``torch.hamming_window(L, periodic=True)`` equal to
    ``torch.hamming_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`\leq 2`, the returned window contains a single value 1.

    .. note::
        This is a generalized version of :meth:`torch.hann_window`.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.

    Returns:
        Tensor: A 1-D tensor of size :math:`(window\_length)` containing the window
    """
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    if window_length == 1:
        return torch.ones(window_length)
    window_length += int(periodic)
    window = torch.arange(window_length).mul_(math.pi * 2 / (window_length - 1)).cos_().mul_(-beta).add_(alpha)
    if periodic:
        return window[:-1]
    else:
        return window


def bartlett_window(window_length, periodic=True):
    r"""Bartlett window function.

    This method computes the Bartlett window function:

    .. math::
        w[n] = 1 - \lvert \frac{2n}{N-1} - 1 \rvert = \begin{cases}
            \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
            2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
        \end{cases}

    , where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`window\_length + 1`. Also, we always have
    ``torch.bartlett_window(L, periodic=True)`` equal to
    ``torch.bartlett_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`\leq 2`, the returned window contains a single value 1.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.

    Returns:
        Tensor: A 1-D tensor of size :math:`(window\_length)` containing the window
    """
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    if window_length == 1:
        return torch.ones(window_length)
    window_length += int(periodic)
    window = torch.arange(window_length).mul_(2.0 / (window_length - 1))
    first_half_size = ((window_length - 1) >> 1) + 1
    window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2)
    if periodic:
        return window[:-1]
    else:
        return window


def isnan(tensor):
    r"""Returns a new tensor with boolean elements representing if each element is `NaN` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `NaN` elements.

    Example::

        >>> torch.isnan(torch.Tensor([1, float('nan'), 2]))
         0
         1
         0
        [torch.ByteTensor of size 3]
    """
    if not torch.is_tensor(tensor):
        raise ValueError("The argument is not a tensor")
    return tensor != tensor


def unique(input, sorted=False, return_inverse=False):
    r"""Returns the unique scalar elements of the input tensor as a 1-D tensor.

    Arguments:
        input (Tensor): the input tensor
        sorted (bool): Whether to sort the unique elements in ascending order
            before returning as output.
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.

    Returns:
        (Tensor, Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be a
              2nd returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.

    Example::

        >>>> output = torch.unique(torch.LongTensor([1, 3, 2, 3]))
        >>>> output

         2
         3
         1
        [torch.LongTensor of size (3,)]

        >>>> output, inverse_indices = torch.unique(
                 torch.LongTensor([1, 3, 2, 3]), sorted=True, return_inverse=True)
        >>>> output

         1
         2
         3
        [torch.LongTensor of size (3,)]

        >>>> inverse_indices

         0
         2
         1
         2
        [torch.LongTensor of size (4,)]

        >>>> output, inverse_indices = torch.unique(
                 torch.LongTensor([[1, 3], [2, 3]]), sorted=True, return_inverse=True)
        >>>> output

         1
         2
         3
        [torch.LongTensor of size (3,)]

        >>>> inverse_indices

         0  2
         1  2
        [torch.LongTensor of size (2,2)]
    """
    output, inverse_indices = torch._C._VariableBase._unique(
        input,
        sorted=sorted,
        return_inverse=return_inverse,
    )
    if return_inverse:
        return output, inverse_indices
    else:
        return output
