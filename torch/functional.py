import torch
from operator import mul
from functools import reduce
import math

__all__ = [
    'argmax',
    'argmin',
    'bartlett_window',
    'btrifact',
    'btriunpack',
    'hamming_window',
    'hann_window',
    'isnan',
    'split',
    'unbind',
    'unique',
]


def split(tensor, split_size_or_sections, dim=0):
    r"""Splits the tensor into chunks.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim= is not divisible by
    :attr:`split_size`.

    If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
    to :attr:`split_size_or_sections`.

    Arguments:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
        list of sizes for each chunk
        dim (int): dimension along which to split the tensor.
    """
    # Overwriting reason:
    # This dispatches to two ATen functions depending on the type of
    # split_size_or_sections. The branching code is in variable.py, which we
    # call here.
    return tensor.split(split_size_or_sections, dim)


def btrifact(A, info=None, pivot=True):
    r"""Batch LU factorization.

    Returns a tuple containing the LU factorization and pivots. Pivoting is done if
    :attr:`pivot` is set.

    The optional argument :attr:`info` stores information if the factorization
    succeeded for each minibatch example. The :attr:`info` is provided as an
    `IntTensor`, its values will be filled from dgetrf and a non-zero value
    indicates an error occurred. Specifically, the values are from cublas if cuda is
    being used, otherwise LAPACK.

    .. warning::
        The :attr:`info` argument is deprecated in favor of :meth:`torch.btrifact_with_info`.

    Arguments:
        A (Tensor): the tensor to factor
        info (IntTensor, optional): (deprecated) an `IntTensor` to store values
            indicating whether factorization succeeds
        pivot (bool, optional): controls whether pivoting is done

    Returns:
        A tuple containing factorization and pivots.

    Example::

        >>> A = torch.randn(2, 3, 3)
        >>> A_LU, pivots = torch.btrifact(A)
        >>> A_LU

        (0 ,.,.) =
          0.7908 -0.0854  0.1522
          0.2757 -1.2942 -1.3715
         -0.6029  0.3609  0.3210

        (1 ,.,.) =
          0.9091  0.1719  0.7741
          0.1625  0.6720  0.1687
         -0.1927 -0.9420 -0.4891
        [torch.FloatTensor of size (2,3,3)]

        >>> pivots

         2  2  3
         1  3  3
        [torch.IntTensor of size (2,3)]

    """
    # Overwriting reason:
    # `info` is being deprecated in favor of `btrifact_with_info`. This warning
    # is in variable.py, which we call here.
    return A.btrifact(info, pivot)


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

    Returns a tuple of tensors as ``(the pivots, the L tensor, the U tensor)``.

    Arguments:
        LU_data (Tensor): the packed LU factorization data
        LU_pivots (Tensor): the packed LU factorization pivots
        unpack_data (bool): flag indicating if the data should be unpacked
        unpack_pivots (bool): flag indicating if the pivots should be unpacked

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


def hann_window(window_length, periodic=True, dtype=torch.float32):
    r"""Hann window function.

    This method computes the Hann window function:

    .. math::
        w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
                \sin^2 \left( \frac{\pi n}{N - 1} \right),

    where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`\text{window_length} + 1`. Also, we always have
    ``torch.hann_window(L, periodic=True)`` equal to
    ``torch.hann_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.
        dtype (:class:`torch.dtype`, optional): the desired type of returned window.
            Default: `torch.float32`

    Returns:
        Tensor: A 1-D tensor of size :math:`(\text{window_length},)` containing the window
    """
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating point type, but got dtype={}".format(dtype))
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    return hamming_window(window_length, periodic=periodic, alpha=0.5, beta=0.5, dtype=dtype)


def hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=torch.float32):
    r"""Hamming window function.

    This method computes the Hamming window function:

    .. math::
        w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

    where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`\text{window_length} + 1`. Also, we always have
    ``torch.hamming_window(L, periodic=True)`` equal to
    ``torch.hamming_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

    .. note::
        This is a generalized version of :meth:`torch.hann_window`.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.
        dtype (:class:`torch.dtype`, optional): the desired type of returned window.
            Default: `torch.float32`

    Returns:
        Tensor: A 1-D tensor of size :math:`(\text{window_length},)` containing the window
    """
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating point type, but got dtype={}".format(dtype))
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    if window_length == 1:
        return torch.ones(window_length, dtype=dtype)
    window_length += int(periodic)
    window = torch.arange(window_length, dtype=dtype)
    window = window.mul_(math.pi * 2 / (window_length - 1)).cos_().mul_(-beta).add_(alpha)
    if periodic:
        return window[:-1]
    else:
        return window


def bartlett_window(window_length, periodic=True, dtype=torch.float32):
    r"""Bartlett window function.

    This method computes the Bartlett window function:

    .. math::
        w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
            \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
            2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
        \end{cases},

    where :math:`N` is the full window size.

    The input :attr:`window_length` is a positive integer controlling the
    returned window size. :attr:`periodic` flag determines whether the returned
    window trims off the last duplicate value from the symmetric window and is
    ready to be used as a periodic window with functions like
    :meth:`torch.stft`. Therefore, if :attr:`periodic` is true, the :math:`N` in
    above formula is in fact :math:`\text{window_length} + 1`. Also, we always have
    ``torch.bartlett_window(L, periodic=True)`` equal to
    ``torch.bartlett_window(L + 1, periodic=False)[:-1])``.

    .. note::
        If :attr:`window_length` :math:`=1`, the returned window contains a single value 1.

    Arguments:
        window_length (int): the size of returned window
        periodic (bool, optional): If True, returns a window to be used as periodic
            function. If False, return a symmetric window.
        dtype (:class:`torch.dtype`, optional): the desired type of returned window.
            Default: `torch.float32`

    Returns:
        Tensor: A 1-D tensor of size :math:`(\text{window_length},)` containing the window
    """
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating point type, but got dtype={}".format(dtype))
    if window_length <= 0:
        raise ValueError('window_length must be positive')
    if window_length == 1:
        return torch.ones(window_length, dtype=dtype)
    window_length += int(periodic)
    window = torch.arange(window_length, dtype=dtype).mul_(2.0 / (window_length - 1))
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
    output, inverse_indices = torch._C._VariableFunctions._unique(
        input,
        sorted=sorted,
        return_inverse=return_inverse,
    )
    if return_inverse:
        return output, inverse_indices
    else:
        return output


def argmax(input, dim=None, keepdim=False):
    """Returns the indices of the maximum values of a tensor across a dimension.

    This is the second value returned by :meth:`torch.max`. See its
    documentation for the exact semantics of this method.

    Args:
        input (Tensor): the input tensor
        dim (int): the dimension to reduce. If ``None``, the argmax of the
            flattened input is returned.
        keepdim (bool): whether the output tensors have :attr:`dim`
            retained or not. Ignored if ``dim=None``.

    Example::

        >>> a = torch.randn(4, 4)
        >>> a

         2.3461  0.0056  1.4846  0.3911
        -1.3584 -1.0066  0.0530  1.1754
        -0.7929 -0.3194 -1.4865  0.4020
         0.1101  0.6694  1.3456  0.8235
        [torch.FloatTensor of size (4,4)]

        >>> torch.argmax(a, dim=1)
        0
        3
        3
        2
        [torch.LongTensor of size (4,)]
    """
    if dim is None:
        return torch._argmax(input.contiguous().view(-1), dim=0, keepdim=False)
    return torch._argmax(input, dim, keepdim)


def argmin(input, dim=None, keepdim=False):
    """Returns the indices of the minimum values of a tensor across a dimension.

    This is the second value returned by :meth:`torch.min`. See its
    documentation for the exact semantics of this method.

    Args:
        input (Tensor): the input tensor
        dim (int): the dimension to reduce. If ``None``, the argmin of the
            flattened input is returned.
        keepdim (bool): whether the output tensors have :attr:`dim`
            retained or not. Ignored if ``dim=None``.

    Example::

        >>> a = torch.randn(4, 4)
        >>> a

         2.3461  0.0056  1.4846  0.3911
        -1.3584 -1.0066  0.0530  1.1754
        -0.7929 -0.3194 -1.4865  0.4020
         0.1101  0.6694  1.3456  0.8235
        [torch.FloatTensor of size (4,4)]

        >>> torch.argmin(a, dim=1)
         1
         0
         2
         0
        [torch.LongTensor of size (4,)]
    """
    if dim is None:
        return torch._argmin(input.contiguous().view(-1), dim=0, keepdim=False)
    return torch._argmin(input, dim, keepdim)
