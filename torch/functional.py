import torch
import torch.nn.functional as F
from torch._six import inf
from torch._C import _add_docstr
from operator import mul
from functools import reduce
from collections import Iterable
from torch._utils import annotate
from itertools import product
import math
import warnings

__all__ = [
    'argmax',
    'argmin',
    'argsort',
    'btriunpack',
    'chain_matmul',
    'einsum',
    'broadcast_tensors',
    'isfinite',
    'isinf',
    'isnan',
    'norm',
    'meshgrid',
    'potrf',
    'potrs',
    'split',
    'stft',
    'tensordot',
    'unique',
    'cartesian_prod',
]


def broadcast_tensors(*tensors):
    r"""broadcast_tensors(*tensors) -> List of Tensors

    Broadcasts the given tensors according to :ref:`_broadcasting-semantics`.

    Args:
        *tensors: any number of tensors of the same type

    Example::

        >>> x = torch.arange(3).view(1, 3)
        >>> y = torch.arange(2).view(2, 1)
        >>> a, b = torch.broadcast_tensors(x, y)
        >>> a.size()
        torch.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]])
    """
    return torch._C._VariableFunctions.broadcast_tensors(tensors)


def split(tensor, split_size_or_sections, dim=0):
    r"""Splits the tensor into chunks.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim` is not divisible by
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
    # split_size_or_sections. The branching code is in tensor.py, which we
    # call here.
    return tensor.split(split_size_or_sections, dim)


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
        >>> P, A_L, A_U = torch.btriunpack(A_LU, pivots)
        >>>
        >>> # can recover A from factorization
        >>> A_ = torch.bmm(P, torch.bmm(A_L, A_U))
    """

    sz = LU_data.size(-1)

    if unpack_data:
        U = LU_data.triu()
        L = LU_data.tril()
        L.diagonal(dim1=-2, dim2=-1).fill_(1)
    else:
        L = U = None

    if unpack_pivots:
        P = torch.eye(sz, device=LU_data.device, dtype=LU_data.dtype).expand_as(LU_data).clone()
        LU_pivots = LU_pivots - 1
        for idx in product(*map(lambda x: list(range(x)), LU_data.shape[:-2])):
            final_order = list(range(sz))
            for k, j in enumerate(LU_pivots[idx]):
                final_order[k], final_order[j] = final_order[j], final_order[k]
            P[idx] = P[idx].index_select(1, torch.as_tensor(final_order, device=LU_pivots.device))
    else:
        P = None

    return P, L, U


def einsum(equation, *operands):
    r"""einsum(equation, *operands) -> Tensor

This function provides a way of computing multilinear expressions (i.e. sums of products) using the
Einstein summation convention.

Args:
    equation (string): The equation is given in terms of lower case letters (indices) to be associated
           with each dimension of the operands and result. The left hand side lists the operands
           dimensions, separated by commas. There should be one index letter per tensor dimension.
           The right hand side follows after `->` and gives the indices for the output.
           If the `->` and right hand side are omitted, it implicitly defined as the alphabetically
           sorted list of all indices appearing exactly once in the left hand side.
           The indices not apprearing in the output are summed over after multiplying the operands
           entries.
           If an index appears several times for the same operand, a diagonal is taken.
           Ellipses `...` represent a fixed number of dimensions. If the right hand side is inferred,
           the ellipsis dimensions are at the beginning of the output.
    operands (list of Tensors): The operands to compute the Einstein sum of.

Examples::

    >>> x = torch.randn(5)
    >>> y = torch.randn(4)
    >>> torch.einsum('i,j->ij', x, y)  # outer product
    tensor([[-0.0570, -0.0286, -0.0231,  0.0197],
            [ 1.2616,  0.6335,  0.5113, -0.4351],
            [ 1.4452,  0.7257,  0.5857, -0.4984],
            [-0.4647, -0.2333, -0.1883,  0.1603],
            [-1.1130, -0.5588, -0.4510,  0.3838]])


    >>> A = torch.randn(3,5,4)
    >>> l = torch.randn(2,5)
    >>> r = torch.randn(2,4)
    >>> torch.einsum('bn,anm,bm->ba', l, A, r) # compare torch.nn.functional.bilinear
    tensor([[-0.3430, -5.2405,  0.4494],
            [ 0.3311,  5.5201, -3.0356]])


    >>> As = torch.randn(3,2,5)
    >>> Bs = torch.randn(3,5,4)
    >>> torch.einsum('bij,bjk->bik', As, Bs) # batch matrix multiplication
    tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
             [-1.6706, -0.8097, -0.8025, -2.1183]],

            [[ 4.2239,  0.3107, -0.5756, -0.2354],
             [-1.4558, -0.3460,  1.5087, -0.8530]],

            [[ 2.8153,  1.8787, -4.3839, -1.2112],
             [ 0.3728, -2.1131,  0.0921,  0.8305]]])

    >>> A = torch.randn(3, 3)
    >>> torch.einsum('ii->i', A) # diagonal
    tensor([-0.7825,  0.8291, -0.1936])

    >>> A = torch.randn(4, 3, 3)
    >>> torch.einsum('...ii->...i', A) # batch diagonal
    tensor([[-1.0864,  0.7292,  0.0569],
            [-0.9725, -1.0270,  0.6493],
            [ 0.5832, -1.1716, -1.5084],
            [ 0.4041, -1.1690,  0.8570]])

    >>> A = torch.randn(2, 3, 4, 5)
    >>> torch.einsum('...ij->...ji', A).shape # batch permute
    torch.Size([2, 3, 5, 4])
"""
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        operands = operands[0]
    return torch._C._VariableFunctions.einsum(equation, operands)


def isfinite(tensor):
    r"""Returns a new tensor with boolean elements representing if each element is `Finite` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of finite elements and 0 otherwise

    Example::

        >>> torch.isfinite(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([ 1,  0,  1,  0,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))

    # Support int input, nan and inf are concepts in floating point numbers.
    # Numpy uses type 'Object' when the int overflows long, but we don't
    # have a similar concept. It's safe to assume any created LongTensor doesn't
    # overflow and it's finite.
    if not tensor.is_floating_point():
        return torch.ones_like(tensor, dtype=torch.uint8)
    return (tensor == tensor) & (tensor.abs() != inf)


def isinf(tensor):
    r"""Returns a new tensor with boolean elements representing if each element is `+/-INF` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `+/-INF` elements and 0 otherwise

    Example::

        >>> torch.isinf(torch.tensor([1, float('inf'), 2, float('-inf'), float('nan')]))
        tensor([ 0,  1,  0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    if tensor.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.zeros_like(tensor, dtype=torch.uint8)
    return tensor.abs() == inf


def meshgrid(*tensors, **kwargs):
    r"""Take :math:`N` tensors, each of which can be either scalar or 1-dimensional
vector, and create :math:`N` N-dimensional grids, where the :math:`i` :sup:`th` grid is defined by
expanding the :math:`i` :sup:`th` input over dimensions defined by other inputs.


    Args:
        tensors (list of Tensor): list of scalars or 1 dimensional tensors. Scalars will be
        treated as tensors of size :math:`(1,)` automatically

    Returns:
        seq (sequence of Tensors): If the input has :math:`k` tensors of size
        :math:`(N_1,), (N_2,), \ldots , (N_k,)`, then the output would also has :math:`k` tensors,
        where all tensors are of size :math:`(N_1, N_2, \ldots , N_k)`.

    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([4, 5, 6])
        >>> grid_x, grid_y = torch.meshgrid(x, y)
        >>> grid_x
        tensor([[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]])
        >>> grid_y
        tensor([[4, 5, 6],
                [4, 5, 6],
                [4, 5, 6]])
    """
    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'" % (list(kwargs)[0],))
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        tensors = tensors[0]
    return torch._C._VariableFunctions.meshgrid(tensors)


def stft(input, n_fft, hop_length=None, win_length=None, window=None,
         center=True, pad_mode='reflect', normalized=False, onesided=True):
    r"""Short-time Fourier transform (STFT).

    Ignoring the optional batch dimension, this method computes the following
    expression:

    .. math::
        X[m, \omega] = \sum_{k = 0}^{\text{win\_length-1}}%
                            \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
                            \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{win\_length}}\right),

    where :math:`m` is the index of the sliding window, and :math:`\omega` is
    the frequency that :math:`0 \leq \omega < \text{n\_fft}`. When
    :attr:`onesided` is the default value ``True``,

    * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
      sequences.

    * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
      ``floor(n_fft / 4)``.

    * If :attr:`win_length` is ``None`` (default), it is treated as equal to
      :attr:`n_fft`.

    * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
      :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
      treated as if having :math:`1` everywhere in the window. If
      :math:`\text{win\_length} < \text{n\_fft}`, :attr:`window` will be padded on
      both sides to length :attr:`n_fft` before being applied.

    * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
      both sides so that the :math:`t`-th frame is centered at time
      :math:`t \times \text{hop\_length}`. Otherwise, the :math:`t`-th frame
      begins at time  :math:`t \times \text{hop\_length}`.

    * :attr:`pad_mode` determines the padding method used on :attr:`input` when
      :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
      all available options. Default is ``"reflect"``.

    * If :attr:`onesided` is ``True`` (default), only values for :math:`\omega`
      in :math:`\left[0, 1, 2, \dots, \left\lfloor \frac{\text{n\_fft}}{2} \right\rfloor + 1\right]`
      are returned because the real-to-complex Fourier transform satisfies the
      conjugate symmetry, i.e., :math:`X[m, \omega] = X[m, \text{n\_fft} - \omega]^*`.

    * If :attr:`normalized` is ``True`` (default is ``False``), the function
      returns the normalized STFT results, i.e., multiplied by :math:`(\text{frame\_length})^{-0.5}`.

    Returns the real and the imaginary parts together as one tensor of size
    :math:`(* \times N \times T \times 2)`, where :math:`*` is the optional
    batch size of :attr:`input`, :math:`N` is the number of frequencies where
    STFT is applied, :math:`T` is the total number of frames used, and each pair
    in the last dimension represents a complex number as the real part and the
    imaginary part.

    .. warning::
      This function changed signature at version 0.4.1. Calling with the
      previous signature may cause error or return incorrect result.

    Arguments:
        input (Tensor): the input tensor
        n_fft (int): size of Fourier transform
        hop_length (int, optional): the distance between neighboring sliding window
            frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``  (treated as equal to :attr:`n_fft`)
        window (Tensor, optional): the optional window function.
            Default: ``None`` (treated as window of all :math:`1` s)
        center (bool, optional): whether to pad :attr:`input` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: ``False``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy Default: ``True``

    Returns:
        Tensor: A tensor containing the STFT result with shape described above

    """
    # TODO: after having proper ways to map Python strings to ATen Enum, move
    #       this and F.pad to ATen.
    if center:
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(n_fft // 2)
        input = F.pad(input.view(extended_shape), (pad, pad), pad_mode)
        input = input.view(input.shape[-signal_dim:])
    return torch._C._VariableFunctions.stft(input, n_fft, hop_length, win_length, window, normalized, onesided)


isnan = _add_docstr(torch.isnan, r"""
Returns a new tensor with boolean elements representing if each element is `NaN` or not.

Arguments:
    tensor (Tensor): A tensor to check

Returns:
    Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `NaN` elements.

Example::

    >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
    tensor([ 0,  1,  0], dtype=torch.uint8)
""")


def unique(input, sorted=True, return_inverse=False, dim=None):
    r"""Returns the unique scalar elements of the input tensor as a 1-D tensor.

    Arguments:
        input (Tensor): the input tensor
        sorted (bool): Whether to sort the unique elements in ascending order
            before returning as output.
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        dim (int): the dimension to apply unique. If ``None``, the unique of the
            flattened input is returned. default: ``None``

    Returns:
        (Tensor, Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be a
              2nd returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.

    Example::

        >>> output = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long))
        >>> output
        tensor([ 2,  3,  1])

        >>> output, inverse_indices = torch.unique(
                torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True)
        >>> output
        tensor([ 1,  2,  3])
        >>> inverse_indices
        tensor([ 0,  2,  1,  2])

        >>> output, inverse_indices = torch.unique(
                torch.tensor([[1, 3], [2, 3]], dtype=torch.long), sorted=True, return_inverse=True)
        >>> output
        tensor([ 1,  2,  3])
        >>> inverse_indices
        tensor([[ 0,  2],
                [ 1,  2]])

    """
    if dim is not None:
        output, inverse_indices = torch._unique_dim(
            input,
            dim,
            sorted=sorted,
            return_inverse=return_inverse
        )
    else:
        output, inverse_indices = torch._unique(
            input,
            sorted=sorted,
            return_inverse=return_inverse,
        )
    if return_inverse:
        return output, inverse_indices
    else:
        return output


def argmax(input, dim=None, keepdim=False):
    r"""Returns the indices of the maximum values of a tensor across a dimension.

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
        tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                [-0.7401, -0.8805, -0.3402, -1.1936],
                [ 0.4907, -1.3948, -1.0691, -0.3132],
                [-1.6092,  0.5419, -0.2993,  0.3195]])


        >>> torch.argmax(a, dim=1)
        tensor([ 0,  2,  0,  1])
    """
    if dim is None:
        return torch._argmax(input.contiguous().view(-1), dim=0, keepdim=False)
    return torch._argmax(input, dim, keepdim)


def argmin(input, dim=None, keepdim=False):
    r"""Returns the indices of the minimum values of a tensor across a dimension.

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
        tensor([[ 0.1139,  0.2254, -0.1381,  0.3687],
                [ 1.0100, -1.1975, -0.0102, -0.4732],
                [-0.9240,  0.1207, -0.7506, -1.0213],
                [ 1.7809, -1.2960,  0.9384,  0.1438]])


        >>> torch.argmin(a, dim=1)
        tensor([ 2,  1,  3,  1])
    """
    if dim is None:
        return torch._argmin(input.contiguous().view(-1), dim=0, keepdim=False)
    return torch._argmin(input, dim, keepdim)


def tensordot(a, b, dims=2):
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalizes the matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or tuple of two lists of integers): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with an integer argument :attr:`dims` = :math:`d`, and the number of
    dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`, respectively,
    it computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match, but :attr:`tensordot` will deal with broadcasted
    dimensions.

    Examples::

        >>> a = torch.arange(60.).reshape(3, 4, 5)
        >>> b = torch.arange(24.).reshape(4, 3, 2)
        >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
        tensor([[4400., 4730.],
                [4532., 4874.],
                [4664., 5018.],
                [4796., 5162.],
                [4928., 5306.]])

        >>> a = torch.randn(3, 4, 5, device='cuda')
        >>> b = torch.randn(4, 5, 6, device='cuda')
        >>> c = torch.tensordot(a, b, dims=2).cpu()
        tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
                [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
                [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

    """
    if isinstance(dims, (list, tuple)) or \
       (isinstance(dims, torch.Tensor) and dims.numel() > 1):
        dims_a, dims_b = dims
    else:
        if isinstance(dims, torch.Tensor):
            dims = dims.item()
        dims_a = list(range(-dims, 0))
        dims_b = list(range(dims))
    return torch._C._VariableFunctions.tensordot(a, b, dims_a, dims_b)


def argsort(input, dim=None, descending=False):
    r"""Returns the indices that sort a tensor along a given dimension in ascending
    order by value.

    This is the second value returned by :meth:`torch.sort`.  See its documentation
    for the exact semantics of this method.

    Args:
        input (Tensor): the input tensor
        dim (int, optional): the dimension to sort along
        descending (bool, optional): controls the sorting order (ascending or descending)

    Example::

        >>> a = torch.randn(4, 4)
        >>> a
        tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
                [ 0.1598,  0.0788, -0.0745, -1.2700],
                [ 1.2208,  1.0722, -0.7064,  1.2564],
                [ 0.0669, -0.2318, -0.8229, -0.9280]])


        >>> torch.argsort(a, dim=1)
        tensor([[2, 0, 3, 1],
                [3, 2, 1, 0],
                [2, 1, 0, 3],
                [3, 2, 1, 0]])
    """
    if dim is None:
        return torch.sort(input, -1, descending)[1]
    return torch.sort(input, dim, descending)[1]


def cartesian_prod(*tensors):
    """Do cartesian product of the given sequence of tensors. The behavior is similar to
    python's `itertools.product`.

    Arguments:
        *tensors: any number of 1 dimensional tensors.

    Returns:
        Tensor: A tensor equivalent to converting all the input tensors into lists,
            do `itertools.product` on these lists, and finally convert the resulting list
            into tensor.

    Example::

        >>> a = [1, 2, 3]
        >>> b = [4, 5]
        >>> list(itertools.product(a, b))
        [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
        >>> tensor_a = torch.tensor(a)
        >>> tensor_b = torch.tensor(b)
        >>> torch.cartesian_prod(tensor_a, tensor_b)
        tensor([[1, 4],
                [1, 5],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5]])
    """
    return torch._C._VariableFunctions.cartesian_prod(tensors)


def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    r"""Returns the matrix norm or vector norm of a given tensor.

    Args:
        input (Tensor): the input tensor
        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            =====  ============================  ==========================
            ord    matrix norm                   vector norm
            =====  ============================  ==========================
            None   Frobenius norm                2-norm
            'fro'  Frobenius norm                --
            'nuc'  nuclear norm                  --
            Other  as vec norm when dim is None  sum(abs(x)**ord)**(1./ord)
            =====  ============================  ==========================

        dim (int, 2-tuple of ints, 2-list of ints, optional): If it is an int,
            vector norm will be calculated, if it is 2-tuple of ints, matrix norm
            will be calculated. If the value is None, matrix norm will be calculated
            when the input tensor only has two dimensions, vector norm will be
            calculated when the input tensor only has one dimension. If the input
            tensor has more than two dimensions, the vector norm will be applied to
            last dimension.
        keepdim (bool, optional): whether the output tensors have :attr:`dim`
            retained or not. Ignored if :attr:`dim` = ``None`` and
            :attr:`out` = ``None``. Default: ``False``
        out (Tensor, optional): the output tensor. Ignored if
            :attr:`dim` = ``None`` and :attr:`out` = ``None``.
        dtype (:class:`torch.dtype`, optional): the desired data type of
            returned tensor. If specified, the input tensor is casted to
            :attr:'dtype' while performing the operation. Default: None.


    Example::

        >>> import torch
        >>> a = torch.arange(9, dtype= torch.float) - 4
        >>> b = a.reshape((3, 3))
        >>> torch.norm(a)
        tensor(7.7460)
        >>> torch.norm(b)
        tensor(7.7460)
        >>> torch.norm(a, float('inf'))
        tensor(4.)
        >>> torch.norm(b, float('inf'))
        tensor(4.)
        >>> c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
        >>> torch.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000])
        >>> torch.norm(c, dim=1)
        tensor([3.7417, 4.2426])
        >>> torch.norm(c, p=1, dim=1)
        tensor([6., 6.])
        >>> d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
        >>> torch.norm(d, dim=(1,2))
        tensor([ 3.7417, 11.2250])
        >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        (tensor(3.7417), tensor(11.2250))
    """
    ndim = input.dim()

    # catch default case
    if dim is None and out is None and dtype is None:
        if p == "fro":
            return torch._C._VariableFunctions.frobenius_norm(input)
        elif p != "nuc":
            return torch._C._VariableFunctions.norm(input, p)

    if p == "fro":
        if dtype is not None:
            raise ValueError("dtype argument is not supported in frobenius norm")
        if dim is None:
            dim = tuple(range(ndim))
        if out is None:
            return torch._C._VariableFunctions.frobenius_norm(input, dim, keepdim=keepdim)
        return torch._C._VariableFunctions.frobenius_norm(input, dim, keepdim=keepdim, out=out)
    elif p == "nuc":
        if dtype is not None:
            raise ValueError("dtype argument is not supported in nuclear norm")
        if out is None:
            torch._C._VariableFunctions.nuclear_norm(input, keepdim=keepdim)
        return torch._C._VariableFunctions.nuclear_norm(input, keepdim=keepdim, out=out)
    else:
        if dim is None:
            dim = tuple(range(ndim))
        if out is None and dtype is None:
            return torch._C._VariableFunctions.norm(input, p, dim, keepdim=keepdim)
        elif out is None:
            return torch._C._VariableFunctions.norm(input, p, dim, keepdim=keepdim, dtype=dtype)
        elif dtype is None:
            return torch._C._VariableFunctions.norm(input, p, dim, keepdim=keepdim, out=out)
    return torch._C._VariableFunctions.norm(input, p, dim, keepdim=keepdim, dtype=dtype, out=out)


def chain_matmul(*matrices):
    r"""Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
    using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
    of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, :math:`N`
    needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
    If :math:`N` is 1, then this is a no-op - the original matrix is returned as is.


    Args:
        matrices (Tensors...): a sequence of 2 or more 2-D tensors whose product is to be determined.


    Returns:
        Tensor: if the :math:`i^{th}` tensor was of dimensions :math:`p_{i} \times p_{i + 1}`, then the product
        would be of dimensions :math:`p_{1} \times p_{N + 1}`.

    Example::

        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(4, 5)
        >>> c = torch.randn(5, 6)
        >>> d = torch.randn(6, 7)
        >>> torch.chain_matmul(a, b, c, d)
        tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
                [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
                [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])

    .. _`[CLRS]`: https://mitpress.mit.edu/books/introduction-algorithms-third-edition
    """
    return torch._C._VariableFunctions.chain_matmul(matrices)


def potrf(a, upper=True, out=None):
    r"""Computes the Cholesky decomposition of a symmetric positive-definite
    matrix :math:`A`.

    For more information, regarding :func:`torch.potrf`, please check :func:`torch.cholesky`.

    .. warning::
        torch.potrf is deprecated in favour of torch.cholesky and will be removed in the next
        release. Please use torch.cholesky instead and note that the :attr:`upper` argument in
        torch.cholesky defaults to ``False``.
    """
    warnings.warn("torch.potrf is deprecated in favour of torch.cholesky and will be removed in the next "
                  "release. Please use torch.cholesky instead and note that the :attr:`upper` argument in"
                  " torch.cholesky defaults to ``False``.", stacklevel=2)
    return torch.cholesky(a, upper=upper, out=out)


def potrs(b, u, upper=True, out=None):
    r"""Solves a linear system of equations with a positive semidefinite
    matrix to be inverted given its Cholesky factor matrix :attr:`u`.

    For more information, regarding :func:`torch.potrs`, please check :func:`torch.cholesky_solve`.

    .. warning::
        torch.potrs is deprecated in favour of torch.cholesky_solve and will be removed in the next
        release. Please use torch.cholesky_solve instead and note that the :attr:`upper` argument in
        torch.cholesky_solve defaults to ``False``.
    """
    warnings.warn("torch.potrs is deprecated in favour of torch.cholesky_solve and will be removed "
                  "in the next release. Please use torch.cholesky instead and note that the "
                  ":attr:`upper` argument in torch.cholesky_solve defaults to ``False``.", stacklevel=2)
    return torch.cholesky_solve(b, u, upper=upper, out=out)
