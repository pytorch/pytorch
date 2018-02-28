import torch
from operator import mul
from functools import reduce
import math

__all__ = [
    'split', 'chunk', 'empty_like', 'stack', 'unbind', 'btriunpack', 'matmul', 'det', 'stft',
    'hann_window', 'hamming_window', 'bartlett_window', 'where', 'isnan'
]


def split(tensor, split_size_or_sections, dim=0):
    """Splits the tensor into chunks.
    If ``split_size_or_sections`` is an integer type, then ``tensor`` will be
    split into equally sized chunks (if possible).
    Last chunk will be smaller if the tensor size along a given dimension
    is not divisible by ``split_size``.
    If ``split_size_or_sections`` is a list, then ``tensor`` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in ``dim`` according
    to ``split_size_or_sections``.

    Arguments:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
        list of sizes for each chunk
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)

    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        num_splits = (dim_size + split_size - 1) // split_size
        last_split_size = split_size - (split_size * num_splits - dim_size)

        def get_split_size(i):
            return split_size if i < num_splits - 1 else last_split_size
        return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i))) for i
                     in range(0, num_splits))

    else:
        if dim_size != sum(split_size_or_sections):
            raise ValueError("Sum of split sizes exceeds tensor dim")
        split_indices = [0] + split_size_or_sections
        split_indices = torch.cumsum(torch.Tensor(split_indices), dim=0)

        return tuple(
            tensor.narrow(int(dim), int(start), int(length))
            for start, length in zip(split_indices, split_size_or_sections))


def chunk(tensor, chunks, dim=0):
    r"""Splits a tensor into a specific number of chunks.

    Arguments:
        tensor (Tensor): the tensor to split
        chunks (int): number of chunks to return
        dim (int): dimension along which to split the tensor
    """
    if dim < 0:
        dim += tensor.dim()
    split_size = (tensor.size(dim) + chunks - 1) // chunks
    return split(tensor, split_size, dim)


def empty_like(input):
    r"""empty_like(input) -> Tensor

    Returns an uninitialized tensor with the same size as :attr:`input`.

    Args:
        input (Tensor): the size of :attr:`input` will determine size of the output tensor

    Example::

        >>> input = torch.LongTensor(2,3)
        >>> input.new(input.size())

        1.3996e+14  1.3996e+14  1.3996e+14
        4.0000e+00  0.0000e+00  0.0000e+00
        [torch.LongTensor of size 2x3]
    """
    return input.new(input.size())


def stack(sequence, dim=0, out=None):
    r"""Concatenates sequence of tensors along a new dimension.

    All tensors need to be of the same size.

    Arguments:
        sequence (Sequence): sequence of tensors to concatenate
        dim (int): dimension to insert. Has to be between 0 and the number
            of dimensions of concatenated tensors (inclusive)
    """
    if len(sequence) == 0:
        raise ValueError("stack expects a non-empty sequence of tensors")
    if dim < 0:
        dim += sequence[0].dim() + 1
    inputs = [t.unsqueeze(dim) for t in sequence]
    if out is None:
        return torch.cat(inputs, dim)
    else:
        return torch.cat(inputs, dim, out=out)


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

    Returns a tuple indexed by:
      0: The pivots.
      1: The L tensor.
      2: The U tensor.

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


def matmul(tensor1, tensor2, out=None):
    r"""Matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors as follows:

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
      must be broadcastable).  For example, if :attr:`tensor1` is a
      :math:`(j \times 1 \times n \times m)` tensor and :attr:`tensor2` is a :math:`(k \times m \times p)`
      tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.

    .. note::

        The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

    Arguments:
        tensor1 (Tensor): the first tensor to be multiplied
        tensor2 (Tensor): the second tensor to be multiplied
        out (Tensor, optional): the output tensor
    """
    dim_tensor1 = tensor1.dim()
    dim_tensor2 = tensor2.dim()
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        if out is None:
            return torch.dot(tensor1, tensor2)
        else:
            raise ValueError("out must be None for 1-d tensor matmul, returns a scalar")
    if dim_tensor1 == 2 and dim_tensor2 == 1:
        if out is None:
            return torch.mv(tensor1, tensor2)
        else:
            return torch.mv(tensor1, tensor2, out=out)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        if out is None:
            return torch.mm(tensor1.unsqueeze(0), tensor2).squeeze_(0)
        else:
            return torch.mm(tensor1.unsqueeze(0), tensor2, out=out).squeeze_(0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        if out is None:
            return torch.mm(tensor1, tensor2)
        else:
            return torch.mm(tensor1, tensor2, out=out)
    elif dim_tensor1 >= 3 and (dim_tensor2 == 1 or dim_tensor2 == 2):
        # optimization: use mm instead of bmm by folding tensor1's batch into
        # its leading matrix dimension.

        if dim_tensor2 == 1:
            tensor2 = tensor2.unsqueeze(-1)

        size1 = tensor1.size()
        size2 = tensor2.size()
        output_size = size1[:-1] + size2[-1:]

        # fold the batch into the first dimension
        tensor1 = tensor1.contiguous().view(-1, size1[-1])

        if out is None or not out.is_contiguous():
            output = torch.mm(tensor1, tensor2)
        else:
            output = torch.mm(tensor1, tensor2, out=out)

        output = output.view(output_size)

        if dim_tensor2 == 1:
            output = output.squeeze(-1)

        if out is not None:
            out.set_(output)
            return out

        return output
    elif (dim_tensor1 >= 1 and dim_tensor2 >= 1) and (dim_tensor1 >= 3 or dim_tensor2 >= 3):
        # ensure each tensor size is at least 3-dimensional
        tensor1_exp_size = torch.Size((1,) * max(3 - tensor1.dim(), 0) + tensor1.size())
        # rhs needs to be a separate case since we can't freely expand 1s on the rhs, but can on lhs
        if dim_tensor2 == 1:
            tensor2 = tensor2.unsqueeze(1)
        tensor2_exp_size = torch.Size((1,) * max(3 - tensor2.dim(), 0) + tensor2.size())

        # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        expand_batch_portion = torch._C._infer_size(tensor1_exp_size[:-2], tensor2_exp_size[:-2])

        # flatten expanded batches
        tensor1_expanded = tensor1.expand(*(expand_batch_portion + tensor1_exp_size[-2:])) \
            .contiguous().view(reduce(mul, expand_batch_portion), *tensor1_exp_size[-2:])
        tensor2_expanded = tensor2.expand(*(expand_batch_portion + tensor2_exp_size[-2:])) \
            .contiguous().view(reduce(mul, expand_batch_portion), *tensor2_exp_size[-2:])

        # reshape batches back into result
        total_expansion = expand_batch_portion + (tensor1_exp_size[-2], tensor2_exp_size[-1])

        def maybeSqueeze(tensor):
            if dim_tensor1 == 1:
                return tensor.squeeze(-2)
            elif dim_tensor2 == 1:
                return tensor.squeeze(-1)
            else:
                return tensor

        if out is None or not out.is_contiguous():
            output = torch.bmm(tensor1_expanded, tensor2_expanded)
        else:
            output = torch.bmm(tensor1_expanded, tensor2_expanded, out=out)

        output = maybeSqueeze(output.view(total_expansion))

        if out is not None:
            out.set_(output)
            return out

        return output

    raise ValueError("both arguments to __matmul__ need to be at least 1D, "
                     "but they are {}D and {}D".format(dim_tensor1, dim_tensor2))


def det(var):
    r"""Calculates determinant of a 2D square Variable.

    .. note::
        Backward through `det` internally uses SVD results. So double backward
        through `det` will need to backward through :meth:`~Tensor.svd`. This
        can be unstable in certain cases. Please see :meth:`~torch.svd` for
        details.

    Arguments:
        var (Variable): The input 2D square Variable.
    """
    if torch.is_tensor(var):
        raise ValueError("det is currently only supported on Variable")
    return var.det()


def stft(var, frame_length, hop, fft_size=None, return_onesided=True, window=None, pad_end=0):
    r"""Short-time Fourier transform (STFT).

    Ignoring the batch dimension, this method computes the following expression:

    .. math::
        X[m, \omega] = \sum_{k = 0}^{frame\_length}%
                            window[k]\ signal[m \times hop + k]\ e^{- j \frac{2 \pi \cdot \omega k}{frame\_length}}

    , where :math:`m` is the index of the sliding window, and :math:`\omega` is
    the frequency that :math:`0 \leq \omega < fft\_size`. When
    :attr:`return_onsesided` is the default value True, only values for
    :math:`\omega` in range :math:`[0, 1, 2, \dots, \lfloor \frac{fft\_size}{2} \rfloor + 1]`
    are returned because the real-to-complex transform satisfies the Hermitian
    symmetry, i.e., :math:`X[m, \omega] = X[m, fft\_length - \omega]^*`.

    The input :attr:`signal` must be 1-D sequence :math:`(T)` or 2-D a batch of
    sequences :math:`(N \times T)`. If :attr:`fft_size` is ``None``, it is
    default to same value as  :attr:``frame_length``. :attr:`window` can be a
    1-D tensor of size :math:`(frame\_length)`, e.g., see
    :meth:`torch.hann_window`. If :attr:`window` is the default value ``None``,
    it is treated as if having :math:`1` everywhere in the frame.
    :attr:`pad_end` indicates the amount of zero padding at the end of
    :attr:`signal` before STFT.

    Returns the real and the imaginary parts together as one tensor of size
    :math:`(* \times N \times 2)`, where :math:`*` is the shape of input :attr:`signal`,
    :math:`N` is the number of :math:`\omega`s considered depending on
    :attr:`fft_size` and :attr:`return_onesided`, and each pair in the last
    dimension represents a complex number as real part and imaginary part.

    Arguments:
        signal (Tensor): the input tensor
        frame_length (int): the size of window frame and STFT filter
        hop (int): the distance between neighboring sliding window frames
        fft_size (int, optional): size of Fourier transform
        return_onesided (bool, optional): controls whether to avoid redundancy in the return value
        window (Tensor, optional): the optional window function
        pad_end (int, optional): implicit zero padding at the end of :attr:`signal`

    Returns:
        Tensor: A tensor containing the STFT result
    """
    if torch.is_tensor(var):
        raise ValueError("stft is currently only supported on Variable")
    return var.stft(frame_length, hop, fft_size, return_onesided, window, pad_end)


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


def where(condition, x, y):
    r"""Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.

    defined as::

         out_i =  x_i        if condition_i
                  y_i        else

    .. note::
        This function only works with ``Variables``.

    .. note::
        The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.

    Arguments:
        condition (ByteTensor): When True (nonzero), yield x, otherwise yield y.
        x (Tensor): values selected at indices where :attr:`condition` is True.
        y (Tensor): values selected at indices where :attr:`condition` is False.

    Returns:
        Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`
    """
    # the parameter order is changed here; the functional order is the same as numpy; the
    # method follows the usual torch mask semantics of x.fn(mask, y)
    return torch._C._VariableBase.where(x, condition, y)


def isnan(tensor):
    r"""Returns a new tensor with boolean elements representing if each element is NaN or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of NaN elements.

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
