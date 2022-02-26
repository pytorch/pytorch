from collections import namedtuple
import warnings

import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional

from typing import List, Tuple



PackedSequence_ = namedtuple('PackedSequence_',
                             ['data', 'batch_sizes', 'sorted_indices', 'unsorted_indices'])

# type annotation for PackedSequence_ to make it compatible with TorchScript
PackedSequence_.__annotations__ = {'data': torch.Tensor, 'batch_sizes': torch.Tensor,
                                   'sorted_indices': Optional[torch.Tensor],
                                   'unsorted_indices': Optional[torch.Tensor]}

def bind(optional, fn):
    if optional is None:
        return None
    return fn(optional)


class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a `:class:PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).

    """
    def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
        return super(PackedSequence, cls).__new__(
            cls,
            *_packed_sequence_init_args(data, batch_sizes, sorted_indices,
                                        unsorted_indices))

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").
    def pin_memory(self):
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.pin_memory(), self.batch_sizes,
                          bind(self.sorted_indices, lambda t: t.pin_memory()),
                          bind(self.unsorted_indices, lambda t: t.pin_memory()))

    def cuda(self, *args, **kwargs):
        # Tests to see if 'cuda' should be added to kwargs
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(*args, device='cuda', **kwargs)

    def cpu(self, *args, **kwargs):

        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        if ex.device.type == 'cpu':
            return self.to(*args, **kwargs)
        return self.to(*args, device='cpu', **kwargs)

    def double(self):
        return self.to(dtype=torch.double)

    def float(self):
        return self.to(dtype=torch.float)

    def half(self):
        return self.to(dtype=torch.half)

    def long(self):
        return self.to(dtype=torch.long)

    def int(self):
        return self.to(dtype=torch.int)

    def short(self):
        return self.to(dtype=torch.short)

    def char(self):
        return self.to(dtype=torch.int8)

    def byte(self):
        return self.to(dtype=torch.uint8)

    def to(self, *args, **kwargs):
        r"""Performs dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`torch.Tensor.to`, except optional
        arguments like `non_blocking` and `copy` should be passed as kwargs,
        not args, or they will not apply to the index tensors.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
            and :class:`torch.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            # Does not forward device or dtype arg/kwargs, device is set from data.device
            kwargs = {k : v for k, v in filter(lambda t: t[0] != 'device' and t[0] != 'dtype', kwargs.items())}
            sorted_indices = bind(self.sorted_indices, lambda t: t.to(data.device, **kwargs))
            unsorted_indices = bind(self.unsorted_indices, lambda t: t.to(data.device, **kwargs))
            return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        r"""Returns true if `self.data` stored on a gpu"""
        return self.data.is_cuda

    def is_pinned(self):
        r"""Returns true if `self.data` stored on in pinned memory"""
        return self.data.is_pinned()


# TorchScript doesn't support constructors on named tuples, so we use this helper
# method to construct PackedSequence
def _packed_sequence_init_args(data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]  # noqa: B950
    # NB: if unsorted_indices is provided, it should be the inverse permutation
    # to sorted_indices. Don't assert it here because the PackedSequence ctor
    # should only be used internally.

    if unsorted_indices is None:
        unsorted_indices = invert_permutation(sorted_indices)

    # support being called as `PackedSequence(data, batch_sizes, sorted_indices)`
    if batch_sizes is not None:
        # TODO: Re-enable this check (.type isn't supported in TorchScript)
        if batch_sizes.device.type != 'cpu':
            raise ValueError(
                "batch_sizes should always be on CPU. "
                "Instances of PackedSequence should never be created manually. "
                "They should be instantiated by functions like pack_sequence "
                "and pack_padded_sequences in nn.utils.rnn. "
                "https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence")
        return data, batch_sizes, sorted_indices, unsorted_indices

    # support being called as `PackedSequence((data, batch_sizes), *, sorted_indices)`
    else:
        assert isinstance(data, (list, tuple)) and len(data) == 2
        return data[0], data[1], sorted_indices, unsorted_indices


def _packed_sequence_init(data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
    # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> PackedSequence
    data, batch_sizes, sorted_indices, unsorted_indices = _packed_sequence_init_args(
        data, batch_sizes, sorted_indices, unsorted_indices)
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


def invert_permutation(permutation):
    # type: (Optional[Tensor]) -> Optional[Tensor]
    if permutation is None:
        return None
    output = torch.empty_like(permutation, memory_format=torch.legacy_contiguous_format)
    output.scatter_(0, permutation,
                    torch.arange(0, permutation.numel(), device=permutation.device))
    return output


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    # type: (Tensor, Tensor, bool, bool) -> PackedSequence
    r"""Packs a Tensor containing padded sequences of variable length.

    :attr:`input` can be of size ``T x B x *`` where `T` is the length of the
    longest sequence (equal to ``lengths[0]``), ``B`` is the batch size, and
    ``*`` is any number of dimensions (including 0). If ``batch_first`` is
    ``True``, ``B x T x *`` :attr:`input` is expected.

    For unsorted sequences, use `enforce_sorted = False`. If :attr:`enforce_sorted` is
    ``True``, the sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the shortest
    one. `enforce_sorted = True` is only necessary for ONNX export.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Args:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor or list(int)): list of sequence lengths of each batch
            element (must be on the CPU if provided as a tensor).
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format.
        enforce_sorted (bool, optional): if ``True``, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            ``False``, the input will get sorted unconditionally. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    if torch._C._get_tracing_state() and not isinstance(lengths, torch.Tensor):
        warnings.warn('pack_padded_sequence has been called with a Python list of '
                      'sequence lengths. The tracer cannot track the data flow of Python '
                      'values, and it will treat them as constants, likely rendering '
                      'the trace incorrect for any other combination of lengths.',
                      stacklevel=2)
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        _VF._pack_padded_sequence(input, lengths, batch_first)
    return _packed_sequence_init(data, batch_sizes, sorted_indices, None)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    # type: (PackedSequence, bool, float, Optional[int]) -> Tuple[Tensor, Tensor]
    r"""Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *``, where `T` is the length
    of the longest sequence and `B` is the batch size. If ``batch_first`` is True,
    the data will be transposed into ``B x T x *`` format.

    Example:
        >>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
        >>> lens = [2, 1, 3]
        >>> packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
        >>> packed
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                       sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[1, 2, 0],
                [3, 0, 0],
                [4, 5, 6]])
        >>> lens_unpacked
        tensor([2, 1, 3])

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`this FAQ section <pack-rnn-unpack-with-data-parallelism>` for
        details.

    Args:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format.
        padding_value (float, optional): values for padded elements.
        total_length (int, optional): if not ``None``, the output will be padded to
            have length :attr:`total_length`. This method will throw :class:`ValueError`
            if :attr:`total_length` is less than the max sequence length in
            :attr:`sequence`.

    Returns:
        Tuple of Tensor containing the padded sequence, and a Tensor
        containing the list of lengths of each sequence in the batch.
        Batch elements will be re-ordered as they were ordered originally when
        the batch was passed to ``pack_padded_sequence`` or ``pack_sequence``.




    """
    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError("Expected total_length to be at least the length "
                             "of the longest sequence in input, but got "
                             "total_length={} and max sequence length being {}"
                             .format(total_length, max_seq_length))
        max_seq_length = total_length
    padded_output, lengths = _VF._pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length)
    unsorted_indices = sequence.unsorted_indices
    if unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return padded_output.index_select(batch_dim, unsorted_indices), lengths[unsorted_indices.cpu()]
    return padded_output, lengths


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise. Default: False.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(sequences, batch_first, padding_value)


def unpad_sequence(padded_sequences, lengths, batch_first=False):
    # type: (Tensor, Tensor, bool) -> List[Tensor]
    r"""Unpad padded Tensor into a list of variable length Tensors

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = torch.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> torch.allclose(sequences[0], unpadded_sequences[0])
        True
        >>> torch.allclose(sequences[1], unpadded_sequences[1])
        True
        >>> torch.allclose(sequences[2], unpadded_sequences[2])
        True

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: False.

    Returns:
        a list of :class:`Tensor` objects
    """

    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def pack_sequence(sequences, enforce_sorted=True):
    # type: (List[Tensor], bool) -> PackedSequence
    r"""Packs a list of variable length Tensors

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including zero.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = torch.tensor([1,2,3])
        >>> b = torch.tensor([4,5])
        >>> c = torch.tensor([6])
        >>> pack_sequence([a, b, c])
        PackedSequence(data=tensor([ 1,  4,  6,  2,  5,  3]), batch_sizes=tensor([ 3,  2,  1]))


    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    lengths = torch.as_tensor([v.size(0) for v in sequences])
    return pack_padded_sequence(pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted)


def unpack_sequence(packed_sequences):
    # type: (PackedSequence) -> List[Tensor]
    r"""Unpacks PackedSequence into a list of variable length Tensors

    ``packed_sequences`` should be a PackedSequence object.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = torch.tensor([1,2,3])
        >>> b = torch.tensor([4,5])
        >>> c = torch.tensor([6])
        >>> sequences = [a, b, c]
        [tensor([ 1,  2,  3]), tensor([ 4,  5]), tensor([ 6])]
        >>> packed_sequences = pack_sequence(sequences)
        PackedSequence(data=tensor([ 1,  4,  6,  2,  5,  3]), batch_sizes=tensor([ 3,  2,  1]))
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        [tensor([ 1,  2,  3]), tensor([ 4,  5]), tensor([ 6])]


    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """

    padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first=True)
    unpacked_sequences = unpad_sequence(padded_sequences, lengths, batch_first=True)
    return unpacked_sequences
