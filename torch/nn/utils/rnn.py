from collections import namedtuple
import warnings

import torch


PackedSequence_ = namedtuple('PackedSequence',
                             ['data', 'batch_sizes', 'sorted_indices', 'unsorted_indices'])


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

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").

    def __new__(cls, data, batch_sizes=None, sorted_indices=None, unsorted_indices=None):
        # PackedSequence used to only have __init__(self, data, batch_sizes)
        # without a __new__ like this. So to preserve BC for calling in keyword
        # arg style (e.g., `PackedSequence(data=..., batch_sizes=...)`), we have
        # to provide two arguments with exact names `data` and `batch_sizes`.

        # NB: if unsorted_indices is provided, it should be the inverse permutation
        # to sorted_indices. Don't assert it here because the PackedSequence ctor
        # should only be used internally.
        if unsorted_indices is None:
            unsorted_indices = invert_permutation(sorted_indices)

        # support being called as `PackedSequence(data, batch_sizes, sorted_indices)`
        if batch_sizes is not None:
            return super(PackedSequence, cls).__new__(
                cls, data, batch_sizes, sorted_indices, unsorted_indices)

        # support being called as `PackedSequence((data, batch_sizes), *, sorted_indices)`
        else:
            assert isinstance(data, (list, tuple)) and len(data) == 2
            return super(PackedSequence, cls).__new__(
                cls, data[0], data[1], sorted_indices)

    def pin_memory(self):
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.pin_memory(), self.batch_sizes,
                          bind(self.sorted_indices, lambda t: t.pin_memory()),
                          bind(self.unsorted_indices, lambda t: t.pin_memory()))

    def cuda(self, *args, **kwargs):
        """Returns a GPU copy if `self.data` not already on the GPU"""
        if self.is_cuda:
            return self
        else:
            # Why not convert `batch_sizes`?
            # See NOTE [ device and dtype of a PackedSequence ]
            return type(self)(self.data.cuda(*args, **kwargs), self.batch_sizes,
                              bind(self.sorted_indices, lambda t: t.cuda(*args, **kwargs)),
                              bind(self.unsorted_indices, lambda t: t.cuda(*args, **kwargs)))

    def cpu(self):
        """Returns a CPU copy if `self.data` not already on the CPU"""
        if self.is_cuda:
            # Why not convert `batch_sizes`?
            # See NOTE [ device and dtype of a PackedSequence ]
            return type(self)(self.data.cpu(), self.batch_sizes,
                              bind(self.sorted_indices, lambda t: t.cpu()),
                              bind(self.unsorted_indices, lambda t: t.cpu()))
        else:
            return self

    def double(self):
        r"""Returns copy with `self.data` cast to double type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.double(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def float(self):
        r"""Returns copy with `self.data` cast to float type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.float(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def half(self):
        r"""Returns copy with `self.data` cast to half type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.half(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def long(self):
        r"""Returns copy with `self.data` cast to long type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.long(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def int(self):
        r"""Returns copy with `self.data` cast to int type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.int(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def short(self):
        r"""Returns copy with `self.data` cast to short type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.short(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def char(self):
        r"""Returns copy with `self.data` cast to char type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.char(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def byte(self):
        r"""Returns copy with `self.data` cast to byte type"""

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(self.data.byte(), self.batch_sizes,
                          self.sorted_indices, self.unsorted_indices)

    def to(self, *args, **kwargs):
        r"""Performs dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`torch.Tensor.to`.

        .. note::

            If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
            and :class:`torch.device`, then ``self`` is returned.
            Otherwise, returns a copy with the desired configuration.
        """

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        data = self.data.to(*args, **kwargs)
        sorted_indices = self.sorted_indices
        unsorted_indices = self.unsorted_indices
        device_kw = 'device'
        if device_kw in kwargs:
            sorted_indices = bind(sorted_indices, lambda t: t.to(kwargs[device_kw]))
            unsorted_indices = bind(unsorted_indices, lambda t: t.to(kwargs[device_kw]))
        if data is self.data:
            return self
        else:
            return type(self)(data, self.batch_sizes,
                              sorted_indices, unsorted_indices)

    @property
    def is_cuda(self):
        r"""Returns true if `self.data` stored on a gpu"""
        return self.data.is_cuda

    def is_pinned(self):
        r"""Returns true if `self.data` stored on in pinned memory"""
        return self.data.is_pinned()


def invert_permutation(permutation):
    if permutation is None:
        return None
    output = torch.empty_like(permutation)
    output.scatter_(0, permutation,
                    torch.arange(0, permutation.numel(), device=permutation.device))
    return output


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
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

    Arguments:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor): list of sequences lengths of each batch element.
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format.
        enforce_sorted (bool, optional): if ``True``, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    if torch._C._get_tracing_state() and not isinstance(lengths, torch.Tensor):
        warnings.warn('pack_padded_sequence has been called with a Python list of '
                      'sequence lengths. The tracer cannot track the data flow of Python '
                      'values, and it will treat them as constants, likely rendering '
                      'the trace incorrect for any other combination of lengths.',
                      category=torch.jit.TracerWarning, stacklevel=2)
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    r"""Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *``, where `T` is the length
    of the longest sequence and `B` is the batch size. If ``batch_first`` is True,
    the data will be transposed into ``B x T x *`` format.

    Batch elements will be ordered decreasingly by their length.

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`this FAQ section <pack-rnn-unpack-with-data-parallelism>` for
        details.

    Arguments:
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

    """
    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError("Expected total_length to be at least the length "
                             "of the longest sequence in input, but got "
                             "total_length={} and max sequence length being {}"
                             .format(total_length, max_seq_length))
        max_seq_length = total_length
    padded_output, lengths = torch._C._VariableFunctions._pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length)
    if sequence.unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return padded_output.index_select(batch_dim, sequence.unsorted_indices), \
            lengths[sequence.unsorted_indices]
    return padded_output, lengths


def pad_sequence(sequences, batch_first=False, padding_value=0):
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

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def pack_sequence(sequences, enforce_sorted=True):
    r"""Packs a list of variable length Tensors

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


    Arguments:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    lengths = [v.size(0) for v in sequences]
    return pack_padded_sequence(pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted)


def get_packed_sequence(data, batch_sizes, sorted_indices, unsorted_indices):
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)
