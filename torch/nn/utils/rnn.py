from collections import namedtuple

import torch
import torch.onnx


from .._functions.packing import PackPadded

PackedSequence_ = namedtuple('PackedSequence', ['data', 'batch_sizes'])


class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data  ``abc`` and `x`
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step

    """
    def __new__(cls, *args):
        # support being called as `PackedSequence(data, batch_sizes)`
        if len(args) == 2:
            return super(PackedSequence, cls).__new__(cls, *args)
        # support being called as `PackedSequence((data, batch_sizes))`
        else:
            assert len(args) == 1
            return super(PackedSequence, cls).__new__(cls, *args[0])

    def cuda(self, *args, **kwargs):
        """Returns a GPU copy if `self.data` not already on the GPU"""
        if self.is_cuda:
            return self
        else:
            return type(self)(self.data.cuda(*args, **kwargs), self.batch_sizes)

    def cpu(self):
        """Returns a CPU copy if `self.data` not already on the CPU"""
        if self.is_cuda:
            return type(self)(self.data.cpu(), self.batch_sizes)
        else:
            return self

    def double(self):
        r"""Returns copy with `self.data` cast to double type"""
        return type(self)(self.data.double(), self.batch_sizes)

    def float(self):
        r"""Returns copy with `self.data` cast to float type"""
        return type(self)(self.data.float(), self.batch_sizes)

    def half(self):
        r"""Returns copy with `self.data` cast to half type"""
        return type(self)(self.data.half(), self.batch_sizes)

    def long(self):
        r"""Returns copy with `self.data` cast to long type"""
        return type(self)(self.data.long(), self.batch_sizes)

    def int(self):
        r"""Returns copy with `self.data` cast to int type"""
        return type(self)(self.data.int(), self.batch_sizes)

    def short(self):
        r"""Returns copy with `self.data` cast to short type"""
        return type(self)(self.data.short(), self.batch_sizes)

    def char(self):
        r"""Returns copy with `self.data` cast to char type"""
        return type(self)(self.data.char(), self.batch_sizes)

    def byte(self):
        r"""Returns copy with `self.data` cast to byte type"""
        return type(self)(self.data.byte(), self.batch_sizes)

    @property
    def is_cuda(self):
        r"""Returns true if `self.data` stored on a gpu"""
        return self.data.is_cuda


def pack_padded_sequence(input, lengths, batch_first=False):
    r"""Packs a Tensor containing padded sequences of variable length.

    Input can be of size ``T x B x *`` where `T` is the length of the longest sequence
    (equal to ``lengths[0]``), `B` is the batch size, and `*` is any number of
    dimensions (including 0). If ``batch_first`` is True ``B x T x *`` inputs are
    expected.

    The sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
    shortest one.

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

    Returns:
        a :class:`PackedSequence` object
    """
    if isinstance(lengths, list):
        lengths = torch.LongTensor(lengths)

    data, batch_sizes = PackPadded.apply(input, lengths, batch_first)

    return PackedSequence(data, batch_sizes)


def _symbolic_pack_padded_sequence(g, input, lengths, batch_first=False, padding_value=0.0, total_length=None):
    if total_length is not None:
        raise ValueError("_symbolic_pad_packed_sequence only supports total_length=None")
    # There currently is no PackPadded operator in ONNX. We rely on an
    # optimization pass to remove this later. It is an error if all
    # PackPadded operators cannot be optimized out.

    def _onnx_symbolic_pack_padded_sequence(g, input, lengths):
        if batch_first:
            input = g.op('Transpose', input, perm_i=[1, 0, 2])
        return g.op("prim::PackPadded", input, lengths, outputs=2)

    def pack_padded_sequence_trace_wrapper(input, lengths):
        return pack_padded_sequence(input, lengths, batch_first=batch_first)

    outputs = g.wrapPyFuncWithSymbolic(
        pack_padded_sequence_trace_wrapper, [input, lengths], 2,
        _onnx_symbolic_pack_padded_sequence)
    return tuple(o for o in outputs)


pack_padded_sequence = torch.onnx.symbolic_override_first_arg_based(
    _symbolic_pack_padded_sequence)(pack_padded_sequence)


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
    var_data, batch_sizes = sequence
    max_batch_size = int(batch_sizes[0])
    max_seq_length = batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError("Expected total_length to be at least the length "
                             "of the longest sequence in input, but got "
                             "total_length={} and max sequence length being {}"
                             .format(total_length, max_seq_length))
        max_seq_length = total_length
    output = var_data.data.new(max_seq_length, max_batch_size, *var_data.size()[1:]).fill_(padding_value)

    lengths = []
    data_offset = 0
    prev_batch_size = int(batch_sizes[0])
    prev_i = 0
    for i, batch_size in enumerate(batch_sizes.tolist() + [0]):
        if batch_size != prev_batch_size:
            l = prev_batch_size * (i - prev_i)
            tmp = var_data[data_offset:data_offset + l]
            output[prev_i:i, :prev_batch_size] = tmp.view(i - prev_i, prev_batch_size, *tmp.size()[1:])
            data_offset += l
            prev_i = i
        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size

    lengths.reverse()

    if batch_first:
        output = output.transpose(0, 1)
    # This Tensor doesn't actually have any history (well,
    # technically it does; it's just untracked), it is purely here to
    # make ONNX export easier. That is to say, from an autodiff
    # standpoint this doesn't make any sense.
    return output, torch.LongTensor(lengths)


def _symbolic_pad_packed_sequence(g, input, batch_first=False, padding_value=0.0):
    def _onnx_symbolic_pad_packed_sequence(g, data, batch_sizes):
        data, lengths = g.op("prim::PadPacked", data, batch_sizes, outputs=2)
        if batch_first:
            data = g.op('Transpose', data, perm_i=[1, 0, 2])
        return data, lengths

    def pad_packed_sequence_trace_wrapper(data, batch_sizes):
        return pad_packed_sequence(PackedSequence(data, batch_sizes),
                                   batch_first=batch_first, padding_value=padding_value)

    data, lengths = g.wrapPyFuncWithSymbolic(
        pad_packed_sequence_trace_wrapper, [input.data, input.batch_sizes], 2,
        _onnx_symbolic_pad_packed_sequence)
    return data, lengths


pad_packed_sequence = torch.onnx.symbolic_override_packed_sequence_based(
    _symbolic_pad_packed_sequence)(pad_packed_sequence)


def pad_sequence(sequences, batch_first=False, padding_value=0):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    `B` is batch size. It's equal to the number of elements in ``sequences``.
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
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
            raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def pack_sequence(sequences):
    r"""Packs a list of variable length Tensors

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including zero. They should be sorted in the order of decreasing length.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = torch.tensor([1,2,3])
        >>> b = torch.tensor([4,5])
        >>> c = torch.tensor([6])
        >>> pack_sequence([a, b, c]])
        PackedSequence(data=tensor([ 1,  4,  6,  2,  5,  3]), batch_sizes=tensor([ 3,  2,  1]))


    Arguments:
        sequences (list[Tensor]): A list of sequences of decreasing length.

    Returns:
        a :class:`PackedSequence` object
    """
    return pack_padded_sequence(pad_sequence(sequences), [v.size(0) for v in sequences])
