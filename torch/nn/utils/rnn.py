from collections import namedtuple

import torch
from torch.autograd import Variable


from .._functions.packing import PackPadded

PackedSequence_ = namedtuple('PackedSequence', ['data', 'batch_sizes'])


class PackedSequence(PackedSequence_):
    r"""Holds the data and list of batch_sizes of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data  ``abc`` and `d`
        the ``PackedSequence`` would be ``adbc`` with ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Variable): Variable containing packed sequence
        batch_sizes (Variable): Variable of integers holding
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


def _pack_padded_sequence(input, lengths, batch_first=False):
    data, batch_sizes = PackPadded.apply(input, lengths, batch_first)

    return PackedSequence(data, batch_sizes)


def _symbolic_pack_padded_sequence(g, input, lengths, batch_first=False):
    # There currently is no PackPadded operator in ONNX. We rely on an
    # optimization pass to remove this later. It is an error if all
    # PackPadded operators cannot be optimized out.
    return g.op("PackPadded", input, lengths, outputs=2)


def pack_padded_sequence(input, *args, **kwargs):
    r"""Packs a Variable containing padded sequences of variable length.

    Input can be of size ``TxBx*`` where T is the length of the longest sequence
    (equal to ``lengths[0]``), B is the batch size, and * is any number of
    dimensions (including 0). If ``batch_first`` is True ``BxTx*`` inputs are
    expected.

    The sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the
    shortest one.

    Note:
        This function accept any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Variable can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Arguments:
        input (Variable): padded batch of variable length sequences.
        lengths (Variable): list of sequences lengths of each batch element.
        batch_first (bool, optional): if ``True``, the input is expected in BxTx*
            format.

    Returns:
        a :class:`PackedSequence` object
    """
    import torch
    if torch._C._jit_is_tracing(input):
        from torch.onnx import symbolic_override
        return symbolic_override(_symbolic_pack_padded_sequence)(_pack_padded_sequence)(input, *args, **kwargs)
    else:
        return _pack_padded_sequence(input, *args, **kwargs)


def _pad_packed_sequence(sequence, batch_first=False, padding_value=0):
    var_data, batch_sizes = sequence
    max_batch_size = int(batch_sizes[0])
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).fill_(padding_value)
    output = Variable(output)

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
    # This Variable doesn't actually have any history (well,
    # technically it does; it's just untracked), it is purely here to
    # make ONNX export easier. That is to say, from an autodiff
    # standpoint this doesn't make any sense.
    return output, Variable(torch.LongTensor(lengths))


def _symbolic_pad_packed_sequence(g, input, batch_first=False, padding_value=0.0):
    # See comment on _symbolic_pack_padded_sequence
    return g.op("PadPacked", input.data, input.batch_sizes, outputs=2)


def pad_packed_sequence(input, *args, **kwargs):
    r"""Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Variable's data will be of size TxBx*, where T is the length
    of the longest sequence and B is the batch size. If ``batch_first`` is True,
    the data will be transposed into BxTx* format.

    Batch elements will be ordered decreasingly by their length.

    Arguments:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in BxTx*
            format.
        padding_value (float, optional): values for padded elements.

    Returns:
        Tuple of Variable containing the padded sequence, and Variable
        containing the list of lengths of each sequence in the batch.

    """
    import torch
    if torch._C._jit_is_tracing(input.data):
        from torch.onnx import symbolic_override
        return symbolic_override(_symbolic_pad_packed_sequence)(_pad_packed_sequence)(input, *args, **kwargs)
    else:
        return _pad_packed_sequence(input, *args, **kwargs)


def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sequences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = Variable(torch.ones(25, 300))
        >>> b = Variable(torch.ones(22, 300))
        >>> c = Variable(torch.ones(15, 300))
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Variable of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable


def pack_sequence(sequences):
    r"""Packs a list of variable length Variables

    ``sequences`` should be a list of Variables of size ``Lx*``, where L is
    the length of a sequence and * is any number of trailing dimensions,
    including zero. They should be sorted in the order of decreasing length.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = Variable(torch.Tensor([1,2,3]))
        >>> b = Variable(torch.Tensor([4,5]))
        >>> c = Variable(torch.Tensor([6]))
        >>> pack_sequence([a, b, c]])
        PackedSequence(data=
         1
         4
         6
         2
         5
         3
        [torch.FloatTensor of size 6]
        , batch_sizes=[3, 2, 1])


    Arguments:
        sequences (list[Variable]): A list of sequences of decreasing length.

    Returns:
        a :class:`PackedSequence` object
    """
    return pack_padded_sequence(pad_sequence(sequences), [v.size(0) for v in sequences])
