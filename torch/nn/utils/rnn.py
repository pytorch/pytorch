from collections import namedtuple
import torch
from torch.autograd import Variable


PackedSequence_ = namedtuple('PackedSequence', ['data', 'batch_sizes'])


class PackedSequence(PackedSequence_):
    """Holds the data and list of batch_sizes of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

    Attributes:
        data (Variable): Variable containing packed sequence
        batch_sizes (list[int]): list of integers holding information about
            the batch size at each sequence step
    """
    pass


def pack_padded_sequence(input, lengths, batch_first=False):
    """Packs a Variable containing padded sequences of variable length.

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
        lengths (list[int]): list of sequences lengths of each batch element.
        batch_first (bool, optional): if True, the input is expected in BxTx*
            format.

    Returns:
        a :class:`PackedSequence` object
    """
    if lengths[-1] <= 0:
        raise ValueError("length of all samples has to be greater than 0, "
                         "but found an element in 'lengths' that is <=0")
    if batch_first:
        input = input.transpose(0, 1)

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    batch_size = input.size(1)
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    prev_l = 0
    for i, l in enumerate(lengths_iter):
        if l > prev_l:
            c_batch_size = batch_size - i
            steps.append(input[prev_l:l, :c_batch_size].contiguous().view(-1, input.size(2)))
            batch_sizes.extend([c_batch_size] * (l - prev_l))
            prev_l = l
        elif prev_l > l:  # remember that new_length is the preceding length in the array
            raise ValueError("lengths array has to be sorted in decreasing order")

    return PackedSequence(torch.cat(steps), batch_sizes)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0):
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Variable's data will be of size TxBx*, where T is the length
    of the longest sequence and B is the batch size. If ``batch_first`` is True,
    the data will be transposed into BxTx* format.

    Batch elements will be ordered decreasingly by their length.

    Arguments:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if True, the output will be in BxTx*
            format.
        padding_value (float, optional): values for padded elements

    Returns:
        Tuple of Variable containing the padded sequence, and a list of lengths
        of each sequence in the batch.
    """
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).fill_(padding_value)
    output = Variable(output)

    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    for i, batch_size in enumerate(batch_sizes):
        output[i, :batch_size] = var_data[data_offset:data_offset + batch_size]
        data_offset += batch_size

        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size
    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()

    if batch_first:
        output = output.transpose(0, 1)
    return output, lengths
