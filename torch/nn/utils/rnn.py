from collections import namedtuple
import torch
from torch.autograd import Variable


PackedSequence = namedtuple('PackedSequence', ['data', 'batch_sizes'])


def pack_padded_sequence(tensor, lengths, batch_first=False):
    if batch_first:
        tensor = tensor.transpose(0, 1)

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    current_length = next(lengths_iter)
    batch_size = tensor.size(1)
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    for step, step_value in enumerate(tensor, 1):
        steps.append(step_value[:batch_size])
        batch_sizes.append(batch_size)

        while step == current_length:
            try:
                new_length = next(lengths_iter)
            except StopIteration:
                current_length = None
                break

            if current_length > new_length:  # remember that new_length is the preceding length in the array
                raise ValueError("lengths array has to be sorted in decreasing order")
            batch_size -= 1
            current_length = new_length
        if current_length is None:
            break
    return PackedSequence(torch.cat(steps), batch_sizes)


def pad_packed_sequence(sequence, batch_first=False):
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).zero_()
    output = Variable(output)

    data_offset = 0
    for i, batch_size in enumerate(batch_sizes):
        output[i,:batch_size] = var_data[data_offset:data_offset + batch_size]
        data_offset += batch_size

    if batch_first:
        output = output.transpose(0, 1)
    return output
