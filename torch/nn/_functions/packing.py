import torch
from torch.autograd import Function


class PackPadded(Function):
    @staticmethod
    def forward(ctx, input, lengths, batch_first):
        if batch_first:
            input = input.transpose(0, 1)

        if lengths[-1] <= 0:
            raise ValueError("Length of all samples has to be greater than 0, "
                             "but found an element in 'lengths' that is <= 0")

        steps = []
        batch_sizes = []
        lengths_iter = reversed(lengths)
        batch_size = input.size(1)

        if len(lengths) != batch_size:
            raise ValueError("Expected `len(lengths)` to be equal to batch_size, but got "
                             "{} (batch_size={}).".format(len(lengths), batch_size))

        prev_l = 0
        for i, l in enumerate(lengths_iter):
            if l > prev_l:
                c_batch_size = batch_size - i
                steps.append(input[prev_l:l, :c_batch_size].contiguous().view(-1, *input.size()[2:]))
                batch_sizes.extend([c_batch_size] * (l - prev_l))
                prev_l = l

            elif prev_l > l:
                raise ValueError("'lengths' array has to be sorted in decreasing order")

        ctx.batch_sizes = batch_sizes
        ctx.batch_first = batch_first
        ctx.input_size = input.size()

        return torch.cat(steps), torch.LongTensor(batch_sizes)

    @staticmethod
    def backward(ctx, grad_steps, grad_batch_sizes):
        grad_input = grad_steps.new(*ctx.input_size).zero_()

        offset = 0
        for i, bs in enumerate(ctx.batch_sizes):
            grad_input[i, :bs] = grad_steps[offset:offset + bs]
            offset += bs

        if ctx.batch_first:
            grad_input = grad_input.transpose(0, 1)

        return grad_input, None, None

class PadPacked(Function):
    @staticmethod
    def forward(ctx, var_data, batch_sizes, batch_first, padding_value):
        max_batch_size = batch_sizes[0]
        output = var_data.new(len(batch_sizes), max_batch_size, *var_data.size()[1:]).fill_(padding_value)

        lengths = []
        data_offset = 0
        prev_batch_size = batch_sizes[0]
        prev_i = 0
        for i, batch_size in enumerate(list(batch_sizes) + [0]):
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

        return output, torch.LongTensor(lengths)

    @staticmethod
    def backward(ctx, *args):
        assert False, "FOO"
