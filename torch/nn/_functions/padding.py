from torch.autograd import Function, Variable
from torch.autograd._functions.utils import prepare_onnx_paddings


class ConstantPadNd(Function):

    @staticmethod
    def symbolic(g, input, pad, value=0):
        paddings = prepare_onnx_paddings(len(input.type().sizes()), pad)
        return g.op("Pad", input, pads_i=paddings, mode_s="constant", value_f=value)

    @staticmethod
    def forward(ctx, input, pad, value=0):
        ctx.pad = pad
        ctx.value = value
        ctx.input_size = input.size()
        ctx.l_inp = len(input.size())
        ctx.pad_tup = tuple([(a, b) for a, b in zip(pad[:-1:2], pad[1::2])][::-1])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i]) for i, d in enumerate(input.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # crop input if necessary
        output = input.new(input.size()[:(ctx.l_diff)] + new_dim).fill_(ctx.value)
        c_input = input

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] < 0:
                c_input = c_input.narrow(i, -p[0], c_input.size(i) + p[0])
            if p[1] < 0:
                c_input = c_input.narrow(i, 0, c_input.size(i) + p[1])

        # crop output if necessary
        c_output = output
        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                c_output = c_output.narrow(i, p[0], c_output.size(i) - p[0])
            if p[1] > 0:
                c_output = c_output.narrow(i, 0, c_output.size(i) - p[1])
        c_output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x,) for x in ctx.input_size]

        def narrow_slice(dim, start, length):
            grad_input_slices[dim] = (slice(grad_input_slices[dim].start + start,
                                            grad_input_slices[dim].start + start + length))

        def slice_length(dim):
            return grad_input_slices[dim].stop - grad_input_slices[dim].start

        #  crop grad_input if necessary
        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] < 0:
                narrow_slice(i, -p[0], slice_length(i) + p[0])
            if p[1] < 0:
                narrow_slice(i, 0, slice_length(i) + p[1])

        # crop grad_output if necessary
        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0], cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0, cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None
