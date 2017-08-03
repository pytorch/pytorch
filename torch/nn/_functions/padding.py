from torch.autograd import Function, Variable


class ConstantPad2d(Function):

    @staticmethod
    def forward(ctx, input, pad, value=0):
        assert input.dim() == 4, 'only 4D supported for padding'
        ctx.pad = pad
        ctx.value = value
        pad_l, pad_r, pad_t, pad_b = ctx.pad
        h = input.size(2) + pad_t + pad_b
        w = input.size(3) + pad_l + pad_r
        assert w > 0 and h > 0, 'input is too small'

        ctx.input_size = input.size()

        # crop input if necessary
        output = input.new(input.size(0), input.size(1), h, w).fill_(ctx.value)
        c_input = input
        if pad_t < 0:
            c_input = c_input.narrow(2, -pad_t, c_input.size(2) + pad_t)
        if pad_b < 0:
            c_input = c_input.narrow(2, 0, c_input.size(2) + pad_b)
        if pad_l < 0:
            c_input = c_input.narrow(3, -pad_l, c_input.size(3) + pad_l)
        if pad_r < 0:
            c_input = c_input.narrow(3, 0, c_input.size(3) + pad_r)

        # crop output if necessary
        c_output = output
        if pad_t > 0:
            c_output = c_output.narrow(2, pad_t, c_output.size(2) - pad_t)
        if pad_b > 0:
            c_output = c_output.narrow(2, 0, c_output.size(2) - pad_b)
        if pad_l > 0:
            c_output = c_output.narrow(3, pad_l, c_output.size(3) - pad_l)
        if pad_r > 0:
            c_output = c_output.narrow(3, 0, c_output.size(3) - pad_r)
        c_output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pad_l, pad_r, pad_t, pad_b = ctx.pad

        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x,) for x in ctx.input_size]

        def narrow_slice(dim, start, length):
            grad_input_slices[dim] = (slice(grad_input_slices[dim].start + start,
                                            grad_input_slices[dim].start + start + length))

        def slice_length(dim):
            return grad_input_slices[dim].stop - grad_input_slices[dim].start

        #  crop grad_input if necessary
        if pad_t < 0:
            narrow_slice(2, -pad_t, slice_length(2) + pad_t)
        if pad_b < 0:
            narrow_slice(2, 0, slice_length(2) + pad_b)
        if pad_l < 0:
            narrow_slice(3, -pad_l, slice_length(3) + pad_l)
        if pad_r < 0:
            narrow_slice(3, 0, slice_length(3) + pad_r)

        # crop grad_output if necessary
        cg_output = grad_output
        if pad_t > 0:
            cg_output = cg_output.narrow(2, pad_t, cg_output.size(2) - pad_t)
        if pad_b > 0:
            cg_output = cg_output.narrow(2, 0, cg_output.size(2) - pad_b)
        if pad_l > 0:
            cg_output = cg_output.narrow(3, pad_l, cg_output.size(3) - pad_l)
        if pad_r > 0:
            cg_output = cg_output.narrow(3, 0, cg_output.size(3) - pad_r)
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None
