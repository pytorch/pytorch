from torch.autograd import Function


class ConstantPad2d(Function):

    def __init__(self, pad, value=0):
        super(ConstantPad2d, self).__init__()
        self.pad = pad
        self.value = value

    def forward(self, input):
        assert input.dim() == 4, 'only 4D supported for padding'
        pad_l, pad_r, pad_t, pad_b = self.pad
        h = input.size(2) + pad_t + pad_b
        w = input.size(3) + pad_l + pad_r
        assert w > 0 and h > 0, 'input is too small'

        self.input_size = input.size()

        # crop input if necessary
        output = input.new(input.size(0), input.size(1), h, w).fill_(self.value)
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

    def backward(self, grad_output):
        pad_l, pad_r, pad_t, pad_b = self.pad

        grad_input = grad_output.new(self.input_size).zero_()

        #  crop grad_input if necessary
        cg_input = grad_input
        if pad_t < 0:
            cg_input = cg_input.narrow(2, -pad_t, cg_input.size(2) + pad_t)
        if pad_b < 0:
            cg_input = cg_input.narrow(2, 0, cg_input.size(2) + pad_b)
        if pad_l < 0:
            cg_input = cg_input.narrow(3, -pad_l, cg_input.size(3) + pad_l)
        if pad_r < 0:
            cg_input = cg_input.narrow(3, 0, cg_input.size(3) + pad_r)

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
        cg_input.copy_(cg_output)
        return grad_input
