from torch.autograd import Variable


def hardtanh_backwards_backwards(ctx, ggI):
    t = ctx.saved_variables
    input, grad_output = t[0], t[1]
    min_val, max_val = ctx.additional_args[0:2]

    max_mask = input <= max_val
    min_mask = input <= min_val
    gI = Variable(ggI.data.new(ggI.size()).zero_())
    ggO = ggI * (max_mask - min_mask).type_as(grad_output)
    return gI, ggO, None, None, None

double_backwards_fns = {
    'Hardtanh': hardtanh_backwards_backwards
}
