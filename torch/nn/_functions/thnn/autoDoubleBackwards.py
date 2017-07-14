from torch.autograd import Variable


def hardtanh_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    input, grad_output = t[0], t[1]
    min_val, max_val = ctx.additional_args[0:2]

    max_mask = input <= max_val
    min_mask = input <= min_val
    gI = Variable(ggI.data.new(ggI.size()).zero_())
    ggO = ggI * (max_mask - min_mask).type_as(grad_output)
    return gI, ggO, None, None, None


def elu_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    input, grad_output = t[0], t[1]
    alpha = ctx.additional_args[0]

    negative_mask = (input < 0).type_as(ggI)
    exp_alpha = input.exp() * alpha * negative_mask
    gI = ggI * grad_output * exp_alpha

    non_negative_mask = (input >= 0).type_as(ggI)
    ggO = ggI * (exp_alpha + non_negative_mask)
    return gI, ggO, None, None, None, None


def l1loss_double_backwards(ctx, ggI):
    size_average = ctx.additional_args[0]
    input, target, grad_output = ctx.saved_variables
    gI = Variable(ggI.data.new(ggI.size()).zero_())

    positive_mask = (input > target).type_as(ggI)
    negative_mask = (input < target).type_as(ggI)
    ggO = (ggI * (positive_mask - negative_mask)).sum()
    if size_average:
        ggO = ggO / input.nelement()
    return gI, None, ggO, None, None

double_backwards_fns = {
    'Hardtanh': hardtanh_double_backwards,
    'ELU': elu_double_backwards,
    'L1Loss': l1loss_double_backwards,
}
