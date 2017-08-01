from torch.autograd import Variable


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


def hardtanh_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    input, grad_output = t[0], t[1]
    min_val, max_val = ctx.additional_args[0:2]

    max_mask = input <= max_val
    min_mask = input <= min_val
    gI = Variable(ggI.data.new(ggI.size()).zero_())
    ggO = ggI * (max_mask - min_mask).type_as(grad_output)
    return gI, ggO, None, None, None


def leakyrelu_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    input = t[0]
    negative_slope = ctx.additional_args[0]

    gI = Variable(ggI.data.new(ggI.size()).zero_())
    input_lt_0 = (input < 0).type_as(ggI)
    input_ge_0 = (input >= 0).type_as(ggI)
    ggO = ggI * (input_lt_0 * negative_slope + input_ge_0)
    return gI, ggO, None, None, None


def logsoftmax_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    gO, output = t[1], t[2]

    output_exp = output.exp()
    gO_sum = gO.sum(dim=1, keepdim=True)
    ggI_output_exp = ggI * output_exp
    ggI_output_exp_sum = ggI_output_exp.sum(dim=1, keepdim=True)

    gI = output_exp * gO_sum * ggI_output_exp_sum - ggI_output_exp * gO_sum
    ggO = ggI - ggI_output_exp_sum

    return gI, ggO, None, None, None, None


def softmax_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    gO, output = t[1], t[2]

    # terms for reuse
    ggI_output = ggI * output
    ggI_out_sum = ggI_output.sum(dim=1, keepdim=True)
    ggI_out_sum_output = ggI_out_sum * output
    gO_out_sum = (gO * output).sum(dim=1, keepdim=True)

    # gI calculation
    gI_t0 = ggI_output * (gO - gO_out_sum)
    gI_t1 = output * ((ggI_output * gO).sum(dim=1, keepdim=True).sub_(gO_out_sum * ggI_out_sum))
    gI_t2 = ggI_out_sum_output * gO
    gI_t3 = ggI_out_sum_output * gO_out_sum
    gI = gI_t0 - gI_t1 - gI_t2 + gI_t3

    # gO calculation
    ggO = output * (ggI - ggI_out_sum)

    return gI, ggO, None, None, None, None


def threshold_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    input = t[0]
    threshold, value = ctx.additional_args[0:2]

    gI = Variable(ggI.data.new(ggI.size()).zero_())
    input_gt_threshold = (input > threshold).type_as(ggI)
    ggO = ggI * input_gt_threshold
    return gI, ggO, None, None, None


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


def nllloss_double_backwards(ctx, ggI):
    t = ctx.saved_variables
    target = t[1]
    weights = Variable(ctx.additional_args[1])
    size_average = ctx.additional_args[0]
    ignore_index = ctx.additional_args[3]

    gI = None

    # can't scatter/gather on indices outside of range, let's just put them in range
    # and 0 out the weights later (so it doesn't matter where in range we put them)
    target_mask = target == ignore_index
    safe_target = target.clone()
    safe_target.masked_fill_(target_mask, 0)

    if weights.dim() == 0:
        weights_to_scatter = Variable(ggI.data.new(safe_target.size()).fill_(1))
    else:
        weights_maybe_resized = weights
        while weights_maybe_resized.dim() < target.dim():
            weights_maybe_resized = weights_maybe_resized.unsqueeze(1)

        weights_maybe_resized = weights_maybe_resized.expand(weights.size()[0:1] + target.size()[1:])
        weights_to_scatter = weights_maybe_resized.gather(0, safe_target)

    weights_to_scatter.masked_fill_(target_mask, 0)
    divisor = weights_to_scatter.sum() if size_average else 1
    weights_to_scatter = -1 * weights_to_scatter / divisor
    zeros = Variable(ggI.data.new(ggI.size()).zero_())
    mask = zeros.scatter_(1, safe_target.unsqueeze(1), weights_to_scatter.unsqueeze(1))

    ggO = (ggI * mask).sum()

    return gI, None, ggO, None, None, None


double_backwards_fns = {
    'ELU': elu_double_backwards,
    'Hardtanh': hardtanh_double_backwards,
    'LeakyReLU': leakyrelu_double_backwards,
    'LogSoftmax': logsoftmax_double_backwards,
    'Softmax': softmax_double_backwards,
    'Threshold': threshold_double_backwards,
    'L1Loss': l1loss_double_backwards,
    'NLLLoss': nllloss_double_backwards,
    'NLLLoss2d': nllloss_double_backwards,
}
