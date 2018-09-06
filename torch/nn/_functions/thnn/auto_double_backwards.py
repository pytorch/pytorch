import torch


def elu_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    input, grad_output = t[0], t[1]
    alpha = ctx.additional_args[0]

    negative_mask = (input < 0).type_as(ggI)
    exp_alpha = input.exp() * alpha * negative_mask
    gI = ggI * grad_output * exp_alpha

    non_negative_mask = (input >= 0).type_as(ggI)
    ggO = ggI * (exp_alpha + non_negative_mask)
    return gI, ggO, None, None, None, None


def gatedlinear_double_backwards(ctx, ggI):
    input, gO = ctx.saved_tensors
    dim = ctx.additional_args[0]

    input_size = input.size(dim) // 2

    first_half = input.narrow(dim, 0, input_size)
    second_half = input.narrow(dim, input_size, input_size)
    sig_second_half = second_half.sigmoid()
    one_sub_sig_second_half = 1 - sig_second_half
    sig_one_sub_sig = sig_second_half * one_sub_sig_second_half

    ggI_first_half = ggI.narrow(dim, 0, input_size)
    ggI_second_half = ggI.narrow(dim, input_size, input_size)
    ggI_second_half_times_first_half = ggI_second_half * first_half

    gI_first_half = ggI_second_half * gO * sig_one_sub_sig
    second_order_sh = sig_one_sub_sig * one_sub_sig_second_half - sig_second_half * sig_one_sub_sig
    gI_second_half = ggI_second_half_times_first_half * gO * second_order_sh + ggI_first_half * gO * sig_one_sub_sig
    gI = torch.cat((gI_first_half, gI_second_half), dim)

    ggO = ggI_first_half * sig_second_half + ggI_second_half_times_first_half * sig_one_sub_sig

    return gI, ggO, None, None, None


def hardshrink_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    input = t[0]
    lambd = ctx.additional_args[0]
    gI = None

    mask = torch.zeros_like(input).masked_fill_(input > lambd, 1).masked_fill_(input < -lambd, 1)
    ggO = ggI * mask

    return gI, ggO, None, None, None


def hardtanh_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    input, grad_output = t[0], t[1]
    min_val, max_val = ctx.additional_args[0:2]

    max_mask = input <= max_val
    min_mask = input <= min_val
    gI = torch.zeros_like(ggI)
    ggO = ggI * (max_mask - min_mask).type_as(grad_output)
    return gI, ggO, None, None, None


def leakyrelu_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    input = t[0]
    negative_slope = ctx.additional_args[0]

    gI = torch.zeros_like(ggI)
    input_lt_0 = (input < 0).type_as(ggI)
    input_ge_0 = (input >= 0).type_as(ggI)
    ggO = ggI * (input_lt_0 * negative_slope + input_ge_0)
    return gI, ggO, None, None, None


def logsigmoid_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    # maybe more efficient in terms of output, but save_output is False
    input, gO = t[0], t[1]

    exp_input = input.exp()
    exp_input_plus_1 = exp_input + 1
    gI = ggI * gO * -1 * exp_input / (exp_input_plus_1.pow(2))
    ggO = ggI / exp_input_plus_1

    return gI, ggO, None, None, None, None


def softplus_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    input, gO, output = t[0], t[1], t[2]
    beta, threshold = ctx.additional_args[0], ctx.additional_args[1]

    input_beta = input * beta
    above_threshold = torch.zeros_like(ggI).masked_fill_(input_beta > threshold, 1)
    below_threshold = torch.zeros_like(ggI).masked_fill_(input_beta <= threshold, 1)

    exp_output_beta = (output * beta).exp()
    first_deriv = (exp_output_beta - 1) / exp_output_beta
    first_deriv_below_threshold = first_deriv * below_threshold

    gI = ggI * gO * first_deriv_below_threshold * beta / exp_output_beta
    ggO = ggI * (above_threshold + first_deriv_below_threshold)

    return gI, ggO, None, None, None, None


def softshrink_double_backwards(ctx, ggI):
    return hardshrink_double_backwards(ctx, ggI)


def threshold_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    input = t[0]
    threshold, value = ctx.additional_args[0:2]

    gI = torch.zeros_like(ggI)
    input_gt_threshold = (input > threshold).type_as(ggI)
    ggO = ggI * input_gt_threshold
    return gI, ggO, None, None, None


def klddivloss_double_backwards(ctx, ggI):
    size_average = ctx.additional_args[0]
    input, target, gO = ctx.saved_tensors
    div_factor = input.nelement() if size_average else 1

    gI = None
    ggO = (ggI * target).sum() / -div_factor

    return gI, None, ggO, None, None


def l1loss_double_backwards(ctx, ggI):
    size_average = ctx.additional_args[0]
    input, target, grad_output = ctx.saved_tensors
    gI = torch.zeros_like(ggI)

    positive_mask = (input > target).type_as(ggI)
    negative_mask = (input < target).type_as(ggI)
    ggO = (ggI * (positive_mask - negative_mask)).sum()
    if size_average:
        ggO = ggO / input.nelement()
    return gI, None, ggO, None, None


def mseloss_double_backwards(ctx, ggI):
    size_average = ctx.additional_args[0]
    reduce = ctx.additional_args[1]
    input, target, gO = ctx.saved_tensors
    div_factor = input.nelement() if size_average and reduce else 1

    gI = ggI * (gO * 2. / div_factor).expand_as(input)
    if reduce:
        ggO = (ggI * (input - target)).sum() * (2. / div_factor)
    else:
        ggO = (ggI * (input - target)) * 2.

    return gI, None, ggO, None, None


def nllloss_double_backwards(ctx, ggI):
    t = ctx.saved_tensors
    target = t[1]
    weights = ctx.additional_args[1]
    size_average = ctx.additional_args[0]
    ignore_index = ctx.additional_args[3]
    reduce = ctx.additional_args[4]

    gI = None

    # can't scatter/gather on indices outside of range, let's just put them in range
    # and 0 out the weights later (so it doesn't matter where in range we put them)
    target_mask = target == ignore_index
    safe_target = target.clone()
    safe_target.masked_fill_(target_mask, 0)

    if weights.dim() == 0:
        weights_to_scatter = torch.ones_like(safe_target)
    else:
        weights_maybe_resized = weights
        while weights_maybe_resized.dim() < target.dim():
            weights_maybe_resized = weights_maybe_resized.unsqueeze(1)

        weights_maybe_resized = weights_maybe_resized.expand(weights.size()[0:1] + target.size()[1:])
        weights_to_scatter = weights_maybe_resized.gather(0, safe_target)

    weights_to_scatter.masked_fill_(target_mask, 0)
    divisor = weights_to_scatter.sum() if size_average and reduce else 1
    weights_to_scatter = -1 * weights_to_scatter / divisor
    zeros = torch.zeros_like(ggI)
    mask = zeros.scatter_(1, safe_target.unsqueeze(1), weights_to_scatter.unsqueeze(1))

    if reduce:
        ggO = (ggI * mask).sum()
    else:
        ggO = (ggI * mask).sum(dim=1)

    return gI, None, ggO, None, None, None


def smoothl1loss_double_backwards(ctx, ggI):
    size_average = ctx.additional_args[0]
    input, target, gO = ctx.saved_tensors
    div_factor = input.nelement() if size_average else 1

    input_sub_target = input - target
    small_error_mask = (input_sub_target.abs() < 1)
    large_error_mask = (small_error_mask == 0)
    large_error_pos_mask = (((input_sub_target > 0) + large_error_mask) == 2).type_as(ggI)
    large_error_neg_mask = (((input_sub_target <= 0) + large_error_mask) == 2).type_as(ggI)
    small_error_mask = small_error_mask.type_as(ggI)

    gI = small_error_mask * ggI * gO / div_factor
    ggO = (ggI * (input_sub_target * small_error_mask + large_error_pos_mask - large_error_neg_mask)).sum() / div_factor

    return gI, None, ggO, None, None, None


def softmarginloss_double_backwards(ctx, ggI):
    size_average = ctx.additional_args[0]
    input, target, gO = ctx.saved_tensors
    div_factor = input.nelement() if size_average else 1

    t0 = (1 + (-target * input).exp()).pow(-1)
    t1 = (-target * (-target * input).exp())
    first_deriv = t0 * t1

    gI = -1 * gO * ggI / div_factor * (first_deriv.pow(2) + first_deriv * target)
    ggO = (ggI * first_deriv).sum() / div_factor

    return gI, None, ggO, None, None, None


double_backwards_fns = {
    'ELU': elu_double_backwards,
    'GatedLinear': gatedlinear_double_backwards,
    'Hardshrink': hardshrink_double_backwards,
    'Hardtanh': hardtanh_double_backwards,
    'LeakyReLU': leakyrelu_double_backwards,
    'LogSigmoid': logsigmoid_double_backwards,
    'Softplus': softplus_double_backwards,
    'Softshrink': softshrink_double_backwards,
    'Threshold': threshold_double_backwards,
    'KLDivLoss': klddivloss_double_backwards,
    'L1Loss': l1loss_double_backwards,
    'MSELoss': mseloss_double_backwards,
    'NLLLoss': nllloss_double_backwards,
    'NLLLoss2d': nllloss_double_backwards,
    'SmoothL1Loss': smoothl1loss_double_backwards,
    'SoftMarginLoss': softmarginloss_double_backwards,
}
