from torch.autograd.variable import Variable
from functools import reduce
from operator import mul


def sum_exclude_dim1(to_sum, keepdim=True):
    to_sum = to_sum.sum(dim=0, keepdim=keepdim)
    start_point_exclusive = 1 if keepdim else 0
    for dim in range(to_sum.dim() - 1, start_point_exclusive, -1):
        to_sum = to_sum.sum(dim=dim, keepdim=keepdim)
    return to_sum


# similar to expand_as below, but doesn't do the expand_as; operates as if
# reductions were done with keepdim=True
def unsqueeze_dim1(src, target):
    src_expanded = src
    while len(src_expanded.size()) < len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(1)
    if len(src_expanded.size()) == len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(0)
    return src_expanded


# because gamma/ggG/ggB are 1-dimensional and represent dim==1, we can't
# do a straight expansion because it won't follow the broadcasting rules.
def expand_as_dim1(src, target):
    src_expanded = src
    while len(src_expanded.size()) < len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(1)
    return src_expanded.expand_as(target)


def batchnorm_double_backwards_fn(input, gamma, ggI, ggG, ggB, gO, eps,
                                  save_mean, save_std, running_mean, running_var, training):
    affine = gamma is not None
    if affine:
        gamma_expanded = expand_as_dim1(gamma, input)

        if ggG is not None:
            ggG_expanded = expand_as_dim1(ggG, input)

        if ggB is not None:
            ggB_expanded = expand_as_dim1(ggB, input)
    else:
        gamma_expanded = 1

    # define some terms we will reuse
    M = reduce(mul, input.size()[0:1] + input.size()[2:])
    mu = unsqueeze_dim1(Variable(save_mean if training else running_mean), input)
    input_sub_mu = input - mu
    sigma2_eps_neg_1_2 = unsqueeze_dim1(Variable(save_std if training else (running_var + eps).pow(-1. / 2)), input)
    sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2)
    sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3)

    # calculate gI
    input_mu_sigma2_neg_3_2 = (input_sub_mu * sigma2_eps_neg_3_2)
    gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu)
    gO_sum = sum_exclude_dim1(gO)

    # start with contribution of input term
    gI = None
    if ggI is not None and training:
        ggI_sum = sum_exclude_dim1(ggI)
        ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu)
        all_sub = ((ggI_sum * gO_sum).div_(M)).sub_(sum_exclude_dim1(gO * ggI)).add_(
                  (sigma2_eps_neg_1 * gOinmu_sum * ggIinmu_sum).mul_(3. / M))
        gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(M)
        gI_1t = (ggIinmu_sum * sigma2_eps_neg_3_2).div_(M) * (gO_sum.div(M) - gO)
        gI_2t = (gOinmu_sum * sigma2_eps_neg_3_2).div_(M) * (ggI_sum.div(M) - ggI)
        gI = gamma_expanded * (gI_0t.add_(gI_1t).add_(gI_2t))

    # add contribution of gamma term to gI
    if affine and ggG is not None:
        if training:
            t0 = gO * sigma2_eps_neg_1_2
            t1 = (sigma2_eps_neg_1_2 * gO_sum).div_(-M)
            t2 = (input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu)).div_(-M)
            gI_G_term = ggG_expanded * (t0.add_(t1).add_(t2))
            gI = gI.add_(gI_G_term) if gI is not None else gI_G_term
        else:
            gI_G_term = ggG_expanded * sigma2_eps_neg_1_2 * gO
            gI = gI.add_(gI_G_term) if gI is not None else gI_G_term

    # this is the first backward's grad_input
    def first_back_grad_input(gO, gamma):
        h0 = (gamma * sigma2_eps_neg_1_2).div_(M)
        h1 = (M * gO).sub_(sum_exclude_dim1(gO)).sub_(
            input_sub_mu.mul(sigma2_eps_neg_1) * sum_exclude_dim1(gO * input_sub_mu))
        return h0 * h1

    # calculate gG
    gG = None
    if affine and ggI is not None:
        if training:
            # gG is just the first backwards with the gamma term removed (then shaped properly)
            gG = ggI * first_back_grad_input(gO, 1)
            gG = sum_exclude_dim1(gG, keepdim=False)
        else:
            gG = sum_exclude_dim1(ggI * gO * sigma2_eps_neg_1_2, keepdim=False)

    # calculate ggO
    ggO = None
    # contribution of input term
    if ggI is not None:
        if training:
            ggO = first_back_grad_input(ggI, gamma_expanded)
        else:
            ggO = ggI * sigma2_eps_neg_1_2 * gamma_expanded
    if ggG is not None:
        ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2
        ggO = ggO.add_(ggO_G_term) if ggO is not None else ggO_G_term
    if ggB is not None:
        ggO_B_term = ggB_expanded
        ggO = ggO.add_(ggO_B_term) if ggO is not None else ggO_B_term

    return gI, gG, ggO
