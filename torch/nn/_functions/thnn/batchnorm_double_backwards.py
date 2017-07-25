from torch.autograd.variable import Variable
from functools import reduce
from operator import mul


def sum_exclude_dim1(to_sum, keepdim=True):
    to_sum = to_sum.sum(dim=0, keepdim=True)
    dim = 2
    for dim in range(2, to_sum.dim()):
        to_sum = to_sum.sum(dim=dim, keepdim=True)
    return to_sum


# because gamma/ggG/ggB are 1-dimensional and represent dim==1, we can't
# do a straight expansion because it won't follow the broadcasting rules.
def expand_as_dim1(src, target):
    src_expanded = src
    while len(src_expanded.size()) < len(target.size()) - 1:
        src_expanded = src_expanded.unsqueeze(1)
    return src_expanded.expand_as(target)


def batchnorm_double_backwards_fn(input, gamma, ggI, ggG, ggB, gO, eps):
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
    mu = sum_exclude_dim1(input).div(M)
    input_sub_mu = input - mu
    sigma2_eps = sum_exclude_dim1(input_sub_mu.pow(2)).div(M) + eps
    sigma2_eps_neg_1_2 = (sigma2_eps).pow(-1. / 2)
    sigma2_eps_neg_3_2 = (sigma2_eps).pow(-3. / 2)

    # calculate gI
    input_mu_sigma2_neg_3_2 = (input_sub_mu * sigma2_eps_neg_3_2)
    gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu)

    # start with contribution of input term
    gI = None
    if ggI is not None:
        ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu)
        all_sub = (1. / M * sum_exclude_dim1(ggI) * sum_exclude_dim1(gO) - sum_exclude_dim1(gO * ggI) +
                   3. / M * (sigma2_eps).pow(-1) * gOinmu_sum * ggIinmu_sum)
        gI_0t = 1. / M * input_mu_sigma2_neg_3_2 * all_sub
        gI_1t = 1. / M * (ggIinmu_sum * sigma2_eps_neg_3_2) * (1. / M * sum_exclude_dim1(gO) - gO)
        gI_2t = 1. / M * (gOinmu_sum * sigma2_eps_neg_3_2) * (1. / M * sum_exclude_dim1(ggI) - ggI)
        gI = gamma_expanded * (gI_0t + gI_1t + gI_2t)

    # add contribution of gamma term to gI
    if affine and ggG is not None:
        t0 = gO * sigma2_eps_neg_1_2
        t1 = -1. / M * sigma2_eps_neg_1_2 * sum_exclude_dim1(gO)
        t2 = -1. / M * input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu)
        gI_G_term = ggG_expanded * (t0 + t1 + t2)
        gI = gI + gI_G_term if gI is not None else gI_G_term

    # this is the first backward's grad_input
    def first_back_grad_input(gO, gamma):
        h0 = (gamma / (sigma2_eps).sqrt()).div(M)
        h1 = M * gO - sum_exclude_dim1(gO) - input_sub_mu.div(sigma2_eps) * sum_exclude_dim1(gO * input_sub_mu)
        return h0 * h1

    # calculate gG
    gG = None
    if affine and ggI is not None:
        # gG is just the first backwards with the gamma term removed (then shaped properly)
        gG = ggI * first_back_grad_input(gO, 1)
        gG = sum_exclude_dim1(gG, keepdim=False)

    # calculate gB
    gB = None

    # calculate ggO
    ggO = None
    # contribution of input term
    if ggI is not None:
        ggO = first_back_grad_input(ggI, gamma_expanded)
    if ggG is not None:
        ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2
        ggO = ggO + ggO_G_term if ggO is not None else ggO_G_term
    if ggB is not None:
        ggO_B_term = ggB_expanded
        ggO = ggO + ggO_B_term if ggO is not None else ggO_B_term

    return gI, gG, gB, ggO
