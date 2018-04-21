import sys
import tempfile
import unittest
from copy import deepcopy
from itertools import product

import torch
import torch.cuda
from common import TestCase, to_gpu, freeze_rng_state, is_iterable
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors, contiguous
import torch.backends.cudnn

# tarfile module tries to obtain a file object name in python 3.3
if sys.version_info[:2] == (3, 3):
    TemporaryFile = tempfile.NamedTemporaryFile
else:
    TemporaryFile = tempfile.TemporaryFile

TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
TEST_CUDNN = TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1))
TEST_CUDNN_VERSION = TEST_CUDNN and torch.backends.cudnn.version()
PRECISION = 1e-5


def get_size_average(m):
    return getattr(m, 'size_average', False) or getattr(m, 'sizeAverage', False)


def get_weight(m):
    result = getattr(m, 'weight', None)
    if result is not None:
        return result
    return getattr(m, 'weights', None)

module_tests = [
    dict(
        module_name='Linear',
        constructor_args=(10, 8),
        input_size=(4, 10),
        reference_fn=lambda i, p: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8)
    ),
    dict(
        module_name='Linear',
        constructor_args=(10, 8, False),
        input_size=(4, 10),
        desc='no_bias',
        reference_fn=lambda i, p: torch.mm(i, p[0].t())
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2, 1),
        input_size=(2, 3, 4, 5),
        check_inplace=True,
        desc='threshold_value'
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2, 10),
        input_size=(2, 3, 4, 5),
        desc='large_value'
    ),
    dict(
        module_name='ReLU',
        input_size=(2, 3, 4, 5),
        check_inplace=True,
    ),
    dict(
        module_name='ReLU6',
        input_size=(2, 3, 4, 5),
        check_inplace=True,
    ),
    dict(
        module_name='RReLU',
        input_size=(1, 2, 2),
        test_cuda=False,
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        input_size=(4, 4, 5),
        desc='with_up_down',
        test_cuda=False,
    ),
    dict(
        module_name='Hardtanh',
        input_size=(3, 2, 5),
        reference_fn=lambda i, _: i.clamp(-1, 1),
    ),
    dict(
        module_name='Sigmoid',
        input_size=(2, 3, 4, 5)
    ),
    dict(
        module_name='Tanh',
        input_size=(2, 3, 4, 5)
    ),
    dict(
        module_name='Softmax',
        constructor_args=(1,),
        input_size=(10, 20),
        reference_fn=lambda i, _: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20)),
    ),
    dict(
        module_name='Softmax2d',
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i, _: torch.exp(i).div(torch.exp(i).sum(1, False)),
    ),
    dict(
        module_name='LogSoftmax',
        constructor_args=(1,),
        input_size=(10, 20),
        reference_fn=lambda i, _: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_(),
    ),
    dict(
        module_name='LogSoftmax',
        constructor_args=(1,),
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i, _: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(),
        desc='multiparam',
    ),
    dict(
        module_name='ELU',
        constructor_args=(2.,),
        input_size=(3, 2, 5),
    ),
    # TODO: reference function
    dict(
        module_name='Hardshrink',
        constructor_args=(2.,),
        input_size=(4, 3, 2, 4),
    ),
    dict(
        module_name='LeakyReLU',
        input_size=(3, 2, 5),
        check_inplace=True
    ),
    dict(
        module_name='LeakyReLU',
        constructor_args=(0.5,),
        input_size=(3, 2, 5),
        check_inplace=True,
        desc='with_negval'
    ),
    dict(
        module_name='LogSigmoid',
        input_size=(2, 3, 4),
        reference_fn=lambda i, _: i.sigmoid().log(),
    ),
    dict(
        module_name='Softplus',
        input_size=(10, 20),
        reference_fn=lambda i, _: torch.log(1 + torch.exp(i)),
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2,),
        input_size=(10, 20),
        reference_fn=lambda i, _: 1. / 2. * torch.log(1 + torch.exp(2 * i)),
        desc='beta',
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2, -100),
        input_size=(10, 20),
        reference_fn=(lambda i, _: ((i * 2) > -100).type_as(i) * i +
                                   ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log(1 + torch.exp(2 * i))),
        desc='beta_threshold',
    ),
    dict(
        module_name='Softshrink',
        input_size=(3, 2, 5),
    ),
    dict(
        module_name='Softshrink',
        constructor_args=(1,),
        input_size=(3, 2, 5),
        desc='lambda',
    ),
    dict(
        module_name='CrossMapLRN2d',
        constructor_args=(5, 5e-3, 1e-3, 2),
        input_size=(2, 3, 6, 6),
        check_gradgrad=False,
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4),
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='1d',
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4),
        desc='1d_multiparam',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4, 5),
        desc='2d',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4, 5),
        desc='2d_multiparam',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4, 5, 6),
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='3d',
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4, 5, 6),
        desc='3d_multiparam',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='Softsign',
        input_size=(3, 2, 5),
        reference_fn=lambda i, _: i.div(1 + torch.abs(i)),
    ),
    dict(
        module_name='Softmin',
        constructor_args=(1,),
        input_size=(10, 20),
    ),
    dict(
        module_name='Softmin',
        constructor_args=(1,),
        input_size=(2, 3, 5, 10),
        desc='multidim',
    ),
    dict(
        module_name='Tanhshrink',
        input_size=(2, 3, 4, 5)
    ),
]


def kldivloss_reference(input, target, size_average=True, reduce=True):
    safe_target = target * (target > 0).type_as(target)
    safe_target_log = (safe_target + (target <= 0).type_as(target)).log()
    result = safe_target * (safe_target_log - input)
    if reduce and size_average:
        return result.mean()
    elif reduce:
        return result.sum()
    return result


def nlllossNd_reference(input, target, weight=None, ignore_index=-100,
                        size_average=True, reduce=True):
    assert input.dim() >= 3
    N = input.size(0)
    C = input.size(1)
    out_size = (N,) + input.size()[2:]
    output = torch.zeros(out_size).type_as(input)

    if weight is None:
        weight = torch.ones(C).type_as(input)
    total_weight = 0
    for tup in product(*[range(size) for size in out_size]):
        t_nx = target[tup]
        norm = 0. if ignore_index == t_nx else weight[t_nx].item()
        input_index = list(tup)
        input_index.insert(1, t_nx)
        output[tup] = -input[tuple(input_index)] * norm
        total_weight += norm

    if reduce and size_average:
        return output.sum() / total_weight
    elif reduce:
        return output.sum()
    return output


def nllloss_reference(input, target, weight=None, ignore_index=-100,
                      size_average=True, reduce=True):

    def nll_loss_helper(input, target, weight, ignore_index):
        if target == ignore_index:
            return (0, 0)
        norm = 1 if weight is None else weight[target]
        result = -input[target] * norm
        return (result, norm)

    losses_and_weights = [nll_loss_helper(i, t, weight, ignore_index)
                          for i, t in zip(input, target)]
    losses, weights = zip(*losses_and_weights)
    losses_tensor = input.new_tensor(losses)
    if reduce and size_average:
        return sum(losses_tensor) / sum(weights)
    elif reduce:
        return sum(losses_tensor)
    else:
        return losses_tensor


def smoothl1loss_reference(input, target, size_average=True, reduce=True):
    abs_diff = (input - target).abs()
    ge_one_mask = (abs_diff >= 1).type_as(abs_diff)
    lt_one_mask = (abs_diff < 1).type_as(abs_diff)
    output = ge_one_mask * (abs_diff - 0.5) + lt_one_mask * 0.5 * (abs_diff ** 2)
    if reduce and size_average:
        return output.mean()
    elif reduce:
        return output.sum()
    return output


def _multilabelmarginloss_reference(input, target):
    targets = []
    for target_index in target:
        if target_index < 0:
            break
        targets.append(target_index)

    sum = 0
    for target_index in targets:
        for i in range(0, len(input)):
            if i not in targets:
                sum += max(0, 1 - input[target_index] + input[i])

    return sum


def multilabelmarginloss_reference(input, target, size_average=True, reduce=True):
    if input.dim() == 1:
        n = 1
        dim = input.size(0)
        output = input.new(n).zero_()
        output[0] = _multilabelmarginloss_reference(input, target)
    else:
        n = input.size(0)
        dim = input.size(1)
        output = input.new(n).zero_()
        for i in range(0, n):
            output[i] = _multilabelmarginloss_reference(input[i], target[i])

    if reduce and size_average:
        return output.mean() / dim
    elif reduce:
        return output.sum() / dim
    return output / dim


def hingeembeddingloss_reference(input, target, margin=1.0, size_average=True, reduce=True):
    margin_clamp = (margin - input).clamp(min=0).type_as(input)
    output = torch.where(target == 1, input, margin_clamp)

    if reduce and size_average:
        return output.mean()
    elif reduce:
        return output.sum()
    return output


def softmarginloss_reference(input, target, size_average=True, reduce=True):
    output = (1 + (-input * target).exp()).log()

    if reduce and size_average:
        return output.mean()
    elif reduce:
        return output.sum()
    return output


def _multimarginloss_reference(input, target_idx, p, margin, weight):
    if weight is None:
        weight = input.new(len(input)).fill_(1)

    output = 0
    for i in range(0, len(input)):
        if i != target_idx:
            output += max(0, weight[target_idx] * (margin - input[target_idx] + input[i]) ** p)
    return output


def multimarginloss_reference(input, target, p=1, margin=1, weight=None, size_average=True,
                              reduce=True):
    if input.dim() == 1:
        n = 1
        dim = input.size(0)
        output = input.new(1)
        output[0] = _multimarginloss_reference(input, target[0], p, margin, weight) / dim
        return output
    else:
        n = input.size(0)
        dim = input.size(1)
        output = input.new(n)
        for x in range(0, n):
            output[x] = _multimarginloss_reference(input[x], target[x], p, margin, weight)

        if reduce and size_average:
            return output.mean() / dim
        elif reduce:
            return output.sum() / dim
        return output / dim


def cosineembeddingloss_reference(input1, input2, target, margin=0, size_average=True, reduce=True):
    def _cos(a, b):
        cos = a.new(a.size(0))
        for i in range(0, a.size(0)):
            cos[i] = (a[i] * b[i]).sum() / ((((a[i] * a[i]).sum() + 1e-12) * ((b[i] * b[i]).sum() + 1e-12)) ** 0.5)
        return cos

    output = torch.where(target == 1, 1 - _cos(input1, input2), (_cos(input1, input2) - margin).clamp(min=0))

    if reduce and size_average:
        return output.mean()
    elif reduce:
        return output.sum()
    return output


def tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False,
                                size_average=True, reduce=True):
    d_p = torch.pairwise_distance(anchor, positive, p, eps)
    d_n = torch.pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = torch.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    output = torch.clamp(margin + d_p - d_n, min=0.0)
    if reduce and size_average:
        return output.mean()
    elif reduce:
        return output.sum()
    return output


def marginrankingloss_reference(input1, input2, target, margin=0, size_average=True, reduce=True):
    output = (-target * (input1 - input2) + margin).clamp(min=0)
    if reduce and size_average:
        return output.mean()
    elif reduce:
        return output.sum()
    return output


loss_reference_fns = {
    'KLDivLoss': kldivloss_reference,
    'NLLLoss': nllloss_reference,
    'NLLLossNd': nlllossNd_reference,
    'SmoothL1Loss': smoothl1loss_reference,
    'MultiLabelMarginLoss': multilabelmarginloss_reference,
    'HingeEmbeddingLoss': hingeembeddingloss_reference,
    'SoftMarginLoss': softmarginloss_reference,
    'MultiMarginLoss': multimarginloss_reference,
    'CosineEmbeddingLoss': cosineembeddingloss_reference,
    'TripletMarginLoss': tripletmarginloss_reference,
    'MarginRankingLoss': marginrankingloss_reference,
}


criterion_tests = [
    dict(
        module_name='L1Loss',
        input_size=(2, 3, 4),
        target_size=(2, 3, 4),
        reference_fn=lambda i, t, _: 1. / i.numel() *
        sum((a - b).abs().sum() for a, b in zip(i, t)),
    ),
    dict(
        module_name='NLLLoss',
        input_fn=lambda: torch.rand(15, 10).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, size_average=get_size_average(m)),
        check_no_size_average=True
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, True, 2),
        input_fn=lambda: torch.rand(15, 10).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, _: nllloss_reference(i, t, ignore_index=2),
        desc='ignore_index'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m)),
        desc='weights',
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10), True, 2),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m), ignore_index=2),
        desc='weights_ignore_index'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10), True, -1),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10 + 1).floor().long() - 1,
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m), ignore_index=-1),
        desc='weights_ignore_index_neg'
    ),
    dict(
        module_name='KLDivLoss',
        input_fn=lambda: torch.rand(10, 10).log(),
        target_fn=lambda: torch.rand(10, 10),
        reference_fn=lambda i, t, m:
            kldivloss_reference(i, t, get_size_average(m), reduce=True),
        check_no_size_average=True,
    ),
    dict(
        module_name='MSELoss',
        input_size=(2, 3, 4, 5),
        target_size=(2, 3, 4, 5),
        reference_fn=lambda i, t, m: (i - t).abs().pow(2).sum() / (i.numel() if get_size_average(m) else 1),
        check_no_size_average=True,
    ),
    dict(
        module_name='BCELoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
        reference_fn=lambda i, t, m: -(t * i.log() + (1 - t) * (1 - i).log()).sum() /
            (i.numel() if get_size_average(m) else 1),
        check_gradgrad=False,
    ),
    dict(
        module_name='BCELoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
        reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() /
            (i.numel() if get_size_average(m) else 1),
        desc='weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(15, 10),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        input_size=(15, 10),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='weights',
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        input_size=(10,),
        target_fn=lambda: torch.randn(10).gt(0).double().mul_(2).sub(1),
        reference_fn=lambda i, t, m:
            hingeembeddingloss_reference(i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        input_size=(10,),
        target_fn=lambda: torch.randn(10).gt(0).double().mul_(2).sub(1),
        reference_fn=lambda i, t, m:
            hingeembeddingloss_reference(i, t, margin=0.5, size_average=get_size_average(m)),
        desc='margin',
        check_no_size_average=True,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(10,),
        target_fn=lambda: torch.rand(10).mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            multilabelmarginloss_reference(i, t, size_average=get_size_average(m)),
        desc="1d",
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            multilabelmarginloss_reference(i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(2).floor(),
        reference_fn=lambda i, t, m: -(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()).sum() / i.numel(),
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(10,),
        target_fn=lambda: torch.rand(1).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, size_average=get_size_average(m)),
        desc='1d',
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(2,),
        input_fn=lambda: torch.rand(5, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, p=2, size_average=get_size_average(m)),
        desc='p',
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(1, 0.5),
        legacy_constructor_args=(1, None, 0.5),
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, margin=0.5, size_average=get_size_average(m)),
        desc='margin',
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(1, 1, torch.rand(10)),
        legacy_constructor_args=(1, torch.rand(10)),
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, weight=get_weight(m), size_average=get_size_average(m)),
        desc='weights',
        check_no_size_average=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(5, 10),
        target_size=(5, 10),
        check_no_size_average=True,
        reference_fn=lambda i, t, m:
            smoothl1loss_reference(i, t, size_average=get_size_average(m)),
    ),
    dict(
        module_name='SoftMarginLoss',
        input_size=(5, 5),
        target_fn=lambda: torch.randn(5, 5).sign(),
        reference_fn=lambda i, t, m:
            softmarginloss_reference(i, t, size_average=get_size_average(m)),
        check_no_size_average=True,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)),
        target_fn=lambda: torch.randn(15).sign(),
        reference_fn=lambda i, t, m:
            cosineembeddingloss_reference(i[0], i[1], t, size_average=get_size_average(m)),
        check_no_size_average=True,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        constructor_args=(0.7,),
        input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)),
        target_fn=lambda: torch.randn(15).sign(),
        reference_fn=lambda i, t, m:
            cosineembeddingloss_reference(i[0], i[1], t, margin=0.7, size_average=get_size_average(m)),
        desc='margin',
        check_no_size_average=True,
    ),
    dict(
        module_name='MarginRankingLoss',
        input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target_fn=lambda: torch.randn(50).sign(),
        reference_fn=lambda i, t, m:
            marginrankingloss_reference(i[0], i[1], t, size_average=get_size_average(m)),
        check_no_size_average=True,
    ),
    dict(
        module_name='MarginRankingLoss',
        constructor_args=(0.5,),
        input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target_fn=lambda: torch.randn(50).sign(),
        reference_fn=lambda i, t, m:
            marginrankingloss_reference(i[0], i[1], t, margin=0.5, size_average=get_size_average(m)),
        desc='margin',
        check_no_size_average=True,
    ),
]


class NNTestCase(TestCase):

    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)

    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        else:
            return tuple(self._flatten_tensors(a) for a in x)

    def _zero_grad_input(self, input):
        if isinstance(input, torch.Tensor):
            if input.requires_grad and input.grad is not None:
                input.grad.zero_()
                input.grad.detach_()
        else:
            for i in input:
                self._zero_grad_input(i)

    def _analytical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nelement()

        if jacobian_input:
            jacobian_inp = self._jacobian(input, output_size)
            flat_jacobian_input = list(iter_tensors(jacobian_inp))

        if jacobian_parameters:
            num_param = sum(p.numel() for p in self._get_parameters(module)[0])
            jacobian_param = torch.zeros(num_param, output_size)

        for i in range(output_size):
            _, d_param = self._get_parameters(module)
            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1

            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # Tensors will accumulate gradient from multiple steps
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)

            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x.contiguous().view(-1)
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)

        res = tuple()
        if jacobian_input:
            res += jacobian_inp,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    def _numerical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        def fw(input):
            return self._forward(module, input).detach()

        res = tuple()
        input = contiguous(input)
        if jacobian_input:
            res += get_numerical_jacobian(fw, input, input, eps=1e-6),
        if jacobian_parameters:
            param, _ = self._get_parameters(module)
            res += torch.cat([get_numerical_jacobian(fw, input, p, eps=1e-6) for p in param], 0),
        return res

    def check_jacobian(self, module, input, jacobian_input=True):
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = list(iter_tensors(analytical))
        numerical_t = list(iter_tensors(numerical))

        # TODO: compare structure
        self.assertLessEqual(
            max(a.add(-1, n).abs().max() for a, n in zip(analytical_t, numerical_t)),
            PRECISION
        )

    def check_criterion_jacobian(self, criterion, input, target):
        eps = 1e-6
        self._forward_criterion(criterion, input, target)
        analytical_d_x = self._backward_criterion(criterion, input, target)
        numerical_d_x = deepcopy(analytical_d_x)

        input_t = iter_tensors(input)
        numerical_t = iter_tensors(numerical_d_x)
        for x, d_x in zip(input_t, numerical_t):
            x = x.view(-1)
            d_x = d_x.view(-1)
            for i in range(x.nelement()):
                original = x[i].item()
                x[i] = original + eps
                fx1 = self._forward_criterion(criterion, input, target)
                x[i] = original - eps
                fx2 = self._forward_criterion(criterion, input, target)
                deriv = (fx1 - fx2) / (2. * eps)
                d_x[i] = float(deriv)
                x[i] = original

        # TODO: check structure
        analytical_t = list(iter_tensors(analytical_d_x))
        numerical_t = list(iter_tensors(numerical_d_x))

        self.assertLessEqual(
            max(a.add(-1, n).abs().max() for a, n in zip(analytical_t, numerical_t)),
            PRECISION
        )


class TestBase(object):

    _required_arg_names = {'constructor_args', 'input'}

    def __init__(self, constructor, desc='', reference_fn=None, fullname=None, **kwargs):
        self.desc = desc
        self.fullname = fullname
        self.constructor = constructor
        self.reference_fn = reference_fn
        for name in self._required_arg_names:
            if name not in kwargs and name + '_fn' not in kwargs and name + '_size' not in kwargs:
                if name == 'constructor_args':
                    kwargs['constructor_args'] = tuple()
                else:
                    raise ValueError("{}: Specify {} by a value, a function to generate it, or it's size!"
                                     .format(self.get_name(), name))
        self._extra_kwargs = kwargs
        self._arg_cache = {}

    def get_name(self):
        if self.fullname is not None:
            return 'test_' + self.fullname

        test_name = 'test_' + self.constructor.__name__
        if self.desc:
            test_name += '_' + self.desc
        return test_name

    def _unpack(self, value):
        if isinstance(value, torch.Tensor):
            return value
        elif is_iterable(value):
            return type(value)(self._unpack(v) for v in value)
        else:
            return value

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', True)

    def _get_arg(self, name, unpack):
        assert name in self._required_arg_names

        if name not in self._arg_cache:
            fn_name = name + '_fn'
            size_name = name + '_size'

            if name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[name]
            elif fn_name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[fn_name]()
            else:
                assert size_name in self._extra_kwargs

                def map_tensor_sizes(sizes):
                    if isinstance(sizes, list):
                        return [map_tensor_sizes(s) for s in sizes]
                    elif isinstance(sizes, torch.Tensor):
                        return sizes.double()
                    else:
                        return torch.randn(sizes)

                self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])

        return self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name]

    def _get_input(self, unpack=True):
        return self._get_arg('input', unpack)

    def __call__(self, test_case):
        raise NotImplementedError


class ModuleTest(TestBase):

    def __init__(self, *args, **kwargs):
        super(ModuleTest, self).__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.should_test_pickle = kwargs.get('pickle', True)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.FIXME_no_cuda_gradgrad_comparison = \
            kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        self.precision = kwargs.get('precision', 2e-4)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        if self.reference_fn is not None:
            out = test_case._forward(module, input)
            ref_input = deepcopy(input)
            expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0])
            test_case.assertEqual(out, expected_out)
        self.test_noncontig(test_case, module, input)

        if self.should_test_pickle:
            # TODO: do this with in-memory files as soon as torch.save will support it
            with TemporaryFile() as f:
                test_case._forward(module, input)
                torch.save(module, f)
                f.seek(0)
                module_copy = torch.load(f)
                test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))

        self._do_test(test_case, module, input)

    def noncontiguize(self, obj):
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        tensor = obj
        ndim = tensor.dim()
        noncontig = torch.stack([torch.zeros_like(tensor), tensor], ndim).select(ndim, 1).detach()
        assert noncontig.numel() == 1 or not noncontig.is_contiguous()
        noncontig.requires_grad = tensor.requires_grad
        return noncontig

    def test_noncontig(self, test_case, module, input):
        # check no scalars, can't make non-contig
        if isinstance(input, torch.Tensor) and input.dim() == 0:
            return
        if any(i.dim() == 0 for i in input if isinstance(i, torch.Tensor)):
            return

        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            grad_output = output.new(output.shape).normal_()
            output = output.clone()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])

        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for contig_i, contig_g in product((True, False), repeat=2):
            i = input if contig_i else nc_input
            go = grad_output if contig_g else nc_grad_output
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                try:
                    out = test_case._forward(module, i)
                except Exception:
                    # Some modules will fail because of non contiguous inputs and we're ok with that
                    continue
                grad = test_case._backward(module, i, out, go)

                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, 1e-4)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {'torch.DoubleTensor': torch.cuda.FloatTensor}
            gpu_input = to_gpu(cpu_input, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0]):
                gpu_p.data.copy_(cpu_p)

            test_case._zero_grad_input(cpu_input)
            test_case._zero_grad_input(gpu_input)
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_output = test_case._forward(cpu_module, cpu_input)
            gpu_output = test_case._forward(gpu_module, gpu_input)
            test_case.assertEqual(cpu_output, gpu_output, self.precision)

            # Run backwards on CPU and GPU and compare results
            for i in range(5):
                cpu_gradOutput = cpu_output.clone().normal_()
                gpu_gradOutput = cpu_gradOutput.type('torch.cuda.FloatTensor')
                cpu_gradInput = test_case._backward(cpu_module, cpu_input, cpu_output, cpu_gradOutput)
                gpu_gradInput = test_case._backward(gpu_module, gpu_input, gpu_output, gpu_gradOutput)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, self.precision)
                for cpu_d_p, gpu_d_p in zip(cpu_param[1], gpu_param[1]):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, self.precision)

            # Run double-backwards on CPU and GPU and compare results
            if self.check_gradgrad and not self.FIXME_no_cuda_gradgrad_comparison:
                cpu_output = cpu_module(cpu_input)
                gpu_output = gpu_module(gpu_input)

                cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output).detach()
                gpu_gradOutput.requires_grad = True

                cpu_gradInputs = torch.autograd.grad(
                    cpu_output,
                    (cpu_input,) + tuple(cpu_module.parameters()),
                    cpu_gradOutput,
                    create_graph=True)
                gpu_gradInputs = torch.autograd.grad(
                    gpu_output,
                    (gpu_input,) + tuple(gpu_module.parameters()),
                    gpu_gradOutput,
                    create_graph=True)

                for cpu_d_i, gpu_d_i in zip(cpu_gradInputs, gpu_gradInputs):
                    test_case.assertEqual(cpu_d_i, gpu_d_i, self.precision)

                # We mix output into the second backwards computation so that
                # torch.autograd.grad doesn't complain that some inputs
                # are unreachable (which can happen if you differentiate
                # only on the gradient.
                cpu_gg = torch.autograd.grad(
                    cpu_output.sum() + sum(map(lambda x: x.sum(), cpu_gradInputs)),
                    (cpu_input, cpu_gradOutput) + tuple(cpu_module.parameters()),
                    retain_graph=True)
                gpu_gg = torch.autograd.grad(
                    gpu_output.sum() + sum(map(lambda x: x.sum(), gpu_gradInputs)),
                    (gpu_input, gpu_gradOutput) + tuple(gpu_module.parameters()),
                    retain_graph=True)

                test_case.assertEqual(cpu_gradInput, gpu_gradInput, self.precision)
                for cpu_d_p, gpu_d_p in zip(cpu_gg, gpu_gg):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, self.precision)

            self.test_noncontig(test_case, gpu_module, gpu_input)
        except NotImplementedError:
            pass
        # TODO: remove this after CUDA scatter_ is implemented
        except AttributeError as e:
            if len(e.args) == 1 and "'FloatTensor' object has no attribute 'scatter_'" in e.args[0]:
                pass
            else:
                raise


class CriterionTest(TestBase):

    _required_arg_names = TestBase._required_arg_names.union({'target'})

    def __init__(self, *args, **kwargs):
        super(CriterionTest, self).__init__(*args, **kwargs)
        self.should_test_cuda = kwargs.get('test_cuda', True)

    def _get_target(self):
        return self._get_arg('target', True)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        # Check that these methods don't raise errors
        module.__repr__()
        str(module)

        target = self._get_target()

        if self.reference_fn is not None:
            out = test_case._forward_criterion(module, input, target)
            expected_out = self.reference_fn(deepcopy(input),
                                             deepcopy(target), module)
            if isinstance(expected_out, torch.Tensor):
                expected_out = expected_out.item()
            test_case.assertEqual(out, expected_out)

        test_case.check_criterion_jacobian(module, input, target)
        self._do_extra_tests(test_case, module, input, target)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {
                'torch.DoubleTensor': torch.cuda.FloatTensor,
            }
            gpu_input = to_gpu(cpu_input, type_map=type_map)

            cpu_target = self._get_target()
            gpu_target = to_gpu(cpu_target, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_output, gpu_output, 4e-4)

            gradOutput = torch.randn(())
            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target, gradOutput)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target, gradOutput)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 4e-4)
        except NotImplementedError:
            pass

    def _do_extra_tests(self, test_case, module, input, target):
        pass
