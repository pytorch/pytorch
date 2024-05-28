# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
from itertools import chain

import pytest

import torch
from torch import nn, optim

from torch.distributed.pipeline.sync.batchnorm import DeferredBatchNorm
from torch.testing._internal.common_utils import run_tests

CHUNKS = 4


def tilt_dist(input):
    # Tilt variance by channel.
    rgb = input.transpose(0, 1)
    rgb[0] *= 1
    rgb[1] *= 10
    rgb[2] *= 100

    # Tilt mean by single batch.
    for i, single in enumerate(input):
        single += 2**i

    return input


def chunked_forward(model, input, chunks=CHUNKS):
    output_chunks = []

    for chunk in input.chunk(chunks):
        output_chunks.append(model(chunk))

    return torch.cat(output_chunks)


@pytest.mark.parametrize("chunks", [1, 4])
@pytest.mark.parametrize("input_requires_grad", [True, False])
def test_transparency(chunks, input_requires_grad):
    bn = nn.BatchNorm2d(3)
    dbn = DeferredBatchNorm.convert_deferred_batch_norm(deepcopy(bn), chunks=chunks)

    input1 = torch.rand(16, 3, 224, 224)
    input1 = tilt_dist(input1)
    input2 = input1.clone()
    input1.requires_grad = input_requires_grad
    input2.requires_grad = input_requires_grad

    output1 = chunked_forward(bn, input1, chunks=chunks)
    output2 = chunked_forward(dbn, input2, chunks=chunks)

    assert torch.allclose(output1, output2, atol=1e-4)

    output1.mean().backward()
    output2.mean().backward()

    assert torch.allclose(bn.weight.grad, dbn.weight.grad, atol=1e-4)

    if input_requires_grad:
        assert input1.grad is not None
        assert input2.grad is not None
        assert torch.allclose(input1.grad, input2.grad, atol=1e-4)


@pytest.mark.parametrize("momentum", [0.1, None])
def test_running_stats(momentum):
    bn = nn.BatchNorm2d(3, momentum=momentum)
    dbn = DeferredBatchNorm.convert_deferred_batch_norm(deepcopy(bn), chunks=CHUNKS)

    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    bn(input)
    chunked_forward(dbn, input)

    assert torch.allclose(bn.running_mean, dbn.running_mean, atol=1e-4)
    assert torch.allclose(bn.running_var, dbn.running_var, atol=1e-4)


def test_convert_deferred_batch_norm():
    bn = nn.BatchNorm2d(3, track_running_stats=False)
    bn = DeferredBatchNorm.convert_deferred_batch_norm(bn, chunks=CHUNKS)
    assert type(bn) is nn.BatchNorm2d  # because of track_running_stats=False

    dbn = DeferredBatchNorm(3, chunks=CHUNKS)
    dbn_again = DeferredBatchNorm.convert_deferred_batch_norm(dbn, chunks=CHUNKS)
    assert dbn is dbn_again

    dbn_again = DeferredBatchNorm.convert_deferred_batch_norm(dbn, chunks=CHUNKS + 1)
    assert dbn is not dbn_again  # because of different chunks


def test_eval():
    bn = nn.BatchNorm2d(3)
    dbn = DeferredBatchNorm.convert_deferred_batch_norm(deepcopy(bn), chunks=CHUNKS)

    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    bn(input)
    chunked_forward(dbn, input)

    bn.eval()
    dbn.eval()

    assert torch.allclose(bn(input), dbn(input), atol=1e-4)


def test_optimize():
    bn = nn.BatchNorm2d(3)
    dbn = DeferredBatchNorm.convert_deferred_batch_norm(deepcopy(bn), chunks=CHUNKS)

    opt = optim.SGD(chain(bn.parameters(), dbn.parameters()), lr=1.0)

    for i in range(5):
        input = torch.rand(16, 3, 224, 224)
        input = tilt_dist(input)

        # train
        y = bn(input)
        a = y.sum()
        a.backward()

        y = chunked_forward(dbn, input)
        b = y.sum()
        b.backward()

        opt.step()

        # eval
        bn.eval()
        dbn.eval()

        with torch.no_grad():
            assert torch.allclose(bn(input), dbn(input), atol=1e-1 * (10**i))


def test_conv_bn():
    bn = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
    dbn = DeferredBatchNorm.convert_deferred_batch_norm(deepcopy(bn), chunks=CHUNKS)

    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)

    opt = optim.SGD(chain(bn.parameters(), dbn.parameters()), lr=0.1)

    # 1st step
    a = bn(input)
    b = chunked_forward(dbn, input)

    # Outputs are different. (per-mini-batch vs. per-micro-batch)
    assert not torch.allclose(a, b)

    a.sum().backward()
    b.sum().backward()
    opt.step()
    opt.zero_grad()

    # Conv layers are also trained differently because of their different outputs.
    assert not torch.allclose(bn[0].weight, dbn[0].weight)

    # But BNs track identical running stats.
    assert torch.allclose(bn[1].running_mean, dbn[1].running_mean, atol=1e-4)
    assert torch.allclose(bn[1].running_var, dbn[1].running_var, atol=1e3)

    # 2nd step
    a = bn(input)
    b = chunked_forward(dbn, input)
    a.sum().backward()
    b.sum().backward()

    # BNs can't track identical running stats due to the different conv layers.
    assert not torch.allclose(bn[1].running_mean, dbn[1].running_mean, atol=1e-4)
    assert not torch.allclose(bn[1].running_var, dbn[1].running_var, atol=1e3)


def test_input_requiring_grad():
    dbn = DeferredBatchNorm(3, chunks=CHUNKS)

    input = torch.rand(16, 3, 224, 224)
    input = tilt_dist(input)
    input.requires_grad = True

    chunked_forward(dbn, input)

    assert not dbn.sum.requires_grad
    assert dbn.sum.grad_fn is None


if __name__ == "__main__":
    run_tests()
