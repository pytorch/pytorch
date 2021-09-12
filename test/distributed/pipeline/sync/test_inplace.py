# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync import Pipe


def test_inplace_on_requires_grad(setup_rpc):
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(inplace=True))
    model = Pipe(model, checkpoint="always")

    x = torch.rand(1)
    y = model(x).local_value()

    message = r"a leaf Variable that requires grad .* used in an in-place operation."
    with pytest.raises(RuntimeError, match=message):
        y.backward()


@pytest.mark.xfail(strict=True)
def test_inplace_on_not_requires_grad(setup_rpc):
    # In-place operation on a tensor not requiring grad doesn't cause a
    # RuntimeError. Currently, we cannot detect this case.
    model = nn.Sequential(nn.ReLU(inplace=True))
    model = Pipe(model, [1], devices=["cpu"], checkpoint="always")

    x = torch.rand(1)
    y = model(x).local_value()
    del model

    message = r"a leaf Variable that requires grad .* used in an in-place operation."
    with pytest.raises(RuntimeError, match=message):
        y.backward()


@pytest.mark.xfail(strict=True)
def test_inplace_incorrect_grad(setup_rpc):
    class M(nn.Module):
        def forward(self, foo_bar):
            # 'foo' requires grad but 'bar' does not. In-place operation on
            # 'bar' won't cause a RuntimeError.
            foo, bar = foo_bar

            # add_(1) is not idempotent, in contrast to relu_(). If it is
            # executed multiple times, it will accumulates each difference onto
            # 'bar'.
            bar.add_(1)

            # 'bar' is still captured by checkpointing. 'foo' will get
            # incorrect grad.
            return foo * bar

    model = nn.Sequential(M())
    model = Pipe(model, [1], devices=["cpu"], checkpoint="always")

    foo = torch.tensor([1.0], requires_grad=True)
    bar = torch.tensor([1.0])

    output = model((foo, bar)).local_value()
    del model
    output.backward()

    # The gradient of 'foo' should be 2, but it is 3 actually because
    # bar.add_(1) was executed twice due to checkpointing.
    assert foo.grad.item() == 2.0
