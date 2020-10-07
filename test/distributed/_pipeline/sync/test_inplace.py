# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import pytest
import torch
from torch import nn

from torch.distributed._pipeline.sync import Pipe


def test_inplace_on_requires_grad():
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(inplace=True))
    model = Pipe(model, [1, 1], devices=["cpu", "cpu"], checkpoint="always")

    x = torch.rand(1)
    y = model(x)

    message = r"a leaf Variable that requires grad .* used in an in-place operation."
    with pytest.raises(RuntimeError, match=message):
        y.backward()


@pytest.mark.xfail(strict=True)
def test_inplace_on_not_requires_grad():
    # In-place operation on a tensor not requiring grad doesn't cause a
    # RuntimeError. Currently, we cannot detect this case.
    model = nn.Sequential(nn.ReLU(inplace=True))
    model = Pipe(model, [1], devices=["cpu"], checkpoint="always")

    x = torch.rand(1)
    y = model(x)
    del model

    message = r"a leaf Variable that requires grad .* used in an in-place operation."
    with pytest.raises(RuntimeError, match=message):
        y.backward()


@pytest.mark.xfail(strict=True)
def test_inplace_incorrect_grad():
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

    output = model((foo, bar))
    del model
    output.backward()

    # The gradient of 'foo' should be 2, but it is 3 actually because
    # bar.add_(1) was executed twice due to checkpointing.
    assert foo.grad.item() == 2.0
