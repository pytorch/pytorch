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
import torch.cuda

from torch.distributed._pipeline.sync.microbatch import Batch, check, gather, scatter


def test_batch_atomic():
    x = torch.tensor(42)
    b = Batch(x)

    assert b.atomic

    assert b.tensor is x
    with pytest.raises(AttributeError):
        b.tensors

    assert list(b) == [x]
    assert len(b) == 1
    assert b[0] is x


def test_batch_non_atomic():
    x, y = torch.tensor(42), torch.tensor(21)
    b = Batch((x, y))

    assert not b.atomic

    with pytest.raises(AttributeError):
        b.tensor
    assert b.tensors == (x, y)

    assert list(b) == [x, y]
    assert len(b) == 2
    assert b[0] is x
    assert b[1] is y


def test_batch_call():
    a = Batch(torch.tensor(42))
    b = Batch((torch.tensor(42), torch.tensor(21)))

    def f(x):
        return x

    assert a.call(f).atomic
    assert not b.call(f).atomic


def test_batch_setitem_by_index():
    a = Batch(torch.tensor(42))
    b = Batch((torch.tensor(42), torch.tensor(21)))

    a[0] = torch.tensor(0)
    b[0] = torch.tensor(0)

    assert a.atomic
    assert a[0].item() == 0

    assert not b.atomic
    assert len(b) == 2
    assert b[0].item() == 0
    assert b[1].item() == 21


def test_batch_setitem_by_slice():
    a = Batch(torch.tensor(42))
    b = Batch((torch.tensor(42), torch.tensor(21)))

    a[:] = (torch.tensor(0),)
    b[:] = (torch.tensor(0),)

    assert a.atomic
    assert a[0].item() == 0

    assert not b.atomic
    assert len(b) == 1
    assert b[0].item() == 0


def test_check():
    check(torch.tensor(42))
    check((torch.tensor(4), torch.tensor(2)))

    with pytest.raises(TypeError):
        check(42)

    with pytest.raises(TypeError):
        check("str")

    with pytest.raises(TypeError):
        check((torch.tensor(4), 2))


def test_gather_tensors():
    a = torch.zeros(1, 1)
    b = torch.zeros(1, 1)

    ab = gather([Batch(a), Batch(b)])

    assert ab.size() == (2, 1)


def test_gather_tuples():
    a = (torch.zeros(1, 1), torch.zeros(2, 2))
    b = (torch.zeros(1, 1), torch.zeros(2, 2))

    ab = gather([Batch(a), Batch(b)])

    assert isinstance(ab, tuple)
    assert ab[0].size() == (2, 1)
    assert ab[1].size() == (4, 2)


def test_scatter_tensor():
    ab = torch.zeros(2, 1)

    a, b = scatter(ab, chunks=2)

    assert a.tensor.size() == (1, 1)
    assert b.tensor.size() == (1, 1)


def test_scatter_tuple():
    ab = (torch.zeros(2, 1), torch.zeros(4, 2))

    a, b = scatter(ab, chunks=2)

    assert a.tensors[0].size() == (1, 1)
    assert b.tensors[0].size() == (1, 1)
    assert a.tensors[1].size() == (2, 2)
    assert b.tensors[1].size() == (2, 2)
