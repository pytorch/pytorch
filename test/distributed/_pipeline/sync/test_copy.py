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

from torch.distributed._pipeline.sync.copy import Copy, Wait
from torch.distributed._pipeline.sync.stream import CPUStream, current_stream, get_device, is_cuda, new_stream, use_stream

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")


def _test_copy_wait(prev_stream, next_stream, cuda_sleep=None):
    device = get_device(prev_stream)

    with use_stream(prev_stream):
        if is_cuda(prev_stream):
            cuda_sleep(0.5)
        x = torch.ones(100, device=device, requires_grad=True)

    (y,) = Copy.apply(prev_stream, next_stream, x)
    (y,) = Wait.apply(prev_stream, next_stream, x)

    with use_stream(next_stream):
        assert torch.allclose(y.sum(), torch.tensor(100.0, device=device))
        y.norm().backward()
    with use_stream(prev_stream):
        assert torch.allclose(x.grad.sum(), torch.tensor(10.0, device=device))


def test_copy_wait_cpu_cpu():
    prev_stream = CPUStream
    next_stream = CPUStream
    _test_copy_wait(prev_stream, next_stream)


@skip_if_no_cuda
def test_copy_wait_cpu_cuda(cuda_sleep):
    prev_stream = CPUStream
    next_stream = current_stream(torch.device("cuda"))
    _test_copy_wait(prev_stream, next_stream, cuda_sleep)


@skip_if_no_cuda
def test_copy_wait_cuda_cpu(cuda_sleep):
    prev_stream = current_stream(torch.device("cuda"))
    next_stream = CPUStream
    _test_copy_wait(prev_stream, next_stream, cuda_sleep)


@skip_if_no_cuda
def test_copy_wait_cuda_cuda(cuda_sleep):
    prev_stream = current_stream(torch.device("cuda"))
    next_stream = new_stream(torch.device("cuda"))
    _test_copy_wait(prev_stream, next_stream, cuda_sleep)


def test_wait_multiple_tensors():
    a = torch.rand(1, requires_grad=True)
    b = torch.rand(1, requires_grad=True)

    a, b = Wait.apply(CPUStream, CPUStream, a, b)

    assert a.grad_fn is b.grad_fn
    assert a.grad_fn.__class__ is Wait._backward_cls
