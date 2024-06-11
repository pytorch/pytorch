# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torch.distributed.pipeline.sync.stream import (
    CPUStream,
    current_stream,
    default_stream,
    get_device,
    is_cuda,
    new_stream,
    record_stream,
    use_device,
    use_stream,
    wait_stream,
)
from torch.testing._internal.common_utils import run_tests

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda required"
)


class TestNewStream:
    def test_new_stream_cpu(self):
        stream = new_stream(torch.device("cpu"))
        assert stream is CPUStream

    @skip_if_no_cuda
    def test_new_stream_cuda(self):
        stream = new_stream(torch.device("cuda"))
        assert isinstance(stream, torch.cuda.Stream)
        assert stream != torch.cuda.default_stream()


class TestCurrentStream:
    def test_current_stream_cpu(self):
        stream = current_stream(torch.device("cpu"))
        assert stream is CPUStream

    @skip_if_no_cuda
    def test_current_stream_cuda(self):
        stream = current_stream(torch.device("cuda"))
        assert isinstance(stream, torch.cuda.Stream)
        assert stream == torch.cuda.current_stream()


class TestDefaultStream:
    def test_default_stream_cpu(self):
        stream = default_stream(torch.device("cpu"))
        assert stream is CPUStream

    @skip_if_no_cuda
    def test_default_stream_cuda(self):
        stream = default_stream(torch.device("cuda"))
        assert isinstance(stream, torch.cuda.Stream)
        assert stream == torch.cuda.default_stream()


class TestUseDevice:
    def test_use_device_cpu(self):
        with use_device(torch.device("cpu")):
            pass

    @skip_if_no_cuda
    def test_use_device_cuda(self):
        with use_device(torch.device("cuda")):
            pass


class TestUseStream:
    def test_use_stream_cpu(self):
        with use_stream(CPUStream):
            pass

    @skip_if_no_cuda
    def test_use_stream_cuda(self):
        stream = new_stream(torch.device("cuda"))
        with use_stream(stream):
            assert current_stream(torch.device("cuda")) == stream


class TestGetDevice:
    def test_get_device_cpu(self):
        assert get_device(CPUStream).type == "cpu"

    @skip_if_no_cuda
    def test_get_device_cuda(self):
        stream = current_stream(torch.device("cuda"))
        assert get_device(stream).type == "cuda"


class TestWaitStream:
    def _test_wait_stream(self, source, target, cuda_sleep=None):
        with use_stream(target):
            if is_cuda(target):
                cuda_sleep(0.5)
            x = torch.ones(100, 100, device=get_device(target))

        wait_stream(source, target)

        with use_stream(source):
            assert x.sum().item() == 10000

    def test_wait_stream_cpu_cpu(self):
        source = CPUStream
        target = CPUStream
        self._test_wait_stream(source, target)

    @skip_if_no_cuda
    def test_wait_stream_cpu_cuda(self, cuda_sleep):
        source = CPUStream
        target = new_stream(torch.device("cuda"))
        self._test_wait_stream(source, target, cuda_sleep)

    @skip_if_no_cuda
    def test_wait_stream_cuda_cpu(self, cuda_sleep):
        source = new_stream(torch.device("cuda"))
        target = CPUStream
        self._test_wait_stream(source, target, cuda_sleep)

    @skip_if_no_cuda
    def test_wait_stream_cuda_cuda(self, cuda_sleep):
        source = current_stream(torch.device("cuda"))
        target = new_stream(torch.device("cuda"))
        self._test_wait_stream(source, target, cuda_sleep)


class TestRecordStream:
    def test_record_stream_cpu(self):
        # It should silently ignore CPU tensors.
        x = torch.rand(1, device=torch.device("cpu"))
        record_stream(x, CPUStream)

    @skip_if_no_cuda
    def test_record_stream_cuda(self, cuda_sleep):
        # This test detects unexpected block reallocation. For reliable test,
        # the stream to allocate tensors is isolated. The allocator will not
        # reuse free blocks which were allocated from another stream.
        stream_alloc = new_stream(torch.device("cuda"))
        with torch.cuda.stream(stream_alloc):
            x = torch.rand(1, device=torch.device("cuda"))

        stream = new_stream(torch.device("cuda"))
        record_stream(x, stream)
        with use_stream(stream):
            cuda_sleep(0.5)

        # 'x' is deleted at Python's perspective. But the block of 'x' is still
        # required for 'stream'. 'y' shouldn't be allocated to the block.
        data_ptr = x.data_ptr()
        del x
        stream_alloc.synchronize()
        with torch.cuda.stream(stream_alloc):
            y = torch.rand(1, device=torch.device("cuda"))
        assert y.data_ptr() != data_ptr

        # Pause Python until 'stream' finishes tasks queued. Now the block of
        # 'x' is free to be reallocated.
        wait_stream(CPUStream, stream)
        with torch.cuda.stream(stream_alloc):
            z = torch.rand(1, device=torch.device("cuda"))
        assert z.data_ptr() == data_ptr

    @skip_if_no_cuda
    def test_record_stream_shifted_view(self, cuda_sleep):
        # Issue: https://github.com/pytorch/pytorch/issues/27366
        stream_alloc = new_stream(torch.device("cuda"))
        with torch.cuda.stream(stream_alloc):
            x = torch.rand(2, device=torch.device("cuda"))

        y = x[1:]
        assert y.data_ptr() > x.data_ptr()

        stream = new_stream(torch.device("cuda"))
        with use_stream(stream):
            cuda_sleep(0.5)
        record_stream(y, stream)

        data_ptr = x.data_ptr()
        del x, y

        stream_alloc.synchronize()
        with torch.cuda.stream(stream_alloc):
            z = torch.rand(2, device=torch.device("cuda"))
        assert z.data_ptr() != data_ptr


if __name__ == "__main__":
    run_tests()
