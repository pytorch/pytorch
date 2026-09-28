# Owner(s): ["oncall: distributed"]

import multiprocessing
import os
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.distributed._transport.nixl import NIXLTransport
from torch.distributed._transport.nixl._cuda_host import CudaHostTransport
from torch.distributed._transport.nixl._memory import NIXLRemoteBuffer
from torch.testing._internal.common_utils import run_tests, TestCase


def _worker(rank, pipe):
    torch.cuda.set_device(rank)
    transport = NIXLTransport("cpu")
    stream = torch.cuda.Stream()
    source = torch.empty(1024, pin_memory=True)
    destination = torch.zeros(1024, pin_memory=True)
    producer = torch.full((1024,), rank + 1.0, device="cuda")
    consumer = torch.zeros_like(producer)
    # Prewarm kernels and memory copies before queuing host callbacks.
    source.copy_(producer)
    consumer.copy_(destination)
    torch.cuda.synchronize()
    source.zero_()
    local_source = transport.register_memory(source)
    local_destination = transport.register_memory(destination)
    pipe.send_bytes(transport.bind())
    transport.connect(pipe.recv_bytes())
    for memory in (local_source, local_destination):
        pipe.send_bytes(memory.to_remote_buffer().serialize())
    remote_source = NIXLRemoteBuffer.deserialize(pipe.recv_bytes())
    remote_destination = NIXLRemoteBuffer.deserialize(pipe.recv_bytes())
    bridge = CudaHostTransport(transport, stream)
    with torch.cuda.stream(stream):
        source.copy_(producer, non_blocking=True)
        done = bridge.write(local_source.to_view(), remote_destination)
    if not done.wait(15):
        raise TimeoutError("callback completion")
    # Local write completion does not imply peer write completion.
    pipe.send_bytes(b"written")
    if pipe.recv_bytes() != b"written":
        raise RuntimeError("unexpected control message")
    torch.testing.assert_close(destination, torch.full_like(destination, 2 - rank))
    with torch.cuda.stream(stream):
        bridge.read(local_destination.to_mutable_view(), remote_source)
        consumer.copy_(destination, non_blocking=True)
    stream.synchronize()  # Verify results only; no synchronization in enqueue.
    torch.testing.assert_close(consumer, torch.full_like(consumer, 2 - rank))
    bridge.close()
    pipe.send_bytes(b"finished")
    pipe.recv_bytes()
    transport.close()


def _callback_worker(fail):
    torch.cuda.set_device(0)
    transport = NIXLTransport("cpu")
    tensor = torch.zeros(8, pin_memory=True)
    memory = transport.register_memory(tensor)
    remote = memory.to_remote_buffer()
    stream = torch.cuda.Stream()
    bridge = CudaHostTransport(transport, stream)
    try:
        bridge.write(memory.to_view(), replace(remote, memory_type="VRAM"))
    except ValueError:
        pass
    else:
        raise AssertionError("VRAM transfer was not rejected before enqueue")
    entered, release = threading.Event(), threading.Event()

    def transfer(*args, **kwargs):
        entered.set()
        if fail:
            raise RuntimeError("injected callback failure")
        if not release.wait(5):
            raise TimeoutError("enqueue blocked the submitting thread")
        return SimpleNamespace(is_completed=lambda: True, wait=lambda: None)

    with patch.object(transport, "write", side_effect=transfer):
        done = bridge.write(memory.to_view(), remote)
        if fail:
            stream.synchronize()  # The child must terminate before this returns.
            raise AssertionError("failed callback returned successfully")
        if not entered.wait(5) or done.is_set():
            raise AssertionError("expected pending callback after enqueue returned")
        release.set()
        bridge.close()
        if not done.is_set():
            raise AssertionError("missing callback completion")
    transport.close()


@unittest.skipUnless(
    os.getenv("TORCH_TEST_CUDA_TRANSPORT") == "1" and torch.cuda.device_count() >= 2,
    "opt-in two-GPU CUDA/NIXL prototype test",
)
class TestCudaHostTransport(TestCase):
    def test_enqueue_returns_before_completion(self):
        self._check_callback_worker(False, 0)

    def test_callback_failure_terminates_process(self):
        self._check_callback_worker(True, 1)

    def _check_callback_worker(self, fail, exitcode):
        process = multiprocessing.get_context("spawn").Process(
            target=_callback_worker, args=(fail,)
        )
        process.start()
        try:
            process.join(30)
            self.assertEqual(process.exitcode, exitcode)
        finally:
            if process.is_alive():
                process.kill()
            process.join()

    def test_native_staged_write_and_read(self):
        ctx = multiprocessing.get_context("spawn")
        pipes = ctx.Pipe()
        processes = [
            ctx.Process(target=_worker, args=(rank, pipes[rank])) for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(40)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join()


if __name__ == "__main__":
    run_tests()
