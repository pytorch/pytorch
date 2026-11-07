# Owner(s): ["oncall: distributed"]

import asyncio
import ctypes
import gc
import sys
import threading
import tomllib
import types
import weakref
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed._transport import _registry, new_transport, wait_all
from torch.distributed._transport.mooncake import (
    _transport as _mooncake,
    MooncakeRemoteBuffer,
)
from torch.distributed._transport.mooncake._work import _live_transports
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class Operation:
    def __init__(self, local, remote, length, read):
        self.local, self.remote, self.length, self.read = (local, remote, length, read)
        self.ready = False
        self.error = ""
        self.query_error = False
        self.completed = False
        self.queried = threading.Event()

    def poll(self):
        if self.query_error:
            self.queried.set()
            raise RuntimeError("status unavailable")
        if self.ready and (not self.completed):
            if not self.error:
                dest, src = (
                    (self.local, self.remote)
                    if self.read
                    else (self.remote, self.local)
                )
                ctypes.memmove(dest, src, self.length)
            self.completed = True
        return (self.completed, self.error if self.completed else "")


class Engine:
    TransferOpcode = types.SimpleNamespace(Read=0, Write=1)
    port = 12300

    def __init__(self):
        Engine.port += 1
        self.endpoint = f"127.0.0.1:{Engine.port}"
        self.operations = []
        self.registrations = {}
        self.ready = False
        self.unregister_error = False
        self.refreshes = []
        self.submitted = threading.Event()

    def initialize(self, *args):
        self.init_args = args
        return 0

    def get_endpoint(self):
        return self.endpoint

    def refresh_remote(self, segment):
        self.refreshes.append(segment)

    def register_memory(self, address, length, location):
        self.registrations[address] = length
        return 0

    def unregister_memory(self, address):
        if self.unregister_error:
            return -1
        del self.registrations[address]
        return 0

    def submit_transfer(self, segment, local, remote, length, opcode):
        operation = Operation(local, remote, length, opcode == 0)
        operation.ready = self.ready
        self.operations.append(operation)
        self.submitted.set()
        return operation


def buffers(pair):
    left, right = pair
    source = torch.arange(16, dtype=torch.float32)
    target = torch.zeros_like(source)
    local = left.register_memory(source)
    remote = right.register_memory(target)
    descriptor = MooncakeRemoteBuffer.deserialize(remote.to_remote_buffer().serialize())
    return (source, target, local, remote, descriptor)


async def _next_loop_turn():
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    loop.call_soon(ready.set_result, None)
    await ready


@contextmanager
def _pair():
    module = types.ModuleType("mooncake.engine")
    module.TransferEngine = Engine
    module.TransferOpcode = Engine.TransferOpcode
    entry = EntryPoint(
        name="mooncake",
        value="torch.distributed._transport.mooncake:MooncakeTransport",
        group="torch.distributed.transports",
    )
    with (
        patch.object(_mooncake, "_load_backend", return_value=module),
        patch.object(_registry, "entry_points", return_value=[entry]),
        patch.object(_registry, "_registered_transports", {}),
    ):
        left = new_transport("mooncake", "cpu", protocol="tcp", timeout=0.5)
        right = new_transport("mooncake", "cpu", protocol="tcp", timeout=0.5)
        left.connect(right.bind())
        right.connect(left.bind())
        try:
            yield (left, right)
        finally:
            for owner in (left, right):
                if owner._engine is not None:
                    owner._engine.unregister_error = False
                    for operation in owner._engine.operations:
                        operation.query_error = False
                        operation.ready = True
                try:
                    owner.close()
                except RuntimeError:
                    owner.close()
            if left in _live_transports or right in _live_transports:
                raise AssertionError("closed transports retained registrations")


class TestMooncakeManagedTransport(TestCase):
    hw_classification = HardwareClassification.CPU

    def setUp(self):
        super().setUp()
        self.pair = self.enterContext(_pair())

    def test_package_entrypoint(self):
        config = tomllib.loads(
            (Path(__file__).parents[2] / "pyproject.toml").read_text()
        )
        self.assertEqual(
            config["project"]["entry-points"]["torch.distributed.transports"][
                "mooncake"
            ],
            "torch.distributed._transport.mooncake:MooncakeTransport",
        )

    def test_read_write_offsets_and_no_nixl(self):
        pair = self.pair
        left, _ = pair
        source, target, local, _, remote = buffers(pair)
        left._engine.ready = True
        self.assertEqual(left.write(local.to_view(offset=8, length=16), remote), 0)
        self.assertEqual(target[:4], source[2:6])
        target.fill_(42)
        self.assertEqual(
            left.read(local.to_mutable_view(offset=4, length=12), remote), 0
        )
        self.assertEqual(source[1:4], target[:3])
        self.assertTrue(
            not any(name == "nixl" or name.startswith("nixl.") for name in sys.modules)
        )

    def test_independent_work_handles_and_zero_length(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, remote = buffers(pair)
        first = left.write(local.to_view(), remote, async_op=True)
        second = left.write(local.to_view(), remote, async_op=True)
        self.assertIsNot(left._engine.operations[0], left._engine.operations[1])
        left._engine.operations[1].ready = True
        self.assertTrue(second.wait())
        self.assertTrue(not first.is_completed())
        zero = left.write(local.to_view(length=0), remote, async_op=True)
        self.assertEqual(zero.get_future().wait(), [])

    def test_timeout_retains_work_tensor_and_rejects_early_unregister(self):
        pair = self.pair
        left, _ = pair
        source, _, local, _, remote = buffers(pair)
        tensor_ref = weakref.ref(source)
        work = left.write(local.to_view(), remote, async_op=True)
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=1))
        with self.assertRaisesRegex(RuntimeError, "pending transfers"):
            left.unregister_memory(local)
        work_ref = weakref.ref(work)
        del work, source
        gc.collect()
        self.assertTrue(tensor_ref() is not None and work_ref() is not None)
        self.assertIn(left, _live_transports)
        left._engine.operations[0].ready = True
        self.assertTrue(work_ref().wait())
        left.unregister_memory(local)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            local.to_view()

    def test_alias_invalidation_and_overlapping_registration(self):
        pair = self.pair
        left, _ = pair
        source, _, local, _, _ = buffers(pair)
        alias = left.register_memory(source.view(-1))
        self.assertIs(alias._registration, local._registration)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            left.register_memory(source[1:])
        left.unregister_memory(alias)
        left.unregister_memory(local)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            local.to_remote_buffer()
        replacement = left.register_memory(source)
        self.assertIsNot(replacement._registration, alias._registration)

    def test_remote_registration_refresh(self):
        pair = self.pair
        left, right = pair
        _, target, local, output, remote = buffers(pair)
        left._engine.ready = True
        left.write(local.to_view(), remote)
        left.write(local.to_view(), remote)
        self.assertEqual(len(left._engine.refreshes), 2)
        right.unregister_memory(output)
        replacement = right.register_memory(target)
        fresh = replacement.to_remote_buffer()
        self.assertEqual(fresh, remote)
        left.write(local.to_view(), fresh)
        self.assertEqual(len(left._engine.refreshes), 3)

    @parametrize("tensor", [torch.empty(0), torch.zeros(4, 4).T])
    def test_bad_tensor(self, tensor):
        pair = self.pair
        with self.assertRaises(ValueError):
            pair[0].register_memory(tensor)

    def test_read_requires_mutable_view_and_valid_peer(self):
        pair = self.pair
        left, _ = pair
        _, _, local, other, remote = buffers(pair)
        with self.assertRaisesRegex(TypeError, "registered"):
            left.read(local.to_view(), remote)
        with self.assertRaises(TypeError):
            left.write(other.to_view(), remote)
        wrong = replace(remote, endpoint="different-peer:1234")
        with self.assertRaisesRegex(ValueError, "connected peer"):
            left.write(local.to_view(), wrong)
        small = replace(remote, length=1)
        with self.assertRaisesRegex(ValueError, "fit"):
            left.write(local.to_view(), small)
        with self.assertRaises(ValueError):
            local.to_view(offset=-1)
        with self.assertRaises(ValueError):
            local.to_view(length=10000)

    def test_resized_tensor_rejected(self):
        pair = self.pair
        left, _ = pair
        source, _, local, _, remote = buffers(pair)
        source.resize_(1)
        with self.assertRaisesRegex(RuntimeError, "resized"):
            left.write(local.to_view(), remote)

    @parametrize("data", [b"{}", b'{"version":1,"version":1}', b"not-json", b"[]"])
    def test_malformed_descriptor(self, data):
        with self.assertRaises(ValueError):
            MooncakeRemoteBuffer.deserialize(data)

    def test_descriptor_type_and_backend_validation(self):
        pair = self.pair
        left, _ = pair
        with self.assertRaises(ValueError):
            MooncakeRemoteBuffer.deserialize(left.bind())
        with self.assertRaises(ValueError):
            MooncakeRemoteBuffer("host:1234", True, 10, 0, "DRAM").serialize()
        with self.assertRaises(TypeError):
            MooncakeRemoteBuffer.deserialize("not-bytes")

    def test_close_timeout_latches_shutdown_and_can_retry(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, remote = buffers(pair)
        left.write(local.to_view(), remote, async_op=True)
        with self.assertRaises(TimeoutError):
            left.close(timeout=0)
        self.assertTrue(local._registration.active)
        self.assertIn(left, _live_transports)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            left.write(local.to_view(), remote)
        left._engine.operations[-1].ready = True
        left.close()
        self.assertTrue(not local._registration.active)
        left.close()

    def test_failed_query_is_not_completion_and_failure_drains(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, remote = buffers(pair)
        work = left.write(local.to_view(), remote, async_op=True)
        operation = left._engine.operations[-1]
        operation.query_error = True
        self.assertTrue(operation.queried.wait(5))
        self.assertTrue(not work.is_completed())
        with self.assertRaises(TimeoutError):
            work.wait(timedelta(milliseconds=1))
        operation.query_error = False
        operation.ready = True
        with self.assertRaisesRegex(RuntimeError, "status unavailable"):
            work.wait()
        self.assertTrue(work.is_completed())
        left.unregister_memory(local)

    def test_close_drains_after_terminal_error(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, remote = buffers(pair)
        left.write(local.to_view(), remote, async_op=True)
        second = left.write(local.to_view(), remote, async_op=True)
        left._engine.operations[0].error = "transfer failed"
        begin_close = left._begin_close

        def ready_on_close(timeout):
            pending = begin_close(timeout)
            left._engine.operations[0].ready = left._engine.operations[1].ready = True
            return pending

        with patch.object(left, "_begin_close", side_effect=ready_on_close):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                left.close()
        self.assertTrue(second.is_completed() and left._closed)

    def test_cleanup_failure_retains_registration_for_retry(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, _ = buffers(pair)
        left._engine.unregister_error = True
        with self.assertRaisesRegex(RuntimeError, "unregistration"):
            left.close()
        self.assertTrue(local._registration.active and left in _live_transports)
        left._engine.unregister_error = False
        left.close()

    def test_future_reentrancy(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, remote = buffers(pair)
        work = left.write(local.to_view(), remote, async_op=True)
        left._engine.operations[-1].ready = True
        future = work.get_future()
        self.assertEqual(future.then(lambda _: work.get_future().wait()).wait(), [])

    def test_async_progress_cancellation_and_close(self):
        pair = self.pair

        async def run():
            left, _ = pair
            _, target, local, _, remote = buffers(pair)
            task = asyncio.create_task(left.write_async(local.to_view(), remote))
            self.assertTrue(await asyncio.to_thread(left._engine.submitted.wait, 5))
            with left._operation_lock:
                self.assertEqual(len(left._pending), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(len(left._pending), 1)
            work = next(iter(left._pending.values()))
            left._engine.operations[-1].ready = True
            await wait_all([work])
            self.assertEqual(target, torch.arange(16, dtype=torch.float32))
            await left.close_async()
            self.assertTrue(left._closed)

        asyncio.run(run())

    def test_async_close_timeout_and_cancellation_retain_resources(self):
        pair = self.pair

        async def run():
            left, _ = pair
            _, _, local, _, remote = buffers(pair)
            left.write(local.to_view(), remote, async_op=True)
            with self.assertRaises(TimeoutError):
                await left.close_async(timeout=0)
            self.assertTrue(left._closing and local._registration.active)
            closer = asyncio.create_task(left.close_async(timeout=10))
            await _next_loop_turn()
            closer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closer
            self.assertTrue(local._registration.active)
            left._engine.operations[-1].ready = True
            await left.close_async()

        asyncio.run(run())

    def test_async_lock_contention_yields_to_event_loop(self):
        pair = self.pair
        left, _ = pair
        _, _, local, _, remote = buffers(pair)
        held, release = (threading.Event(), threading.Event())

        def holder():
            with left._operation_lock:
                held.set()
                release.wait(5)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(held.wait(5))

        async def run():
            asyncio.get_running_loop().call_soon(release.set)
            left._engine.ready = True
            await left.write_async(local.to_view(), remote, timeout=0.5)

        try:
            asyncio.run(run())
        finally:
            release.set()
            thread.join(5)

    @parametrize("timeout", [-1, float("nan"), float("inf")])
    def test_invalid_timeout(self, timeout):
        pair = self.pair
        with self.assertRaises(ValueError):
            pair[0].bind(timeout=timeout)

    def test_multi_peer_and_idempotent_connect(self):
        from torch.distributed._transport.mooncake import MooncakeTransport

        left, right = self.pair
        with MooncakeTransport("cpu", protocol="tcp") as third:
            left.connect(third.bind())
            left.connect(right.bind())
            self.assertTrue(left.connected())
            source = torch.arange(8, dtype=torch.float32)
            local = left.register_memory(source)
            outputs = [torch.zeros_like(source), torch.zeros_like(source)]
            descriptors = [
                owner.register_memory(output).to_remote_buffer()
                for owner, output in zip((right, third), outputs)
            ]
            left._engine.ready = True
            for descriptor in descriptors:
                left.write(local.to_view(), descriptor)
            for output in outputs:
                self.assertEqual(output, source)
            self.assertEqual(len(left._engine.refreshes), 2)

    def test_nested_views_and_remote_slices(self):
        left, _ = self.pair
        source, target, local, _, remote = buffers(self.pair)
        left._engine.ready = True
        view = local.to_view()[4:28].slice(4, 8)
        descriptor = MooncakeRemoteBuffer.deserialize(remote[12:20].serialize())
        left.write(view, descriptor)
        self.assertEqual(target[3:5], source[2:4])
        target.fill_(17)
        mutable = local.to_mutable_view()[8:24][-8:]
        left.read(mutable, remote.slice(4, 8))
        self.assertEqual(source[4:6], target[1:3])
        self.assertEqual(local.to_view()[100:200].size(), 0)
        self.assertEqual(remote[20:10].length, 0)
        for value in (local.to_view(), remote):
            with self.assertRaises(ValueError):
                value.slice(-1)
            with self.assertRaises(ValueError):
                value.slice(0, 10000)
            with self.assertRaises(ValueError):
                value[::2]
            with self.assertRaises(TypeError):
                value[0]
        left.unregister_memory(local)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            view.slice()


instantiate_parametrized_tests(TestMooncakeManagedTransport)

if __name__ == "__main__":
    run_tests()
