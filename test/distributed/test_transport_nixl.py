# Owner(s): ["oncall: distributed"]

import asyncio
import gc
import json
import threading
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import new_transport, wait_all
from torch.distributed._transport.nixl import _transport as _nixl
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@dataclass
class _Descriptors:
    memory_type: str
    entries: list[tuple[int, int, int]]


@dataclass(eq=False)
class _RegistrationDescriptors:
    tensor: torch.Tensor
    memory_type: str


@dataclass
class _Transfer:
    operation: str
    local: _Descriptors
    remote: _Descriptors
    remote_agent: str


@dataclass
class _AgentConfig:
    backends: list[str]
    num_threads: int
    enable_prog_thread: bool
    capture_telemetry: bool = False


class _Agent:
    agents = {}
    plugins = ("UCX", "LIBFABRIC", "UCCL")
    transfer_state = "DONE"

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.backends = dict.fromkeys(
            plugin for plugin in config.backends if plugin in self.plugins
        )
        self.registrations = []
        self.remote_metadata = {}
        self.released = 0
        self.agents[name] = self

    def create_backend(self, plugin, options):
        self.backends[plugin] = dict(options)

    def _metadata(self, registrations):
        entries = [
            (
                registration.tensor.data_ptr(),
                registration.tensor.nbytes,
                max(registration.tensor.get_device(), 0),
                registration.memory_type,
            )
            for registration in registrations
        ]
        return json.dumps((self.name, entries)).encode()

    def get_agent_metadata(self):
        return self._metadata(self.registrations)

    def get_partial_agent_metadata(self, registrations, *, inc_conn_info, backends):
        if not inc_conn_info or backends != list(self.backends):
            raise AssertionError("unexpected metadata options")
        return self._metadata([registrations])

    def add_remote_agent(self, metadata):
        name, entries = json.loads(metadata)
        self.remote_metadata.setdefault(name, set()).update(map(tuple, entries))
        return name.encode()

    def remove_remote_agent(self, name):
        self.remote_metadata.pop(name, None)

    def register_memory(self, tensor, *, backends):
        if backends != list(self.backends):
            raise AssertionError("unexpected registration backend")
        memory_type = "VRAM" if tensor.is_cuda else "DRAM"
        descs = _RegistrationDescriptors(tensor, memory_type)
        self.registrations.append(descs)
        return descs

    def deregister_memory(self, descs, *, backends):
        if backends != list(self.backends):
            raise AssertionError("unexpected deregistration backend")
        self.registrations.remove(descs)

    def get_xfer_descs(self, entries, *, mem_type):
        return _Descriptors(mem_type, entries)

    def initialize_xfer(self, operation, local, remote, remote_agent, *, backends):
        if backends != list(self.backends):
            raise AssertionError("unexpected transfer backend")
        address, length, device_id = remote.entries[0]
        registered = self.remote_metadata.get(remote_agent, set())
        if not any(
            start <= address
            and address + length <= start + registered_length
            and device_id == registered_device
            and remote.memory_type == memory_type
            for start, registered_length, registered_device, memory_type in registered
        ):
            raise RuntimeError("remote memory is absent from NIXL metadata")
        return _Transfer(operation, local, remote, remote_agent)

    def _view(self, descs):
        address, length, _ = descs.entries[0]
        for registration in self.registrations:
            tensor = registration.tensor
            offset = address - tensor.data_ptr()
            if 0 <= offset and offset + length <= tensor.nbytes:
                return tensor.view(torch.uint8).flatten()[offset : offset + length]
        raise RuntimeError("unknown local memory")

    def transfer(self, handle):
        if self.transfer_state != "DONE":
            return self.transfer_state
        peer = self.agents[handle.remote_agent]
        local = self._view(handle.local)
        remote = peer._view(handle.remote)
        if handle.operation == "WRITE":
            remote.copy_(local.clone())
        else:
            local.copy_(remote.clone())
        return "DONE"

    def check_xfer_state(self, handle):
        return self.transfer_state

    def release_xfer_handle(self, handle):
        self.released += 1


@instantiate_parametrized_tests
class TestNIXLTransport(TransportTestMixin, TestCase):
    plugin = "UCX"

    def setUp(self):
        super().setUp()
        _Agent.agents.clear()
        _Agent.transfer_state = "DONE"

    @staticmethod
    def backend():
        return SimpleNamespace(nixl_agent=_Agent, nixl_agent_config=_AgentConfig)

    def make_transport_pair(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            first = new_transport("nixl", agent_name="first", plugin=self.plugin)
            second = new_transport("nixl", agent_name="second", plugin=self.plugin)
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        self.assertTrue(first._agent.config.enable_prog_thread)
        return first, second

    def registered_pair(self, tensor=None):
        first, second = self.make_transport_pair()
        tensor = torch.ones(8) if tensor is None else tensor
        source = first.register_memory(tensor)
        target = second.register_memory(torch.zeros_like(tensor))
        return first, second, source, target.to_remote_buffer()

    def test_unregister_invalidates_aliases_and_views(self):
        tensor = torch.ones(8)
        first, second, memory, remote = self.registered_pair(tensor)
        alias = first.register_memory(tensor)
        view = alias.to_view()
        agent = first._agent
        with patch.object(
            agent, "deregister_memory", wraps=agent.deregister_memory
        ) as deregister:
            first.unregister_memory(memory)
            first.unregister_memory(alias)
            self.assertEqual(deregister.call_count, 1)
        self.assertFalse(agent.registrations)
        self.assertFalse(first._registrations)
        for handle in (memory, alias):
            with self.assertRaisesRegex(RuntimeError, "unregistered"):
                handle.to_view()
            with self.assertRaisesRegex(RuntimeError, "unregistered"):
                handle.to_mutable_view()
            with self.assertRaisesRegex(RuntimeError, "unregistered"):
                handle.to_remote_buffer()
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            first.write(view, remote)
        fresh = first.register_memory(tensor)
        self.assertFalse(fresh.reused_registration())
        first.unregister_memory(memory)
        self.assertEqual(first.write(fresh.to_view(), remote), 0)
        first.unregister_memory(fresh)
        first.close()
        self.assertFalse(agent.registrations)

    @parametrize("operation", ["read", "write"])
    def test_unregister_rejects_pending_dma(self, operation):
        first, second, memory, remote = self.registered_pair()
        other = first.register_memory(torch.ones(16))
        view = memory.to_mutable_view() if operation == "read" else memory.to_view()
        with patch.object(first._agent, "transfer_state", "PROC"):
            work = getattr(first, operation)(view, remote, async_op=True)
            first.unregister_memory(other)
            with self.assertRaisesRegex(RuntimeError, "pending transfers"):
                first.unregister_memory(memory)
            self.assertTrue(memory._registration.active)
            self.assertEqual(len(first._agent.registrations), 1)
        work.wait()
        first.unregister_memory(memory)
        self.assertFalse(first._agent.registrations)

    def test_unregister_failure_can_be_retried(self):
        first, second, memory, remote = self.registered_pair()
        with patch.object(
            first._agent,
            "deregister_memory",
            side_effect=RuntimeError("deregister failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "deregister failed"):
                first.unregister_memory(memory)
        self.assertTrue(memory._registration.active)
        self.assertEqual(len(first._registrations), 1)
        self.assertEqual(first.write(memory.to_view(), remote), 0)
        first.unregister_memory(memory)
        self.assertFalse(first._registrations)

    def test_unregister_validates_owner(self):
        first, second, memory, remote = self.registered_pair()
        for invalid in (None, remote):
            with self.assertRaisesRegex(TypeError, "not registered"):
                first.unregister_memory(invalid)
        with self.assertRaisesRegex(TypeError, "not registered"):
            second.unregister_memory(memory)
        with self.assertRaises(ValueError):
            first.unregister_memory(memory, timeout=-1)
        first.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            first.unregister_memory(memory)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            memory.to_view()

    def test_unregister_lock_timeout(self):
        first, second, memory, remote = self.registered_pair()
        acquired, release = threading.Event(), threading.Event()

        def hold_lock():
            with first._operation_lock:
                acquired.set()
                release.wait(5)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        try:
            self.assertTrue(acquired.wait(5))
            with self.assertRaises(TimeoutError):
                first.unregister_memory(memory, timeout=0.001)
            self.assertTrue(memory._registration.active)
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        first.unregister_memory(memory)

    def test_raw_tensors_are_not_implicitly_registered(self):
        first, second, source, remote = self.registered_pair()
        registered = len(first._agent.registrations)
        for operation in (first.write, first.read):
            with self.assertRaisesRegex(TypeError, "not registered"):
                operation(torch.ones(8), remote)
        self.assertEqual(len(first._agent.registrations), registered)

    def test_registrations_outlive_last_outgoing_work(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            first = new_transport("nixl", agent_name="retained")
            second = new_transport("nixl", agent_name="peer")
        ref = weakref.ref(first)
        try:
            first.connect(second.bind())
            second.connect(first.bind())
            source = first.register_memory(torch.ones(8))
            target = second.register_memory(torch.zeros(8))
            work = first.write(
                source.to_view(), target.to_remote_buffer(), async_op=True
            )
            work.wait()
            del work, source, first
            gc.collect()
            self.assertIsNotNone(ref())
            self.assertIn(ref(), _nixl._live_transports)
            ref().close()
            gc.collect()
            self.assertIsNone(ref())
        finally:
            if ref() is not None:
                ref().close()
            second.close()

    def test_mismatched_peer_cleanup_failure_is_retryable(self):
        first, second, source, remote = self.registered_pair()
        wrong = replace(remote, metadata=json.dumps(("unexpected", [])).encode())
        with patch.object(
            first._agent,
            "remove_remote_agent",
            side_effect=RuntimeError("cleanup failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                first.write(source.to_view(), wrong)
        self.assertIn("unexpected", first._remote_agents)
        self.assertIn(first, _nixl._live_transports)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            first.bind()
        agent = first._agent
        first.close()
        self.assertFalse(agent.remote_metadata)
        self.assertNotIn(first, _nixl._live_transports)

    def test_supported(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            self.assertTrue(_nixl.NIXLTransport.supported())

    @parametrize("plugin", ["LIBFABRIC", "UCCL"])
    def test_plugin_without_ucx(self, plugin):
        with (
            patch.object(_Agent, "plugins", (plugin,)),
            patch.object(self, "plugin", plugin.lower()),
        ):
            self.test_transport_asyncio_put_get()
        for agent in _Agent.agents.values():
            self.assertEqual(list(agent.backends), [plugin])
            self.assertEqual(agent.registrations, [])

    def test_unavailable_plugin(self):
        with (
            patch.object(_nixl, "_load_backend", return_value=self.backend()),
            self.assertRaisesRegex(RuntimeError, "failed to create transport") as error,
        ):
            new_transport("nixl", plugin="missing")
        self.assertRegex(
            str(error.exception.__cause__), "plugin 'MISSING' is unavailable"
        )

    def test_refreshes_metadata_after_connect(self):
        first, second = self.make_transport_pair()
        first_agent = first._agent
        try:
            source = first.register_memory(torch.arange(8, dtype=torch.uint8))
            target_tensor = torch.zeros(8, dtype=torch.uint8)
            target = second.register_memory(target_tensor)
            descriptor = target.to_remote_buffer()
            remote = type(descriptor).deserialize(descriptor.serialize())

            self.assertEqual(first.write(source.to_view(2, 4), remote), 0)
            self.assertEqual(first.write(source.to_view(2, 4), remote), 0)
            self.assertEqual(target_tensor[:4], torch.arange(2, 6, dtype=torch.uint8))
            self.assertEqual(first._agent.released, 2)
        finally:
            first.close()
            second.close()
        self.assertEqual(first_agent.released, 2)

    def test_rejects_invalid_inputs_and_cleans_up(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(4, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            first.write(source.to_view(), target.to_remote_buffer())
        with self.assertRaisesRegex(TypeError, "local_buffer"):
            first.write(target.to_view(), target.to_remote_buffer())
        with self.assertRaisesRegex(ValueError, "outside"):
            source.to_view(9)

        first_agent = first._agent
        first.close()
        second.close()
        self.assertFalse(first.connected())
        self.assertEqual(first_agent.registrations, [])

    def test_timeout_retains_pending_transfer(self):
        first, second, source, remote = self.registered_pair(
            torch.arange(8, dtype=torch.uint8)
        )
        agent = first._agent
        with patch.object(agent, "transfer_state", "PROC"):
            work = first.write(
                source.to_view(),
                remote,
                async_op=True,
                timeout=0.001,
            )
            with self.assertRaises(TimeoutError):
                work.wait()
            self.assertFalse(work.is_completed())
            self.assertEqual(agent.released, 0)
            with self.assertRaises(TimeoutError):
                first.close(timeout=0.001)
            self.assertTrue(agent.registrations)
            self.assertIn(first, _nixl._live_transports)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                first.bind()
        first.close(timeout=1)
        self.assertTrue(work.is_completed())
        self.assertEqual(agent.registrations, [])
        self.assertEqual(agent.released, 1)
        self.assertNotIn(first, _nixl._live_transports)

    @parametrize("operation", ["read", "write", "read_async", "write_async"])
    def test_operation_timeout(self, operation):
        first, second, source, remote = self.registered_pair(
            torch.arange(8, dtype=torch.uint8)
        )
        view = (
            source.to_mutable_view()
            if operation.startswith("read")
            else source.to_view()
        )
        with patch.object(first._agent, "transfer_state", "PROC"):
            with self.assertRaises(TimeoutError):
                result = getattr(first, operation)(view, remote, timeout=0.001)
                if operation.endswith("_async"):
                    asyncio.run(result)
            self.assertTrue(first._pending)
            self.assertEqual(first._agent.released, 0)
        first.close()

    def test_native_setup_runs_on_calling_thread(self):
        first, second = self.make_transport_pair()
        self.assertFalse(hasattr(first, "_work_queue"))
        caller = threading.get_ident()
        original = first._agent.register_memory

        def register(*args, **kwargs):
            self.assertEqual(threading.get_ident(), caller)
            return original(*args, **kwargs)

        with patch.object(first._agent, "register_memory", side_effect=register):
            first.register_memory(torch.ones(8))

    def test_distinct_handles_for_overlapping_requests(self):
        first, second, source, remote = self.registered_pair()
        with patch.object(first._agent, "transfer_state", "PROC"):
            one = first.write(source.to_view(), remote, async_op=True)
            two = first.write(source.to_view(), remote, async_op=True)
            self.assertIsNot(one._handle, two._handle)
            self.assertEqual(len(first._pending), 2)
            self.assertFalse(one.is_completed())
            self.assertFalse(two.is_completed())
        two.wait()
        one.wait()
        self.assertEqual(first._agent.released, 2)
        self.assertFalse(first._pending)

    def test_asyncio_polling_and_future(self):
        first, second, source, remote = self.registered_pair()

        async def run():
            first._agent.transfer_state = "PROC"
            work = first.write(source.to_view(), remote, async_op=True)
            future = work.get_future()
            ready = asyncio.Event()
            future.add_done_callback(lambda _: ready.set())

            async def finish():
                await asyncio.sleep(0.01)
                first._agent.transfer_state = "DONE"

            task = asyncio.create_task(finish())
            await asyncio.wait_for(ready.wait(), 1)
            await task
            self.assertEqual(future.wait(), [])
            await wait_all([work])

        asyncio.run(run())

    def test_async_close_timeout_and_retry(self):
        first, second, source, remote = self.registered_pair()
        agent = first._agent

        async def run():
            with patch.object(agent, "transfer_state", "PROC"):
                work = first.write(source.to_view(), remote, async_op=True)
                with self.assertRaises(TimeoutError):
                    await first.close_async(timeout=0.001)
                self.assertTrue(agent.registrations)
                self.assertFalse(work.is_completed())
            await first.close_async()
            self.assertTrue(work.is_completed())
            self.assertFalse(agent.registrations)

        asyncio.run(run())

    def test_cancellation_retains_buffers(self):
        tensor = torch.ones(8)
        ref = weakref.ref(tensor)
        first, second, source, remote = self.registered_pair(tensor)

        async def run(memory):
            with patch.object(first._agent, "transfer_state", "PROC"):
                task = asyncio.create_task(first.write_async(memory.to_view(), remote))
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(first._pending)
                self.assertEqual(first._agent.released, 0)

        asyncio.run(run(source))
        del source, tensor
        gc.collect()
        self.assertIsNotNone(ref())
        first.close()
        gc.collect()
        self.assertIsNone(ref())

    def test_dropped_work_and_transport_retained_until_dma_completion(self):
        first, second, source, remote = self.registered_pair()
        with patch.object(first._agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remote, async_op=True)
            ref = weakref.ref(work)
            del work
            gc.collect()
            self.assertIsNotNone(ref())
            self.assertIn(first, _nixl._live_transports)
        first.close()
        gc.collect()
        self.assertIsNone(ref())

    def test_concurrent_waiters_release_handle_once(self):
        first, second, source, remote = self.registered_pair()
        work = first.write(source.to_view(), remote, async_op=True)
        errors = []

        def wait():
            try:
                work.wait()
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=wait) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(first._agent.released, 1)

    def test_native_terminal_failure_is_released(self):
        first, second, source, remote = self.registered_pair()
        with patch.object(first._agent, "transfer_state", "ERR"):
            work = first.write(source.to_view(), remote, async_op=True)
            with self.assertRaisesRegex(RuntimeError, "NIXL transfer failed"):
                work.wait()
        self.assertTrue(work.is_completed())
        self.assertFalse(work.is_success())
        self.assertEqual(first._agent.released, 1)

    def test_cancelled_async_close_can_be_retried(self):
        first, second, source, remote = self.registered_pair()

        async def run():
            with patch.object(first._agent, "transfer_state", "PROC"):
                work = first.write(source.to_view(), remote, async_op=True)
                task = asyncio.create_task(first.close_async())
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(first._agent.registrations)
                self.assertFalse(work.is_completed())
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    first.register_memory(torch.ones(8))
            await first.close_async()
            self.assertTrue(work.is_completed())

        asyncio.run(run())

    def test_handle_release_failure_rejects_new_work_until_close(self):
        first, second, source, remote = self.registered_pair()
        agent = first._agent
        work = first.write(source.to_view(), remote, async_op=True)
        with patch.object(
            agent, "release_xfer_handle", side_effect=RuntimeError("release failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                work.wait()
            self.assertTrue(work.is_completed())
            self.assertFalse(first._pending)
            self.assertEqual(len(first._transfers), 1)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                first.write(source.to_view(), remote)
        first.close()
        self.assertEqual(agent.released, 1)
        self.assertFalse(first._transfers)

    def test_native_cleanup_failure_can_be_retried(self):
        first, second = self.make_transport_pair()
        first.register_memory(torch.ones(8))
        first.register_memory(torch.ones(16))
        agent = first._agent
        original = agent.deregister_memory
        calls = 0

        def deregister(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("cleanup failed")
            return original(*args, **kwargs)

        with patch.object(agent, "deregister_memory", side_effect=deregister):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                first.close()
            self.assertEqual(len(agent.registrations), 1)
            first.close()
        self.assertEqual(agent.registrations, [])
        self.assertEqual(calls, 3)

    def test_async_poll_does_not_block_on_native_lock(self):
        first, second, source, remote = self.registered_pair()
        work = first.write(source.to_view(), remote, async_op=True)
        locked, release = threading.Event(), threading.Event()

        def hold_lock():
            with first._operation_lock:
                locked.set()
                release.wait(5)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        try:
            self.assertTrue(locked.wait(5))
            self.assertFalse(work.is_completed())

            async def run():
                with self.assertRaises(TimeoutError):
                    await wait_all([work], timeout=0.001)

            asyncio.run(run())
            self.assertEqual(first._agent.released, 0)
        finally:
            release.set()
            thread.join(5)
        work.wait()
        self.assertEqual(first._agent.released, 1)

    @parametrize("timeout", [-1, float("nan"), float("inf")])
    def test_invalid_timeout(self, timeout):
        with (
            patch.object(_nixl, "_load_backend", return_value=self.backend()),
            self.assertRaises(ValueError),
        ):
            _nixl.NIXLTransport(timeout=timeout)
        first, second = self.make_transport_pair()
        with self.assertRaises(ValueError):
            first.bind(timeout=timeout)
        with self.assertRaises(ValueError):
            first.close(timeout=timeout)

    def test_backend_options(self):
        options = {"NET_DEVICES": "all"}
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            with _nixl.NIXLTransport(
                backend_options=options, capture_telemetry=True
            ) as transport:
                self.assertEqual(transport._agent.backends["UCX"], options)
                self.assertTrue(transport._agent.config.capture_telemetry)
                options.clear()
                self.assertEqual(
                    transport._agent.backends["UCX"], {"NET_DEVICES": "all"}
                )

    def test_empty_view_and_replaced_storage(self):
        tensor = torch.ones(8)
        first, second, source, remote = self.registered_pair(tensor)
        self.assertEqual(first.write(source.to_view(0, 0), remote), 0)
        self.assertEqual(first._transfers, {})
        with self.assertRaises(TypeError):
            source.to_view(0.5)
        tensor.set_(torch.zeros(8))
        with self.assertRaisesRegex(RuntimeError, "storage was replaced"):
            first.write(source.to_view(), remote)

    def test_changed_remote_metadata_rebuilds_handle(self):
        first, second, source, remote = self.registered_pair()
        first.write(source.to_view(), remote)
        name, entries = json.loads(remote.metadata)
        updated = replace(
            remote, metadata=json.dumps((name, entries + entries)).encode()
        )
        first.write(source.to_view(), updated)
        self.assertEqual(first._agent.released, 2)
        self.assertFalse(first._transfers)

    def test_peer_validation(self):
        first, second, source, remote = self.registered_pair()
        with self.assertRaisesRegex(RuntimeError, "already connected"):
            first.connect(second.bind())
        with self.assertRaisesRegex(ValueError, "connected peer"):
            first.write(source.to_view(), replace(remote, agent_name="other"))
        with self.assertRaisesRegex(ValueError, "different agent"):
            first.write(
                source.to_view(),
                replace(remote, metadata=json.dumps(("other", [])).encode()),
            )
        self.assertEqual(first._transfers, {})

        self.assertNotIn("other", first._agent.remote_metadata)

    @parametrize("failure", ["dispatch", "query"])
    def test_error_drains_before_releasing_transfer(self, failure):
        first, second, source, remote = self.registered_pair(
            torch.arange(8, dtype=torch.uint8)
        )
        error = RuntimeError(f"{failure} failed")
        states = ["PROC", "DONE"] if failure == "dispatch" else [error, "PROC", "DONE"]
        with (
            patch.object(
                first._agent,
                "transfer",
                side_effect=error if failure == "dispatch" else None,
                return_value="PROC",
            ),
            patch.object(first._agent, "check_xfer_state", side_effect=states) as check,
            self.assertRaisesRegex(RuntimeError, f"{failure} failed"),
        ):
            first.write(source.to_view(), remote, async_op=True).wait()
        self.assertEqual(check.call_count, len(states))
        self.assertEqual(first._agent.released, 1)
        self.assertEqual(first._transfers, {})


if __name__ == "__main__":
    run_tests()
