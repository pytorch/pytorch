# Owner(s): ["oncall: distributed"]

import asyncio
import gc
import itertools
import json
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
import torch.distributed as dist
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
    resolved: tuple[int, int]
    state: str = "PROC"


class nixlNotFoundError(RuntimeError):
    pass


class nixlNotAllowedError(RuntimeError):
    pass


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
        self.peers = {}
        self.released = 0
        self.loads = 0
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

    def get_partial_agent_metadata(self, registrations, *, inc_conn_info, backends):
        if not inc_conn_info or backends != list(self.backends):
            raise AssertionError("unexpected metadata options")
        if not isinstance(registrations, list):
            registrations = [registrations]
        return self._metadata(registrations)

    def add_remote_agent(self, metadata):
        self.loads += 1
        name, entries = json.loads(metadata)
        # Like NIXL, refuse a restarted peer's connection info until it is reset.
        peer = self.agents.get(name)
        if self.peers.setdefault(name, peer) is not peer:
            raise nixlNotAllowedError("NIXL_ERR_NOT_ALLOWED")
        self.remote_metadata.setdefault(name, set()).update(map(tuple, entries))
        return name.encode()

    def remove_remote_agent(self, name):
        self.remote_metadata.pop(name, None)
        self.peers.pop(name, None)

    def check_remote_metadata(self, name):
        return name in self.remote_metadata

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
        # Like NIXL, resolve with the nearest containing descriptor, even stale.
        candidates = [
            (start, -registered_length)
            for start, registered_length, registered_device, memory_type in registered
            if start <= address
            and address + length <= start + registered_length
            and device_id == registered_device
            and remote.memory_type == memory_type
        ]
        if not candidates:
            raise RuntimeError("remote memory is absent from NIXL metadata")
        start, negative_length = max(candidates)
        return _Transfer(
            operation, local, remote, remote_agent, (start, -negative_length)
        )

    def _view(self, descs):
        address, length, _ = descs.entries[0]
        for registration in self.registrations:
            tensor = registration.tensor
            offset = address - tensor.data_ptr()
            if 0 <= offset and offset + length <= tensor.nbytes:
                return tensor.view(torch.uint8).flatten()[offset : offset + length]
        raise RuntimeError("unknown local memory")

    def transfer(self, handle):
        handle.state = self._transfer(handle)
        return self.check_xfer_state(handle)

    def _transfer(self, handle):
        if self.transfer_state != "DONE":
            return self.transfer_state
        peer = self.agents[handle.remote_agent]
        if handle.resolved not in {
            (registration.tensor.data_ptr(), registration.tensor.nbytes)
            for registration in peer.registrations
        }:
            return "ERR_REMOTE_DISCONNECT"
        local = self._view(handle.local)
        remote = peer._view(handle.remote)
        if handle.operation == "WRITE":
            remote.copy_(local.clone())
        else:
            local.copy_(remote.clone())
        return "DONE"

    def check_xfer_state(self, handle):
        # Like NIXL, a peer whose descriptors were dropped refuses status queries.
        if handle.state == "PROC" and handle.remote_agent not in self.remote_metadata:
            raise nixlNotFoundError("NIXL_ERR_NOT_FOUND")
        if handle.state == "PROC":
            handle.state = self.transfer_state
        # Like NIXL, raise for a failed transfer and keep that state final.
        if handle.state not in ("PROC", "DONE"):
            raise RuntimeError(f"NIXL_{handle.state}")
        return handle.state

    def release_xfer_handle(self, handle):
        self.released += 1


def no_sleep():
    # A zero timeout must not wait, so sleeping means the deadline was ignored.
    def sleep(seconds):
        raise AssertionError("deadline ignored")

    return patch.object(
        _nixl, "time", SimpleNamespace(monotonic=time.monotonic, sleep=sleep)
    )


@instantiate_parametrized_tests
class TestNIXLTransport(TransportTestMixin, TestCase):
    plugin = "UCX"

    def setUp(self):
        super().setUp()
        _Agent.agents.clear()
        _Agent.transfer_state = "DONE"

    @staticmethod
    def backend():
        return SimpleNamespace(
            nixl_agent=_Agent,
            nixl_agent_config=_AgentConfig,
            nixlNotAllowedError=nixlNotAllowedError,
            nixlNotFoundError=nixlNotFoundError,
            nixlRegDList=lambda memory_type: [],
            DRAM_SEG="DRAM",
        )

    def make_transport_pair(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            # Bounded so a lost completion fails rather than stalls.
            first, second = (
                new_transport("nixl", agent_name=name, plugin=self.plugin, timeout=5)
                for name in ("first", "second")
            )
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
        with patch.object(first._agent, "register_memory") as register:
            alias = first.register_memory(tensor)
            register.assert_not_called()
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
        self.assertIsNot(fresh._registration, memory._registration)
        self.assertEqual(len(agent.registrations), 1)
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
        first.write(source.to_view(), remote)
        wrong = replace(
            remote,
            address=remote.address + remote.length,
            region_address=remote.address + remote.length,
            metadata=json.dumps(("unexpected", [])).encode(),
        )
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
        required = (
            "nixlRegDList",
            "DRAM_SEG",
            "nixlNotAllowedError",
            "nixlNotFoundError",
        )
        for name in required:
            backend = self.backend()
            delattr(backend, name)
            with patch.object(_nixl, "_load_backend", return_value=backend):
                self.assertFalse(_nixl.NIXLTransport.supported())

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

    def test_skips_loaded_remote_metadata(self):
        first, second, source, remote = self.registered_pair()
        self.assertEqual(json.loads(second.bind()), ["second", []])
        loads = first._agent.loads
        with patch.object(
            first._agent, "remove_remote_agent", side_effect=AssertionError
        ):
            for _ in range(3):
                first.write(source.to_view(), remote)
        self.assertEqual(first._agent.loads, loads + 1)

    def test_remote_slices_reuse_loaded_metadata(self):
        first, second, source, remote = self.registered_pair()
        loads = first._agent.loads
        first.write(source.to_view(0, 2), remote.slice(2, 2))
        first.write(source.to_view(), remote)
        first.write(source.to_view(0, 2), remote.slice(0, 2))
        self.assertEqual(first._agent.loads, loads + 1)

    @parametrize("start", [0, 4])
    def test_reregistered_remote_memory_resets_peer(self, start):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = torch.zeros(8, dtype=torch.uint8)
        half = second.register_memory(target[start : start + 4])
        first.write(source.to_view(0, 4), half.to_remote_buffer())
        second.unregister_memory(half)
        whole = second.register_memory(target)
        first.write(source.to_view(4, 4), whole.to_remote_buffer())
        self.assertEqual(target[:4], torch.arange(4, 8, dtype=torch.uint8))
        self.assertEqual(len(first._agent.remote_metadata["second"]), 1)
        self.assertEqual(len(first._loaded["second"]), 1)
        self.assertEqual(first._remote_agents, {"second"})

    def test_overlap_requires_same_memory(self):
        key = ("DRAM", 0, 100, 8)
        self.assertTrue(_nixl._overlaps(key, ("DRAM", 0, 107, 8)))
        self.assertFalse(_nixl._overlaps(key, ("DRAM", 0, 108, 8)))
        self.assertFalse(_nixl._overlaps(key, ("DRAM", 0, 92, 8)))
        self.assertFalse(_nixl._overlaps(key, ("VRAM", 0, 100, 8)))
        self.assertFalse(_nixl._overlaps(key, ("DRAM", 1, 100, 8)))

    @parametrize("capped", [False, True])
    @parametrize("use_async", [False, True])
    def test_reset_waits_for_peer_transfers(self, use_async, capped):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = torch.zeros(8, dtype=torch.uint8)
        half = second.register_memory(target[:4])
        whole = second.register_memory(target)
        other = second.register_memory(torch.zeros(4, dtype=torch.uint8))

        def write(timeout=None):
            if use_async:
                remote = whole.to_remote_buffer()
                write = first.write_async(source.to_view(), remote, timeout=timeout)
                asyncio.run(asyncio.wait_for(write, 5))
            else:
                first.write(source.to_view(), whole.to_remote_buffer(), timeout=timeout)

        # Past the loaded limit, stale peers still wait.
        limit = 1 if capped else _nixl._MAX_LOADED
        with (
            patch.object(_nixl, "_MAX_LOADED", limit),
            patch.object(_Agent, "transfer_state", "PROC"),
            no_sleep(),
        ):
            work = first.write(
                source.to_view(0, 4), half.to_remote_buffer(), async_op=True
            )
            loads = first._agent.loads
            with self.assertRaisesRegex(TimeoutError, "peer transfers remain pending"):
                write(timeout=0)
            # Transfers to loaded memory wait for the pending reset too.
            with self.assertRaisesRegex(TimeoutError, "peer transfers remain pending"):
                first.write(
                    source.to_view(0, 4),
                    half.to_remote_buffer(),
                    async_op=True,
                    timeout=0,
                )
            # So do transfers to newly registered memory on the peer.
            with self.assertRaisesRegex(TimeoutError, "peer transfers remain pending"):
                first.write(source.to_view(0, 4), other.to_remote_buffer(), timeout=0)
            self.assertEqual(first._agent.loads, loads)
            self.assertEqual(
                first._agent.remote_metadata["second"],
                {(target.data_ptr(), 4, 0, "DRAM")},
            )
        write()
        self.assertTrue(work.is_completed())
        self.assertNotIn("second", first._resetting)
        self.assertEqual(target, torch.arange(8, dtype=torch.uint8))

    def test_loaded_limit_resets_only_idle_peer(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        tensors = [torch.zeros(8, dtype=torch.uint8) for _ in range(3)]
        targets = [second.register_memory(tensor) for tensor in tensors]
        with patch.object(_nixl, "_MAX_LOADED", 1):
            # Past the cap, a busy peer loads without waiting for its transfers.
            with patch.object(first._agent, "transfer_state", "PROC"), no_sleep():
                works = [
                    first.write(source.to_view(), t.to_remote_buffer(), async_op=True)
                    for t in targets[:2]
                ]
            self.assertEqual(len(first._agent.remote_metadata["second"]), 2)
            for work in works:
                work.wait()
            first.write(source.to_view(), targets[2].to_remote_buffer())
            # Loaded registrations are reused at the limit, idle or busy.
            loads = first._agent.loads
            first.write(source.to_view(), targets[2].to_remote_buffer())
            with patch.object(first._agent, "transfer_state", "PROC"), no_sleep():
                works = [
                    first.write(
                        source.to_view(), targets[2].to_remote_buffer(), async_op=True
                    )
                    for _ in range(2)
                ]
            self.assertEqual(first._agent.loads, loads)
            for work in works:
                work.wait()
        self.assertEqual(len(first._agent.remote_metadata["second"]), 1)
        self.assertEqual(tensors[2], torch.arange(8, dtype=torch.uint8))

    @parametrize("use_async", [False, True])
    def test_reset_ignores_peer_transfer_errors(self, use_async):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = torch.zeros(8, dtype=torch.uint8)
        half = second.register_memory(target[:4])
        whole = second.register_memory(target)
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(
                source.to_view(0, 4), half.to_remote_buffer(), async_op=True
            )
        work._handle.state = "ERR"
        if use_async:
            asyncio.run(first.write_async(source.to_view(), whole.to_remote_buffer()))
        else:
            first.write(source.to_view(), whole.to_remote_buffer())
        self.assertEqual(target, torch.arange(8, dtype=torch.uint8))
        with self.assertRaisesRegex(RuntimeError, "NIXL_ERR"):
            work.wait()

    def test_async_work_keeps_remaining_timeout(self):
        first, _, source, remote = self.registered_pair()
        clock = itertools.chain([0.0], itertools.repeat(60.0))
        with patch.object(
            _nixl, "time", SimpleNamespace(monotonic=lambda: next(clock))
        ):
            work = first.write(source.to_view(), remote, async_op=True, timeout=100)
        self.assertEqual(work._timeout, 40)
        work.wait()

    def test_restarted_peer_reloads_after_reset(self):
        first, second, source, remote = self.registered_pair()
        with patch.object(_Agent, "transfer_state", "PROC"), no_sleep():
            work = first.write(source.to_view(), remote, async_op=True)
            with patch.object(_nixl, "_load_backend", return_value=self.backend()):
                restarted = new_transport(
                    "nixl", agent_name="second", plugin=self.plugin
                )
            self.addCleanup(restarted.close)
            target = torch.zeros(8)
            moved = restarted.register_memory(target).to_remote_buffer()
            with self.assertRaisesRegex(nixlNotAllowedError, "NOT_ALLOWED"):
                first.write(source.to_view(), moved)
            self.assertIn("second", first._resetting)
            with self.assertRaisesRegex(TimeoutError, "remain pending"):
                first.write(source.to_view(), moved, timeout=0)
        work.wait()
        first.write(source.to_view(), moved)
        self.assertNotIn("second", first._resetting)
        self.assertEqual(target, torch.ones(8))

    def test_connect_restarted_peer_then_retry(self):
        first, second, source, remote = self.registered_pair()
        first.write(source.to_view(), remote)
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            restarted = new_transport("nixl", agent_name="second", plugin=self.plugin)
        self.addCleanup(restarted.close)
        with self.assertRaisesRegex(nixlNotAllowedError, "NOT_ALLOWED"):
            first.connect(restarted.bind())
        target = torch.zeros(8)
        moved = restarted.register_memory(target).to_remote_buffer()
        with self.assertRaisesRegex(nixlNotAllowedError, "NOT_ALLOWED"):
            first.write(source.to_view(), moved)
        first.write(source.to_view(), moved)
        self.assertEqual(target, torch.ones(8))

    @parametrize("error", [RuntimeError, nixlNotAllowedError])
    @parametrize("dropped", ["second", "third"])
    def test_load_dropping_peer_closes_transport(self, error, dropped):
        first, source, _, remotes = self.connect_peers()
        first.write(source.to_view(), remotes[1])
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remotes[0], async_op=True)
        metadata = first._agent.remote_metadata
        saved = metadata[dropped]

        def drop_peer(_):
            del metadata[dropped]
            raise error("load failed")

        with patch.object(first._agent, "add_remote_agent", side_effect=drop_peer):
            with self.assertRaisesRegex(error, "load failed"):
                first.write(source.to_view(), remotes[3])
        self.assertFalse(first.connected())
        self.assertFalse(first._resetting)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            first.write(source.to_view(), remotes[1])
        if dropped == "second":
            # The dropped peer's transfer can never be confirmed.
            with self.assertRaises(TimeoutError):
                first.close(timeout=0)
            self.assertFalse(work.is_completed())
            metadata[dropped] = saved
        work.wait()
        first.close(timeout=0)

    @parametrize("reset", [False, True])
    def test_load_dropping_unloaded_peer_closes_transport(self, reset):
        # A peer counts whether it was only connected or was just reset.
        first, source, _, remotes = self.connect_peers()
        if reset:
            first.write(source.to_view(), remotes[0])
            first._resetting.add("second")
        metadata = first._agent.remote_metadata

        def drop_peer(_):
            metadata.pop("second", None)
            raise RuntimeError("load failed")

        with patch.object(first._agent, "add_remote_agent", side_effect=drop_peer):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                first.write(source.to_view(), remotes[0])
        self.assertFalse(first.connected())

    def test_connect_dropping_peer_closes_transport(self):
        first, source, _, remotes = self.connect_peers()

        def drop_peer(_):
            del first._agent.remote_metadata["third"]
            raise RuntimeError("load failed")

        with patch.object(first._agent, "add_remote_agent", side_effect=drop_peer):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                first.connect(b"bad")
        self.assertFalse(first.connected())

    def test_async_reset_wait_does_not_block(self):
        first, _, source, remote = self.registered_pair()
        first._agent.transfer_state = "PROC"
        work = first.write(source.to_view(), remote, async_op=True)
        first._resetting.add("second")

        async def finish():
            await asyncio.sleep(0)
            first._agent.transfer_state = "DONE"

        async def main():
            write = first.write_async(source.to_view(), remote)
            await asyncio.wait_for(asyncio.gather(write, finish()), 5)

        with no_sleep():
            asyncio.run(main())
        self.assertTrue(work.is_success())
        self.assertNotIn("second", first._resetting)

    def test_rejected_load_keeps_transport(self):
        first, second, source, remote = self.registered_pair()
        first.write(source.to_view(), remote)
        far = remote.region_address + 4096
        with patch.object(
            first._agent, "add_remote_agent", side_effect=RuntimeError("bad metadata")
        ):
            with self.assertRaisesRegex(RuntimeError, "bad metadata"):
                first.write(
                    source.to_view(), replace(remote, address=far, region_address=far)
                )
        self.assertTrue(first.connected())
        self.assertFalse(first._resetting)
        first.write(source.to_view(), remote)

    def test_failed_transfer_resets_peer(self):
        first, _, source, remote = self.registered_pair()
        with patch.object(first._agent, "transfer_state", "ERR"):
            with self.assertRaisesRegex(RuntimeError, "NIXL_ERR"):
                first.write(source.to_view(), remote)
        self.assertIn("second", first._resetting)
        loads = first._agent.loads
        first.write(source.to_view(), remote)
        self.assertEqual(first._agent.loads, loads + 1)
        self.assertNotIn("second", first._resetting)

    def test_status_query_propagates_base_exception(self):
        first, _, source, remote = self.registered_pair()
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remote, async_op=True)
        with patch.object(
            first._agent, "check_xfer_state", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                work.is_completed()
        work.wait()
        self.assertTrue(work.is_success())

    def test_not_found_keeps_transfer_pending(self):
        first, _, source, remote = self.registered_pair()
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remote, async_op=True)
        released = first._agent.released
        with patch.object(
            first._agent, "check_xfer_state", side_effect=nixlNotFoundError("gone")
        ):
            self.assertFalse(work.is_completed())
        self.assertEqual(first._agent.released, released)
        self.assertIn(id(work), first._pending)
        work.wait()
        self.assertEqual(first._agent.released, released + 1)

    def test_mismatched_reload_after_reset_reloads_peer(self):
        first, _, source, remote = self.registered_pair()
        first.write(source.to_view(), remote)
        first._resetting.add("second")
        wrong = replace(remote, metadata=json.dumps(("unexpected", [])).encode())
        with self.assertRaisesRegex(ValueError, "different agent"):
            first.write(source.to_view(), wrong)
        loads = first._agent.loads
        first.write(source.to_view(), remote)
        self.assertEqual(first._agent.loads, loads + 1)

    def test_close_clears_peer_state(self):
        first, _, source, remote = self.registered_pair()
        first.write(source.to_view(), remote)
        first._resetting.add("second")
        first.close()
        self.assertFalse(first._loaded)
        self.assertFalse(first._resetting)

    @parametrize("use_async", [False, True])
    def test_peer_reset_wait_honors_deadline(self, use_async):
        first, _, source, remote = self.registered_pair()
        # Each clock read advances it, so only polling can reach the deadline.
        clock = itertools.count(0, 0.4)

        def sleep(seconds):
            if next(clock) > 10:
                raise AssertionError("deadline ignored")

        fake_time = SimpleNamespace(monotonic=lambda: next(clock), sleep=sleep)
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remote, async_op=True)
            first._resetting.add("second")
            with patch.object(_nixl, "time", fake_time):
                with self.assertRaisesRegex(TimeoutError, "remain pending"):
                    if use_async:
                        asyncio.run(
                            asyncio.wait_for(
                                first.write_async(source.to_view(), remote, timeout=1),
                                5,
                            )
                        )
                    else:
                        first.write(source.to_view(), remote, timeout=1)
        self.assertGreater(next(clock), 1)
        work.wait()

    @parametrize("use_async", [False, True])
    def test_zero_timeout_resets_finished_peer(self, use_async):
        first, _, source, remote = self.registered_pair()
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remote, async_op=True)
        # Finished natively but not yet polled.
        first._resetting.add("second")
        loads = first._agent.loads
        with no_sleep():
            if use_async:
                write = first.write_async(source.to_view(), remote, timeout=0)
                asyncio.run(asyncio.wait_for(write, 5))
            else:
                first.write(source.to_view(), remote, timeout=0)
        self.assertEqual(first._agent.loads, loads + 1)
        self.assertTrue(work.is_success())

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

    def test_ordinary_transfers_do_not_use_cuda_streams(self):
        first, second, memory, remote = self.registered_pair()
        with patch.object(
            first, "_cuda_stream", side_effect=AssertionError("used a CUDA stream")
        ):
            self.assertEqual(first.write(memory.to_view(), remote), 0)
            first.read(memory.to_mutable_view(), remote, async_op=True).wait()
        self.assertFalse(first._cuda_streams)

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
            with self.assertRaisesRegex(RuntimeError, "NIXL_ERR"):
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

    @parametrize("state,length", [("DONE", None), ("ERR", None), ("DONE", 0)])
    def test_completed_work_does_not_require_transport_lock(self, state, length):
        first, second, source, remote = self.registered_pair()
        with patch.object(first._agent, "transfer_state", state):
            work = first.write(source.to_view(length=length), remote, async_op=True)
            self.assertTrue(work.is_completed())
        future = work.get_future()
        error = work.exception()
        locked, release = threading.Event(), threading.Event()

        def hold_lock():
            with first._operation_lock:
                locked.set()
                release.wait(5)

        thread = threading.Thread(target=hold_lock)
        thread.start()
        try:
            self.assertTrue(locked.wait(5))
            self.assertTrue(work.is_completed())
            self.assertEqual(work.is_success(), state == "DONE")
            self.assertIs(work.exception(), error)
            self.assertIs(work.get_future(), future)
            self.assertTrue(future.done())
            if state == "DONE":
                self.assertTrue(work.wait(timedelta(milliseconds=1)))
                self.assertEqual(future.wait(), [])
            else:
                with self.assertRaisesRegex(RuntimeError, "NIXL_ERR"):
                    work.wait(timedelta(milliseconds=1))
                with self.assertRaisesRegex(RuntimeError, "NIXL_ERR"):
                    future.wait()
            self.assertEqual(first._pending, {})
            self.assertEqual(first._transfers, {})
            self.assertEqual(first._agent.released, int(length != 0))
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())

    @parametrize("state", ["DONE", "ERR"])
    def test_completion_publication_waits_for_pending_cleanup(self, state):
        first, second, source, remote = self.registered_pair()
        entered, release = threading.Event(), threading.Event()
        results, errors = [], []

        class PausedPending(dict):
            def pop(self, key):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("pending cleanup was not released")
                return super().pop(key)

        with patch.object(first, "_pending", PausedPending()):
            with patch.object(first._agent, "transfer_state", state):
                work = first.write(source.to_view(), remote, async_op=True)

            def complete():
                try:
                    results.append(work.is_completed())
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=complete)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertIn(id(work), first._pending)
                self.assertEqual(first._agent.released, 1)
                self.assertFalse(work.is_completed())
                self.assertFalse(work._future.done())
            finally:
                release.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())

            self.assertEqual(errors, [])
            self.assertEqual(results, [True])
            self.assertTrue(work.is_completed())
            future = work.get_future()
            self.assertTrue(future.done())
            if state == "DONE":
                self.assertEqual(future.wait(), [])
            else:
                with self.assertRaisesRegex(RuntimeError, "NIXL_ERR"):
                    future.wait()
            self.assertEqual(first._pending, {})
            self.assertEqual(first._transfers, {})
            self.assertEqual(first._agent.released, 1)

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
        with patch.object(
            first._agent, "remove_remote_agent", wraps=first._agent.remove_remote_agent
        ) as remove:
            first.write(source.to_view(), updated)
        remove.assert_called_once_with("second")
        self.assertEqual(first._agent.released, 2)
        self.assertFalse(first._transfers)

    def test_remote_slice_ranges(self):
        _, _, _, remote = self.registered_pair()
        part = remote.slice(2, 4).slice(1, 2)
        self.assertEqual(
            (part.address, part.length, part.region_address, part.region_length),
            (remote.address + 3, 2, remote.address, remote.length),
        )
        self.assertEqual(part.metadata, remote.metadata)
        self.assertEqual(remote.slice(remote.length).length, 0)

    def test_rejects_descriptor_outside_registration(self):
        first, _, source, remote = self.registered_pair()
        for wrong in (
            replace(remote, address=remote.address - 1),
            replace(remote, length=remote.length + 1),
        ):
            with self.assertRaisesRegex(ValueError, "outside its registration"):
                wrong.serialize()
            with self.assertRaisesRegex(ValueError, "outside its registration"):
                first.write(source.to_view(0, 1), wrong)
        data = json.loads(remote.serialize())
        data["fields"]["region_length"] = 0
        with self.assertRaisesRegex(ValueError, "outside its registration"):
            type(remote).deserialize(json.dumps(data).encode())

    def test_slice_unregistered_view(self):
        first, _, source, _ = self.registered_pair()
        view = source.to_view()
        first.unregister_memory(source)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            view.slice(1)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            view[1:]

    def connect_peers(self):
        first, second = self.make_transport_pair()
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            third, fourth = (
                new_transport("nixl", agent_name=name, plugin=self.plugin, timeout=5)
                for name in ("third", "fourth")
            )
        self.addCleanup(fourth.close)
        self.addCleanup(third.close)
        self.assertEqual(first.connect(third.bind()), 0)
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        # The last target is a second registration on the first peer.
        targets = [torch.zeros(8, dtype=torch.uint8) for _ in range(4)]
        remotes = [
            peer.register_memory(target).to_remote_buffer()
            for peer, target in zip((second, third, fourth, second), targets)
        ]
        return first, source, targets, remotes

    def test_connects_multiple_peers(self):
        first, source, targets, remotes = self.connect_peers()
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remotes[0], async_op=True)
            other = first.write(source.to_view(), remotes[1], async_op=True, timeout=0)
            self.assertIn("second", first._agent.remote_metadata)
        work.wait()
        other.wait()
        for remote in remotes[:2]:
            first.write(source.to_view(), remote)
        self.assertEqual(targets[0], torch.arange(8, dtype=torch.uint8))
        self.assertEqual(targets[1], targets[0])
        self.assertEqual(first._peers, {"second", "third"})
        self.assertTrue(first.connected())
        loads = first._agent.loads
        self.assertEqual(first.connect(json.dumps(["third", []]).encode()), 0)
        first.write(source.to_view(), remotes[1])
        self.assertEqual(first._agent.loads, loads + 1)
        with self.assertRaisesRegex(ValueError, "connected peer"):
            first.write(source.to_view(), remotes[2])
        agent = first._agent
        first.close()
        self.assertFalse(agent.remote_metadata)

    def test_connect_rank_multiple_peers(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            transports = [
                new_transport("nixl", agent_name=f"rank{rank}", plugin=self.plugin)
                for rank in range(3)
            ]
        for transport in transports:
            self.addCleanup(transport.close)
        store = dist.HashStore()
        calls = [(0, 1), (0, 2), (1, 0), (2, 0)]
        with ThreadPoolExecutor(len(calls)) as pool:
            futures = [
                pool.submit(
                    transports[rank].connect_rank,
                    peer,
                    store=store,
                    rank=rank,
                    timeout=5,
                )
                for rank, peer in calls
            ]
            for future in futures:
                future.result(timeout=10)
        self.assertEqual(transports[0]._peers, {"rank1", "rank2"})
        source = transports[0].register_memory(torch.arange(8, dtype=torch.uint8))
        for transport in transports[1:]:
            target = torch.zeros(8, dtype=torch.uint8)
            remote = transport.register_memory(target).to_remote_buffer()
            transports[0].write(source.to_view(), remote)
            self.assertEqual(target, torch.arange(8, dtype=torch.uint8))
            self.assertEqual(transport._peers, {"rank0"})

    def test_mismatched_metadata_keeps_other_peer(self):
        first, source, targets, remotes = self.connect_peers()
        first.write(source.to_view(), remotes[1])
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remotes[0], async_op=True)
            wrong = replace(remotes[1], metadata=remotes[0].metadata)
            with self.assertRaisesRegex(ValueError, "different agent"):
                first.write(source.to_view(), wrong)
            self.assertIn("second", first._agent.remote_metadata)
        work.wait()
        # The other peer is reset before its next new descriptor is loaded.
        self.assertIn("second", first._resetting)
        loads = first._agent.loads
        first.write(source.to_view(), remotes[0])
        self.assertEqual(first._agent.loads, loads + 1)
        self.assertNotIn("second", first._resetting)
        self.assertEqual(targets[0], torch.arange(8, dtype=torch.uint8))

    def test_reset_waits_only_for_its_peer(self):
        first, source, targets, remotes = self.connect_peers()
        first.write(source.to_view(), remotes[1])
        with patch.object(_Agent, "transfer_state", "PROC"):
            work = first.write(source.to_view(), remotes[0], async_op=True)
        first._resetting.add("third")
        loads = first._agent.loads
        first.write(source.to_view(), remotes[1], timeout=0)
        self.assertEqual(first._agent.loads, loads + 1)
        self.assertFalse(work._done)
        work.wait()

    def test_reset_marks_only_its_peer(self):
        first, source, targets, remotes = self.connect_peers()
        first.write(source.to_view(), remotes[0])
        first.write(source.to_view(), remotes[1])
        first._resetting.add("third")
        loads = first._agent.loads
        first.write(source.to_view(), remotes[0])
        self.assertEqual(first._agent.loads, loads)
        first._resetting.add("second")
        first.write(source.to_view(), remotes[0])
        self.assertEqual(first._agent.loads, loads + 1)
        self.assertEqual(first._resetting, {"third"})
        # A new registration of an idle peer loads without resetting it.
        first.write(source.to_view(), remotes[3])
        self.assertEqual(first._agent.loads, loads + 2)
        self.assertEqual(len(first._loaded["second"]), 2)
        self.assertEqual(targets[3], targets[0])

    def test_peer_validation(self):
        first, second, source, remote = self.registered_pair()
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
        # NIXL keeps a failed status final, so a raising query is terminal.
        states = ["PROC", "DONE"] if failure == "dispatch" else [error]
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
