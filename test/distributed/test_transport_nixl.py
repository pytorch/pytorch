# Owner(s): ["oncall: distributed"]

import asyncio
import pickle
import threading
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import _nixl, new_transport
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


@dataclass
class _RegistrationDescriptors:
    tensor: torch.Tensor
    memory_type: str


@dataclass
class _Transfer:
    operation: str
    local: _Descriptors
    remote: _Descriptors
    remote_agent: str


class _AgentConfig:
    def __init__(
        self, *, backends, num_threads, enable_prog_thread, capture_telemetry=False
    ):
        self.backends = backends
        self.num_threads = num_threads
        self.enable_prog_thread = enable_prog_thread
        self.capture_telemetry = capture_telemetry


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
        return pickle.dumps((self.name, entries))

    def get_agent_metadata(self):
        return self._metadata(self.registrations)

    def get_partial_agent_metadata(self, registrations, *, inc_conn_info, backends):
        if not inc_conn_info or backends != list(self.backends):
            raise AssertionError("unexpected metadata options")
        return self._metadata([registrations])

    def add_remote_agent(self, metadata):
        name, entries = pickle.loads(metadata)
        self.remote_metadata.setdefault(name, set()).update(entries)
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
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        self.assertTrue(first._agent.config.enable_prog_thread)
        return first, second

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
            remote = pickle.loads(pickle.dumps(target.to_remote_buffer()))

            self.assertEqual(first.write(source.to_view(2, 4), remote), 0)
            self.assertEqual(first.write(source.to_view(2, 4), remote), 0)
            self.assertEqual(target_tensor[:4], torch.arange(2, 6, dtype=torch.uint8))
            self.assertEqual(first._agent.released, 0)
        finally:
            first.close()
            second.close()
        self.assertEqual(first_agent.released, 1)

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
        import threading

        first, second = self.make_transport_pair()
        release = threading.Event()
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        self.addCleanup(release.set)
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(8, dtype=torch.uint8))
        remote = target.to_remote_buffer()
        agent = first._agent
        with patch.object(
            agent, "transfer", side_effect=lambda handle: (release.wait(5) and "DONE")
        ):
            work = first.write(source.to_view(), remote, async_op=True, timeout=0.01)
            with self.assertRaises(TimeoutError):
                work.wait()
            self.assertFalse(work.is_completed())
            self.assertEqual(agent.released, 0)
            with self.assertRaises(TimeoutError):
                first.close(timeout=0.01)
            self.assertTrue(agent.registrations)
            release.set()
            first.close(timeout=5)
        self.assertTrue(work.is_completed())
        self.assertEqual(agent.registrations, [])
        self.assertEqual(agent.released, 1)

    @parametrize(
        "operation",
        [
            "bind",
            "connect",
            "register",
            "descriptor",
            "read",
            "write",
            "read_async",
            "write_async",
        ],
    )
    def test_operation_timeout(self, operation):
        first, second = self.make_transport_pair()
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(8, dtype=torch.uint8))
        remote = target.to_remote_buffer()
        release = threading.Event()
        self.addCleanup(release.set)
        methods = {
            "bind": ("get_agent_metadata", lambda: first.bind(timeout=0.01)),
            "connect": (
                "add_remote_agent",
                lambda: first.connect(second.bind(), timeout=0.01),
            ),
            "register": (
                "register_memory",
                lambda: first.register_memory(torch.ones(9), timeout=0.01),
            ),
            "descriptor": (
                "get_partial_agent_metadata",
                lambda: source.to_remote_buffer(timeout=0.01),
            ),
            "read": (
                "transfer",
                lambda: first.read(source.to_mutable_view(), remote, timeout=0.01),
            ),
            "write": (
                "transfer",
                lambda: first.write(source.to_view(), remote, timeout=0.01),
            ),
            "read_async": (
                "transfer",
                lambda: asyncio.run(
                    first.read_async(source.to_mutable_view(), remote, timeout=0.01)
                ),
            ),
            "write_async": (
                "transfer",
                lambda: asyncio.run(
                    first.write_async(source.to_view(), remote, timeout=0.01)
                ),
            ),
        }
        method, call = methods[operation]
        if operation == "connect":
            first._peer_name = None
        original = getattr(first._agent, method)

        def blocked(*args, **kwargs):
            if not release.wait(5):
                raise RuntimeError("test did not release operation")
            return original(*args, **kwargs)

        with patch.object(first._agent, method, side_effect=blocked):
            with self.assertRaises(TimeoutError):
                call()
            release.set()
            first.close(timeout=5)

    @parametrize("timeout", [-1, float("nan"), float("inf")])
    def test_invalid_timeout(self, timeout):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            with self.assertRaises(ValueError):
                _nixl.NIXLTransport(timeout=timeout)
        first, second = self.make_transport_pair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
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
        first, second = self.make_transport_pair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        tensor = torch.ones(8)
        source = first.register_memory(tensor)
        target = second.register_memory(torch.zeros(8))
        remote = target.to_remote_buffer()
        self.assertEqual(first.write(source.to_view(0, 0), remote), 0)
        self.assertEqual(first._transfers, {})
        with self.assertRaises(TypeError):
            source.to_view(0.5)
        tensor.set_(torch.zeros(8))
        with self.assertRaisesRegex(RuntimeError, "storage was replaced"):
            first.write(source.to_view(), remote)

    def test_changed_remote_metadata_rebuilds_handle(self):
        first, second = self.make_transport_pair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        source = first.register_memory(torch.ones(8))
        target = second.register_memory(torch.zeros(8))
        remote = target.to_remote_buffer()
        first.write(source.to_view(), remote)
        name, entries = pickle.loads(remote.metadata)
        updated = replace(remote, metadata=pickle.dumps((name, entries + entries)))
        first.write(source.to_view(), updated)
        self.assertEqual(len(first._transfers), 2)

    def test_peer_validation(self):
        first, second = self.make_transport_pair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        source = first.register_memory(torch.ones(8))
        target = second.register_memory(torch.zeros(8))
        remote = target.to_remote_buffer()
        with self.assertRaisesRegex(RuntimeError, "already connected"):
            first.connect(second.bind())
        with self.assertRaisesRegex(ValueError, "connected peer"):
            first.write(source.to_view(), replace(remote, agent_name="other"))
        with self.assertRaisesRegex(ValueError, "different agent"):
            first.write(
                source.to_view(), replace(remote, metadata=pickle.dumps(("other", [])))
            )
        self.assertEqual(first._transfers, {})

    @parametrize("failure", ["dispatch", "query"])
    def test_error_drains_before_releasing_transfer(self, failure):
        first, second = self.make_transport_pair()
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(8, dtype=torch.uint8))
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
            first.write(
                source.to_view(), target.to_remote_buffer(), async_op=True
            ).wait()
        self.assertEqual(check.call_count, len(states))
        self.assertEqual(first._agent.released, 1)
        self.assertEqual(first._transfers, {})


if __name__ == "__main__":
    run_tests()
