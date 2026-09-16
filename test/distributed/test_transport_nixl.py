# Owner(s): ["oncall: distributed"]

import pickle
from dataclasses import dataclass
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
    def __init__(self, *, backends, num_threads, enable_prog_thread):
        self.backends = backends
        self.num_threads = num_threads
        self.enable_prog_thread = enable_prog_thread


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

    def test_timeout_drains_before_releasing_transfer(self):
        first, second = self.make_transport_pair()
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        first._timeout = 1
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(8, dtype=torch.uint8))
        _Agent.transfer_state = "PROC"
        with (
            patch.object(_nixl.time, "monotonic", side_effect=[0, 2, 3]),
            patch.object(
                first._agent, "check_xfer_state", side_effect=["PROC", "DONE"]
            ) as check,
            self.assertRaisesRegex(TimeoutError, "write timed out"),
        ):
            first.write(
                source.to_view(), target.to_remote_buffer(), async_op=True
            ).wait()
        self.assertEqual(check.call_count, 2)
        self.assertEqual(first._agent.released, 1)
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
