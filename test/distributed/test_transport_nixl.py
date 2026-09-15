# Owner(s): ["oncall: distributed"]

import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import _nixl
from torch.testing._internal.common_utils import run_tests, TestCase


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
    transfer_state = "DONE"

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.backends = dict.fromkeys(config.backends, 1)
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
        if not inc_conn_info or backends != ["UCX"]:
            raise AssertionError("unexpected metadata options")
        return self._metadata([registrations])

    def add_remote_agent(self, metadata):
        name, entries = pickle.loads(metadata)
        self.remote_metadata.setdefault(name, set()).update(entries)
        return name.encode()

    def remove_remote_agent(self, name):
        self.remote_metadata.pop(name, None)

    def register_memory(self, tensor, *, backends):
        if backends != ["UCX"]:
            raise AssertionError("unexpected registration backend")
        memory_type = "VRAM" if tensor.is_cuda else "DRAM"
        descs = _RegistrationDescriptors(tensor, memory_type)
        self.registrations.append(descs)
        return descs

    def deregister_memory(self, descs, *, backends):
        self.registrations.remove(descs)

    def get_xfer_descs(self, entries, *, mem_type):
        return _Descriptors(mem_type, entries)

    def initialize_xfer(self, operation, local, remote, remote_agent, *, backends):
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


class TestNIXLTransport(TransportTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        _Agent.agents.clear()
        _Agent.transfer_state = "DONE"

    @staticmethod
    def backend():
        return SimpleNamespace(nixl_agent=_Agent, nixl_agent_config=_AgentConfig)

    def make_transport_pair(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            first = _nixl.NIXLTransport(agent_name="first")
            second = _nixl.NIXLTransport(agent_name="second")
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        self.assertTrue(first._agent.config.enable_prog_thread)
        return first, second

    def test_supported(self):
        with patch.object(_nixl, "_load_backend", return_value=self.backend()):
            self.assertTrue(_nixl.NIXLTransport.supported())

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

    def test_timeout_releases_transfer(self):
        first, second = self.make_transport_pair()
        first._timeout = 0.001
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(8, dtype=torch.uint8))
        _Agent.transfer_state = "PROC"
        with self.assertRaisesRegex(TimeoutError, "write timed out"):
            first.write(source.to_view(), target.to_remote_buffer())
        self.assertEqual(first._agent.released, 1)
        self.assertEqual(first._transfers, {})
        first.close()
        second.close()


if __name__ == "__main__":
    run_tests()
