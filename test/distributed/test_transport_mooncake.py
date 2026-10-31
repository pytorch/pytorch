# Owner(s): ["oncall: distributed"]

import ctypes
import os
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import new_transport, register_transport
from torch.distributed._transport.mooncake import (
    _transport as _mooncake,
    MooncakeMemory,
    MooncakeRemoteBuffer,
    MooncakeTransport,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class _Opcode:
    Read = "READ"
    Write = "WRITE"


class _Engine:
    """In-process fake of mooncake.engine.TransferEngine.

    Transfers copy bytes directly between the real process addresses of the
    registered tensors, so the shared conformance assertions observe real data
    movement without RDMA hardware.
    """

    _next_port = 20000

    def __init__(self):
        self._registered: set[int] = set()
        self._port = 0
        self.options = None

    def initialize(self, host, metadata_server, protocol, device_name):
        self.options = (metadata_server, protocol, device_name)
        type(self)._next_port += 1
        self._port = type(self)._next_port
        return 0

    def get_rpc_port(self):
        return self._port

    def register_memory(self, address, length, location="*"):
        if address in self._registered:
            raise AssertionError("allocation registered twice")
        self._registered.add(address)
        return 0

    def unregister_memory(self, address):
        self._registered.discard(address)
        return 0

    def transfer_sync(self, peer, buffer, peer_buffer, length, opcode, notify=None):
        if opcode == _Opcode.Write:
            ctypes.memmove(peer_buffer, buffer, length)
        else:
            ctypes.memmove(buffer, peer_buffer, length)
        return 0


def _backend():
    return SimpleNamespace(TransferEngine=_Engine, TransferOpcode=_Opcode)


class TestMooncakeTransport(TransportTestMixin, TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        register_transport("mooncake", MooncakeTransport, replace=True)

    def make_transport_pair(self):
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            first = new_transport("mooncake", host="::1")
            second = new_transport("mooncake", host="::1")
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        return first, second

    def test_initialize_uses_p2p_handshake(self):
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            transport = new_transport(
                "mooncake", "cpu", host="::1", protocol="tcp", device_name="mlx5_0"
            )
        self.addCleanup(transport.close)
        self.assertEqual(transport._engine.options, ("P2PHANDSHAKE", "tcp", "mlx5_0"))
        self.assertTrue(transport._endpoint.startswith("[::1]:"))

    def test_initialization_failure_raises(self):
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            with patch.object(_Engine, "initialize", return_value=-1):
                with self.assertRaises(RuntimeError):
                    new_transport("mooncake")

    def test_descriptor_roundtrip_excludes_metadata_blob(self):
        descriptor = MooncakeRemoteBuffer("[::1]:20001", 4096, 64, 0, "DRAM")
        restored = MooncakeRemoteBuffer.deserialize(descriptor.serialize())
        self.assertEqual(descriptor, restored)
        self.assertEqual(
            set(MooncakeRemoteBuffer._fields),
            {
                "endpoint",
                "address",
                "length",
                "device_id",
                "memory_type",
            },
        )

    def test_registration_reuse_and_validation(self):
        first, _ = self.make_transport_pair()
        tensor = torch.zeros(32, dtype=torch.uint8)
        memory = first.register_memory(tensor)
        self.assertIsInstance(memory, MooncakeMemory)
        self.assertIs(memory._registration, first.register_memory(tensor)._registration)
        with self.assertRaises(ValueError):
            first.register_memory(torch.empty(0))
        with self.assertRaises(ValueError):
            first.register_memory(torch.zeros(4, 4).t())
        with self.assertRaises(TypeError):
            first.register_memory(None)

    def test_transfer_to_unconnected_peer_rejected(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.ones(8, dtype=torch.uint8))
        stranger = MooncakeRemoteBuffer("[::1]:9", 0, 8, 0, "DRAM")
        with self.assertRaises(ValueError):
            first.write(source.to_view(), stranger)

    def test_unregister_then_use_fails(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.ones(8, dtype=torch.uint8))
        remote = second.register_memory(
            torch.zeros(8, dtype=torch.uint8)
        ).to_remote_buffer()
        stale_view = source.to_view()
        first.unregister_memory(source)
        with self.assertRaises(RuntimeError):
            source.to_view()
        with self.assertRaises(RuntimeError):
            first.write(stale_view, remote)


_RUN_INTEGRATION = os.environ.get("RUN_MOONCAKE_INTEGRATION") == "1"


@skipUnless(
    _RUN_INTEGRATION and MooncakeTransport.supported(),
    "set RUN_MOONCAKE_INTEGRATION=1 with a working Mooncake install to run",
)
class TestMooncakeTransportIntegration(TransportTestMixin, TestCase):
    hw_classification = HardwareClassification.GENERIC
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "rdma")

    def make_transport_pair(self):
        first = new_transport(
            "mooncake", "cpu", host="127.0.0.1", protocol=self.protocol
        )
        second = new_transport(
            "mooncake", "cpu", host="127.0.0.1", protocol=self.protocol
        )
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        return first, second


if __name__ == "__main__":
    run_tests()
