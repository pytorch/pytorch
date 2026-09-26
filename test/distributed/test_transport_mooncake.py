# Owner(s): ["oncall: distributed"]

import ctypes
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest import skipIf
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import _mooncake, new_transport
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _Engine:
    engines = {}

    def __init__(self):
        self.registrations = {}
        self.transfers = 0

    def initialize(self, host, metadata_server, protocol, device_name):
        self.options = metadata_server, protocol, device_name
        self.port = len(self.engines) + 1
        self.engines[f"{host}:{self.port}"] = self
        return 0

    def get_rpc_port(self):
        return self.port

    def register_memory(self, address, length):
        if address in self.registrations:
            raise AssertionError("allocation registered twice")
        self.registrations[address] = length
        return 0

    def unregister_memory(self, address):
        del self.registrations[address]
        return 0

    def _validate(self, address, length):
        if not any(
            start <= address and address + length <= start + capacity
            for start, capacity in self.registrations.items()
        ):
            raise AssertionError("transfer exceeds registered memory")

    def _transfer(self, peer, local, remote, length, read):
        self._validate(local, length)
        self.engines[peer]._validate(remote, length)
        self.transfers += 1
        if read:
            ctypes.memmove(local, remote, length)
        else:
            ctypes.memmove(remote, local, length)
        return 0

    def transfer_sync_write(self, peer, local, remote, length):
        return self._transfer(peer, local, remote, length, False)

    def transfer_sync_read(self, peer, local, remote, length):
        return self._transfer(peer, local, remote, length, True)


class TestMooncakeTransport(TransportTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        _Engine.engines.clear()
        backend = SimpleNamespace(TransferEngine=_Engine)
        self.load_backend = _mooncake._load_backend
        self.backend_patch = patch.object(
            _mooncake, "_load_backend", return_value=backend
        )
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.addCleanup(_Engine.engines.clear)

    def make_transport_pair(self):
        first = new_transport("mooncake", host="first")
        second = new_transport("mooncake", host="second")
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        return first, second

    def test_optional_dependency(self):
        with patch.object(_mooncake, "_load_backend", side_effect=RuntimeError):
            self.assertFalse(_mooncake.MooncakeTransport.supported())
        with patch.object(_mooncake, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "mooncake-transfer-engine"):
                self.load_backend()

    def test_initialization_and_ipv6(self):
        with _mooncake.MooncakeTransport(
            "cpu", host="::1", protocol="tcp", device_name="mlx5_0"
        ) as transport:
            self.assertEqual(transport.bind(), b"[::1]:1")
            self.assertEqual(
                transport._engine.options, ("P2PHANDSHAKE", "tcp", "mlx5_0")
            )
        with patch.object(_Engine, "initialize", return_value=-1):
            with self.assertRaisesRegex(RuntimeError, "initialization.*-1"):
                _mooncake.MooncakeTransport("cpu")

    def test_registers_overlapping_views_once(self):
        first, second = self.make_transport_pair()
        tensor = torch.arange(32, dtype=torch.uint8)
        source = first.register_memory(tensor[8:24])
        overlapping = first.register_memory(tensor[16:])
        self.assertFalse(source.reused_registration())
        self.assertTrue(overlapping.reused_registration())
        self.assertEqual(first._engine.registrations, {tensor.data_ptr(): 32})
        destination = torch.zeros(16, dtype=torch.uint8)
        target = second.register_memory(destination)
        first.write(source.to_view(), target.to_remote_buffer())
        self.assertEqual(destination, tensor[8:24])

    @parametrize("read", [False, True])
    def test_validates_transfer(self, read):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.zeros(8, dtype=torch.uint8))
        target = second.register_memory(torch.zeros(4, dtype=torch.uint8))
        transfer = first.read if read else first.write
        remote = target.to_remote_buffer()
        with self.assertRaisesRegex(ValueError, "does not fit"):
            transfer(source.to_mutable_view(), remote)
        with self.assertRaisesRegex(TypeError, "local_buffer"):
            transfer(target.to_mutable_view(), remote)
        with self.assertRaisesRegex(TypeError, "remote_buffer"):
            transfer(source.to_mutable_view(), object())
        with self.assertRaisesRegex(ValueError, "connected peer"):
            transfer(source.to_mutable_view(), replace(remote, endpoint="other"))
        if read:
            with self.assertRaisesRegex(TypeError, "local_buffer"):
                first.read(source.to_view(), remote)
        self.assertEqual(transfer(source.to_mutable_view(8, 0), remote), 0)
        self.assertEqual(first._engine.transfers, 0)

    @parametrize("offset,length", [(-1, None), (9, None), (0, -1), (4, 5)])
    def test_validates_view(self, offset, length):
        first, _ = self.make_transport_pair()
        memory = first.register_memory(torch.zeros(8, dtype=torch.uint8))
        with self.assertRaises(ValueError):
            memory.to_view(offset, length)

    def test_registration_failure(self):
        first, _ = self.make_transport_pair()
        with patch.object(first._engine, "register_memory", return_value=-2):
            with self.assertRaisesRegex(RuntimeError, "registration.*-2"):
                first.register_memory(torch.zeros(8))
        self.assertEqual(first._registrations, {})

    def test_invalid_registration(self):
        first, _ = self.make_transport_pair()
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            first.register_memory(None)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            first.register_memory(torch.zeros(4, 4).t())
        with self.assertRaisesRegex(ValueError, "empty tensor"):
            first.register_memory(torch.empty(0))
        with self.assertRaisesRegex(ValueError, "CPU or CUDA tensors"):
            first.register_memory(torch.empty(4, device="meta"))

    @parametrize("read", [False, True])
    def test_transfer_failure(self, read):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.zeros(8))
        target = second.register_memory(torch.zeros(8))
        operation = "read" if read else "write"
        with patch.object(first._engine, f"transfer_sync_{operation}", return_value=-3):
            with self.assertRaisesRegex(RuntimeError, f"{operation}.*-3"):
                transfer = first.read if read else first.write
                transfer(source.to_mutable_view(), target.to_remote_buffer())

    @parametrize("status", [0, -4])
    def test_close(self, status):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.zeros(8))
        first.register_memory(torch.zeros(4))
        remote = second.register_memory(torch.zeros(8)).to_remote_buffer()
        engine = first._engine
        with patch.object(
            engine, "unregister_memory", return_value=status
        ) as unregister:
            if status:
                with self.assertRaisesRegex(RuntimeError, "unregistration.*-4"):
                    first.close()
            else:
                first.close()
            self.assertEqual(unregister.call_count, 2)
        first.close()
        self.assertFalse(first.connected())
        self.assertIsNone(first._engine)
        self.assertEqual(first._registrations, {})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            first.write(source.to_view(), remote)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            source.to_remote_buffer()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            first.bind()


@skipIf(not _mooncake.MooncakeTransport.supported(), "Mooncake is unavailable")
class TestMooncakeTransportDevice(TestCase):
    @parametrize("read", [False, True])
    def test_transfer(self, device, read):
        is_cpu = torch.device(device).type == "cpu"
        if not is_cpu and not os.environ.get("MOONCAKE_TEST_RDMA"):
            self.skipTest("set MOONCAKE_TEST_RDMA=1 to test GPU RDMA")
        env = {"MC_FORCE_TCP": "1"} if is_cpu else {}
        rdma_device = os.environ.get("MOONCAKE_TEST_DEVICE_NAME", "")
        with (
            patch.dict(os.environ, env),
            _mooncake.MooncakeTransport(
                host="127.0.0.1",
                protocol="tcp" if is_cpu else "rdma",
                device_name="" if is_cpu else rdma_device,
            ) as first,
            _mooncake.MooncakeTransport(
                host="127.0.0.1",
                protocol="tcp" if is_cpu else "rdma",
                device_name="" if is_cpu else rdma_device,
            ) as second,
        ):
            first.connect(second.bind())
            source = torch.arange(32, dtype=torch.uint8, device=device)
            source_memory = first.register_memory(source[8:24])
            self.assertTrue(first.register_memory(source).reused_registration())
            destination = torch.zeros(16, dtype=torch.uint8, device=device)
            target = second.register_memory(destination)
            descriptor = target.to_remote_buffer()
            remote = type(descriptor).deserialize(descriptor.serialize())
            first.write(source_memory.to_view(), remote)
            self.assertEqual(destination, source[8:24])

            # Register another buffer after the engine has cached the peer segment.
            other = torch.full((16,), 42, dtype=torch.uint8, device=device)
            other_memory = second.register_memory(other)
            if read:
                first.read(
                    source_memory.to_mutable_view(), other_memory.to_remote_buffer()
                )
                self.assertEqual(source[8:24], other)
            else:
                first.write(source_memory.to_view(), other_memory.to_remote_buffer())
                self.assertEqual(other, source[8:24])

            cpu = torch.zeros(16, dtype=torch.uint8)
            cpu_memory = first.register_memory(cpu)
            first.read(cpu_memory.to_mutable_view(), remote)
            self.assertEqual(cpu, destination.cpu())
            self.assertIsNone(first.device)


instantiate_parametrized_tests(TestMooncakeTransport)
instantiate_device_type_tests(TestMooncakeTransportDevice, globals())


if __name__ == "__main__":
    run_tests()
