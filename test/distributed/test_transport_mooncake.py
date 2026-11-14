# Owner(s): ["oncall: distributed"]

import asyncio
import ctypes
import itertools
import os
from dataclasses import replace
from datetime import timedelta
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
    movement without RDMA hardware. Like Mooncake, a transfer resolves ``peer``
    to that engine and fails unless the local range is registered here and the
    remote range is registered on the peer. Mooncake batch ids are freed
    BatchDesc addresses, so a settled batch id is handed out again.
    """

    _next_port = 20000
    _by_port: dict[int, "_Engine"] = {}

    def __init__(self):
        self._regions: dict[int, int] = {}
        self._port = 0
        self._batches: set[int] = set()
        self._failed: set[int] = set()
        self.options = None

    def initialize(self, host, metadata_server, protocol, device_name):
        self.options = (metadata_server, protocol, device_name)
        type(self)._next_port += 1
        self._port = type(self)._next_port
        self._by_port[self._port] = self
        return 0

    def get_rpc_port(self):
        return self._port

    def register_memory(self, address, length, location="*"):
        if address in self._regions:
            raise AssertionError("allocation registered twice")
        self._regions[address] = length
        return 0

    def unregister_memory(self, address):
        self._regions.pop(address, None)
        return 0

    def _covers(self, address, length):
        return any(
            start <= address and address + length <= start + size
            for start, size in self._regions.items()
        )

    def batch_transfer_async(self, peer, buffers, peer_buffers, lengths, opcode):
        batch_id = next(i for i in itertools.count(1) if i not in self._batches)
        self._batches.add(batch_id)
        target = self._by_port.get(int(peer.rpartition(":")[2]))
        requests = list(zip(buffers, peer_buffers, lengths))
        if target is None or not all(
            self._covers(buffer, length) and target._covers(peer_buffer, length)
            for buffer, peer_buffer, length in requests
        ):
            self._failed.add(batch_id)
            return batch_id
        for buffer, peer_buffer, length in requests:
            if opcode == _Opcode.Write:
                ctypes.memmove(peer_buffer, buffer, length)
            else:
                ctypes.memmove(buffer, peer_buffer, length)
        return batch_id

    def transfer_check_status(self, batch_id):
        self._batches.discard(batch_id)
        if batch_id in self._failed:
            self._failed.discard(batch_id)
            return -1
        return 1


def _backend():
    return SimpleNamespace(TransferEngine=_Engine, TransferOpcode=_Opcode)


class TestMooncakeTransport(TransportTestMixin, TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        register_transport("mooncake", MooncakeTransport, replace=True)
        self.addCleanup(_mooncake._pinned_registrations.clear)

    def make_transport_pair(self):
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            first = new_transport("mooncake", host="127.0.0.1")
            second = new_transport("mooncake", host="127.0.0.1")
        self.addCleanup(second.close)
        self.addCleanup(first.close)
        self.assertEqual(first.connect(second.bind()), 0)
        self.assertEqual(second.connect(first.bind()), 0)
        return first, second

    def test_initialize_uses_p2p_handshake(self):
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            transport = new_transport(
                "mooncake", host="127.0.0.1", protocol="tcp", device_name="mlx5_0"
            )
        self.addCleanup(transport.close)
        self.assertEqual(transport._engine.options, ("P2PHANDSHAKE", "tcp", "mlx5_0"))
        self.assertTrue(transport._endpoint.startswith("127.0.0.1:"))

    def test_ipv6_host_requires_mc_use_ipv6(self):
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            with patch.dict(os.environ, {"MC_USE_IPV6": "1"}):
                transport = new_transport("mooncake", host="::1")
            self.addCleanup(transport.close)
            self.assertTrue(transport._endpoint.startswith("[::1]:"))
            with patch.dict(os.environ):
                os.environ.pop("MC_USE_IPV6", None)
                with self.assertRaises(RuntimeError) as ctx:
                    new_transport("mooncake", host="::1")
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

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
        with patch.object(_Engine, "register_memory", return_value=0) as native:
            memory = first.register_memory(tensor)
            self.assertIs(
                memory._registration, first.register_memory(tensor)._registration
            )
        native.assert_called_once_with(tensor.data_ptr(), 32, "*")
        self.assertIsInstance(memory, MooncakeMemory)
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

    def test_connects_multiple_peers(self):
        first, second = self.make_transport_pair()
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            peers = [new_transport("mooncake", host="127.0.0.1") for _ in range(2)]
        third, fourth = peers
        self.addCleanup(fourth.close)
        self.addCleanup(third.close)
        self.assertEqual(first.connect(third.bind()), 0)
        self.assertEqual(first.connect(third.bind()), 0)
        self.assertEqual(first._peers, {second._endpoint, third._endpoint})
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        targets = [torch.zeros(8, dtype=torch.uint8) for _ in range(3)]
        remotes = [
            peer.register_memory(target).to_remote_buffer()
            for peer, target in zip((second, third, fourth), targets)
        ]
        works = [first.write(source.to_view(), r, async_op=True) for r in remotes[:2]]
        for work in works:
            work.wait()
        self.assertEqual(targets[0], torch.arange(8, dtype=torch.uint8))
        self.assertEqual(targets[1], targets[0])
        with self.assertRaisesRegex(ValueError, "connected peer"):
            first.write(source.to_view(), remotes[2])
        first.close()
        self.assertFalse(first.connected())
        self.assertEqual(first._peers, set())

    def test_transfers_reach_the_descriptor_peer(self):
        first, second = self.make_transport_pair()
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            third = new_transport("mooncake", host="127.0.0.1")
        self.addCleanup(third.close)
        first.connect(third.bind())
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        targets = [torch.zeros(8, dtype=torch.uint8) for _ in range(2)]
        memories = [p.register_memory(t) for p, t in zip((second, third), targets)]
        remotes = [memory.to_remote_buffer() for memory in memories]
        first.write(source.to_view(), remotes[1])
        self.assertEqual(targets[1], torch.arange(8, dtype=torch.uint8))
        self.assertEqual(targets[0].count_nonzero(), 0)
        misrouted = replace(remotes[1], endpoint=remotes[0].endpoint)
        targets[1].zero_()
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            first.write(source.to_view(), misrouted)
        self.assertEqual(targets[1].count_nonzero(), 0)
        second.unregister_memory(memories[0])
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            first.write(source.to_view(), remotes[0])
        self.assertEqual(targets[0].count_nonzero(), 0)

    def test_remote_slice_ranges(self):
        _, second = self.make_transport_pair()
        remote = second.register_memory(
            torch.zeros(8, dtype=torch.uint8)
        ).to_remote_buffer()
        part = remote.slice(2, 4).slice(1, 2)
        self.assertEqual((part.address, part.length), (remote.address + 3, 2))
        self.assertEqual(part.endpoint, remote.endpoint)
        self.assertEqual(remote.slice(remote.length).length, 0)

    def test_slice_unregistered_view(self):
        first, _ = self.make_transport_pair()
        source = first.register_memory(torch.ones(8, dtype=torch.uint8))
        view = source.to_view()
        first.unregister_memory(source)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            view.slice(1)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            view[1:]

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

    def test_close_after_handled_failure_does_not_reraise(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.ones(8, dtype=torch.uint8))
        remote = second.register_memory(
            torch.zeros(8, dtype=torch.uint8)
        ).to_remote_buffer()
        with patch.object(_Engine, "transfer_check_status", return_value=-1):
            with self.assertRaises(RuntimeError):
                first.write(source.to_view(), remote)
        # The handled failure settled and was dropped; close must not resurface
        # it, and a second close/__exit__ must stay a no-op.
        self.assertEqual(first._pending, {})
        first.close()
        first.close()

    def test_failed_transfer_fails_only_its_work(self):
        first, second = self.make_transport_pair()
        with patch.object(_mooncake, "_load_backend", return_value=_backend()):
            third = new_transport("mooncake", host="127.0.0.1")
        self.addCleanup(third.close)
        first.connect(third.bind())
        source = first.register_memory(torch.arange(8, dtype=torch.uint8))
        target = torch.zeros(8, dtype=torch.uint8)
        dead = second.register_memory(torch.zeros(8, dtype=torch.uint8))
        live = third.register_memory(target).to_remote_buffer()
        with patch.object(_Engine, "transfer_check_status", return_value=-1):
            with self.assertRaisesRegex(RuntimeError, "transfer failed"):
                first.write(source.to_view(), dead.to_remote_buffer())
        first.write(source.to_view(), live)
        self.assertEqual(target, torch.arange(8, dtype=torch.uint8))
        with patch.object(_Engine, "unregister_memory", return_value=0) as native:
            first.unregister_memory(source)
        native.assert_called_once_with(source._registration.address)

    def test_timeout_status_pins_memory(self):
        first, second = self.make_transport_pair()
        timed_out = first.register_memory(torch.ones(8, dtype=torch.uint8))
        other = first.register_memory(torch.ones(8, dtype=torch.uint8))
        remote = second.register_memory(
            torch.zeros(8, dtype=torch.uint8)
        ).to_remote_buffer()
        with patch.object(_Engine, "transfer_check_status", return_value=-2) as check:
            with self.assertRaisesRegex(RuntimeError, "stays pinned"):
                first.write(timed_out.to_view(), remote)
        check.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "pinned after a timed-out"):
            first.write(timed_out.to_view(), remote)
        first.write(other.to_view(), remote)
        engine = first._engine
        with patch.object(_Engine, "unregister_memory", return_value=0) as native:
            first.unregister_memory(timed_out)
            first.close()
        native.assert_called_once_with(other._registration.address)
        self.assertEqual(
            _mooncake._pinned_registrations, [(engine, timed_out._registration)]
        )

    def test_connect_rejects_malformed_peer_url(self):
        first, _ = self.make_transport_pair()
        peers = set(first._peers)
        for url in (b"", b"host", b"host:", b":1234", b"host:port", b"\xff:1"):
            with self.assertRaisesRegex(ValueError, r"bind\(\) endpoint"):
                first.connect(url)
        self.assertEqual(first._peers, peers)
        self.assertEqual(first.connect(b"[::1]:1234"), 0)

    def test_reused_batch_id_is_not_dropped(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.ones(8, dtype=torch.uint8))
        remote = second.register_memory(
            torch.zeros(8, dtype=torch.uint8)
        ).to_remote_buffer()
        check = _Engine.transfer_check_status
        resubmitted = []

        def check_then_submit(engine, batch_id):
            # Submit while the poller is between the status check, which frees
            # batch_id, and its bookkeeping, as a concurrent caller could.
            status = check(engine, batch_id)
            if not resubmitted:
                work = first.write(source.to_view(), remote, async_op=True)
                resubmitted.append((work, batch_id in first._batches))
            return status

        with patch.object(_Engine, "transfer_check_status", check_then_submit):
            first.write(source.to_view(), remote)
            work, reused = resubmitted[0]
            self.assertTrue(reused)
            work.wait(timedelta(seconds=5))
        self.assertEqual(first._batches, {})

    def test_close_async_after_handled_failure_does_not_reraise(self):
        first, second = self.make_transport_pair()
        source = first.register_memory(torch.ones(8, dtype=torch.uint8))
        remote = second.register_memory(
            torch.zeros(8, dtype=torch.uint8)
        ).to_remote_buffer()

        async def run():
            with patch.object(_Engine, "transfer_check_status", return_value=-1):
                with self.assertRaises(RuntimeError):
                    await first.write_async(source.to_view(), remote)
            self.assertEqual(first._pending, {})
            await first.close_async()
            await first.close_async()

        asyncio.run(run())


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

    def test_connects_multiple_peers(self):
        first, second = self.make_transport_pair()
        third = new_transport(
            "mooncake", "cpu", host="127.0.0.1", protocol=self.protocol
        )
        self.addCleanup(third.close)
        self.assertEqual(first.connect(third.bind()), 0)
        source = torch.arange(64, dtype=torch.uint8)
        view = first.register_memory(source).to_view()
        targets = [torch.zeros(64, dtype=torch.uint8) for _ in range(2)]
        remotes = [
            peer.register_memory(target).to_remote_buffer()
            for peer, target in zip((second, third), targets)
        ]
        works = [
            first.write(view[:32], remotes[0][16:48], async_op=True),
            first.write(view[32:], remotes[1][:32], async_op=True),
        ]
        for work in works:
            work.wait()
        self.assertEqual(targets[0][16:48], source[:32])
        self.assertEqual(targets[1][:32], source[32:])
        self.assertEqual(targets[0][:16].count_nonzero(), 0)
        self.assertEqual(targets[1][32:].count_nonzero(), 0)


if __name__ == "__main__":
    run_tests()
