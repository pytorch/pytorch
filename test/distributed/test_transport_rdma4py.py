# Owner(s): ["oncall: distributed"]

from __future__ import annotations

import ctypes
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport import _rdma4py
from torch.testing._internal.common_utils import run_tests, TestCase


class _QPInfo:
    _STRUCT = struct.Struct("!IIH16sBB")

    def __init__(
        self,
        qp_num: int,
        psn: int = 0,
        lid: int = 1,
        gid: bytes = b"\x01" * 16,
        port: int = 1,
        mtu: int = 4,
    ) -> None:
        self.values = qp_num, psn, lid, gid, port, mtu

    def to_bytes(self) -> bytes:
        return self._STRUCT.pack(*self.values)

    @classmethod
    def from_bytes(cls, value: bytes) -> _QPInfo:
        return cls(*cls._STRUCT.unpack(value))


class _Completion:
    def raise_for_status(self) -> None:
        pass


class _CQ:
    def __init__(self) -> None:
        self.completions: list[_Completion] = []
        self.closed = False

    def poll(self, count: int) -> list[_Completion]:
        result = self.completions[:count]
        del self.completions[:count]
        return result

    def close(self) -> None:
        self.closed = True


class _QP:
    _next_qp = 1

    def __init__(self, cq: _CQ) -> None:
        self.qp_num = self._next_qp
        _QP._next_qp += 1
        self.cq = cq
        self.writes: list[SimpleNamespace] = []
        self.closed = False

    def post_send(self, requests: SimpleNamespace | list[SimpleNamespace]) -> None:
        if not isinstance(requests, list):
            requests = [requests]
        self.writes.extend(requests)
        for request in requests:
            local = request.sg_list[0]
            if request.opcode == _IBVerbs.WROpcode.RDMA_WRITE:
                ctypes.memmove(request.remote_addr, local.addr, local.length)
            elif request.opcode == _IBVerbs.WROpcode.RDMA_READ:
                ctypes.memmove(local.addr, request.remote_addr, local.length)
        if requests[-1].send_flags:
            self.cq.completions.append(_Completion())

    def close(self) -> None:
        self.closed = True


class _MR:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.addr = tensor.data_ptr()
        self.lkey = 11
        self.rkey = 12
        self.closed = False

    def sge(self, length: int, offset: int = 0) -> SimpleNamespace:
        return SimpleNamespace(addr=self.addr + offset, length=length, lkey=self.lkey)

    def close(self) -> None:
        self.closed = True


class _PD:
    def __init__(self) -> None:
        self.closed = False

    def create_qp(self, init: SimpleNamespace) -> _QP:
        return _QP(init.send_cq)

    def close(self) -> None:
        self.closed = True


class _Context:
    def __init__(self) -> None:
        self.pd = _PD()
        self.closed = False

    def alloc_pd(self) -> _PD:
        return self.pd

    def query_port(self, port: int) -> SimpleNamespace:
        return SimpleNamespace(lid=1, active_mtu=4, gid_tbl_len=1)

    def query_gid(self, port: int, index: int) -> SimpleNamespace:
        return SimpleNamespace(raw=b"\x01" * 16)

    def create_cq(self, depth: int) -> _CQ:
        return _CQ()

    def close(self) -> None:
        self.closed = True


class _Device:
    name = "mlx5_0"

    def __init__(self) -> None:
        self.context = _Context()

    def open(self) -> _Context:
        return self.context


class _IBVerbs:
    AccessFlags = SimpleNamespace(LOCAL_WRITE=1, REMOTE_WRITE=2, REMOTE_READ=4)
    QPType = SimpleNamespace(RC=1)
    SendFlags = SimpleNamespace(SIGNALED=1)
    WROpcode = SimpleNamespace(RDMA_WRITE=2, RDMA_READ=3)
    QPInfo = _QPInfo

    def __init__(self) -> None:
        self.device = _Device()
        self.connections: list[tuple[_QP, _QPInfo]] = []

    def get_device_list(self) -> list[_Device]:
        return [self.device]

    def QPInitAttr(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)

    def local_qp_info(
        self, qp: _QP, port_attr: object, gid: object, *, port: int
    ) -> _QPInfo:
        return _QPInfo(qp.qp_num, port=port)

    def connect_rc(self, qp: _QP, remote: _QPInfo, **kwargs: object) -> None:
        self.connections.append((qp, remote))

    def reg_tensor(self, pd: _PD, tensor: torch.Tensor, access: int) -> _MR:
        return _MR(tensor)

    def SendWR(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(**kwargs)


class TestIBVerbsTransport(TransportTestMixin, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ibverbs = _IBVerbs()
        self.import_module = mock.patch.object(
            _rdma4py.importlib,
            "import_module",
            side_effect=lambda name: self.ibverbs if name == "ibverbs" else None,
        )
        self.import_module.start()

    def tearDown(self) -> None:
        self.import_module.stop()
        super().tearDown()

    def _connected(self, **kwargs: object) -> _rdma4py.IBVerbsTransport:
        transport = _rdma4py.IBVerbsTransport("cpu", **kwargs)
        url = transport.bind()
        self.assertEqual(transport.connect(url), 0)
        self.assertTrue(transport.connected())
        return transport

    def make_transport_pair(self):
        first = _rdma4py.IBVerbsTransport("cpu", num_qps=2)
        second = _rdma4py.IBVerbsTransport("cpu", num_qps=2)
        first_url = first.bind()
        second_url = second.bind()
        self.assertEqual(first.connect(second_url), 0)
        self.assertEqual(second.connect(first_url), 0)
        return first, second

    def test_supported(self) -> None:
        self.assertTrue(_rdma4py.IBVerbsTransport.supported())

    def test_selects_topology_local_hca(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = root / "devices/pci0/switch0/gpu"
            near = root / "devices/pci0/switch0/nic"
            far = root / "devices/pci1/nic"
            for path in (gpu, near, far):
                path.mkdir(parents=True, exist_ok=True)
            pci = root / "bus/pci/devices"
            pci.mkdir(parents=True)
            (pci / "0000:01:00.0").symlink_to(gpu)
            infiniband = root / "class/infiniband"
            for name, target in (("far", far), ("near", near)):
                (infiniband / name).mkdir(parents=True)
                (infiniband / name / "device").symlink_to(target)
            properties = SimpleNamespace(pci_domain_id=0, pci_bus_id=1, pci_device_id=0)
            devices = [SimpleNamespace(name="far"), SimpleNamespace(name="near")]
            with mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ):
                selected = _rdma4py._select_rdma_device(
                    devices, torch.device("cuda:0"), root
                )
            self.assertEqual(selected.name, "near")

    def test_registration_views_and_reuse(self) -> None:
        transport = self._connected(num_qps=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        memory = transport.register_memory(tensor)
        reused = transport.register_memory(tensor)

        self.assertFalse(memory.reused_registration())
        self.assertTrue(reused.reused_registration())
        self.assertEqual(memory.to_view(3, 7).size(), 7)
        self.assertEqual(memory.to_mutable_view(5).size(), 11)
        self.assertEqual(memory.to_remote_buffer().length, 16)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            memory.to_view(10, 7)
        transport.close()

    def test_large_transfers_are_striped_across_queue_pairs(self) -> None:
        transport = self._connected(num_qps=4, stripe_threshold=16)
        memory = transport.register_memory(torch.arange(64, dtype=torch.uint8))
        remote = memory.to_remote_buffer()

        self.assertEqual(transport.write(memory.to_view(), remote), 0)
        self.assertEqual(transport.read(memory.to_mutable_view(), remote), 0)
        self.assertEqual([len(qp.writes) for qp in transport._qps], [2, 2, 2, 2])
        self.assertEqual(
            [qp.writes[0].sg_list[0].addr - remote.address for qp in transport._qps],
            [0, 16, 32, 48],
        )
        self.assertEqual(
            [qp.writes[0].opcode for qp in transport._qps],
            [self.ibverbs.WROpcode.RDMA_WRITE] * 4,
        )
        self.assertEqual(
            [qp.writes[1].opcode for qp in transport._qps],
            [self.ibverbs.WROpcode.RDMA_READ] * 4,
        )
        transport.close()

    def test_small_transfer_uses_one_queue_pair(self) -> None:
        transport = self._connected(num_qps=4, stripe_threshold=1024)
        memory = transport.register_memory(torch.arange(64, dtype=torch.uint8))

        transport.write(memory.to_view(), memory.to_remote_buffer())

        self.assertEqual([len(qp.writes) for qp in transport._qps], [1, 0, 0, 0])
        transport.close()

    def test_rejects_peer_with_different_queue_pair_count(self) -> None:
        transport = _rdma4py.IBVerbsTransport("cpu", num_qps=2)
        peer_url = transport.bind()
        header = _rdma4py._URL_HEADER.pack(
            _rdma4py._URL_MAGIC, _rdma4py._URL_VERSION, 1, _QPInfo._STRUCT.size
        )

        with self.assertRaisesRegex(ValueError, "peer uses 1 queue pairs"):
            transport.connect(header + peer_url[_rdma4py._URL_HEADER.size : -28])
        transport.close()

    def test_rejects_buffer_from_another_transport(self) -> None:
        first = self._connected(num_qps=1)
        second = self._connected(num_qps=1)
        memory = first.register_memory(torch.arange(8, dtype=torch.uint8))

        with self.assertRaisesRegex(ValueError, "another transport"):
            second.write(memory.to_view(), memory.to_remote_buffer())
        first.close()
        second.close()

    def test_cuda_graph_provider_receives_multi_qp_transfer(self) -> None:
        provider = mock.Mock()
        transport = _rdma4py.IBVerbsTransport(
            "cuda:0",
            num_qps=4,
            stripe_threshold=16,
            cuda_graph=True,
            gpunetio_provider=provider,
        )
        transport.connect(transport.bind())
        registration = _rdma4py._Registration(
            transport,
            mock.Mock(),
            SimpleNamespace(lkey=11, rkey=12),
            1000,
            64,
        )
        memory = _rdma4py.IBVerbsMemory(registration)

        transport.write(memory.to_view(), _rdma4py.IBVerbsRemoteBuffer(2000, 64, 13))

        provider.transfer.assert_called_once_with("write", 1000, 11, 2000, 13, 64, 4)
        transport.close()
        provider.close.assert_called_once_with()

    def test_gpunetio_triton_launch_options(self) -> None:
        self.assertEqual(
            _rdma4py._triton_put_kernel.__annotations__["local_address"], "uint64"
        )
        self.assertEqual(_rdma4py._triton_put_kernel.__annotations__["rkey"], "uint32")
        self.assertEqual(
            _rdma4py._triton_put_kernel.__annotations__["assert_status"], "uint32"
        )

        provider = _rdma4py._TritonGPUNetIOProvider.__new__(
            _rdma4py._TritonGPUNetIOProvider
        )
        provider._device = torch.device("cuda:0")
        provider._queue_depth = 128
        provider._timeout_ns = 30_000_000_000
        provider._failure = None
        provider._qp_addresses = mock.Mock()
        provider._statuses = mock.MagicMock()
        provider._statuses.__getitem__.return_value.tolist.return_value = [0]
        provider._libraries = {"rdma4py_gpunetio": "device.bc"}
        provider._warmed = set()
        provider._dump_mr = None
        provider._put = mock.MagicMock()
        launch = provider._put.__getitem__.return_value

        with (
            mock.patch.object(torch.cuda, "device"),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            mock.patch.object(torch.cuda, "current_stream"),
        ):
            provider.transfer("write", 1000, 11, 2000, 13, 64, 1)

        self.assertFalse(launch.call_args.kwargs["enable_reflect_ftz"])
        self.assertTrue(launch.call_args.kwargs["debug"])
        self.assertEqual(launch.call_args.args[-2:], (30_000_000_000, False))

        with self.assertRaisesRegex(ValueError, "queue depth"):
            provider.transfer(
                "write",
                1000,
                11,
                2000,
                13,
                129 * _rdma4py._GPUNETIO_MAX_TRANSFER_SIZE,
                1,
            )

        provider._statuses.__getitem__.return_value.tolist.return_value = [-110]
        with (
            mock.patch.object(torch.cuda, "device"),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            mock.patch.object(torch.cuda, "current_stream"),
            self.assertRaisesRegex(TimeoutError, "timed out"),
        ):
            provider.transfer("write", 1000, 11, 2000, 13, 64, 1)
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            provider.transfer("write", 1000, 11, 2000, 13, 64, 1)

    def test_host_path_rejects_cuda_graph_capture(self) -> None:
        transport = _rdma4py.IBVerbsTransport("cuda:0", num_qps=1)
        transport.connect(transport.bind())
        registration = _rdma4py._Registration(
            transport,
            mock.Mock(),
            SimpleNamespace(lkey=11, rkey=12),
            1000,
            16,
        )
        memory = _rdma4py.IBVerbsMemory(registration)

        with (
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
            self.assertRaisesRegex(RuntimeError, "cuda_graph=True"),
        ):
            transport.write(
                memory.to_view(), _rdma4py.IBVerbsRemoteBuffer(2000, 16, 13)
            )
        transport.close()


class TestGPUNetIOCompilation(TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_triton_kernel_compiles(self) -> None:
        major, minor = torch.cuda.get_device_capability(0)
        if major < 8:
            self.skipTest("GPUNetIO requires sm_80+")
        try:
            import ibverbs.gpunetio as runtime
            import ibverbs.gpunetio.triton as gpunetio
            import triton
            import triton.language as language

            libraries = gpunetio.external_libraries(arch=f"sm_{major}{minor}")
        except (ImportError, OSError, RuntimeError) as error:
            self.skipTest(str(error))
        if runtime.BITCODE_ABI < 2:
            self.skipTest("requires GPUNetIO bitcode ABI 2")

        put, _, _ = _rdma4py._create_triton_kernels(triton, language, gpunetio)
        qps = torch.empty(1, dtype=torch.int64, device="cuda:0")
        statuses = torch.empty(1, dtype=torch.int32, device="cuda:0")
        put.warmup(
            qps,
            0,
            0,
            0,
            0,
            64,
            64,
            statuses,
            1_000_000,
            0,
            grid=(1,),
            num_warps=1,
            extern_libs=libraries,
            enable_reflect_ftz=False,
            debug=True,
        )


if __name__ == "__main__":
    run_tests()
