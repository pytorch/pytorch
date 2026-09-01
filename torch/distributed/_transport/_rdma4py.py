from __future__ import annotations

import importlib
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

import torch

from ._api import Memory, MemoryView, MutableMemoryView, RemoteBuffer, Transport


if TYPE_CHECKING:
    from triton.language import uint32, uint64
else:
    uint32 = None
    uint64 = None


_URL_HEADER = struct.Struct("!4sBHH")
_URL_MAGIC = b"PTIB"
_URL_VERSION = 1
_MAX_SGE_LENGTH = (1 << 32) - 1
_GPUNETIO_MAX_TRANSFER_SIZE = 1 << 30
_GPUNETIO_SCALAR_ARGUMENTS = (
    "local_address",
    "lkey",
    "remote_address",
    "rkey",
    "length",
    "stripe_length",
    "timeout_ns",
    "assert_status",
)


def _select_rdma_device(
    devices: list[Any], device: torch.device, sysfs_root: Path = Path("/sys")
) -> Any:
    if device.type != "cuda":
        return devices[0]
    try:
        properties: Any = torch.cuda.get_device_properties(device)
        pci_address = (
            f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
            f"{properties.pci_device_id:02x}.0"
        )
        gpu_path = (sysfs_root / "bus/pci/devices" / pci_address).resolve(strict=True)

        def proximity(candidate: Any) -> int:
            try:
                hca_path = (
                    sysfs_root / "class/infiniband" / candidate.name / "device"
                ).resolve(strict=True)
            except OSError:
                return -1
            return next(
                (
                    index
                    for index, paths in enumerate(zip(gpu_path.parts, hca_path.parts))
                    if paths[0] != paths[1]
                ),
                min(len(gpu_path.parts), len(hca_path.parts)),
            )

        return max(devices, key=proximity)
    except (AttributeError, OSError, RuntimeError):
        return devices[0]


@dataclass(frozen=True)
class IBVerbsRemoteBuffer:
    address: int
    length: int
    rkey: int


@dataclass
class _Registration:
    owner: IBVerbsTransport
    tensor: torch.Tensor
    mr: Any
    address: int
    length: int


class IBVerbsMemoryView:
    def __init__(self, registration: _Registration, offset: int, length: int) -> None:
        self._registration = registration
        self._offset = offset
        self._length = length

    def size(self) -> int:
        return self._length

    @property
    def address(self) -> int:
        return self._registration.address + self._offset

    @property
    def lkey(self) -> int:
        return int(self._registration.mr.lkey)

    def _sge(self, offset: int, length: int) -> Any:
        return self._registration.mr.sge(length, offset=self._offset + offset)


class IBVerbsMutableMemoryView(IBVerbsMemoryView):
    pass


class IBVerbsMemory:
    def __init__(self, registration: _Registration, reused: bool = False) -> None:
        self._registration = registration
        self._reused = reused

    def _range(self, offset: int | None, length: int | None) -> tuple[int, int]:
        offset = 0 if offset is None else int(offset)
        if offset < 0 or offset > self._registration.length:
            raise ValueError("offset is outside the registered memory")
        length = self._registration.length - offset if length is None else int(length)
        if length < 0 or length > self._registration.length - offset:
            raise ValueError("view exceeds the registered memory")
        return offset, length

    def to_view(
        self, offset: int | None = None, length: int | None = None
    ) -> IBVerbsMemoryView:
        return IBVerbsMemoryView(self._registration, *self._range(offset, length))

    def to_mutable_view(
        self, offset: int | None = None, length: int | None = None
    ) -> IBVerbsMutableMemoryView:
        return IBVerbsMutableMemoryView(
            self._registration, *self._range(offset, length)
        )

    def to_remote_buffer(self) -> IBVerbsRemoteBuffer:
        return IBVerbsRemoteBuffer(
            self._registration.address,
            self._registration.length,
            int(self._registration.mr.rkey),
        )

    def reused_registration(self) -> bool:
        return self._reused


class _GPUNetIOProvider(Protocol):
    def transfer(
        self,
        opcode: str,
        local_address: int,
        lkey: int,
        remote_address: int,
        rkey: int,
        length: int,
        lanes: int,
    ) -> None: ...

    def close(self) -> None: ...


_triton_language: Any = None
_triton_gpunetio: Any = None
_triton_wait_send: Any = None


def _triton_wait_send_impl(qp: uint64, ticket: uint64, timeout_ns: uint64):
    start = _triton_language.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        "=l",
        [],
        dtype=_triton_language.uint64,
        is_pure=False,
        pack=1,
    )
    status = _triton_gpunetio.test_send(qp, ticket)
    now = start
    while (status > 0) & (
        now - start < _triton_language.cast(timeout_ns, _triton_language.uint64)
    ):
        status = _triton_gpunetio.test_send(qp, ticket)
        now = _triton_language.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l",
            [],
            dtype=_triton_language.uint64,
            is_pure=False,
            pack=1,
        )
    return _triton_language.where(status > 0, -110, status)


def _triton_put_kernel(
    qps,
    local_address: uint64,
    lkey: uint32,
    remote_address: uint64,
    rkey: uint32,
    length: uint64,
    stripe_length: uint64,
    statuses,
    timeout_ns: uint64,
    assert_status: uint32,
):
    qp_index = _triton_language.program_id(0)
    qp = _triton_language.cast(
        _triton_language.load(qps + qp_index), _triton_language.uint64
    )
    offset = qp_index * stripe_length
    lane_length = _triton_language.minimum(stripe_length, length - offset)
    physical_lane = _triton_language.inline_asm_elementwise(
        "mov.u32 $0, %laneid;",
        "=r",
        [],
        dtype=_triton_language.int32,
        is_pure=True,
        pack=1,
    )
    if physical_lane == 0:
        ticket = _triton_gpunetio.put(
            qp,
            _triton_language.cast(remote_address + offset, _triton_language.uint64),
            _triton_language.cast(rkey, _triton_language.uint32),
            _triton_language.cast(local_address + offset, _triton_language.uint64),
            _triton_language.cast(lkey, _triton_language.uint32),
            _triton_language.cast(lane_length, _triton_language.uint64),
        )
        status = _triton_wait_send(qp, ticket, timeout_ns)
        _triton_language.store(statuses + qp_index, status)
        _triton_language.device_assert(
            (assert_status == 0) | (status == 0), "GPUNetIO RDMA write failed"
        )


def _triton_get_kernel(
    qps,
    local_address: uint64,
    lkey: uint32,
    remote_address: uint64,
    rkey: uint32,
    length: uint64,
    stripe_length: uint64,
    statuses,
    timeout_ns: uint64,
    assert_status: uint32,
):
    qp_index = _triton_language.program_id(0)
    qp = _triton_language.cast(
        _triton_language.load(qps + qp_index), _triton_language.uint64
    )
    offset = qp_index * stripe_length
    lane_length = _triton_language.minimum(stripe_length, length - offset)
    physical_lane = _triton_language.inline_asm_elementwise(
        "mov.u32 $0, %laneid;",
        "=r",
        [],
        dtype=_triton_language.int32,
        is_pure=True,
        pack=1,
    )
    if physical_lane == 0:
        ticket = _triton_gpunetio.get(
            qp,
            _triton_language.cast(remote_address + offset, _triton_language.uint64),
            _triton_language.cast(rkey, _triton_language.uint32),
            _triton_language.cast(local_address + offset, _triton_language.uint64),
            _triton_language.cast(lkey, _triton_language.uint32),
            _triton_language.cast(lane_length, _triton_language.uint64),
        )
        status = _triton_wait_send(qp, ticket, timeout_ns)
        _triton_language.store(statuses + qp_index, status)
        _triton_language.device_assert(
            (assert_status == 0) | (status == 0), "GPUNetIO RDMA read failed"
        )


def _triton_get_mcst_kernel(
    qps,
    local_address: uint64,
    lkey: uint32,
    remote_address: uint64,
    rkey: uint32,
    length: uint64,
    stripe_length: uint64,
    statuses,
    dump_address: uint64,
    dump_lkey: uint32,
    timeout_ns: uint64,
    assert_status: uint32,
):
    qp_index = _triton_language.program_id(0)
    qp = _triton_language.cast(
        _triton_language.load(qps + qp_index), _triton_language.uint64
    )
    offset = qp_index * stripe_length
    lane_length = _triton_language.minimum(stripe_length, length - offset)
    physical_lane = _triton_language.inline_asm_elementwise(
        "mov.u32 $0, %laneid;",
        "=r",
        [],
        dtype=_triton_language.int32,
        is_pure=True,
        pack=1,
    )
    if physical_lane == 0:
        ticket = _triton_gpunetio.get_mcst(
            qp,
            _triton_language.cast(remote_address + offset, _triton_language.uint64),
            _triton_language.cast(rkey, _triton_language.uint32),
            _triton_language.cast(local_address + offset, _triton_language.uint64),
            _triton_language.cast(lkey, _triton_language.uint32),
            _triton_language.cast(lane_length, _triton_language.uint64),
            _triton_language.cast(dump_address + qp_index, _triton_language.uint64),
            _triton_language.cast(dump_lkey, _triton_language.uint32),
        )
        status = _triton_wait_send(qp, ticket, timeout_ns)
        _triton_language.store(statuses + qp_index, status)
        _triton_language.device_assert(
            (assert_status == 0) | (status == 0), "GPUNetIO RDMA read failed"
        )


def _create_triton_kernels(
    triton: Any, language: Any, gpunetio: Any
) -> tuple[Any, Any, Any]:
    global _triton_gpunetio, _triton_language, _triton_wait_send, uint32, uint64
    _triton_language = language
    _triton_gpunetio = gpunetio
    uint32 = language.uint32
    uint64 = language.uint64
    _triton_wait_send = triton.jit(_triton_wait_send_impl)
    jit_options = {
        "do_not_specialize": _GPUNETIO_SCALAR_ARGUMENTS,
        "do_not_specialize_on_alignment": _GPUNETIO_SCALAR_ARGUMENTS,
    }
    put = triton.jit(_triton_put_kernel, **jit_options)
    get = triton.jit(_triton_get_kernel, **jit_options)
    get_mcst = triton.jit(
        _triton_get_mcst_kernel,
        do_not_specialize=(*_GPUNETIO_SCALAR_ARGUMENTS, "dump_address", "dump_lkey"),
        do_not_specialize_on_alignment=(
            *_GPUNETIO_SCALAR_ARGUMENTS,
            "dump_address",
            "dump_lkey",
        ),
    )
    return put, get, get_mcst


class _TritonGPUNetIOProvider:
    def __init__(
        self,
        ibverbs: Any,
        qps: list[Any],
        pd: Any,
        device: torch.device,
        *,
        queue_depth: int,
        timeout: float,
        bitcode: str | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("GPUNetIO requires a CUDA transport device")
        major, minor = torch.cuda.get_device_capability(device)
        if major < 8:
            raise RuntimeError("GPUNetIO requires an NVIDIA sm_80+ GPU")

        try:
            triton = importlib.import_module("triton")
            language = importlib.import_module("triton.language")
            gpunetio = importlib.import_module("ibverbs.gpunetio.triton")
            runtime = importlib.import_module("ibverbs.gpunetio")
        except ImportError as error:
            raise RuntimeError(
                "CUDA graph mode requires ibverbs[gpunetio-triton] and DOCA GPUNetIO"
            ) from error
        if getattr(runtime, "BITCODE_ABI", 0) < 2:
            raise RuntimeError(
                "CUDA graph mode requires an ibverbs release with GPUNetIO "
                "bitcode ABI 2 or newer"
            )

        self._device = device
        self._queue_depth = queue_depth
        self._timeout_ns = int(timeout * 1e9)
        self._failure: BaseException | None = None
        self._device_qps: list[Any] = []
        self._dump_mr: Any = None
        self._dump_tensor: torch.Tensor | None = None
        self._warmed: set[str] = set()
        arch = f"sm_{major}{minor}"
        self._libraries = gpunetio.external_libraries(bitcode, arch=arch)
        self._put, self._get, self._get_mcst = _create_triton_kernels(
            triton, language, gpunetio
        )

        with torch.cuda.device(device):
            self._statuses = torch.empty(len(qps), dtype=torch.int32, device=device)
            self._qp_addresses = torch.tensor(
                [0] * len(qps), dtype=torch.int64, device=device
            )
            if major < 9:
                self._dump_tensor = torch.empty(
                    len(qps), dtype=torch.uint8, device=device
                )
                cuda = importlib.import_module("ibverbs.cuda")
                self._dump_mr = cuda.register_tensor(
                    pd, self._dump_tensor, ibverbs.AccessFlags.LOCAL_WRITE
                )
            try:
                for qp in qps:
                    self._device_qps.append(
                        runtime.DeviceQP.export(qp, gpu=device.index)
                    )
                self._qp_addresses.copy_(
                    torch.tensor(
                        [handle.device_ptr for handle in self._device_qps],
                        dtype=torch.int64,
                        device=device,
                    )
                )
            except Exception:
                self.close()
                raise

    def transfer(
        self,
        opcode: str,
        local_address: int,
        lkey: int,
        remote_address: int,
        rkey: int,
        length: int,
        lanes: int,
    ) -> None:
        if self._failure is not None:
            raise RuntimeError("GPUNetIO provider failed") from self._failure
        stripe_length = (length + lanes - 1) // lanes
        segments = (
            stripe_length + _GPUNETIO_MAX_TRANSFER_SIZE - 1
        ) // _GPUNETIO_MAX_TRANSFER_SIZE
        if opcode == "read" and self._dump_mr is not None:
            segments += 1
        if segments > self._queue_depth:
            raise ValueError("transfer exceeds the configured queue depth")
        args = (
            self._qp_addresses,
            local_address,
            lkey,
            remote_address,
            rkey,
            length,
            stripe_length,
            self._statuses,
        )
        with torch.cuda.device(self._device):
            capturing = torch.cuda.is_current_stream_capturing()
            kernel_name = opcode if self._dump_mr is None else f"{opcode}_mcst"
            if capturing and kernel_name not in self._warmed:
                raise RuntimeError(
                    f"warm up one GPUNetIO {opcode} before CUDA graph capture"
                )
            if opcode == "write":
                self._put[(lanes,)](
                    *args,
                    self._timeout_ns,
                    int(capturing),
                    num_warps=1,
                    extern_libs=self._libraries,
                    enable_reflect_ftz=False,
                    debug=True,
                )
            elif self._dump_mr is None:
                self._get[(lanes,)](
                    *args,
                    self._timeout_ns,
                    int(capturing),
                    num_warps=1,
                    extern_libs=self._libraries,
                    enable_reflect_ftz=False,
                    debug=True,
                )
            else:
                self._get_mcst[(lanes,)](
                    *args,
                    self._dump_mr.addr,
                    self._dump_mr.lkey,
                    self._timeout_ns,
                    int(capturing),
                    num_warps=1,
                    extern_libs=self._libraries,
                    enable_reflect_ftz=False,
                    debug=True,
                )
            if capturing:
                return
            torch.cuda.current_stream(self._device).synchronize()
            statuses = self._statuses[:lanes].tolist()
            self._warmed.add(kernel_name)
        if -110 in statuses:
            error = TimeoutError(f"GPUNetIO {opcode} timed out")
            self._failure = error
            raise error
        if any(statuses):
            error = RuntimeError(f"GPUNetIO {opcode} failed: statuses={statuses}")
            self._failure = error
            raise error

    def close(self) -> None:
        if self._device_qps:
            with torch.cuda.device(self._device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "cannot close GPUNetIO during CUDA graph capture"
                    )
                torch.cuda.synchronize(self._device)
        for device_qp in reversed(self._device_qps):
            device_qp.close()
        self._device_qps.clear()
        if self._dump_mr is not None:
            self._dump_mr.close()
            self._dump_mr = None
        self._dump_tensor = None


class IBVerbsTransport(Transport):
    """RDMA transport backed by the ``ibverbs`` package from rdma4py."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        hca: str | None = None,
        port: int = 1,
        gid_index: int | None = None,
        num_qps: int = 4,
        stripe_threshold: int = 1 << 20,
        queue_depth: int = 128,
        timeout: float = 30.0,
        cuda_graph: bool = False,
        gpunetio_provider: _GPUNetIOProvider | None = None,
    ) -> None:
        super().__init__(device)
        if num_qps <= 0:
            raise ValueError("num_qps must be positive")
        if stripe_threshold <= 0:
            raise ValueError("stripe_threshold must be positive")
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if gpunetio_provider is not None and not cuda_graph:
            raise ValueError("gpunetio_provider requires cuda_graph=True")
        if cuda_graph and self.device.type != "cuda":
            raise ValueError("cuda_graph mode requires a CUDA device")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        try:
            self._ib = importlib.import_module("ibverbs")
        except ImportError as error:
            raise RuntimeError(
                "ibverbs transport requires the 'ibverbs' distribution from rdma4py"
            ) from error

        devices = self._ib.get_device_list()
        if not devices:
            raise RuntimeError("ibverbs found no RDMA devices")
        if hca is None:
            rdma_device = _select_rdma_device(devices, self.device)
        else:
            rdma_device = next(
                (candidate for candidate in devices if candidate.name == hca), None
            )
            if rdma_device is None:
                names = ", ".join(candidate.name for candidate in devices)
                raise RuntimeError(f"RDMA device {hca!r} not found; available: {names}")

        self._port = int(port)
        self._hca = rdma_device.name
        self._gid_index = gid_index
        self._num_qps = int(num_qps)
        self._stripe_threshold = int(stripe_threshold)
        self._queue_depth = int(queue_depth)
        self._timeout = float(timeout)
        self._cuda_graph = cuda_graph
        self._graph_provider = gpunetio_provider
        self._context = rdma_device.open()
        try:
            self._pd = self._context.alloc_pd()
        except Exception:
            self._context.close()
            raise
        self._send_cqs: list[Any] = []
        self._recv_cqs: list[Any] = []
        self._qps: list[Any] = []
        self._registrations: dict[tuple[int, int], _Registration] = {}
        self._bound_url: bytes | None = None
        self._connected = False
        self._closed = False
        self._next_wr_id = 1
        self._transfer_lock = threading.Lock()

    @staticmethod
    def supported() -> bool:
        try:
            ibverbs = importlib.import_module("ibverbs")
            return bool(ibverbs.get_device_list())
        except (ImportError, OSError, RuntimeError):
            return False

    def _select_gid_index(self, port_attr: Any) -> int:
        if self._gid_index is not None:
            return int(self._gid_index)
        selected: tuple[int, int] | None = None
        for index in range(int(port_attr.gid_tbl_len)):
            raw = bytes(self._context.query_gid(self._port, index).raw)
            if raw == bytes(16):
                continue
            score = 0 if raw.startswith(b"\xfe\x80") else 1
            gid_type = Path(
                f"/sys/class/infiniband/{self._hca}/ports/{self._port}/"
                f"gid_attrs/types/{index}"
            )
            try:
                if "v2" in gid_type.read_text().lower():
                    score += 2
            except OSError:
                pass
            if selected is None or score > selected[0]:
                selected = score, index
        if selected is not None:
            return selected[1]
        raise RuntimeError(f"RDMA port {self._port} has no usable GID")

    def bind(self) -> bytes:
        self._ensure_open()
        if self._bound_url is not None:
            return self._bound_url
        port_attr = self._context.query_port(self._port)
        self._gid_index = self._select_gid_index(port_attr)
        gid = self._context.query_gid(self._port, self._gid_index)
        try:
            for _ in range(self._num_qps):
                send_cq = self._context.create_cq(self._queue_depth + 1)
                recv_cq = self._context.create_cq(self._queue_depth + 1)
                self._send_cqs.append(send_cq)
                self._recv_cqs.append(recv_cq)
                self._qps.append(
                    self._pd.create_qp(
                        self._ib.QPInitAttr(
                            send_cq=send_cq,
                            recv_cq=recv_cq,
                            qp_type=self._ib.QPType.RC,
                            max_send_wr=self._queue_depth,
                            max_recv_wr=1,
                        )
                    )
                )
            infos = [
                self._ib.local_qp_info(qp, port_attr, gid, port=self._port).to_bytes()
                for qp in self._qps
            ]
        except Exception:
            self.close()
            raise
        info_size = len(infos[0])
        if any(len(info) != info_size for info in infos):
            self.close()
            raise RuntimeError("ibverbs returned inconsistent QPInfo encodings")
        self._bound_url = _URL_HEADER.pack(
            _URL_MAGIC, _URL_VERSION, len(infos), info_size
        ) + b"".join(infos)
        return self._bound_url

    def connect(self, peer_url: bytes) -> int:
        self._ensure_open()
        if self._connected:
            return 0
        if self._bound_url is None:
            raise RuntimeError("bind() must be called before connect()")
        peer_url = bytes(peer_url)
        if len(peer_url) < _URL_HEADER.size:
            raise ValueError("invalid ibverbs peer URL")
        magic, version, count, qp_info_size = _URL_HEADER.unpack_from(peer_url)
        if magic != _URL_MAGIC or version != _URL_VERSION:
            raise ValueError("unsupported ibverbs peer URL")
        if count != len(self._qps):
            raise ValueError(
                f"peer uses {count} queue pairs; local transport uses {len(self._qps)}"
            )
        expected_length = _URL_HEADER.size + count * qp_info_size
        if qp_info_size == 0 or len(peer_url) != expected_length:
            raise ValueError("invalid ibverbs peer URL length")
        access = (
            self._ib.AccessFlags.LOCAL_WRITE
            | self._ib.AccessFlags.REMOTE_WRITE
            | self._ib.AccessFlags.REMOTE_READ
        )
        try:
            for index, qp in enumerate(self._qps):
                start = _URL_HEADER.size + index * qp_info_size
                remote = self._ib.QPInfo.from_bytes(
                    peer_url[start : start + qp_info_size]
                )
                self._ib.connect_rc(
                    qp,
                    remote,
                    port=self._port,
                    sgid_index=self._gid_index,
                    access=access,
                )
        except Exception:
            self.close()
            raise
        self._connected = True
        return 0

    def connected(self) -> bool:
        return self._connected and not self._closed

    def register_memory(self, tensor: torch.Tensor) -> Memory:
        self._ensure_open()
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")
        if tensor.numel() == 0:
            raise ValueError("cannot register an empty tensor")
        if tensor.device.type != self.device.type or (
            self.device.index is not None and tensor.device.index != self.device.index
        ):
            raise ValueError(
                f"tensor is on {tensor.device}; transport uses {self.device}"
            )
        length = tensor.numel() * tensor.element_size()
        key = (tensor.data_ptr(), length)
        if registration := self._registrations.get(key):
            return IBVerbsMemory(registration, reused=True)
        access = (
            self._ib.AccessFlags.LOCAL_WRITE
            | self._ib.AccessFlags.REMOTE_WRITE
            | self._ib.AccessFlags.REMOTE_READ
        )
        if tensor.is_cuda:
            try:
                cuda = importlib.import_module("ibverbs.cuda")
            except ImportError as error:
                raise RuntimeError(
                    "CUDA tensors require ibverbs GPUDirect support"
                ) from error
            mr = cuda.register_tensor(self._pd, tensor, access)
            address = int(mr.addr)
        else:
            mr = self._ib.reg_tensor(self._pd, tensor, access)
            address = int(mr.addr)
        registration = _Registration(self, tensor, mr, address, length)
        self._registrations[key] = registration
        return IBVerbsMemory(registration)

    def _validate_transfer(
        self,
        local_buffer: MemoryView,
        remote_buffer: RemoteBuffer,
        *,
        mutable: bool,
    ) -> tuple[IBVerbsMemoryView, IBVerbsRemoteBuffer]:
        self._ensure_open()
        if not self.connected():
            raise RuntimeError("ibverbs transport is not connected")
        expected = IBVerbsMutableMemoryView if mutable else IBVerbsMemoryView
        if not isinstance(local_buffer, expected):
            raise TypeError(f"local buffer must be {expected.__name__}")
        if local_buffer._registration.owner is not self:
            raise ValueError("local buffer belongs to another transport")
        if not isinstance(remote_buffer, IBVerbsRemoteBuffer):
            raise TypeError("remote buffer must be IBVerbsRemoteBuffer")
        if local_buffer.size() > remote_buffer.length:
            raise ValueError("local buffer exceeds remote buffer")
        if local_buffer.size() <= 0:
            raise ValueError("transfer length must be positive")
        return local_buffer, remote_buffer

    def _lanes(self, length: int) -> int:
        return min(self._num_qps, length) if length >= self._stripe_threshold else 1

    def _host_transfer(
        self,
        opcode: Any,
        local: IBVerbsMemoryView,
        remote: IBVerbsRemoteBuffer,
    ) -> None:
        length = local.size()
        lanes = self._lanes(length)
        stripe_length = (length + lanes - 1) // lanes
        active: list[int] = []
        for lane in range(lanes):
            lane_offset = lane * stripe_length
            lane_length = min(stripe_length, length - lane_offset)
            segment_count = (lane_length + _MAX_SGE_LENGTH - 1) // _MAX_SGE_LENGTH
            if segment_count > self._queue_depth:
                raise ValueError("transfer exceeds the configured queue depth")
            requests = []
            for segment in range(segment_count):
                offset = lane_offset + segment * _MAX_SGE_LENGTH
                segment_length = min(
                    _MAX_SGE_LENGTH, lane_offset + lane_length - offset
                )
                wr_id = self._next_wr_id
                self._next_wr_id += 1
                requests.append(
                    self._ib.SendWR(
                        wr_id=wr_id,
                        sg_list=[local._sge(offset, segment_length)],
                        opcode=opcode,
                        send_flags=(
                            self._ib.SendFlags.SIGNALED
                            if segment == segment_count - 1
                            else 0
                        ),
                        remote_addr=remote.address + offset,
                        rkey=remote.rkey,
                    )
                )
            self._qps[lane].post_send(requests)
            active.append(lane)
        deadline = time.monotonic() + self._timeout
        for lane in active:
            while True:
                completions = self._send_cqs[lane].poll(1)
                if completions:
                    completions[0].raise_for_status()
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for ibverbs completion")

    def _transfer(
        self,
        operation: str,
        local: IBVerbsMemoryView,
        remote: IBVerbsRemoteBuffer,
    ) -> int:
        length = local.size()
        with self._transfer_lock:
            if self._cuda_graph and self._graph_provider is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm up one GPUNetIO transfer before CUDA graph capture"
                    )
                self._graph_provider = _TritonGPUNetIOProvider(
                    self._ib,
                    self._qps,
                    self._pd,
                    self.device,
                    queue_depth=self._queue_depth,
                    timeout=self._timeout,
                )
            if self._graph_provider is not None:
                self._graph_provider.transfer(
                    operation,
                    local.address,
                    local.lkey,
                    remote.address,
                    remote.rkey,
                    length,
                    self._lanes(length),
                )
                return 0
            if self.device.type == "cuda":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "CUDA graph capture requires cuda_graph=True when creating "
                        "the ibverbs transport"
                    )
                torch.cuda.current_stream(self.device).synchronize()
            opcode = (
                self._ib.WROpcode.RDMA_WRITE
                if operation == "write"
                else self._ib.WROpcode.RDMA_READ
            )
            self._host_transfer(opcode, local, remote)
            if operation == "read" and self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    importlib.import_module("ibverbs.cuda").flush_gpudirect_writes()
        return 0

    def write(self, local_buffer: MemoryView, remote_buffer: RemoteBuffer) -> int:
        local, remote = self._validate_transfer(
            local_buffer, remote_buffer, mutable=False
        )
        return self._transfer("write", local, remote)

    def read(self, local_buffer: MutableMemoryView, remote_buffer: RemoteBuffer) -> int:
        local, remote = self._validate_transfer(
            local_buffer, remote_buffer, mutable=True
        )
        return self._transfer("read", local, remote)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ibverbs transport is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._connected = False
        resources: list[Any] = []
        if self._graph_provider is not None:
            resources.append(self._graph_provider)
        resources.extend(
            registration.mr for registration in self._registrations.values()
        )
        resources.extend(reversed(self._qps))
        resources.extend(reversed(self._recv_cqs))
        resources.extend(reversed(self._send_cqs))
        resources.extend((self._pd, self._context))
        for resource in resources:
            resource.close()
        self._graph_provider = None
        self._registrations.clear()
        self._qps.clear()
        self._recv_cqs.clear()
        self._send_cqs.clear()
        self._closed = True


__all__ = [
    "IBVerbsMemory",
    "IBVerbsMemoryView",
    "IBVerbsMutableMemoryView",
    "IBVerbsRemoteBuffer",
    "IBVerbsTransport",
]
