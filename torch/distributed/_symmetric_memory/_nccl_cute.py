import ctypes

import cutlass
import cutlass.cute as cute
import nccl.core.device.cute as nccl_cute  # pyrefly: ignore [missing-import]
from cuda.bindings import driver
from cutlass.cute.runtime import from_dlpack
from nccl.bindings import nccl as nccl_bindings  # pyrefly: ignore [missing-import]
from packaging.version import Version

import torch
from torch.distributed._symmetric_memory import rendezvous


# Borrow the existing window and device communicator. C++ owns their resources;
# the launch arguments retain the symmetric memory handle through the launch.
class _Window(nccl_cute.Window):
    def __init__(self, pointer, owner):
        self._storage = ctypes.c_void_p(pointer)
        self._owner = owner
        self._resource = None
        self._ptr = None

    def __c_pointers__(self):
        return [ctypes.addressof(self._storage)]


class _DevComm(nccl_cute.DevComm):
    def __init__(self, data, owner):
        self._storage = ctypes.create_string_buffer(data)
        self._owner = owner
        self._resource = None
        self._value = None
        self._materialized_ptr = None

    def __c_pointers__(self):
        return [ctypes.addressof(self._storage)]


def _context(input, group):
    handle = rendezvous(input, group=group)
    if handle is None or not hasattr(handle, "_get_nccl_cute_context"):
        raise RuntimeError("NCCL CuTeDSL requires NCCL symmetric memory")
    major, minor, patch = Version(nccl_bindings.__version__).release
    version = major * 10000 + minor * 100 + patch
    pointer, offset, data, multimem = handle._get_nccl_cute_context(version)
    offset += input.storage_offset() * input.element_size()
    return handle, _DevComm(data, handle), _Window(pointer, handle), offset, multimem


@cute.kernel
def _all_to_all_kernel(
    comm: nccl_cute.DevComm,
    window: nccl_cute.Window,
    out: cute.Tensor,
    base: cutlass.Int64,
    rows: cutlass.Int32,
    src_stride: cutlass.Int64,
    width: cutlass.Int32,
    peer_stride: cutlass.Int64,
    dst_stride: cutlass.Int64,
):
    peer, tile, _ = cute.arch.block_idx()
    _, tiles, _ = cute.arch.grid_dim()
    tid, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.cta()
    bar = nccl_cute.lsa_default(coop, comm, index=peer * tiles + tile)
    bar.sync(coop, nccl_cute.MemoryOrder.ACQUIRE)
    vec = 128 // out.element_type.width
    for row in cutlass.range(tile, rows, tiles):
        ptr = cute.make_ptr(
            out.element_type,
            window.lsa_pointer(base + row * src_stride, peer),
            assumed_align=16,
        )
        for col in cutlass.range(tid * vec, width, 128 * vec):
            src = cute.make_tensor(ptr + col, cute.make_layout(vec))
            dst = cute.make_tensor(
                out.iterator + peer * peer_stride + row * dst_stride + col,
                cute.make_layout(vec),
            )
            cute.autovec_copy(src, dst)
    bar.sync(coop, nccl_cute.MemoryOrder.RELEASE)
    bar.destroy()


@cute.jit
def _all_to_all(
    comm,
    window,
    out,
    base,
    rows,
    src_stride,
    width,
    peer_stride,
    dst_stride,
    peers: cutlass.Constexpr,
    tiles: cutlass.Constexpr,
    stream,
):
    _all_to_all_kernel(
        comm,
        window,
        out,
        base,
        rows,
        src_stride,
        width,
        peer_stride,
        dst_stride,
    ).launch(grid=(peers, tiles, 1), block=(128, 1, 1), stream=stream)


def all_to_all_nd(input, out, scatter_dim, gather_dim, *, group):
    if input.ndim not in (2, 3) or input.stride(-1) != 1:
        raise ValueError(
            "all_to_all_nd requires a 2-D or 3-D input with contiguous inner dimension"
        )
    if (scatter_dim, gather_dim) not in ((1, 0), (0, 1)):
        raise ValueError("all_to_all_nd supports only (1, 0) and (0, 1)")
    if (
        out.device != input.device
        or out.dtype != input.dtype
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous and have the input's device and dtype")
    with torch.cuda.device(input.device):
        handle, comm, window, base, _ = _context(input, group)
        peers, rank = handle.world_size, handle.rank
        if peers > 64:
            raise ValueError("all_to_all_nd supports at most 64 ranks")
        if input.ndim == 3 and not input.is_contiguous():
            raise ValueError("3-D input must be contiguous")
        if scatter_dim == 1:
            rows = input.shape[0]
            if input.ndim == 3:
                if input.shape[1] != peers:
                    raise ValueError("input.size(1) must equal group size")
                width = input.shape[2]
            else:
                if input.shape[1] % peers:
                    raise ValueError("input columns must be divisible by group size")
                width = input.shape[1] // peers
            shapes = ((peers, rows, width), (peers * rows, width))
            base += rank * width * input.element_size()
            src_stride = input.stride(0) * input.element_size()
            peer_stride, dst_stride = rows * width, width
        else:
            width = input.shape[-1]
            if input.ndim == 3:
                if input.shape[0] != peers:
                    raise ValueError("input.size(0) must equal group size")
                rows = input.shape[1]
                row_stride = input.stride(1)
            else:
                if input.shape[0] % peers:
                    raise ValueError("input rows must be divisible by group size")
                rows = input.shape[0] // peers
                row_stride = input.stride(0)
            shapes = ((rows, peers, width), (rows, peers * width))
            base += rank * rows * row_stride * input.element_size()
            src_stride = row_stride * input.element_size()
            peer_stride, dst_stride = width, peers * width
        if tuple(out.shape) not in shapes:
            raise ValueError(f"out must have shape {shapes[0]} or {shapes[1]}")
        if any(
            x % 16
            for x in (base, src_stride, width * input.element_size(), out.data_ptr())
        ):
            raise ValueError(
                "all_to_all_nd requires 16-byte aligned rows and addresses"
            )
        tiles = max(1, min((rows * width * input.element_size() + 8191) // 8192, 16))
        stream = driver.CUstream(torch.cuda.current_stream(input.device).cuda_stream)
        _all_to_all(
            comm,
            window,
            from_dlpack(out.view(torch.uint8).view(-1), assumed_align=16),
            cutlass.Int64(base),
            cutlass.Int32(rows),
            cutlass.Int64(src_stride),
            cutlass.Int32(width * input.element_size()),
            cutlass.Int64(peer_stride * input.element_size()),
            cutlass.Int64(dst_stride * input.element_size()),
            peers,
            tiles,
            stream,
        )
