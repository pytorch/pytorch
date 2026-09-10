"""Pure Python c10d backend implemented on top of CUDA SymmetricMemory

Uses the PyBackend trampoline so that ProcessGroup dispatches collective calls
into Python overrides in this class. Each communicator allocates a
symmetric-memory workspace shared across all participating ranks via CUDA IPC;
collectives copy data through the workspace and synchronize with
barrier/signal-pad primitives. Single-node multi-GPU only; Gloo is used only
for rendezvous, with no NCCL dependency on the data path.

Registers via the c10d extended API (extended_api=True), so the factory
receives a _DistributedBackendOptions object:
    dist.Backend.register_backend(
        "symmem", _create_symmem_backend, extended_api=True, devices=["cuda"]
    )

Or use
    dist.init_process_group("symmem", ...)
"""

import atexit
import hashlib
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    _register_process_group,
    _SymmetricMemory,
    _unregister_process_group,
    Backend as C10DBackend,
    ProcessGroup,
    ProcessGroupGloo,
    ReduceOp,
)
from torch.distributed import PrefixStore, Store


__all__ = ["SymmemBackend", "cast_buffer", "nbytes_of", "reduce_op_name"]


def nbytes_of(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def cast_buffer(buf: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """View a uint8 symm-mem ``buf`` as the dtype of ``like``.

    Returns a 1-D contiguous tensor; callers slice it to the element count
    they need.
    """
    if buf.dtype != torch.uint8:
        raise RuntimeError("cast_buffer expects a uint8 buffer")
    if like.element_size() == 1:
        return buf
    capacity = buf.numel() - (buf.numel() % like.element_size())
    return buf[:capacity].view(like.dtype)


def reduce_op_name(op) -> str:
    """Return the canonical lowercase name of a ReduceOp or RedOpType."""
    t = op.op if hasattr(op, "op") else op
    return t.name.lower()


class _SymmemWork(dist._Work):
    """Async work handle backed by a CUDA event."""

    def __init__(self, event: torch.cuda.Event | None = None, device=None):
        super().__init__()
        self._event = event
        self._device = device

    def wait(self, timeout=None):
        if self._event is not None:
            self._event.synchronize()
            self._event = None
        return True

    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result(True)
        return fut


_DEFAULT_WORKSPACE_BYTES: int = 128 * 1024 * 1024
_SEND_SLOT_BYTES: int = 64 * 1024


class _GroupResources:
    __slots__ = (
        "group_name",
        "store",
        "pg",
        "workspace_tensor",
        "symm_mem",
        "workspace_bytes",
        "_pointer_cache",
    )

    def __init__(
        self,
        group_name: str,
        store,
        pg,
        workspace_tensor,
        symm_mem,
        workspace_bytes: int,
    ) -> None:
        self.group_name = group_name
        self.store = store
        self.pg = pg
        self.workspace_tensor = workspace_tensor
        self.symm_mem = symm_mem
        self.workspace_bytes = workspace_bytes
        self._pointer_cache = None


_GROUP_RESOURCES: dict[tuple, _GroupResources] = {}


def _shutdown_all_resources() -> None:
    for res in list(_GROUP_RESOURCES.values()):
        try:
            if res.pg is not None:
                res.pg.shutdown()
        except Exception:
            pass
        try:
            # pyrefly: ignore [bad-argument-type]
            _unregister_process_group(res.group_name)
        except Exception:
            pass
        res.workspace_tensor = None
        res.symm_mem = None
        res.pg = None
        res.store = None
    _GROUP_RESOURCES.clear()


atexit.register(_shutdown_all_resources)


_GROUP_ALLOC_SEQ: dict[str, int] = {}
_COMM_ID_COUNTER: int = 0


def _next_comm_id() -> int:
    global _COMM_ID_COUNTER
    _COMM_ID_COUNTER += 1
    return _COMM_ID_COUNTER


def _alloc_id_for_group(group_name: str) -> int:
    seq = _GROUP_ALLOC_SEQ.get(group_name, 0) + 1
    _GROUP_ALLOC_SEQ[group_name] = seq
    h = hashlib.sha256(f"{group_name}#{seq}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _build_process_group(
    store: Store,
    rank: int,
    size: int,
    group_name: str,
    timeout: timedelta,
) -> ProcessGroup:
    pg = ProcessGroup(store, rank, size)
    gloo = ProcessGroupGloo(store, rank, size, timeout)
    backend_type = ProcessGroup.BackendType.GLOO
    for dev in (torch.device("cpu"), torch.device("cuda")):
        pg._register_backend(dev, backend_type, gloo)
    pg._set_default_backend(backend_type)
    # pyrefly: ignore [bad-argument-type]
    pg._set_group_name(group_name)
    pg._set_group_desc(group_name)
    return pg


class SymmemBackend(C10DBackend):
    """Pure-Python c10d Symmmem Backend backed by CUDA SymmetricMemory."""

    _resources: _GroupResources | None

    def __init__(self, dist_backend_opts, backend_options=None):
        store = dist_backend_opts.store
        rank = dist_backend_opts.group_rank
        size = dist_backend_opts.group_size
        timeout = dist_backend_opts.timeout
        super().__init__(rank, size)
        self._store = store
        self._options = C10DBackend.Options("symmem", timeout=timeout)
        self._timeout = timeout

        device_count = torch.cuda.device_count()
        self._device = torch.device("cuda", rank % device_count)
        torch.cuda.set_device(self._device)

        self._rank = rank
        self._size = size
        # global_ranks_in_group is empty for the default, world-spanning group.
        global_ranks = tuple(dist_backend_opts.global_ranks_in_group)
        self._global_ranks: tuple = global_ranks or tuple(range(size))
        self._world_rank = self._global_ranks[rank]
        self._world_size = len(self._global_ranks)
        self._barrier_channel = 0
        self._sendrecv_channel_base = 16 + _next_comm_id() * size * size

        try:
            _SymmetricMemory.set_backend("CUDA")
        except Exception:
            pass
        try:
            if _SymmetricMemory.signal_pad_size < 1 << 20:
                _SymmetricMemory.signal_pad_size = 1 << 20
        except Exception:
            pass

        workspace_bytes = int(
            os.environ.get("SYMMEM_WORKSPACE_BYTES", _DEFAULT_WORKSPACE_BYTES)
        )

        self._resources = self._get_or_create_resources(
            global_ranks=self._global_ranks,
            rank_in_group=rank,
            device=self._device,
            root_store=store,
            workspace_bytes=workspace_bytes,
            timeout=timeout,
        )
        self._scratch_bytes = (
            self._resources.workspace_bytes - self._world_size * _SEND_SLOT_BYTES
        )
        self._send_region_offset = self._scratch_bytes

    @property
    def options(self):
        return self._options

    @property
    def supports_splitting(self):
        return True

    @property
    def supports_coalescing(self):
        return False

    def getBackendName(self):
        return "symmem"

    def _err(self, msg: str) -> RuntimeError:
        return RuntimeError(f"[symmem] {msg}")

    def _check_device(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        if tensor.device != self._device:
            raise self._err(
                f"{name} device {tensor.device} != backend device {self._device}"
            )

    @property
    def _symm_mem(self) -> _SymmetricMemory | None:
        return self._resources.symm_mem if self._resources else None

    @property
    def _group_name(self) -> str:
        return self._resources.group_name if self._resources else ""

    @property
    def _workspace_tensor(self) -> torch.Tensor | None:
        return self._resources.workspace_tensor if self._resources else None

    @property
    def _workspace_bytes(self) -> int:
        return self._resources.workspace_bytes if self._resources else 0

    def _ensure_workspace(self, min_scratch_bytes: int) -> _SymmetricMemory:
        needed = min_scratch_bytes + self._world_size * _SEND_SLOT_BYTES
        if self._workspace_tensor is None:
            raise self._err("workspace is not allocated")
        if needed > self._workspace_bytes:
            raise self._err(
                f"collective requires {needed} bytes of scratch but workspace "
                f"is {self._workspace_bytes} bytes. Set SYMMEM_WORKSPACE_BYTES "
                f"to at least {needed}."
            )
        if self._symm_mem is None:
            raise self._err("symm_mem is None")
        return self._symm_mem

    def _scratch(self) -> torch.Tensor:
        if self._workspace_tensor is None:
            raise self._err("workspace is not allocated")
        return self._workspace_tensor[: self._scratch_bytes]

    def _peer_global(self, peer_local: int) -> int:
        return self._global_ranks[peer_local]

    def _peer_scratch(self, peer_local: int) -> torch.Tensor:
        if self._symm_mem is None:
            raise self._err("symm_mem is None")
        peer_global = self._peer_global(peer_local)
        return self._symm_mem.get_buffer(
            peer_global, (self._scratch_bytes,), torch.uint8, 0
        )

    def _group_barrier(self) -> None:
        if self._symm_mem is None:
            raise self._err("symm_mem is None")
        if self._size == self._world_size:
            self._symm_mem.barrier()
            return
        chan = self._barrier_channel
        my_global = self._world_rank
        for peer_global in self._global_ranks:
            if peer_global != my_global:
                self._symm_mem.put_signal(peer_global, channel=chan)
        for peer_global in self._global_ranks:
            if peer_global != my_global:
                self._symm_mem.wait_signal(peer_global, channel=chan)

    def _send_slot_offset(self, dst_world_rank: int) -> int:
        return self._send_region_offset + dst_world_rank * _SEND_SLOT_BYTES

    def _send_slot(self, sender_world_rank: int, dst_world_rank: int) -> torch.Tensor:
        if self._symm_mem is None:
            raise self._err("symm_mem is None")
        return self._symm_mem.get_buffer(
            sender_world_rank,
            (_SEND_SLOT_BYTES,),
            torch.uint8,
            self._send_slot_offset(dst_world_rank),
        )

    def _sendrecv_channel(self, src_world: int, dst_world: int) -> int:
        return self._sendrecv_channel_base + src_world * self._world_size + dst_world

    def _make_work(self, async_op: bool = False) -> _SymmemWork:
        if async_op:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self._device))
            return _SymmemWork(event, self._device)
        return _SymmemWork(None, self._device)

    @staticmethod
    def _resource_key(global_ranks: tuple, device: torch.device) -> tuple:
        return (device.type, device.index, tuple(sorted(global_ranks)))

    def _get_or_create_resources(
        self,
        global_ranks: tuple,
        rank_in_group: int,
        device: torch.device,
        root_store: Store,
        workspace_bytes: int,
        timeout: timedelta,
    ) -> _GroupResources:
        key = self._resource_key(global_ranks, device)
        cached = _GROUP_RESOURCES.get(key)
        if cached is not None:
            if cached.workspace_bytes < workspace_bytes:
                raise self._err(
                    f"existing workspace for ranks {global_ranks} is "
                    f"{cached.workspace_bytes} bytes but {workspace_bytes} was "
                    f"requested. Symmem reallocation is not supported."
                )
            return cached

        size = len(global_ranks)
        group_name = "symmem_grp_" + "_".join(str(r) for r in sorted(global_ranks))
        prefix_store = PrefixStore(group_name + "/", root_store)
        pg_store = PrefixStore("pg/", prefix_store)
        pg = _build_process_group(
            pg_store,
            rank_in_group,
            size,
            group_name,
            timeout,
        )
        # pyrefly: ignore [bad-argument-type]
        _register_process_group(group_name, pg)
        _SymmetricMemory.set_group_info(
            group_name,
            rank_in_group,
            size,
            PrefixStore("sm/", prefix_store),
        )
        alloc_id = _alloc_id_for_group(group_name)
        workspace_tensor = _SymmetricMemory.empty_strided_p2p(
            (workspace_bytes,),
            (1,),
            torch.uint8,
            device,
            group_name,
            alloc_id,
        )
        symm_mem = _SymmetricMemory.rendezvous(workspace_tensor)
        symm_mem.barrier()
        torch.cuda.current_stream(device).synchronize()
        cached = _GroupResources(
            group_name=group_name,
            store=prefix_store,
            pg=pg,
            workspace_tensor=workspace_tensor,
            symm_mem=symm_mem,
            workspace_bytes=workspace_bytes,
        )
        _GROUP_RESOURCES[key] = cached
        return cached

    def allreduce(self, tensor_list, opts=None):
        tensor = tensor_list[0]
        self._check_device(tensor)
        if tensor.numel() == 0:
            return self._make_work(opts.asyncOp if opts else False)
        op = opts.reduceOp if opts else ReduceOp.SUM
        result = self._reduce_impl(tensor, op, root=None)
        tensor.copy_(result.view(tensor.shape))
        return self._make_work(opts.asyncOp if opts else False)

    def broadcast(self, tensor_list, opts=None):
        tensor = tensor_list[0]
        self._check_device(tensor)
        root = opts.rootRank if opts else 0
        async_op = opts.asyncOp if opts else False
        if tensor.numel() == 0:
            return self._make_work(async_op)
        nbytes = nbytes_of(tensor)
        self._ensure_workspace(nbytes)

        if self._rank == root:
            cast_buffer(self._scratch(), tensor)[: tensor.numel()].copy_(
                tensor.reshape(-1).contiguous()
            )
        self._group_barrier()
        if self._rank != root:
            root_buf = cast_buffer(self._peer_scratch(root), tensor)[: tensor.numel()]
            tensor.copy_(root_buf.view(tensor.shape))
        self._group_barrier()
        return self._make_work(async_op)

    def reduce(self, tensor_list, opts=None):
        tensor = tensor_list[0]
        self._check_device(tensor)
        root = opts.rootRank if opts else 0
        op = opts.reduceOp if opts else ReduceOp.SUM
        async_op = opts.asyncOp if opts else False
        if tensor.numel() == 0:
            return self._make_work(async_op)
        result = self._reduce_impl(tensor, op, root=root)
        if self._rank == root:
            tensor.copy_(result.view(tensor.shape))
        return self._make_work(async_op)

    _SUPPORTED_REDUCE_DTYPES = (torch.float32, torch.bfloat16)

    def _reduce_impl(
        self,
        tensor: torch.Tensor,
        op: ReduceOp,
        root: int | None,
    ) -> torch.Tensor:
        op_name = reduce_op_name(op)
        if tensor.dtype not in self._SUPPORTED_REDUCE_DTYPES or op_name != "sum":
            raise self._err(
                f"symmem backend currently supports SUM on fp32/bf16 only, "
                f"got op={op_name} dtype={tensor.dtype}. "
                f"Triton kernel support for other ops/dtypes is planned."
            )

        nbytes = nbytes_of(tensor)
        self._ensure_workspace(nbytes)

        local_view = cast_buffer(self._scratch(), tensor)[: tensor.numel()]
        local_view.copy_(tensor.reshape(-1))
        self._group_barrier()

        out = torch.empty_like(tensor)
        if self._size == self._world_size:
            # Fast path: the group spans the whole symmetric-memory rank set, so
            # the kernel's group-wide reduction matches this group exactly.
            result = torch.ops.symm_mem.one_shot_all_reduce(
                local_view.view(tensor.shape),
                "sum",
                self._group_name,
            )
            out.copy_(result)
        else:
            # Strict subgroup: the kernel would reduce over the parent rank set,
            # so sum each participating peer's scratch manually instead.
            self._reduce_peers(out, op, 0, tensor.numel())
        self._group_barrier()
        return out

    def allgather(self, output_tensors, input_tensors, opts=None):
        tensor = input_tensors[0]
        output_list = output_tensors[0]
        self._check_device(tensor)
        async_op = opts.asyncOp if opts else False
        if len(output_list) != self._size:
            raise self._err(
                f"allgather: output_list has {len(output_list)} != size {self._size}"
            )
        if tensor.numel() == 0:
            return self._make_work(async_op)
        nbytes = nbytes_of(tensor)
        self._ensure_workspace(nbytes)

        local_view = cast_buffer(self._scratch(), tensor)[: tensor.numel()]
        local_view.copy_(tensor.reshape(-1))
        self._group_barrier()
        for peer in range(self._size):
            peer_buf = cast_buffer(self._peer_scratch(peer), tensor)[: tensor.numel()]
            output_list[peer].copy_(peer_buf.view(output_list[peer].shape))
        self._group_barrier()
        return self._make_work(async_op)

    def all_gather_single(self, output, input, opts=None):
        self._check_device(input)
        self._check_device(output)
        async_op = opts.asyncOp if opts else False
        if output.numel() != input.numel() * self._size:
            raise self._err(
                f"all_gather_single: output numel {output.numel()} != "
                f"input numel * size {input.numel() * self._size}"
            )
        if input.numel() == 0:
            return self._make_work(async_op)
        nbytes = nbytes_of(input)
        self._ensure_workspace(nbytes)

        cast_buffer(self._scratch(), input)[: input.numel()].copy_(input.reshape(-1))
        self._group_barrier()
        out_flat = output.reshape(-1)
        for peer in range(self._size):
            peer_buf = cast_buffer(self._peer_scratch(peer), input)[: input.numel()]
            out_flat[peer * input.numel() : (peer + 1) * input.numel()].copy_(peer_buf)
        self._group_barrier()
        return self._make_work(async_op)

    def _reduce_peers(self, output, op, elem_offset, n_elements):
        """Sum n_elements from each peer's scratch into output."""
        op_name = reduce_op_name(op)
        if op_name != "sum":
            raise self._err(
                f"reduce_scatter currently supports SUM only, got op={op_name}. "
                f"Triton kernel support for other ops is planned."
            )
        out_flat = output.reshape(-1)
        peer0_buf = cast_buffer(self._peer_scratch(0), output)
        out_flat.copy_(peer0_buf[elem_offset : elem_offset + n_elements])
        for peer in range(1, self._size):
            peer_buf = cast_buffer(self._peer_scratch(peer), output)
            out_flat.add_(peer_buf[elem_offset : elem_offset + n_elements])

    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        output = output_tensors[0]
        input_list = input_tensors[0]
        self._check_device(output)
        op = opts.reduceOp if opts else ReduceOp.SUM
        async_op = opts.asyncOp if opts else False
        if len(input_list) != self._size:
            raise self._err(
                f"reduce_scatter: input_list has {len(input_list)} != size {self._size}"
            )
        nbytes = sum(nbytes_of(t) for t in input_list)
        self._ensure_workspace(max(nbytes, 1))

        scratch_typed = cast_buffer(self._scratch(), output)
        offset = 0
        offsets = []
        for t in input_list:
            self._check_device(t)
            offsets.append(offset)
            if t.numel() > 0:
                scratch_typed[offset : offset + t.numel()].copy_(t.reshape(-1))
            offset += t.numel()
        self._group_barrier()

        my_count = input_list[self._rank].numel()
        if my_count > 0:
            self._reduce_peers(output, op, offsets[self._rank], my_count)
        self._group_barrier()
        return self._make_work(async_op)

    def reduce_scatter_single(self, output, input, opts=None):
        self._check_device(output)
        self._check_device(input)
        op = opts.reduceOp if opts else ReduceOp.SUM
        async_op = opts.asyncOp if opts else False
        if input.numel() != output.numel() * self._size:
            raise self._err(
                f"reduce_scatter_single: input numel {input.numel()} != "
                f"output numel * size {output.numel() * self._size}"
            )
        if output.numel() == 0:
            return self._make_work(async_op)
        nbytes = nbytes_of(input)
        self._ensure_workspace(nbytes)

        cast_buffer(self._scratch(), input)[: input.numel()].copy_(input.reshape(-1))
        self._group_barrier()
        my_count = output.numel()
        start = self._rank * my_count
        self._reduce_peers(output, op, start, my_count)
        self._group_barrier()
        return self._make_work(async_op)

    def all_to_all_single(
        self, output, input, output_split_sizes, input_split_sizes, opts=None
    ):
        self._check_device(output)
        self._check_device(input)
        async_op = opts.asyncOp if opts else False

        if not input_split_sizes and not output_split_sizes:
            if input.numel() % self._size != 0:
                raise self._err(
                    f"all_to_all_single: input numel {input.numel()} "
                    f"not divisible by size {self._size}"
                )
            if input.numel() == 0:
                return self._make_work(async_op)
            per_rank = input.numel() // self._size
            nbytes = nbytes_of(input)
            self._ensure_workspace(nbytes)

            cast_buffer(self._scratch(), input)[: input.numel()].copy_(
                input.reshape(-1)
            )
            self._group_barrier()
            out_flat = output.reshape(-1)
            for peer in range(self._size):
                peer_buf = cast_buffer(self._peer_scratch(peer), input)
                chunk = peer_buf[self._rank * per_rank : (self._rank + 1) * per_rank]
                out_flat[peer * per_rank : (peer + 1) * per_rank].copy_(chunk)
            self._group_barrier()
            return self._make_work(async_op)

        if (
            len(input_split_sizes) != self._size
            or len(output_split_sizes) != self._size
        ):
            raise self._err("all_to_all_single: split-size list length must equal size")
        if input.numel() == 0 and output.numel() == 0:
            return self._make_work(async_op)

        in_stride = (
            input.numel() // input.size(0)
            if input.dim() > 0 and input.size(0) > 0
            else 1
        )
        out_stride = (
            output.numel() // output.size(0)
            if output.dim() > 0 and output.size(0) > 0
            else 1
        )

        self._ensure_workspace(
            max(nbytes_of(input), output.element_size() * output.numel())
        )

        all_splits = torch.zeros(
            (self._size, self._size), dtype=torch.int64, device=self._device
        )
        my_splits = torch.tensor(
            [int(s) * in_stride for s in input_split_sizes],
            dtype=torch.int64,
            device=self._device,
        )
        self.all_gather_single(all_splits.reshape(-1), my_splits)

        my_input_offsets = [0]
        for i in range(self._size - 1):
            my_input_offsets.append(
                my_input_offsets[-1] + int(input_split_sizes[i]) * in_stride
            )
        scratch_typed = cast_buffer(self._scratch(), input)
        if input.numel() > 0:
            in_flat = input.reshape(-1)
            for peer in range(self._size):
                n = int(input_split_sizes[peer]) * in_stride
                if n == 0:
                    continue
                src_off = my_input_offsets[peer]
                scratch_typed[src_off : src_off + n].copy_(
                    in_flat[src_off : src_off + n]
                )
        self._group_barrier()

        out_flat = output.reshape(-1)
        out_pos = 0
        for peer in range(self._size):
            recv_count = int(output_split_sizes[peer]) * out_stride
            if recv_count == 0:
                continue
            peer_splits = all_splits[peer]
            peer_off_for_me = int(peer_splits[: self._rank].sum().item())
            peer_buf = cast_buffer(self._peer_scratch(peer), output)
            out_flat[out_pos : out_pos + recv_count].copy_(
                peer_buf[peer_off_for_me : peer_off_for_me + recv_count]
            )
            out_pos += recv_count
        self._group_barrier()
        return self._make_work(async_op)

    def alltoall(self, output_tensors, input_tensors, opts=None):
        async_op = opts.asyncOp if opts else False
        if len(output_tensors) != self._size or len(input_tensors) != self._size:
            raise self._err("alltoall: list lengths must equal size")

        offsets = [0]
        for t in input_tensors:
            self._check_device(t)
            offsets.append(offsets[-1] + t.numel())
        total = offsets[-1]
        if total == 0 and all(t.numel() == 0 for t in output_tensors):
            return self._make_work(async_op)

        dtype_template = next(
            (t for t in input_tensors if t.numel() > 0),
            next((t for t in output_tensors if t.numel() > 0), input_tensors[0]),
        )
        nbytes = total * dtype_template.element_size()
        self._ensure_workspace(nbytes)

        my_counts = torch.tensor(
            [t.numel() for t in input_tensors],
            dtype=torch.int64,
            device=self._device,
        )
        all_counts = torch.zeros(
            (self._size, self._size), dtype=torch.int64, device=self._device
        )
        self.all_gather_single(all_counts.reshape(-1), my_counts)

        scratch_typed = cast_buffer(self._scratch(), dtype_template)
        for i, t in enumerate(input_tensors):
            if t.numel() > 0:
                scratch_typed[offsets[i] : offsets[i] + t.numel()].copy_(t.reshape(-1))
        self._group_barrier()

        for peer in range(self._size):
            out = output_tensors[peer]
            if out.numel() == 0:
                continue
            self._check_device(out)
            peer_off = int(all_counts[peer, : self._rank].sum().item())
            peer_buf = cast_buffer(self._peer_scratch(peer), out)
            out.copy_(peer_buf[peer_off : peer_off + out.numel()].view(out.shape))
        self._group_barrier()
        return self._make_work(async_op)

    def scatter(self, output_tensors, input_tensors, opts=None):
        output = output_tensors[0]
        self._check_device(output)
        root = opts.rootRank if opts else 0
        async_op = opts.asyncOp if opts else False
        input_list = input_tensors[0] if input_tensors and input_tensors[0] else []

        if output.numel() == 0:
            return self._make_work(async_op)
        self._ensure_workspace(self._size * nbytes_of(output))
        if self._rank == root:
            if len(input_list) != self._size:
                raise self._err(
                    f"scatter: input_list has {len(input_list)} != size {self._size}"
                )
            scratch_typed = cast_buffer(self._scratch(), output)
            off = 0
            for t in input_list:
                self._check_device(t)
                if t.numel() == 0:
                    continue
                scratch_typed[off : off + t.numel()].copy_(t.reshape(-1))
                off += t.numel()
        self._group_barrier()
        root_buf = cast_buffer(self._peer_scratch(root), output)
        start = self._rank * output.numel()
        output.copy_(root_buf[start : start + output.numel()].view(output.shape))
        self._group_barrier()
        return self._make_work(async_op)

    def gather(self, output_tensors, input_tensors, opts=None):
        input_tensor = input_tensors[0]
        self._check_device(input_tensor)
        root = opts.rootRank if opts else 0
        async_op = opts.asyncOp if opts else False

        if input_tensor.numel() == 0:
            return self._make_work(async_op)
        self._ensure_workspace(nbytes_of(input_tensor))

        cast_buffer(self._scratch(), input_tensor)[: input_tensor.numel()].copy_(
            input_tensor.reshape(-1)
        )
        self._group_barrier()
        if self._rank == root:
            output_list = output_tensors[0]
            if len(output_list) != self._size:
                raise self._err(
                    f"gather: output_list has {len(output_list)} != size {self._size}"
                )
            for peer in range(self._size):
                out = output_list[peer]
                self._check_device(out)
                if out.numel() == 0:
                    continue
                peer_buf = cast_buffer(self._peer_scratch(peer), input_tensor)[
                    : input_tensor.numel()
                ]
                out.copy_(peer_buf.view(out.shape))
        self._group_barrier()
        return self._make_work(async_op)

    def barrier(self, opts=None):
        self._group_barrier()
        return self._make_work(opts.asyncOp if opts else False)

    def send(self, tensor_list, dst, tag=0):
        tensor = tensor_list[0]
        self._check_device(tensor)
        if not 0 <= dst < self._size:
            raise self._err(f"send: dst {dst} out of range")
        nbytes = nbytes_of(tensor)
        if nbytes > _SEND_SLOT_BYTES:
            raise self._err(
                f"send: tensor size {nbytes} exceeds slot capacity {_SEND_SLOT_BYTES}"
            )
        self._ensure_workspace(0)
        if self._symm_mem is None:
            raise self._err("symm_mem is None")
        dst_world = self._peer_global(dst)

        if tensor.numel() > 0:
            slot = self._send_slot(self._world_rank, dst_world)
            slot_typed = cast_buffer(slot, tensor)[: tensor.numel()]
            slot_typed.copy_(tensor.reshape(-1).contiguous())
        self._symm_mem.put_signal(
            dst_world, channel=self._sendrecv_channel(self._world_rank, dst_world)
        )
        return self._make_work()

    def recv(self, tensor_list, src, tag=0):
        tensor = tensor_list[0]
        self._check_device(tensor)
        if not 0 <= src < self._size:
            raise self._err(f"recv: src {src} out of range")
        nbytes = nbytes_of(tensor)
        if nbytes > _SEND_SLOT_BYTES:
            raise self._err(
                f"recv: tensor size {nbytes} exceeds slot capacity {_SEND_SLOT_BYTES}"
            )
        self._ensure_workspace(0)
        if self._symm_mem is None:
            raise self._err("symm_mem is None")
        src_world = self._peer_global(src)

        self._symm_mem.wait_signal(
            src_world, channel=self._sendrecv_channel(src_world, self._world_rank)
        )
        if tensor.numel() > 0:
            src_slot = self._send_slot(src_world, self._world_rank)
            slot_typed = cast_buffer(src_slot, tensor)[: tensor.numel()]
            tensor.copy_(slot_typed.view(tensor.shape))
        return self._make_work()

    def split(self, store, ranks, opts=None):
        ranks_list = list(ranks)
        if len(set(ranks_list)) != len(ranks_list):
            raise self._err("split: ranks list contains duplicates")
        for r in ranks_list:
            if not 0 <= r < self._size:
                raise self._err(f"split: rank {r} out of range [0, {self._size})")

        if self._rank not in ranks_list:
            return None

        sub_global_ranks = tuple(self._global_ranks[r] for r in ranks_list)
        sub_rank = ranks_list.index(self._rank)
        sub_size = len(ranks_list)

        child = SymmemBackend.__new__(SymmemBackend)
        C10DBackend.__init__(child, sub_rank, sub_size)
        child._store = store
        child._options = opts if opts is not None else C10DBackend.Options("symmem")
        child._timeout = self._timeout
        child._device = self._device
        child._rank = sub_rank
        child._size = sub_size
        child._global_ranks = sub_global_ranks
        child._world_rank = self._world_rank
        child._world_size = self._world_size

        if self._resources is None:
            raise self._err("split: parent resources missing")
        child._resources = self._resources
        child._scratch_bytes = (
            child._resources.workspace_bytes - child._world_size * _SEND_SLOT_BYTES
        )
        child._send_region_offset = child._scratch_bytes

        chan_space = max(_SymmetricMemory.signal_pad_size // 4 - 32, 64)
        h = hashlib.sha256(
            ("sub_barrier:" + "_".join(map(str, sub_global_ranks))).encode()
        ).digest()
        child._barrier_channel = int.from_bytes(h[:4], "big") % chan_space + 16
        child._sendrecv_channel_base = (
            16 + _next_comm_id() * child._world_size * child._world_size
        )
        return child

    def shutdown(self):
        self._resources = None

    def abort(self):
        self._resources = None


def _create_symmem_backend(dist_backend_opts, backend_options):
    return SymmemBackend(dist_backend_opts, backend_options)


dist.Backend.register_backend(
    "symmem", _create_symmem_backend, extended_api=True, devices=["cuda"]
)
