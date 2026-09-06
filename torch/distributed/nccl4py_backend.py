"""Pure Python c10d backend implemented on top of nccl4py (nccl.core)

Wraps NVIDIA's official nccl4py bindings to provide a Python NCCL backend for
torch.distributed. Uses the PyBackend trampoline so that ProcessGroup dispatches
collective calls into Python overrides in this class.

Register with
    dist.Backend.register_backend("nccl4py", _create_nccl4py_backend, devices=["cuda"])

Or use
    dist.init_process_group("nccl4py", ...)
"""

# pyre-unsafe

__all__ = ["NCCL4PyBackend"]

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import Backend as C10DBackend, ReduceOp
from torch.distributed._watchdog import _get_watchdog, stream_complete, stream_timeout


try:
    import nccl.core as nccl
except ModuleNotFoundError:
    nccl = None  # type: ignore[assignment]


class _NcclWork(dist._Work):
    """Work handle backed by CUDA events recorded after the NCCL operation(s).

    A single collective records one event. A coalesced window records one event
    per distinct stream it ran on, and waiting joins all of them (the analog
    of ProcessGroupNCCL(nccl2)'s coalesceWorks)
    """

    def __init__(self, events, device, wd_handles=None, is_barrier=False):
        super().__init__()
        self._events = list(events)
        self._device = device
        self._wd_handles = list(wd_handles) if wd_handles else []
        self._is_barrier = is_barrier

    def wait(self, timeout=None):
        current = torch.cuda.current_stream(self._device)
        for event in self._events:
            event.wait(current)
        if self._is_barrier:
            current.synchronize()
        self._events = []
        for handle in self._wd_handles:
            handle.cancel()
        self._wd_handles = []
        return True

    def get_future(self):
        fut = torch.futures.Future(devices=[self._device])
        if not self._events:
            fut.set_result(True)
            return fut
        remaining = [len(self._events)]

        def on_complete():
            remaining[0] -= 1
            if remaining[0] == 0:
                fut.set_result(True)

        for event in self._events:
            stream_complete(on_complete, event=event)
        return fut


class NCCL4PyBackend(C10DBackend):
    """Python c10d Backend backed by NVIDIA's nccl4py bindings."""

    _UID_STORE_KEY = "nccl4py_uid"

    def __init__(self, store, rank, size, timeout):
        if nccl is None:
            raise RuntimeError(
                "nccl4py backend requires the 'nccl4py' package. "
                "Install it with: pip install nccl4py"
            )
        super().__init__(rank, size)
        self._store = store
        self._options = C10DBackend.Options("nccl4py", timeout=timeout)

        # TODO (thisisatharva-rh): workaround. The basic creator API only passes the group-local
        # rank, which is not a valid device index for a subgroup, so we recover
        # the device from the default group. The proper fix is to migrate to the
        # extended_api=True, which supplies the resolved device directly

        device_count = torch.cuda.device_count()
        if dist.is_initialized():
            default_pg = dist.distributed_c10d._get_default_group()
            self._device = default_pg.bound_device_id or torch.device(
                "cuda", default_pg.rank() % device_count
            )
        else:
            self._device = torch.device("cuda", rank % device_count)
        torch.cuda.set_device(self._device)

        if rank == 0:
            uid = nccl.get_unique_id()
            store.set(self._UID_STORE_KEY, bytes(uid))
        else:
            store.wait([self._UID_STORE_KEY])
            uid = nccl.UniqueId.from_bytes(bytes(store.get(self._UID_STORE_KEY)))

        self._comm = nccl.Communicator.init(nranks=size, rank=rank, unique_id=uid)
        self._internal_stream = torch.cuda.Stream(device=self._device)
        self._barrier_tensor = torch.zeros(1, dtype=torch.float32, device=self._device)
        self._coalescing = False
        self._coalesced_streams = set()
        _get_watchdog()

    @property
    def options(self):
        return self._options

    @property
    def supports_splitting(self):
        return True

    @property
    def supports_coalescing(self):
        return True

    def getBackendName(self):
        return "nccl4py"

    def _check_tensor(self, tensor):
        if not tensor.is_cuda:
            raise ValueError(f"nccl4py: expected CUDA tensor, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError("nccl4py: expected contiguous tensor")

    def _check_tensors(self, tensors):
        for t in tensors:
            self._check_tensor(t)

    def _op_stream(self, async_op):
        if async_op:
            current = torch.cuda.current_stream(self._device)
            self._internal_stream.wait_stream(current)
            return self._internal_stream
        return torch.cuda.current_stream(self._device)

    def _record_stream(self, stream, *tensors):
        # User tensors and temp buffers are allocated on the compute stream but
        # consumed on the internal NCCL stream; recording the NCCL stream keeps
        # the caching allocator from reusing those blocks before the collective
        # finishes. Mirrors CUDACachingAllocator::recordStream in
        # ProcessGroupNCCL.
        if stream == torch.cuda.current_stream(self._device):
            return
        for t in tensors:
            t.record_stream(stream)

    def _make_work(self, *streams, is_barrier=False):
        # Inside a coalescing window the kernels are only enqueued at group_end,
        # so a per-op event here would be meaningless. Just record which streams
        # the op used; end_coalescing records the events after group_end.
        if self._coalescing:
            self._coalesced_streams.update(streams)
            return _NcclWork((), self._device)
        events = []
        wd_handles = []
        for stream in streams:
            event = torch.cuda.Event()
            event.record(stream)
            with torch.cuda.stream(stream):
                wd_handles.append(stream_timeout(self._options._timeout))
            events.append(event)
        return _NcclWork(events, self._device, wd_handles, is_barrier=is_barrier)

    def _get_nccl_redop(self, reduce_op, tensor):
        """Returns (nccl_op, custom_op_or_None).

        The caller must close custom_op after the NCCL call is enqueued.
        """
        op_type = reduce_op.op
        if op_type == ReduceOp.RedOpType.SUM:
            return nccl.SUM, None
        if op_type == ReduceOp.RedOpType.PRODUCT:
            return nccl.PROD, None
        if op_type == ReduceOp.RedOpType.MIN:
            return nccl.MIN, None
        if op_type == ReduceOp.RedOpType.MAX:
            return nccl.MAX, None
        if op_type == ReduceOp.RedOpType.AVG:
            return nccl.AVG, None
        if op_type == ReduceOp.RedOpType.PREMUL_SUM:
            factor = reduce_op.factor
            if isinstance(factor, torch.Tensor):
                scalar = factor
            else:
                scalar = torch.tensor(
                    [float(factor)], dtype=tensor.dtype, device=self._device
                )
            custom = self._comm.create_pre_mul_sum(scalar)
            return custom, custom
        raise RuntimeError(f"nccl4py backend: unsupported reduce op {reduce_op}")

    def start_coalescing(self):
        self._coalescing = True
        self._coalesced_streams = set()
        nccl.group_start()

    def end_coalescing(self):
        nccl.group_end()
        self._coalescing = False
        streams = self._coalesced_streams
        self._coalesced_streams = set()
        if not streams:
            streams = {torch.cuda.current_stream(self._device)}
        return self._make_work(*streams)

    def send(self, tensor_list, dst, tag=0):
        self._check_tensors(tensor_list)
        stream = torch.cuda.current_stream(self._device)
        s = stream.cuda_stream
        with nccl.group():
            for t in tensor_list:
                self._comm.send(t, dst, stream=s)
        return self._make_work(stream)

    def recv(self, tensor_list, src, tag=0):
        self._check_tensors(tensor_list)
        stream = torch.cuda.current_stream(self._device)
        s = stream.cuda_stream
        with nccl.group():
            for t in tensor_list:
                self._comm.recv(t, src, stream=s)
        return self._make_work(stream)

    def broadcast(self, tensor_list, opts):
        self._check_tensors(tensor_list)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        self._record_stream(stream, *tensor_list)
        with nccl.group():
            for t in tensor_list:
                self._comm.broadcast(t, t, opts.rootRank, stream=s)
        return self._make_work(stream)

    def allreduce(self, tensor_list, opts):
        self._check_tensors(tensor_list)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        nccl_op, custom = self._get_nccl_redop(opts.reduceOp, tensor_list[0])
        self._record_stream(stream, *tensor_list)
        try:
            with nccl.group():
                for t in tensor_list:
                    self._comm.allreduce(t, t, nccl_op, stream=s)
        finally:
            if custom is not None:
                custom.close()
        return self._make_work(stream)

    def reduce(self, tensor_list, opts):
        self._check_tensors(tensor_list)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        nccl_op, custom = self._get_nccl_redop(opts.reduceOp, tensor_list[0])
        self._record_stream(stream, *tensor_list)
        try:
            with nccl.group():
                for t in tensor_list:
                    self._comm.reduce(t, t, nccl_op, root=opts.rootRank, stream=s)
        finally:
            if custom is not None:
                custom.close()
        return self._make_work(stream)

    def allgather(self, output_tensors, input_tensors, opts):
        self._check_tensors(input_tensors)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        self._record_stream(stream, *input_tensors)
        flats = []
        with nccl.group():
            for output_list, inp in zip(output_tensors, input_tensors):
                flat = torch.empty(
                    inp.numel() * self.size(),
                    dtype=inp.dtype,
                    device=self._device,
                )
                self._record_stream(stream, flat)
                self._comm.allgather(inp, flat, stream=s)
                flats.append((flat, output_list, inp.numel()))
        with torch.cuda.stream(stream):
            for flat, output_list, chunk in flats:
                self._record_stream(stream, *output_list)
                for i, out in enumerate(output_list):
                    out.copy_(flat.narrow(0, i * chunk, chunk).view_as(out))
        return self._make_work(stream)

    def all_gather_single(self, output, input, opts):
        self._check_tensor(output)
        self._check_tensor(input)
        stream = self._op_stream(opts.asyncOp)
        self._record_stream(stream, input, output)
        self._comm.allgather(input, output, stream=stream.cuda_stream)
        return self._make_work(stream)

    def gather(self, output_tensors, input_tensors, opts):
        self._check_tensors(input_tensors)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        root = opts.rootRank
        self._record_stream(stream, *input_tensors)
        flats = []
        with nccl.group():
            for idx, inp in enumerate(input_tensors):
                if self.rank() == root:
                    flat = torch.empty(
                        inp.numel() * self.size(),
                        dtype=inp.dtype,
                        device=self._device,
                    )
                    self._record_stream(stream, flat)
                    self._comm.gather(inp, flat, root=root, stream=s)
                    flats.append((flat, output_tensors[idx], inp.numel()))
                else:
                    self._comm.gather(inp, inp, root=root, stream=s)
        if self.rank() == root:
            with torch.cuda.stream(stream):
                for flat, output_list, chunk in flats:
                    self._record_stream(stream, *output_list)
                    for i, out in enumerate(output_list):
                        out.copy_(flat.narrow(0, i * chunk, chunk).view_as(out))
        return self._make_work(stream)

    def scatter(self, output_tensors, input_tensors, opts):
        self._check_tensors(output_tensors)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        root = opts.rootRank
        self._record_stream(stream, *output_tensors)
        with torch.cuda.stream(stream):
            with nccl.group():
                for idx, out in enumerate(output_tensors):
                    if self.rank() == root:
                        self._record_stream(stream, *input_tensors[idx])
                        flat = torch.cat(
                            [t.contiguous().view(-1) for t in input_tensors[idx]]
                        )
                        self._comm.scatter(flat, out, root=root, stream=s)
                    else:
                        self._comm.scatter(out, out, root=root, stream=s)
        return self._make_work(stream)

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        self._check_tensors(output_tensors)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        nccl_op, custom = self._get_nccl_redop(opts.reduceOp, output_tensors[0])
        self._record_stream(stream, *output_tensors)
        try:
            with torch.cuda.stream(stream):
                with nccl.group():
                    for out, inp_list in zip(output_tensors, input_tensors):
                        self._record_stream(stream, *inp_list)
                        flat = torch.cat([t.contiguous().view(-1) for t in inp_list])
                        self._comm.reduce_scatter(flat, out, nccl_op, stream=s)
        finally:
            if custom is not None:
                custom.close()
        return self._make_work(stream)

    def reduce_scatter_single(self, output, input, opts):
        self._check_tensor(output)
        self._check_tensor(input)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        nccl_op, custom = self._get_nccl_redop(opts.reduceOp, output)
        self._record_stream(stream, input, output)
        try:
            self._comm.reduce_scatter(input, output, nccl_op, stream=s)
        finally:
            if custom is not None:
                custom.close()
        return self._make_work(stream)

    def all_to_all_single(
        self, output, input, output_split_sizes, input_split_sizes, opts
    ):
        self._check_tensor(output)
        self._check_tensor(input)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        self._record_stream(stream, input, output)
        if not output_split_sizes and not input_split_sizes:
            self._comm.alltoall(input, output, stream=s)
        else:
            size = self.size()
            in_row = input.numel() // input.size(0) if input.numel() else 1
            out_row = output.numel() // output.size(0) if output.numel() else 1
            if not input_split_sizes:
                input_split_sizes = [input.size(0) // size] * size
            if not output_split_sizes:
                output_split_sizes = [output.size(0) // size] * size
            flat_in = input.view(-1)
            flat_out = output.view(-1)
            in_off = 0
            out_off = 0
            with nccl.group():
                for i in range(size):
                    sc = input_split_sizes[i] * in_row
                    rc = output_split_sizes[i] * out_row
                    if sc > 0:
                        self._comm.send(flat_in.narrow(0, in_off, sc), i, stream=s)
                    if rc > 0:
                        self._comm.recv(flat_out.narrow(0, out_off, rc), i, stream=s)
                    in_off += sc
                    out_off += rc
        return self._make_work(stream)

    def alltoall(self, output_tensors, input_tensors, opts):
        self._check_tensors(output_tensors)
        self._check_tensors(input_tensors)
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        self._record_stream(stream, *input_tensors, *output_tensors)
        with nccl.group():
            for i in range(self.size()):
                self._comm.send(input_tensors[i], i, stream=s)
                self._comm.recv(output_tensors[i], i, stream=s)
        return self._make_work(stream)

    def barrier(self, opts):
        stream = self._op_stream(opts.asyncOp)
        s = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._barrier_tensor.zero_()
        self._comm.allreduce(
            self._barrier_tensor, self._barrier_tensor, nccl.SUM, stream=s
        )
        return self._make_work(stream, is_barrier=True)

    def split(self, store, ranks, opts):
        ranks_list = list(ranks)
        if self.rank() in ranks_list:
            color = min(ranks_list)
            key = ranks_list.index(self.rank())
        else:
            color = None
            key = 0

        new_comm = self._comm.split(color=color, key=key)

        if self.rank() not in ranks_list:
            return None

        child = NCCL4PyBackend.__new__(NCCL4PyBackend)
        C10DBackend.__init__(child, key, len(ranks_list))
        child._store = store
        child._options = opts if opts is not None else C10DBackend.Options("nccl4py")
        child._device = self._device
        child._comm = new_comm
        child._internal_stream = torch.cuda.Stream(device=self._device)
        child._barrier_tensor = torch.zeros(1, dtype=torch.float32, device=self._device)
        child._coalescing = False
        child._coalescing_stream = None
        return child

    def shutdown(self):
        if self._comm is not None and self._comm.is_valid:
            try:
                self._comm.finalize()
                self._comm.destroy()
            except Exception:
                pass
            self._comm = None

    def abort(self):
        if self._comm is not None and self._comm.is_valid:
            try:
                self._comm.abort()
            except Exception:
                pass
            self._comm = None


def _create_nccl4py_backend(store, group_rank, group_size, timeout):
    return NCCL4PyBackend(store, group_rank, group_size, timeout)


def _register_nccl4py_backend():
    dist.Backend.register_backend("nccl4py", _create_nccl4py_backend, devices=["cuda"])
