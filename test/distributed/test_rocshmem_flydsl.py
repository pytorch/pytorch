# Owner(s): ["oncall: distributed"]
# To run:
# python test/distributed/test_rocshmem_flydsl.py

import functools
import sys
import unittest

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


if (
    not torch.backends.cuda.is_built()
    or torch.version.hip is None
    or not symm_mem.is_nvshmem_available()
    or not PLATFORM_SUPPORTS_SYMM_MEM
):
    print("rocSHMEM symmetric memory not available, skipping tests")
    sys.exit(0)


import torch.distributed as dist
import torch.distributed._symmetric_memory._rocshmem_flydsl as rocshmem_flydsl


def _flydsl_available() -> bool:
    try:
        import flydsl.compiler  # noqa: F401

        return True
    except ImportError:
        return False


requires_flydsl = skip_but_pass_in_sandcastle_if(
    not _flydsl_available(),
    "FlyDSL is not installed",
)


# rocshmem_common.hpp: ROCSHMEM_SIGNAL_OPS enumerators, in order.
_ROCSHMEM_SIGNAL_SET = 0
_ROCSHMEM_SIGNAL_ADD = 1
# rocshmem_common.hpp: ROCSHMEM_CMP_EQ is the first rocshmem_cmps enumerator.
_ROCSHMEM_CMP_EQ = 0
# Bit 31 set, as a signed 32-bit kernel argument. FlyDSL lowers a Python int
# to Int32, whose host ABI is c_int32, so 1 << 31 cannot be passed directly.
# uint64 zero-extension turns this bit pattern into 1 << 31.
_SIG_VAL_BIT31 = -2147483648


@functools.cache
def _kernels():
    import flydsl.compiler as flyc

    @flyc.kernel
    def put_kernel(dest, src, nelems, pe):
        rocshmem_flydsl.put(dest, src, nelems, pe)
        rocshmem_flydsl.quiet()

    @flyc.kernel
    def get_kernel(dest, src, nelems, pe):
        rocshmem_flydsl.get(dest, src, nelems, pe)
        rocshmem_flydsl.quiet()

    @flyc.kernel
    def get_nbi_kernel(dest, src, nelems, pe):
        rocshmem_flydsl.get_nbi(dest, src, nelems, pe)
        rocshmem_flydsl.quiet()

    @flyc.kernel
    def putmem_signal_block_kernel(dest, src, nbytes, signal, sig_val, sig_op, pe):
        rocshmem_flydsl.putmem_signal_block(
            dest, src, nbytes, signal, sig_val, sig_op, pe
        )

    @flyc.kernel
    def fence_kernel(dst1, src1, dst2, src2, flag_dst, flag_src, nelems, peer):
        rocshmem_flydsl.put(dst1, src1, nelems, peer)
        rocshmem_flydsl.fence()
        rocshmem_flydsl.put(dst2, src2, nelems, peer)
        rocshmem_flydsl.fence()
        rocshmem_flydsl.put(flag_dst, flag_src, 1, peer)

    @flyc.kernel
    def quiet_kernel(dst, src, flag_dst, flag_src, nelems, peer):
        rocshmem_flydsl.put(dst, src, nelems, peer)
        rocshmem_flydsl.quiet()
        rocshmem_flydsl.put(flag_dst, flag_src, 1, peer)

    @flyc.kernel
    def barrier_kernel(dst, src, seen):
        me = rocshmem_flydsl.my_pe()
        if me == 0:
            rocshmem_flydsl.put(dst, src, 1, 1)
        rocshmem_flydsl.barrier_all()
        if me != 0:
            seen[0] = dst[0]

    @flyc.kernel
    def sync_kernel(local_data, remote_data, next_pe):
        local_data[0] = rocshmem_flydsl.my_pe() + 100
        rocshmem_flydsl.sync_all()
        rocshmem_flydsl.get(remote_data, local_data, 1, next_pe)
        rocshmem_flydsl.quiet()

    @flyc.kernel
    def wait_until_kernel(ivar, cmp_op, cmp_val):
        rocshmem_flydsl.wait_until(ivar, cmp_op, cmp_val)

    @flyc.kernel
    def signal_wait_until_kernel(signal, cmp_op, cmp_val):
        rocshmem_flydsl.signal_wait_until(signal, cmp_op, cmp_val)

    @flyc.kernel
    def my_pe_kernel(out):
        out[0] = rocshmem_flydsl.my_pe()

    @flyc.kernel
    def n_pes_kernel(out):
        out[0] = rocshmem_flydsl.n_pes()

    @flyc.jit
    def launch_put(dest, src, nelems, pe, stream):
        put_kernel(dest, src, nelems, pe).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_get(dest, src, nelems, pe, stream):
        get_kernel(dest, src, nelems, pe).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_get_nbi(dest, src, nelems, pe, stream):
        get_nbi_kernel(dest, src, nelems, pe).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_putmem_signal_block(
        dest, src, nbytes, signal, sig_val, sig_op, pe, stream
    ):
        putmem_signal_block_kernel(
            dest, src, nbytes, signal, sig_val, sig_op, pe
        ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    @flyc.jit
    def launch_fence(dst1, src1, dst2, src2, flag_dst, flag_src, nelems, peer, stream):
        fence_kernel(dst1, src1, dst2, src2, flag_dst, flag_src, nelems, peer).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_quiet(dst, src, flag_dst, flag_src, nelems, peer, stream):
        quiet_kernel(dst, src, flag_dst, flag_src, nelems, peer).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_barrier(dst, src, seen, stream):
        barrier_kernel(dst, src, seen).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_sync(local_data, remote_data, next_pe, stream):
        sync_kernel(local_data, remote_data, next_pe).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_wait_until(ivar, cmp_op, cmp_val, stream):
        wait_until_kernel(ivar, cmp_op, cmp_val).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_signal_wait_until(signal, cmp_op, cmp_val, stream):
        signal_wait_until_kernel(signal, cmp_op, cmp_val).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_my_pe(out, stream):
        my_pe_kernel(out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    @flyc.jit
    def launch_n_pes(out, stream):
        n_pes_kernel(out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    return {
        "put": launch_put,
        "get": launch_get,
        "get_nbi": launch_get_nbi,
        "putmem_signal_block": launch_putmem_signal_block,
        "fence": launch_fence,
        "quiet": launch_quiet,
        "barrier": launch_barrier,
        "sync": launch_sync,
        "wait_until": launch_wait_until,
        "signal_wait_until": launch_signal_wait_until,
        "my_pe": launch_my_pe,
        "n_pes": launch_n_pes,
    }


@unittest.skipIf(torch.cuda.device_count() < 2, "requires at least 2 GPUs")
class RocshmemFlyDSLTest(MultiProcContinuousTest):
    world_size = 2

    def _init_device(self) -> None:
        torch.cuda.set_device(self.rank)
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @requires_flydsl
    def test_flydsl_put(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        nelems = 5
        dtype = torch.int64
        val = 42 + self.rank
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)
        for i in range(nelems):
            src[i] = val * 10 + i
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        dist.barrier()

        if self.rank == 0:
            _kernels()["put"](dst, src, nelems, 1, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()

        dist.barrier()
        if self.rank == 1:
            expected = [420 + i for i in range(nelems)]
            self.assertEqual(
                dst, torch.tensor(expected, device=self.device, dtype=dtype)
            )

    @requires_flydsl
    def test_flydsl_put_dtype_mismatch(self) -> None:
        self._init_device()
        import flydsl.compiler as flyc

        @flyc.kernel
        def mismatch_kernel(dest, src, nelems, pe):
            rocshmem_flydsl.put(dest, src, nelems, pe)

        @flyc.jit
        def launch(dest, src, nelems, pe, stream):
            mismatch_kernel(dest, src, nelems, pe).launch(
                grid=(1, 1, 1), block=(1, 1, 1), stream=stream
            )

        dest = torch.empty(4, dtype=torch.int8, device=self.device)
        src = torch.empty(4, dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(
            RuntimeError, "dest and source element types must match"
        ):
            launch(dest, src, 4, 1, stream=torch.cuda.current_stream())

    @requires_flydsl
    def test_flydsl_get(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val = 7
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if self.rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        dist.barrier()

        if self.rank == 1:
            _kernels()["get"](out, inp, numel, 0, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            self.assertEqual(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_my_pe(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        # Host init happens on rendezvous. The symmetric buffer is otherwise unused.
        token = symm_mem.empty(1, dtype=torch.int32, device=self.device)
        symm_mem.rendezvous(token, group=group_name)
        out = torch.empty(1, dtype=torch.int32, device=self.device)
        _kernels()["my_pe"](out, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        self.assertEqual(int(out.cpu()[0]), self.rank)

    @requires_flydsl
    def test_flydsl_get_nbi(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val = 7
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if self.rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        dist.barrier()

        if self.rank == 1:
            _kernels()["get_nbi"](
                out, inp, numel, 0, stream=torch.cuda.current_stream()
            )
            torch.cuda.synchronize()
            self.assertEqual(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_n_pes(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        token = symm_mem.empty(1, dtype=torch.int32, device=self.device)
        symm_mem.rendezvous(token, group=group_name)
        out = torch.empty(1, dtype=torch.int32, device=self.device)
        _kernels()["n_pes"](out, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        self.assertEqual(int(out.cpu()[0]), self.world_size)

    @requires_flydsl
    def test_flydsl_putmem_signal_block(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val = 11
        signal_val = 1
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(self.rank, (1,), dtype=torch.int64).fill_(0)
        dist.barrier()

        if self.rank == 0:
            _kernels()["putmem_signal_block"](
                out,
                inp,
                numel * dtype.itemsize,
                flag,
                signal_val,
                _ROCSHMEM_SIGNAL_SET,
                1,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()

        dist.barrier()
        if self.rank == 1:
            self.assertEqual(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            self.assertEqual(
                flag, torch.tensor([signal_val], dtype=torch.int64, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_wait_until(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        dtype = torch.int32
        flag_val = 42
        flag = symm_mem.empty(1, dtype=dtype, device=self.device).fill_(0)
        src = symm_mem.empty(1, dtype=dtype, device=self.device).fill_(flag_val)
        symm_mem.rendezvous(flag, group=group_name)
        symm_mem.rendezvous(src, group=group_name)
        dist.barrier()

        stream = torch.cuda.current_stream()
        # Rank 0 blocks inside the kernel until rank 1's put lands. A host
        # barrier between the two launches would deadlock that wait.
        if self.rank == 0:
            _kernels()["wait_until"](flag, _ROCSHMEM_CMP_EQ, flag_val, stream=stream)
        else:
            _kernels()["put"](flag, src, 1, 0, stream=stream)
        torch.cuda.synchronize()
        dist.barrier()

        if self.rank == 0:
            self.assertEqual(
                flag, torch.tensor([flag_val], dtype=dtype, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_signal_wait_until(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val = 123
        signal_val = 1
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(self.rank, (1,), dtype=torch.int64).fill_(0)
        dist.barrier()

        stream = torch.cuda.current_stream()
        if self.rank == 0:
            _kernels()["putmem_signal_block"](
                out,
                inp,
                numel * dtype.itemsize,
                flag,
                signal_val,
                _ROCSHMEM_SIGNAL_SET,
                1,
                stream=stream,
            )
        else:
            _kernels()["signal_wait_until"](
                flag, _ROCSHMEM_CMP_EQ, signal_val, stream=stream
            )
        torch.cuda.synchronize()
        dist.barrier()

        if self.rank == 1:
            self.assertEqual(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            self.assertEqual(
                flag, torch.tensor([signal_val], dtype=torch.int64, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_putmem_signal_add(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val = 11
        signal_val = _SIG_VAL_BIT31
        expected_signal = 1 << 31
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(self.rank, (1,), dtype=torch.int64).fill_(0)
        dist.barrier()

        stream = torch.cuda.current_stream()
        if self.rank == 0:
            _kernels()["putmem_signal_block"](
                out,
                inp,
                numel * dtype.itemsize,
                flag,
                signal_val,
                _ROCSHMEM_SIGNAL_ADD,
                1,
                stream=stream,
            )
        else:
            _kernels()["signal_wait_until"](
                flag, _ROCSHMEM_CMP_EQ, signal_val, stream=stream
            )
        torch.cuda.synchronize()
        dist.barrier()

        if self.rank == 1:
            self.assertEqual(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            self.assertEqual(
                flag,
                torch.tensor([expected_signal], dtype=torch.int64, device=self.device),
            )

    @requires_flydsl
    def test_flydsl_wait_until_width(self) -> None:
        self._init_device()
        import flydsl.compiler as flyc

        @flyc.kernel
        def width_kernel(ivar, cmp_op, cmp_val):
            rocshmem_flydsl.wait_until(ivar, cmp_op, cmp_val)

        @flyc.jit
        def launch(ivar, cmp_op, cmp_val, stream):
            width_kernel(ivar, cmp_op, cmp_val).launch(
                grid=(1, 1, 1), block=(1, 1, 1), stream=stream
            )

        ivar = torch.empty(1, dtype=torch.int64, device=self.device)
        with self.assertRaisesRegex(
            RuntimeError,
            "wait_until expects a 32-bit synchronization variable",
        ):
            launch(ivar, _ROCSHMEM_CMP_EQ, 1, stream=torch.cuda.current_stream())

    @requires_flydsl
    def test_flydsl_signal_wait_until_width(self) -> None:
        self._init_device()
        import flydsl.compiler as flyc

        @flyc.kernel
        def width_kernel(signal, cmp_op, cmp_val):
            rocshmem_flydsl.signal_wait_until(signal, cmp_op, cmp_val)

        @flyc.jit
        def launch(signal, cmp_op, cmp_val, stream):
            width_kernel(signal, cmp_op, cmp_val).launch(
                grid=(1, 1, 1), block=(1, 1, 1), stream=stream
            )

        signal = torch.empty(1, dtype=torch.int32, device=self.device)
        with self.assertRaisesRegex(
            RuntimeError,
            "signal_wait_until expects a 64-bit signal variable",
        ):
            launch(signal, _ROCSHMEM_CMP_EQ, 1, stream=torch.cuda.current_stream())

    @requires_flydsl
    def test_flydsl_fence(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val1 = 10
        val2 = 20
        flag_val = 1
        inp1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val1)
        inp2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val2)
        out1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        flag_update = torch.tensor([flag_val], dtype=torch.int32, device=self.device)
        for tensor in (inp1, inp2, out1, out2, flag):
            symm_mem.rendezvous(tensor, group=group_name)
        dist.barrier()

        stream = torch.cuda.current_stream()
        # Rank 1 blocks until the flag put is visible. fence() orders both
        # payload puts ahead of that flag, so its arrival implies they landed.
        # A host barrier between the two launches would deadlock the wait.
        if self.rank == 0:
            _kernels()["fence"](
                out1,
                inp1,
                out2,
                inp2,
                flag,
                flag_update,
                numel,
                1,
                stream=stream,
            )
        else:
            _kernels()["wait_until"](flag, _ROCSHMEM_CMP_EQ, flag_val, stream=stream)
        torch.cuda.synchronize()
        dist.barrier()

        if self.rank == 1:
            self.assertEqual(
                out1, val1 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            self.assertEqual(
                out2, val2 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            self.assertEqual(
                flag, torch.tensor([flag_val], dtype=torch.int32, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_quiet(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        numel = 8
        dtype = torch.int8
        val = 15
        flag_val = 42
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        flag_update = torch.tensor([flag_val], dtype=torch.int32, device=self.device)
        for tensor in (inp, out, flag):
            symm_mem.rendezvous(tensor, group=group_name)
        dist.barrier()

        stream = torch.cuda.current_stream()
        # Rank 0 blocks until the flag put is visible. quiet() completes the
        # payload put before that flag, so its arrival implies the payload landed.
        if self.rank == 1:
            _kernels()["quiet"](
                out,
                inp,
                flag,
                flag_update,
                numel,
                0,
                stream=stream,
            )
        else:
            _kernels()["wait_until"](flag, _ROCSHMEM_CMP_EQ, flag_val, stream=stream)
        torch.cuda.synchronize()
        dist.barrier()

        if self.rank == 0:
            self.assertEqual(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            self.assertEqual(
                flag, torch.tensor([flag_val], dtype=torch.int32, device=self.device)
            )

    @requires_flydsl
    def test_flydsl_barrier_all(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        dtype = torch.int32
        src = symm_mem.empty(1, dtype=dtype, device=self.device).fill_(
            42 if self.rank == 0 else 0
        )
        dst = symm_mem.empty(1, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        seen = torch.zeros(1, dtype=dtype, device=self.device)
        dist.barrier()

        _kernels()["barrier"](dst, src, seen, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        dist.barrier()

        if self.rank != 0:
            self.assertEqual(seen, torch.tensor([42], dtype=dtype, device=self.device))
            self.assertEqual(dst, torch.tensor([42], dtype=dtype, device=self.device))

    @requires_flydsl
    def test_flydsl_sync_all(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        dtype = torch.int32
        local_data = symm_mem.empty(1, dtype=dtype, device=self.device).fill_(0)
        remote_data = symm_mem.empty(1, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(local_data, group=group_name)
        symm_mem.rendezvous(remote_data, group=group_name)
        dist.barrier()

        next_pe = (self.rank + 1) % self.world_size
        _kernels()["sync"](
            local_data,
            remote_data,
            next_pe,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        dist.barrier()

        self.assertEqual(
            local_data,
            torch.tensor([self.rank + 100], dtype=dtype, device=self.device),
        )
        next_rank = (self.rank + 1) % self.world_size
        self.assertEqual(
            remote_data,
            torch.tensor([next_rank + 100], dtype=dtype, device=self.device),
        )


if __name__ == "__main__":
    run_tests()
