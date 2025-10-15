# Owner(s): ["oncall: distributed"]
# To run:
# python test/distributed/test_nvshmem_triton.py

import sys

import triton.language as tl

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch._inductor.runtime.triton_compat import triton
from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import IS_H100, requires_triton


if not symm_mem.is_nvshmem_available():
    print("NVSHMEM not available, skipping tests")
    sys.exit(0)


def requires_h100():
    return skip_but_pass_in_sandcastle_if(
        not IS_H100,
        "NVSHMEM requires H100. Skipping test on non-H100 GPU.",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


# Shared Triton JIT kernels


@requires_nvshmem
@triton.jit
def my_put_kernel(
    dest,
    src,
    nelems,
    pe,
):
    nvshmem.put(dest, src, nelems, pe)


@requires_nvshmem
@triton.jit
def my_get_kernel(
    dest,
    src,
    nelems,
    pe,
    nbi: tl.constexpr,  # use nonblocking interface if True
):
    if nbi:
        nvshmem.get_nbi(dest, src, nelems, pe)
        nvshmem.quiet()
    else:
        nvshmem.get(dest, src, nelems, pe)


@requires_nvshmem
@triton.jit
def my_putmem_signal_block_kernel(
    dst,
    src,
    size_bytes,
    signal,
    sig_val,
    sig_op,
    peer,
):
    nvshmem.putmem_signal_block(dst, src, size_bytes, signal, sig_val, sig_op, peer)


@requires_nvshmem
@triton.jit
def my_signal_wait_until_kernel(signal, cmp_op, cmp_val):
    nvshmem.signal_wait_until(signal, cmp_op, cmp_val)


@requires_nvshmem
@triton.jit
def my_signal_op_kernel(
    sig_addr,
    signal,
    sig_op,
    peer,
):
    nvshmem.signal_op(sig_addr, signal, sig_op, peer)


@requires_nvshmem
@triton.jit
def my_wait_until_kernel(
    ivar,
    cmp_op,
    cmp_val,
):
    nvshmem.wait_until(ivar, cmp_op, cmp_val)


@requires_nvshmem
@triton.jit
def my_fence_kernel():
    nvshmem.fence()


@requires_nvshmem
@triton.jit
def my_put_with_fence_kernel(
    dst1,
    src1,
    dst2,
    src2,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    # First put
    nvshmem.put(dst1, src1, nelems, peer)
    # Ensure the first put is ordered before the next.
    nvshmem.fence()
    # Second put
    nvshmem.put(dst2, src2, nelems, peer)
    # Order the second put before flag update.
    nvshmem.fence()
    # Write the flag (single int64) to signal completion.
    nvshmem.put(flag_dst, flag_src, 1, peer)


@requires_nvshmem
@triton.jit
def my_put_with_quiet_kernel(
    dst,
    src,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    # Put data
    nvshmem.put(dst, src, nelems, peer)
    # Call quiet to ensure put is complete
    nvshmem.quiet()
    # Only after quiet, set the completion flag
    # This ensures the data put is complete before flag is set
    nvshmem.put(flag_dst, flag_src, 1, peer)


@requires_nvshmem
@triton.jit
def my_barrier_test_kernel(
    dst,
    src,
    nelems,
):
    # Testing barrier_all() requires coordinated operations across PEs within
    # the same kernel execution. Unlike other kernels that just wrap NVSHMEM
    # primitives, this one implements the full test logic to properly verify
    # device-side barrier synchronization.
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Rank 0 broadcasts its value to all other ranks
    if my_pe == 0:
        # Write initial value
        p_src = src.to(tl.pointer_type(tl.int32))
        tl.store(p_src, 42)
        # Put to all other ranks
        i = 1
        while i < n_pes:
            nvshmem.put(dst, src, nelems, i)
            i += 1

    # Synchronize all PEs
    nvshmem.barrier_all()

    # Non-zero ranks increment the received value
    if my_pe != 0:
        p_dst = dst.to(tl.pointer_type(tl.int32))
        received = tl.load(p_dst)
        tl.store(p_dst, received + 1)


@requires_nvshmem
@triton.jit
def my_barrier_all_kernel():
    nvshmem.barrier_all()


@requires_nvshmem
@triton.jit
def my_sync_test_kernel(
    local_data,
    remote_data,
    nelems,
):
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Each PE writes a unique value to its local memory
    p_local = local_data.to(tl.pointer_type(tl.int32))
    unique_value = my_pe + 100  # PE 0 writes 100, PE 1 writes 101, etc.
    tl.store(p_local, unique_value)

    # sync_all() ensures local stores are visible to other PEs
    # but doesn't guarantee completion of any remote operations
    nvshmem.sync_all()

    # Now each PE reads from the next PE's memory to verify visibility
    # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
    next_pe = (my_pe + 1) % n_pes
    nvshmem.get(remote_data, local_data, nelems, next_pe)

    # The get should now see the value that the next PE wrote locally
    # because sync_all() made those local stores visible


@requires_nvshmem
@triton.jit
def my_alltoall_kernel(
    team_handle,
    dst,
    src,
    nelems_per_pe,
):
    nvshmem.alltoall(team_handle, dst, src, nelems_per_pe)


@requires_nvshmem
@triton.jit
def my_broadcast_kernel(
    team_handle,
    dst,
    src,
    nelems,
    pe_root,
):
    nvshmem.broadcast(team_handle, dst, src, nelems, pe_root)


@requires_nvshmem
@triton.jit
def my_reduce_kernel(
    team_handle,
    dest_tensor,
    source_tensor,
    nreduce,
    operation: tl.constexpr,
):
    nvshmem.reduce(team_handle, dest_tensor, source_tensor, nreduce, operation)


@instantiate_parametrized_tests
class NVSHMEMTritonTest(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_put(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        # Configuration
        nelems = 5  # number of elements to transfer
        dtype = torch.int64
        val = 42 + rank  # Each rank has different data

        # Create symmetric tensors
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)

        # Fill source tensor with rank-specific pattern
        for i in range(nelems):
            src[i] = (
                val * 10 + i
            )  # Rank 0: [420, 421, 422, 423, 424], Rank 1: [430, 431, ...]

        # Rendezvous
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)

        # Synchronize before operation
        dist.barrier()

        peer = 1 - rank
        if rank == 0:
            # Rank 0 puts its data to Rank 1
            my_put_kernel[(1,)](
                dst,
                src,
                nelems,
                peer,
            )

        # Synchronize after operation
        dist.barrier()

        if rank == 1:
            # Verify that rank 1 received rank 0's data
            expected = [420 + i for i in range(nelems)]
            torch.testing.assert_close(
                dst, torch.tensor(expected, device=self.device, dtype=dtype)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    @parametrize("nbi", [False, True])  # Test both blocking and nonblocking interfaces
    def test_triton_get(self, nbi: bool) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        # Configuration
        numel = 8
        dtype = torch.int8
        val = 7

        # Create symmetric tensors
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = 1 - rank
        if rank == 1:
            # Rank 1 gets data from rank 0 using tensor-aware API
            my_get_kernel[(1,)](
                out,
                inp,
                numel,
                peer,
                nbi=nbi,
            )
        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_get_ring(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        world_size = dist.get_world_size()

        # Configuration
        numel = 8
        dtype = torch.int8

        # Each rank fills its input buffer with its own rank value
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()

        # Ring topology: each rank gets data from the rank to its left
        # rank 0 gets from rank (world_size-1), rank 1 gets from rank 0, etc.
        peer = (rank - 1) % world_size

        # All ranks execute the get operation using tensor-aware API
        my_get_kernel[(1,)](
            out,
            inp,
            numel,
            peer,
            nbi=False,
        )

        expected_value = peer
        torch.testing.assert_close(
            out, expected_value * torch.ones(numel, dtype=dtype, device=self.device)
        )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_put_signal_set(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Data buffers
        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)

        # Use the signal pad attached to the output symmetric memory handle
        # as the flag buffer for signaling completion.
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_SET = 0  # value defined by NVSHMEM for atomic set
        SIGNAL_VAL = 1  # Signal completion value
        NVSHMEM_CMP_EQ = 0  # compare equal for signal wait until

        if rank == 0:
            # Rank 0 puts into Rank 1
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
            )

        if rank == 1:
            # Wait until signal flag is set by Rank 0
            my_signal_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
            )
            # After wait completes, verify data and flag contents
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_put_signal_add(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Data buffers
        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)

        # Use the signal pad attached to the output symmetric memory handle
        # as the flag buffer for signaling completion.
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_ADD = 5  # atomic add operation
        SIGNAL_VAL = 16  # val + NVSHMEM_SIGNAL_ADD
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
            # Rank 0 puts into Rank 1
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_ADD,
                peer=peer,
            )

        if rank == 1:
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_wait_until(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        rank = self.rank
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0  # equal comparison
        FLAG_INITIAL_VALUE = 0
        FLAG_FINAL_VALUE = 42

        # Use a single int64 symmetric tensor as our synchronization flag.
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(
            FLAG_INITIAL_VALUE
        )
        symm_mem.rendezvous(flag, group=group_name)
        expected_flag = torch.tensor(
            [FLAG_FINAL_VALUE], dtype=torch.int32, device=self.device
        )

        if rank == 0:
            # Rank 0 (the waiter)
            my_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=FLAG_FINAL_VALUE,
            )

            # Verification
            torch.testing.assert_close(
                flag,
                expected_flag,
            )

        if rank == 1:
            # Rank 1 (the signaler)
            # Launch a kernel to put the value to Rank 0's flag tensor.
            my_put_kernel[(1,)](
                flag,  # Destination symmetric tensor on the remote PE
                expected_flag,  # Source data tensor (local)
                1,  # Number of elements
                peer,  # The target PE (Rank 0)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_signal_wait_until(self) -> None:
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        peer = 1 - rank

        # NVSHMEM constants from documentation
        NVSHMEM_CMP_EQ = 0  # equal comparison
        NVSHMEM_SIGNAL_SET = 0  # atomic set operation

        # Message configuration
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        val_to_put = 123  # arbitrary test value
        COMPLETION_FLAG_VAL = 1

        # Producer (rank 0) prepares the data to send
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val_to_put)
        symm_mem.rendezvous(inp, group=group_name)
        # Consumer (rank 1) prepares the destination buffer
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        # Use the signal pad for synchronization, as in previous tests
        flag_dtype = torch.int64
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=flag_dtype).fill_(0)

        if rank == 0:
            # Producer (rank 0): Puts data into rank 1's `out` buffer and then sets the flag
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=COMPLETION_FLAG_VAL,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
            )
        elif rank == 1:
            # Consumer (rank 1): Waits on the signal variable using `signal_wait_until`.
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=COMPLETION_FLAG_VAL,
            )
            # After the wait returns, verify data and flag
            torch.testing.assert_close(
                out, val_to_put * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag,
                torch.tensor(
                    [COMPLETION_FLAG_VAL], dtype=flag_dtype, device=self.device
                ),
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_fence(self) -> None:
        """
        Rank 0 performs two put operations into Rank 1's buffers with a fence
        between them, followed by another fence and a flag update. Rank 1 waits
        for the flag, then verifies that both destination buffers contain the
        expected values. The flag is transferred after the final fence, so
        its arrival implies that both preceding puts have been delivered in
        order.
        """
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        peer = 1 - rank
        # Message configuration
        dtype = torch.int8
        numel = 8

        val1 = 10
        val2 = 20
        flag_val = 1
        # Symmetric buffers
        inp1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val1)
        inp2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val2)
        out1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp1, group=group_name)
        symm_mem.rendezvous(inp2, group=group_name)
        symm_mem.rendezvous(out1, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)

        # Use regular symmetric memory tensor for flag
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        symm_mem.rendezvous(flag, group=group_name)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int32, device=self.device
        )
        NVSHMEM_CMP_EQ = 0  # compare equal

        if rank == 0:
            my_put_with_fence_kernel[(1,)](
                out1,
                inp1,
                out2,
                inp2,
                flag,
                flag_update_val,
                nelems=numel,
                peer=peer,
            )
        elif rank == 1:
            # Wait until flag is set by Rank 0
            my_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=flag_val,
            )

            # Verify ordered data arrival.
            torch.testing.assert_close(
                out1, val1 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                out2, val2 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([flag_val], dtype=torch.int32, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_quiet(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        peer = 1 - rank

        dtype = torch.int8
        numel = 8
        val = 15
        flag_val = 42

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int32, device=self.device
        )

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(flag, group=group_name)

        NVSHMEM_CMP_EQ = 0

        dist.barrier()
        if rank == 1:
            my_put_with_quiet_kernel[(1,)](
                out,
                inp,
                flag,
                flag_update_val,
                nelems=numel,
                peer=peer,
            )
        elif rank == 0:
            my_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=flag_val,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
        dist.barrier()

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_barrier(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        numel = 1
        dtype = torch.int32

        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        dst = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)

        my_barrier_test_kernel[(1,)](
            dst,
            src,
            nelems=numel,
            launch_cooperative_grid=True,
            num_ctas=1,
        )
        dist.barrier()

        if rank == 0:
            torch.testing.assert_close(
                src, torch.tensor([42], device=self.device, dtype=dtype)
            )
        else:
            torch.testing.assert_close(
                dst, torch.tensor([43], device=self.device, dtype=dtype)
            )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_sync(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        numel = 1
        dtype = torch.int32

        # Create symmetric buffers
        local_data = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        remote_data = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(local_data, group=group_name)
        symm_mem.rendezvous(remote_data, group=group_name)

        # Launch kernel with cooperative grid
        my_sync_test_kernel[(1,)](
            local_data,
            remote_data,
            nelems=numel,
            launch_cooperative_grid=True,
            num_ctas=1,
        )

        # Verify results
        # Each PE should have written rank + 100 to its local_data
        expected_local = rank + 100
        torch.testing.assert_close(
            local_data, torch.tensor([expected_local], device=self.device, dtype=dtype)
        )

        # Each PE should have read (next_rank + 100) into its remote_data
        # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
        next_rank = (rank + 1) % self.world_size
        expected_remote = next_rank + 100
        torch.testing.assert_close(
            remote_data,
            torch.tensor([expected_remote], device=self.device, dtype=dtype),
        )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_alltoall(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        world_size = dist.get_world_size()
        rank = self.rank
        # Each PE will send 2 int64 elements to every other PE
        nelems_per_pe = 2
        dtype = torch.int64
        # Source buffer: contains data for all PEs
        # Layout: [data_for_pe0, data_for_pe1, ...]
        src_size = nelems_per_pe * world_size
        src = symm_mem.empty(src_size, dtype=dtype, device=self.device)
        # Fill source with rank-specific data
        # Formula: rank * 100 + destination_pe
        for i in range(world_size):
            value = rank * 100 + i
            src[i * nelems_per_pe : (i + 1) * nelems_per_pe] = value
        # Destination buffer
        dst = symm_mem.empty(src_size, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Synchronize before alltoall
        dist.barrier()
        team_handle = 0  # NVSHMEM_TEAM_WORLD handle is 0
        # Launch the kernel using new tensor-aware API
        my_alltoall_kernel[(1,)](
            team_handle,
            dst,
            src,
            nelems_per_pe,
            launch_cooperative_grid=True,
        )
        # Synchronize after alltoall
        dist.barrier()
        # Verify results
        for i in range(world_size):
            # After alltoall, we should receive data from PE i that was intended for us
            # PE i sends (i * 100 + rank) to us
            expected = i * 100 + rank
            actual = dst[i * nelems_per_pe : (i + 1) * nelems_per_pe]
            torch.testing.assert_close(actual, torch.full_like(actual, expected))

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    def test_triton_broadcast(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        # Configuration
        nelems = 4  # number of elements
        dtype = torch.int64

        # Source buffer - only root will have meaningful data
        pe_root = 0  # PE 0 will be the root
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        # Destination buffer
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)

        if rank == pe_root:
            # Root fills with specific pattern
            for i in range(nelems):
                src[i] = 100 + i
        else:
            # Non-root PEs have dummy data
            src.fill_(-1)

        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)

        # Synchronize before broadcast
        dist.barrier()

        # Execute broadcast
        team_handle = 0  # NVSHMEM_TEAM_WORLD
        my_broadcast_kernel[(1,)](
            team_handle,
            dst,
            src,
            nelems,
            pe_root,
            launch_cooperative_grid=True,
        )

        # Synchronize after broadcast
        dist.barrier()

        # Verify results - all ranks should have the root's data
        expected = [100 + i for i in range(nelems)]
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.float16,
            torch.float32,
            # torch.float64,  # Tensor-likes are not close
            torch.bfloat16,
        ],
    )
    def test_triton_sum_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        world_size = dist.get_world_size()
        rank = self.rank
        # Configuration
        nreduce = 3  # number of separate reductions
        # Source buffer - each rank contributes different values
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            src[i] = (rank + 1) * (i + 1)  # Rank 0: [1,2,3], Rank 1: [2,4,6], etc.
        # Destination buffer
        dst = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Calculate expected results
        expected = []
        for i in range(nreduce):
            # Sum across all ranks: sum((rank+1)*(i+1) for rank in range(world_size))
            total = sum((r + 1) * (i + 1) for r in range(world_size))
            expected.append(total)

        # Synchronize before reduction
        dist.barrier()

        # Execute sum reduction across all ranks
        team_handle = 0  # NVSHMEM_TEAM_WORLD
        my_reduce_kernel[(1,)](
            team_handle,
            dst,
            src,
            nreduce,
            operation="sum",
            launch_cooperative_grid=True,
        )

        # Synchronize after reduction
        dist.barrier()

        # Verify results
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ],
    )
    def test_triton_minmax_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        world_size = dist.get_world_size()
        rank = self.rank
        # Configuration
        nreduce = 2  # number of values to reduce
        # Source buffers for min and max
        src_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        src_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        # Each rank contributes different values
        # For min: rank 0: [10, 20], rank 1: [15, 5], etc.
        # For max: same values
        for i in range(nreduce):
            if i == 0:
                src_min[i] = 10 + rank * 5  # 10, 15, 20, ...
                src_max[i] = 10 + rank * 5
            else:
                src_min[i] = 20 - rank * 15  # 20, 5, -10, ...
                src_max[i] = 20 - rank * 15
        # Destination buffers
        dst_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        dst_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src_min, group=group_name)
        symm_mem.rendezvous(src_max, group=group_name)
        symm_mem.rendezvous(dst_min, group=group_name)
        symm_mem.rendezvous(dst_max, group=group_name)
        # Calculate expected results
        all_values = []
        for i in range(nreduce):
            values = []
            for r in range(world_size):
                if i == 0:
                    values.append(10 + r * 5)
                else:
                    values.append(20 - r * 15)
            all_values.append(values)
        expected_min = [min(vals) for vals in all_values]
        expected_max = [max(vals) for vals in all_values]
        dist.barrier()
        # Execute MIN reduction
        team_handle = 0
        my_reduce_kernel[(1,)](
            team_handle,
            dst_min,
            src_min,
            nreduce,
            operation="min",
            launch_cooperative_grid=True,
        )
        # Execute MAX reduction
        my_reduce_kernel[(1,)](
            team_handle,
            dst_max,
            src_max,
            nreduce,
            operation="max",
            launch_cooperative_grid=True,
        )
        dist.barrier()
        # Verify results
        torch.testing.assert_close(
            dst_min, torch.tensor(expected_min, device=self.device, dtype=dtype)
        )
        torch.testing.assert_close(
            dst_max, torch.tensor(expected_max, device=self.device, dtype=dtype)
        )

    @skipIfRocm
    @requires_triton()
    @requires_h100()
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            # torch.float64,  # Tensor-likes are not close
            torch.bfloat16,
        ],
    )
    def test_triton_prod_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        world_size = dist.get_world_size()
        rank = self.rank
        # Configuration
        nreduce = 3  # number of separate reductions
        # Source buffer - each rank contributes different values
        # Use very small values to avoid overflow, especially for small integer types
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            # Use values that won't overflow even for int8: all values 1 or 2
            if i == 0:
                # For first element: rank 0,2,4... gets 1, rank 1,3,5... gets 2
                src[i] = 1 if rank % 2 == 0 else 2
            elif i == 1:
                # For second element: all get 1 (no multiplication effect)
                src[i] = 1
            else:
                # For third element: rank 0,1 get 1, rank 2,3 get 2, etc. (groups of 2)
                src[i] = 1 if (rank // 2) % 2 == 0 else 2
        # Destination buffer
        dst = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Calculate expected results
        vals = torch.empty(nreduce, world_size, dtype=dtype)
        vals[0, ::2] = 1
        vals[0, 1::2] = 2
        vals[1] = 1
        vals2 = vals[2].view(-1, 2, 2)
        vals2[:, 0] = 1
        vals2[:, 1] = 2
        expected = vals.prod(-1).tolist()

        # Synchronize before reduction
        dist.barrier()

        # Execute product reduction across all ranks
        team_handle = 0  # NVSHMEM_TEAM_WORLD
        my_reduce_kernel[(1,)](
            team_handle,
            dst,
            src,
            nreduce,
            operation="prod",
            launch_cooperative_grid=True,
        )

        # Synchronize after reduction
        dist.barrier()

        # Verify results
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )


if __name__ == "__main__":
    run_tests()
