# Owner(s): ["oncall: distributed"]

# To run:
# TORCH_SYMMMEM=NVSHMEM python test/distributed/test_nvshmem_triton.py


import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch._inductor.runtime.triton_compat import triton
from torch.testing._internal.common_distributed import MultiProcContinousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import requires_triton


# Decorator
def requires_nvshmem():
    return skip_but_pass_in_sandcastle_if(
        not symm_mem.is_nvshmem_available(),
        "test_nvshmem requires NVSHMEM, skipping tests",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


# Shared Triton JIT kernels
@triton.jit
def put_kernel(
    dst_ptr,
    src_ptr,
    numel,
    peer,
):
    nvshmem.putmem_block(dst_ptr, src_ptr, numel, peer)


@triton.jit
def get_kernel(
    dst_ptr,
    src_ptr,
    numel,
    peer,
):
    nvshmem.getmem_block(dst_ptr, src_ptr, numel, peer)


@triton.jit
def put_signal_kernel(
    dst_ptr,
    src_ptr,
    numel,
    sig_ptr,
    signal_val,
    sig_op,
    peer,
):
    nvshmem.putmem_signal_block(
        dst_ptr, src_ptr, numel, sig_ptr, signal_val, sig_op, peer
    )


@triton.jit
def signal_wait_until_kernel(sig_ptr, cmp_op, cmp_val):
    nvshmem.signal_wait_until(sig_ptr, cmp_op, cmp_val)


@triton.jit
def signal_op_kernel(
    sig_addr,
    signal,
    sig_op,
    peer,
):
    nvshmem.signal_op(sig_addr, signal, sig_op, peer)


@triton.jit
def wait_until_kernel(
    ivar_ptr,
    cmp_op,
    cmp_val,
):
    nvshmem.wait_until(ivar_ptr, cmp_op, cmp_val)


@triton.jit
def put_and_signal_kernel(
    dst_ptr,
    src_ptr,
    numel,
    sig_ptr,
    signal_val,
    sig_op,
    peer,
):
    nvshmem.putmem_signal_block(
        dst_ptr, src_ptr, numel, sig_ptr, signal_val, sig_op, peer
    )


@triton.jit
def put_with_fence_kernel(
    dst_ptr1,
    dst_ptr2,
    src_ptr1,
    src_ptr2,
    flag_ptr,
    flag_src_ptr,
    numel,
    peer,
):
    # First put
    nvshmem.putmem_block(dst_ptr1, src_ptr1, numel, peer)
    # Ensure the first put is ordered before the next.
    nvshmem.fence()
    # Second put
    nvshmem.putmem_block(dst_ptr2, src_ptr2, numel, peer)
    # Order the second put before flag update.
    nvshmem.fence()
    # Write the flag (single int64) to signal completion.
    nvshmem.putmem_block(flag_ptr, flag_src_ptr, 1, peer)


@triton.jit
def put_with_quiet_kernel(
    dst_ptr,
    src_ptr,
    flag_dst_ptr,
    flag_src_ptr,
    numel,
    peer,
):
    # Put data
    nvshmem.putmem_block(dst_ptr, src_ptr, numel, peer)
    # Call quiet to ensure put is complete
    nvshmem.quiet()
    # Only after quiet, set the completion flag
    # This ensures the data put is complete before flag is set
    nvshmem.putmem_block(flag_dst_ptr, flag_src_ptr, 1, peer)


@instantiate_parametrized_tests
@requires_nvshmem()
class NVSHMEMTritonTest(MultiProcContinousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # NOTE: required for nvshmem allocation
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    @requires_triton()
    def test_triton_put(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        # Enable NVSHMEM for Triton
        nvshmem_lib = nvshmem.enable_triton()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        val = 5
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)

        peer = 1 - rank
        if rank == 0:
            dst_ptr = out_hdl.buffer_ptrs[rank]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            put_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel=numel,
                peer=peer,
                extern_libs=nvshmem_lib,
            )

        dist.barrier()
        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    def test_triton_get(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        nvshmem_lib = nvshmem.enable_triton()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        val = 7
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        dist.barrier()
        peer = 1 - rank
        if rank == 1:
            # Rank 1 gets data from rank 0
            dst_ptr = out_hdl.buffer_ptrs[rank]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            get_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel=numel,
                peer=peer,
                extern_libs=nvshmem_lib,
            )
        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    def test_triton_get_ring(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        nvshmem_lib = nvshmem.enable_triton()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        world_size = dist.get_world_size()
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Each rank fills its input buffer with its own rank value
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        dist.barrier()

        # Ring topology: each rank gets data from the rank to its left
        # rank 0 gets from rank (world_size-1), rank 1 gets from rank 0, etc.
        peer = (rank - 1) % world_size

        # All ranks execute the get operation
        dst_ptr = out_hdl.buffer_ptrs[rank]
        src_ptr = inp_hdl.buffer_ptrs[rank]
        get_kernel[(1, 1, 1)](
            dst_ptr,
            src_ptr,
            numel=numel,
            peer=peer,
            extern_libs=nvshmem_lib,
        )

        expected_value = peer
        torch.testing.assert_close(
            out, expected_value * torch.ones(numel, dtype=dtype, device=self.device)
        )

    @skipIfRocm
    @requires_triton()
    def test_triton_put_signal_set(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        nvshmem_lib = nvshmem.enable_triton()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Data buffers
        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
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
            dst_ptr = out_hdl.buffer_ptrs[peer]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            sig_ptr = out_hdl.signal_pad_ptrs[peer]
            put_signal_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel=numel,
                sig_ptr=sig_ptr,
                signal_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
                extern_libs=nvshmem_lib,
            )

        if rank == 1:
            # Wait until signal flag is set by Rank 0
            sig_ptr_local = out_hdl.signal_pad_ptrs[rank]
            signal_wait_until_kernel[(1,)](
                sig_ptr_local,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
                extern_libs=nvshmem_lib,
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
    def test_triton_put_signal_add(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        nvshmem_lib = nvshmem.enable_triton()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Data buffers
        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
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
            dst_ptr = out_hdl.buffer_ptrs[peer]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            sig_ptr = out_hdl.signal_pad_ptrs[peer]
            put_signal_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel=numel,
                sig_ptr=sig_ptr,
                signal_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_ADD,
                peer=peer,
                extern_libs=nvshmem_lib,
            )

        if rank == 1:
            sig_ptr_local = out_hdl.signal_pad_ptrs[rank]
            signal_wait_until_kernel[(1, 1, 1)](
                sig_ptr_local,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
                extern_libs=nvshmem_lib,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    def test_triton_wait_until(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        nvshmem_lib = nvshmem.enable_triton()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank

        # Data buffers
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        val = 13
        flag_val = 21
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)

        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0  # from nvshmem.h
        NVSHMEM_SIGNAL_SET = 0  # atomic set operation

        if rank == 0:
            # Rank 0 waits for the flag to be set by Rank 1, then checks the data
            ivar_ptr = out_hdl.signal_pad_ptrs[rank]
            wait_until_kernel[(1, 1, 1)](
                ivar_ptr,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=flag_val,
                extern_libs=nvshmem_lib,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

        if rank == 1:
            # Rank 1 puts data into Rank 0's output buffer
            dst_ptr = out_hdl.buffer_ptrs[rank]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            put_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel=numel,
                peer=peer,
                extern_libs=nvshmem_lib,
            )

            # Rank 1 sets the flag on Rank 0 using nvshmemx_signal_op
            sig_addr = out_hdl.signal_pad_ptrs[rank]
            signal_op_kernel[(1, 1, 1)](
                sig_addr,
                signal=flag_val,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
                extern_libs=nvshmem_lib,
            )

    @skipIfRocm
    @requires_triton()
    def test_triton_signal_wait_until(self) -> None:
        self._init_device()
        # Enable NVSHMEM for Triton
        nvshmem_lib = nvshmem.enable_triton()
        group_name = dist.group.WORLD.group_name
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
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
        # Consumer (rank 1) prepares the destination buffer
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        # Use the signal pad for synchronization, as in previous tests
        flag_dtype = torch.int64
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=flag_dtype).fill_(0)

        if rank == 0:
            # Producer (rank 0): Puts data into rank 1's `out` buffer and then sets the flag
            dst_ptr = out_hdl.buffer_ptrs[peer]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            sig_ptr = out_hdl.signal_pad_ptrs[peer]
            put_and_signal_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel,
                sig_ptr,
                signal_val=COMPLETION_FLAG_VAL,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
                extern_libs=nvshmem_lib,
            )
        elif rank == 1:
            # Consumer (rank 1): Waits on the signal variable using `signal_wait_until`.
            sig_ptr = out_hdl.signal_pad_ptrs[rank]
            signal_wait_until_kernel[(1, 1, 1)](
                sig_ptr,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=COMPLETION_FLAG_VAL,
                extern_libs=nvshmem_lib,
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
        nvshmem_lib = nvshmem.enable_triton()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        peer = 1 - rank
        # Message configuration
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        val1 = 10
        val2 = 20
        flag_val = 1
        # Symmetric buffers
        inp1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val1)
        inp2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val2)
        out1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp1_hdl = symm_mem.rendezvous(inp1, group=group_name)
        inp2_hdl = symm_mem.rendezvous(inp2, group=group_name)
        out1_hdl = symm_mem.rendezvous(out1, group=group_name)
        out2_hdl = symm_mem.rendezvous(out2, group=group_name)

        # Flag buffer resides in the signal pad of out2.
        flag = out2_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int64, device=self.device
        )
        NVSHMEM_CMP_EQ = 0  # compare equal

        if rank == 0:
            dst_ptr1 = out1_hdl.buffer_ptrs[rank]
            dst_ptr2 = out2_hdl.buffer_ptrs[rank]
            src_ptr1 = inp1_hdl.buffer_ptrs[rank]
            src_ptr2 = inp2_hdl.buffer_ptrs[rank]
            flag_ptr = out2_hdl.signal_pad_ptrs[rank]
            flag_src_ptr = flag_update_val.data_ptr()

            put_with_fence_kernel[(1, 1, 1)](
                dst_ptr1,
                dst_ptr2,
                src_ptr1,
                src_ptr2,
                flag_ptr,
                flag_src_ptr,
                numel,
                peer=peer,
                extern_libs=nvshmem_lib,
            )
        elif rank == 1:
            # Wait until flag is set by Rank 0.
            ivar_ptr = out2_hdl.signal_pad_ptrs[rank]
            wait_until_kernel[(1, 1, 1)](
                ivar_ptr,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=flag_val,
                extern_libs=nvshmem_lib,
            )

            # Verify ordered data arrival.
            torch.testing.assert_close(
                out1, val1 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                out2, val2 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([flag_val], dtype=torch.int64, device=self.device)
            )

    @skipIfRocm
    @requires_triton()
    def test_triton_quiet(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        # Enable NVSHMEM for Triton
        nvshmem_lib = nvshmem.enable_triton()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)
        rank = self.rank
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        # Data buffers
        val = 15
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        inp_hdl = symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        # Use signal pad as completion flag
        flag_val = 42
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
            # Rank 0 waits for flag from Rank 1
            ivar_ptr = out_hdl.signal_pad_ptrs[rank]
            wait_until_kernel[(1, 1, 1)](
                ivar_ptr,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=flag_val,
                extern_libs=nvshmem_lib,
            )
            # After flag is set, data should be complete due to quiet
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
        if rank == 1:
            # Rank 1 puts data and flag with quiet in between
            dst_ptr = out_hdl.buffer_ptrs[rank]
            src_ptr = inp_hdl.buffer_ptrs[rank]
            flag_dst_ptr = out_hdl.signal_pad_ptrs[rank]
            # Create a tensor for the flag value
            flag_update_val = torch.tensor(
                [flag_val], dtype=torch.int64, device=self.device
            )
            flag_src_ptr = flag_update_val.data_ptr()
            put_with_quiet_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                flag_dst_ptr,
                flag_src_ptr,
                numel=numel,
                peer=peer,
                extern_libs=nvshmem_lib,
            )


if __name__ == "__main__":
    run_tests()
