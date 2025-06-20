# Owner(s): ["oncall: distributed"]

# To run:
# TORCH_SYMMMEM=NVSHMEM python test/distributed/test_nvshmem.py


import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch._inductor.runtime.triton_compat import tl, triton
from torch.testing._internal.common_distributed import MultiProcContinousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
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


@instantiate_parametrized_tests
@requires_nvshmem()
class NVSHMEMSymmetricMemoryTest(MultiProcContinousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # NOTE: required for nvshmem allocation
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    def test_alloc(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skipIfRocm
    def test_nvshmem_all_to_all(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel_per_peer = 10
        numel = self.world_size * numel_per_peer
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        torch.ops.symm_mem.nvshmem_all_to_all(inp, out, group_name)

        expected = torch.cat(
            [
                torch.empty(numel_per_peer, dtype=dtype, device=self.device).fill_(i)
                for i in range(self.world_size)
            ]
        )
        torch.testing.assert_close(out, expected)

    @skipIfRocm
    def test_nvshmem_all_to_all_vdev(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        # Number of elements for a peer is random between [0, k)
        k = 10
        inp_splits = torch.randint(k, (self.world_size,), device=self.device)
        inp_numel = inp_splits.sum().item()
        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        dist.all_to_all_single(out_splits, inp_splits)
        out_numel = out_splits.sum().item()

        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * self.world_size
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).fill_(
            self.rank
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        in_out_splits = symm_mem.empty(
            (3, self.world_size), dtype=torch.int64, device=self.device
        )
        # Row 0 is input splits
        in_out_splits[0].copy_(inp_splits)

        torch.ops.symm_mem.nvshmem_all_to_all_vdev(inp, out, in_out_splits, group_name)

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_out_splits[0], inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(in_out_splits[1], out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `nvshmem_all_to_all_vdev` is exclusive scan
        self.assertEqual(in_out_splits[2][0], 0)
        torch.testing.assert_close(in_out_splits[2][1:], out_offsets[:-1])

        # Check data
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        dist.all_to_all_single(
            expected, inp[:inp_numel], out_splits.tolist(), inp_splits.tolist()
        )
        torch.testing.assert_close(out[:out_numel], expected)

    @skipIfRocm
    @parametrize("align", [1, 8, 16])  # `major_align` of output
    def test_nvshmem_all_to_all_vdev_2d(self, align: int) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        # Number of experts per rank
        ne = 8
        nsplits = ne * self.world_size

        # Number of elements for an expert is random between [0, k)
        k = 10
        inp_splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=self.device)

        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        dist.all_to_all_single(out_splits, inp_splits)
        # We do a .t() here because there is a rank-major to expert-major shuffle
        out_splits_t = out_splits.reshape(self.world_size, ne).t()

        # Actual number of input elements
        inp_numel = inp_splits.sum().item()
        # Actual number of output elements
        out_numel = out_splits.sum().item()
        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * nsplits
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).fill_(
            self.rank
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        # 3 rows: input splits, output splits, output offsets
        # Initiallizing all values to -1 to check if they are updated
        in_out_splits = symm_mem.empty(
            (3, nsplits), dtype=torch.int64, device=self.device
        ).fill_(-1)
        # Row 0 is input splits
        in_out_splits[0].copy_(inp_splits)

        torch.ops.symm_mem.nvshmem_all_to_all_vdev_2d(
            inp, out, in_out_splits, group_name, major_align=align
        )
        received_out_splits = in_out_splits[1]
        received_out_offsets = in_out_splits[2]

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_out_splits[0], inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(received_out_splits, out_splits_t.reshape(-1))

        # Check output offsets (row 2)
        out_split_list = out_splits_t.tolist()
        for i in range(ne):
            expert_sum = 0
            for j in range(self.world_size):
                expert_sum += out_split_list[i][j]
            # Align up expert_sum
            expert_sum_aligned = (expert_sum + align - 1) // align * align
            # If 0, make it at least `align` (bc cutlass currently does not support empty bins)
            expert_sum_aligned = max(expert_sum_aligned, align)
            # last element absorbs the padding
            out_split_list[i][-1] += expert_sum_aligned - expert_sum

        out_splits_padded = torch.tensor(out_split_list, device=self.device).reshape(-1)
        out_offsets = torch.cumsum(out_splits_padded, dim=0)  # inclusive scan
        # Make it exclusive scan because that's what `nvshmem_all_to_all_vdev_2d` returns
        out_offsets = torch.cat(
            [torch.zeros(1, device=self.device), out_offsets[:-1]]
        ).to(torch.int64)
        torch.testing.assert_close(received_out_offsets, out_offsets)

        # Check data
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        inp_splits_rank = inp_splits.reshape(self.world_size, ne).sum(1)
        out_splits_rank = out_splits.reshape(self.world_size, ne).sum(1)
        dist.all_to_all_single(
            expected,
            inp[:inp_numel],
            out_splits_rank.tolist(),
            inp_splits_rank.tolist(),
        )
        # We still need to shuffle `expected`
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        result_list = []
        for j in range(ne):
            for i in range(self.world_size):
                chunk_id = i * ne + j
                offset = out_offsets[chunk_id]
                chunk = expected[offset - out_splits[chunk_id] : offset]
                result_list.append(chunk)

        # Do a chunk-wise comparison
        for c, chunk in enumerate(result_list):
            start = received_out_offsets[c].item()
            split = received_out_splits[c].item()
            received_chunk = out[start : start + split]
            torch.testing.assert_close(received_chunk, chunk)

    @skipIfRocm
    @requires_triton()
    def test_triton_put(self) -> None:
        # A Triton kernel that calls nvshmem device side API
        @triton.jit
        def put_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.putmem_block(dst_ptr, src_ptr, numel, peer)

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
        # A Triton kernel that calls nvshmem device side API for GET
        @triton.jit
        def get_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.getmem_block(dst_ptr, src_ptr, numel, peer)

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
        # A Triton kernel that calls nvshmem device side API for GET
        # with ring topology
        @triton.jit
        def get_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.getmem_block(dst_ptr, src_ptr, numel, peer)

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
        # A Triton kernel that calls nvshmem device side API for PUT with SIGNAL
        @triton.jit
        def put_signal_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            sig_ptr,
            signal_val: tl.constexpr,
            sig_op: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.putmem_signal_block(
                dst_ptr, src_ptr, numel, sig_ptr, signal_val, sig_op, peer
            )

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

        # Kernel for waiting on the signal locally (Rank 1).
        @triton.jit
        def signal_wait_until_kernel(
            sig_ptr, cmp_op: tl.constexpr, cmp_val: tl.constexpr
        ):
            nvshmem.signal_wait_until(sig_ptr, cmp_op, cmp_val)

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
        # A Triton kernel that calls nvshmem device side API for PUT with SIGNAL
        @triton.jit
        def put_signal_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            sig_ptr,
            signal_val: tl.constexpr,
            sig_op: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.putmem_signal_block(
                dst_ptr, src_ptr, numel, sig_ptr, signal_val, sig_op, peer
            )

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

        @triton.jit
        def signal_wait_until_kernel(
            sig_ptr, cmp_op: tl.constexpr, cmp_val: tl.constexpr
        ):
            nvshmem.signal_wait_until(sig_ptr, cmp_op, cmp_val)

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
        # A Triton kernel that calls nvshmem device side API for PUT
        @triton.jit
        def put_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.putmem_block(dst_ptr, src_ptr, numel, peer)

        # A Triton kernel that calls nvshmem device side API for WAIT_UNTIL
        @triton.jit
        def wait_until_kernel(
            ivar_ptr,
            cmp_op: tl.constexpr,
            cmp_val: tl.constexpr,
        ):
            nvshmem.wait_until(ivar_ptr, cmp_op, cmp_val)

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
        dist.barrier()

        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0  # from nvshmem.h

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
            # Rank 1 sets the flag on Rank 0
            # We use a temporary tensor for the value to put.
            flag_update_val = torch.tensor(
                [flag_val], dtype=torch.int64, device=self.device
            )
            dst_ptr = out_hdl.signal_pad_ptrs[rank]
            src_ptr = flag_update_val.data_ptr()
            put_kernel[(1, 1, 1)](
                dst_ptr,
                src_ptr,
                numel=1,
                peer=peer,
                extern_libs=nvshmem_lib,
            )

    @skipIfRocm
    @requires_triton()
    def test_triton_signal_wait_until(self) -> None:
        # A Triton kernel that waits on a signal variable until it meets the compare condition.
        @triton.jit
        def signal_wait_until_kernel(
            sig_ptr,
            cmp_op: tl.constexpr,
            cmp_val: tl.constexpr,
        ):
            nvshmem.signal_wait_until(sig_ptr, cmp_op, cmp_val)

        # A Triton kernel for the producer that puts data and then signals completion.
        @triton.jit
        def put_and_signal_kernel(
            dst_ptr,
            src_ptr,
            numel: tl.constexpr,
            sig_ptr,
            signal_val: tl.constexpr,
            sig_op: tl.constexpr,
            peer: tl.constexpr,
        ):
            nvshmem.putmem_signal_block(
                dst_ptr, src_ptr, numel, sig_ptr, signal_val, sig_op, peer
            )

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
        # Ensure setup is complete on all ranks before proceeding
        dist.barrier()

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
        # Final barrier to ensure the test does not exit before assertions complete
        dist.barrier()


if __name__ == "__main__":
    run_tests()
