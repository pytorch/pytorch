# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_nvshmem.py


import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)


# Decorator
def requires_nvshmem():
    return skip_but_pass_in_sandcastle_if(
        not symm_mem.is_nvshmem_available(),
        "test_nvshmem requires NVSHMEM, skipping tests",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMSymmetricMemoryTest(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

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
    def test_alloc_without_device_context(self) -> None:
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        self.assertEqual(out.device, self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skipIfRocm
    def test_mempool_tensor_factory(self) -> None:
        """
        Test the effectiveness of MemPool on tensor factory ops.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        src_rank = 0

        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            if self.rank == src_rank:
                tensor = torch.arange(numel, dtype=dtype, device=self.device)
            else:
                tensor = torch.zeros(numel, dtype=dtype, device=self.device)

        symm_mem.rendezvous(tensor, group=group_name)
        torch.ops.symm_mem.nvshmem_broadcast(tensor, src_rank, group_name)
        self.assertEqual(tensor, torch.arange(numel, dtype=dtype, device=self.device))

    @skipIfRocm
    def test_mempool_compute_ops(self) -> None:
        """
        Apply MemPool context to a compute op that creates input to collective.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        dim = 1024
        w = torch.ones(dim, dim, dtype=dtype, device=self.device)
        x0 = torch.ones(1, dim, dtype=dtype, device=self.device)

        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            x = x0 + self.rank
            y = torch.mm(x, w)

        # y should be a symm tensor
        torch.ops.symm_mem.nvshmem_broadcast(y, 0, group_name)
        expected = torch.mm(x0, w)
        self.assertEqual(y, expected)

    @skipIfRocm
    def test_handle_offset(self) -> None:
        """
        Test if handle offset is correctly set.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            x0 = torch.empty(numel, dtype=dtype, device=self.device)
            x1 = torch.empty_like(x0)

        hdl0 = symm_mem.rendezvous(x0, group=group_name)
        hdl1 = symm_mem.rendezvous(x1, group=group_name)
        self.assertEqual(hdl0.offset, 0)
        self.assertEqual(hdl1.offset, x0.untyped_storage().nbytes())

    def test_get_remote_tensor(self) -> None:
        """
        Get a remote tensor and use regular aten ops to write to it.
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        allocator = symm_mem.get_mempool_allocator(self.device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            # src data stores my rank
            x = torch.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
            y = torch.empty_like(x)

        hdl_y = symm_mem.rendezvous(y, group=group_name)
        peer = (self.rank + 1) % self.world_size  # Shifting pattern
        y_remote = hdl_y.get_remote_tensor(peer, y.size(), y.dtype)
        y_remote.copy_(x)
        dist.barrier()
        # Expecting data from -1 rank
        expected = torch.empty(numel, dtype=dtype, device=self.device).fill_(
            (self.rank - 1) % self.world_size
        )
        self.assertEqual(y, expected)

    @skipIfRocm
    def test_nvshmem_put(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(tensor, group=group_name)
        signal_pad = hdl.get_signal_pad(self.rank)
        signal_val = 5

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_put_with_signal(
                tensor, signal_pad, signal_val, 1
            )
        elif self.rank == 1:
            torch.ops.symm_mem.nvshmem_wait_for_signal(signal_pad, signal_val, 0)
            torch.testing.assert_close(
                tensor, torch.zeros(numel, dtype=dtype, device=self.device)
            )

    @skipIfRocm
    def test_nvshmem_get(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_get(tensor, 1)
            # TODO: remove after we have wait_signal
            dist.barrier()
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        else:
            # handle.wait_signal(src_rank=0)
            # TODO: remove after we have wait_signal
            dist.barrier()


def get_occurrence_numbers(tensor):
    """
    Transform tensor to show which occurrence each element is.

    Example: tensor([1, 2, 1, 3, 1, 2]) -> tensor([1, 1, 2, 1, 3, 2])
    """
    device = tensor.device
    # Get unique values and their inverse mapping
    unique_vals, inverse = torch.unique(tensor, return_inverse=True)

    # Create a tensor to count occurrences for each unique value
    n_unique = len(unique_vals)
    n_elements = len(tensor)

    # Create a matrix where each row corresponds to a unique value
    # and columns correspond to positions in the original tensor
    indicator_matrix = torch.zeros(
        n_unique, n_elements, dtype=torch.float, device=device
    )
    indicator_matrix[inverse, torch.arange(n_elements)] = 1.0

    # Cumulative sum along columns gives us occurrence numbers
    occurrence_counts = torch.cumsum(indicator_matrix, dim=1) - indicator_matrix

    # Extract the occurrence number for each position
    result = occurrence_counts[inverse, torch.arange(n_elements, device=device)]

    return result.long()


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMAll2AllTest(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

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
    def test_all_to_all_vdev(self) -> None:
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

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        in_splits = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        out_splits_offsets = symm_mem.empty(
            (2, self.world_size), dtype=torch.int64, device=self.device
        )
        # Row 0 is input splits
        in_splits.copy_(inp_splits)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev(
            inp, out, in_splits, out_splits_offsets, group_name
        )

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_splits, inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(out_splits_offsets[0], out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `all_to_all_vdev` is exclusive scan
        self.assertEqual(out_splits_offsets[1][0], 0)
        torch.testing.assert_close(out_splits_offsets[1][1:], out_offsets[:-1])

        # Check data
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        dist.all_to_all_single(
            expected, inp[:inp_numel], out_splits.tolist(), inp_splits.tolist()
        )
        torch.testing.assert_close(out[:out_numel], expected)

    @skipIfRocm
    @parametrize("align", [1, 8, 16])  # `major_align` of output
    def test_all_to_all_vdev_2d(self, align: int) -> None:
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

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        in_splits = symm_mem.empty(
            nsplits, dtype=torch.int64, device=self.device
        ).copy_(inp_splits)
        # 2 rows: output splits, output offsets
        # Initiallizing all values to -1 to check if they are updated
        out_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        ).fill_(-1)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d(
            inp, out, in_splits, out_splits_offsets, group_name, major_align=align
        )
        received_out_splits = out_splits_offsets[0]
        received_out_offsets = out_splits_offsets[1]

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_splits, inp_splits)

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
        # Make it exclusive scan because that's what `all_to_all_vdev_2d` returns
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
    def test_all_to_all_vdev_2d_offset(self) -> None:
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
        # Each split up align to k, as the offset, i.e. [0, k, 2k, 3k, ...]
        inp_offsets = torch.arange(
            0, k * nsplits, k, dtype=torch.int64, device=self.device
        )

        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        # Remember that we up-align each input split to k?
        max_inp_numel = k * nsplits
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)
        # 2 rows: input splits, input offsets
        in_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        )
        # 2 rows: output splits, output offsets
        # Initiallizing all values to -1 to check if they are updated
        out_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        ).fill_(-1)

        # Row 0 is input splits
        in_splits_offsets[0].copy_(inp_splits)
        # Row 1 is input offsets
        in_splits_offsets[1].copy_(inp_offsets)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            inp, out, in_splits_offsets, out_splits_offsets, group_name
        )
        received_out_splits = out_splits_offsets[0]
        received_out_offsets = out_splits_offsets[1]

        # Check input splits and offsets -- should not change
        torch.testing.assert_close(in_splits_offsets[0], inp_splits)
        torch.testing.assert_close(in_splits_offsets[1], inp_offsets)

        # Check output splits (row 1)
        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        # First need to transpose the input splits
        inp_splits_t = inp_splits.reshape(ne, self.world_size).t().contiguous()
        dist.all_to_all_single(out_splits, inp_splits_t)
        torch.testing.assert_close(received_out_splits, out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `all_to_all_vdev_2d_offset` is exclusive scan
        self.assertEqual(received_out_offsets[0], 0)
        torch.testing.assert_close(received_out_offsets[1:], out_offsets[:-1])

        # Check data
        # Let's "squeeze" the padding out of the input data first
        inp_chunks = []  # (ne, nranks)
        for i in range(ne):
            inp_chunks_e = []  # (nranks,)
            for j in range(self.world_size):
                chunk_id = i * self.world_size + j
                offset = in_splits_offsets[1][chunk_id]
                chunk = inp[offset : offset + inp_splits[chunk_id]]
                inp_chunks_e.append(chunk)
            inp_chunks.append(inp_chunks_e)

        # Transpose the 2D input chunks
        inp_chunks_t = list(zip(*inp_chunks))
        # Now it is (nranks, ne), concatenate the e's
        inp_chunks_t = [torch.cat(row) for row in inp_chunks_t]

        # Create empty output tensors -- each tensor is data to be received from a peer
        out_splits = out_splits.reshape(self.world_size, ne)
        # Sum the split sizes of all experts, per peer
        receive_size_per_peer = out_splits.sum(1)
        out_chunks = []  # (nranks,)
        for i in range(self.world_size):
            out_chunks.append(
                torch.empty(
                    receive_size_per_peer[i].item(), dtype=dtype, device=self.device
                )
            )

        # All-to-all
        dist.all_to_all(out_chunks, inp_chunks_t)

        # Concatenate the output chunks received from all peers
        out_expected = torch.cat(out_chunks)
        # Actual number of output elements
        out_numel = out_splits.sum().item()
        self.assertEqual(out_expected.shape[0], out_numel)

        # Check data
        torch.testing.assert_close(out_expected, out[:out_numel])

    @skipIfRocm
    def test_make_a2a_exchange_plan(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # Number of elements for a peer is random between [0, k)
        k = 10
        orig_inp_splits = torch.randint(k, (self.world_size,), device=self.device)

        # Create symm_mem tensors
        in_splits = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        src_offsets = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        out_splits = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        dst_offsets = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )

        in_splits.copy_(orig_inp_splits)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        symm_mem.make_a2a_exchange_plan(
            in_splits, src_offsets, out_splits, dst_offsets, group_name
        )

        # Check input splits -- should not change
        torch.testing.assert_close(in_splits, orig_inp_splits)

        # Check output splits
        # Exchange input splits to get output splits
        expected_out_splits = torch.zeros_like(orig_inp_splits)
        dist.all_to_all_single(expected_out_splits, orig_inp_splits)
        torch.testing.assert_close(expected_out_splits, out_splits)

        # Check src offsets
        orig_src_offsets = torch.cumsum(orig_inp_splits, dim=0)  # inclusive scan
        # Make it exclusive
        orig_src_offsets = torch.cat(
            [torch.zeros(1, device=self.device), orig_src_offsets[:-1]]
        ).to(torch.int64)
        expected_src_offsets = torch.empty_like(orig_src_offsets)
        dist.all_to_all_single(expected_src_offsets, orig_src_offsets)
        torch.testing.assert_close(src_offsets, expected_src_offsets)

        # Check dst offsets
        expected_dst_offsets = torch.cumsum(
            expected_out_splits, dim=0
        )  # inclusive scan
        self.assertEqual(dst_offsets[0], 0)
        torch.testing.assert_close(dst_offsets[1:], expected_dst_offsets[:-1])

    @skipIfRocm
    def test_a2a_with_exchange_plan(self) -> None:
        self._init_device()

        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # Number of elements for a peer is random between [0, k)
        k = 10
        orig_inp_splits = torch.randint(k, (self.world_size,), device=self.device)

        # Create splits and offsets
        in_splits = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        src_offsets = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        out_splits = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )
        dst_offsets = symm_mem.empty(
            self.world_size, dtype=torch.int64, device=self.device
        )

        # Create data
        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * self.world_size
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor
        dtype = torch.float
        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device).fill_(-1)

        in_splits.copy_(orig_inp_splits)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        # Create exchange plan
        plan = symm_mem.make_a2a_exchange_plan(
            in_splits, src_offsets, out_splits, dst_offsets, group_name
        )

        # Prepare expected output
        inp_numel = in_splits.sum().item()
        out_numel = out_splits.sum().item()
        expected = torch.empty(out_numel, dtype=dtype, device=self.device)
        dist.all_to_all_single(
            expected, inp[:inp_numel], out_splits.tolist(), in_splits.tolist()
        )

        # Exchange data with plan
        # Loop a couple times to ensure the plan is reusable
        for _ in range(3):
            symm_mem.all_to_all_v(inp, out, plan, group_name)
            torch.testing.assert_close(out[:out_numel], expected)

    @skipIfRocm
    @parametrize("align", [1])  # `major_align` of output
    def test_make_a2a_2d_exchange_plan(self, align: int) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # Number of experts per rank
        ne = 8
        nsplits = ne * self.world_size

        # Number of elements for an expert is random between [0, k)
        k = 10
        orig_inp_splits = torch.randint(
            k, (nsplits,), dtype=torch.int64, device=self.device
        )

        # Create symm_mem tensors
        in_splits = symm_mem.empty(nsplits, dtype=torch.int64, device=self.device)
        src_offsets = symm_mem.empty(nsplits, dtype=torch.int64, device=self.device)
        out_splits = symm_mem.empty(
            nsplits, dtype=torch.int64, device=self.device
        ).fill_(0)
        dst_offsets = symm_mem.empty(
            nsplits, dtype=torch.int64, device=self.device
        ).fill_(0)

        in_splits.copy_(orig_inp_splits)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        plan = symm_mem.make_a2a_2d_exchange_plan(
            in_splits, src_offsets, out_splits, dst_offsets, group_name
        )

        # Exchange input splits to get output splits
        expected_out_splits = torch.zeros_like(orig_inp_splits)
        dist.all_to_all_single(expected_out_splits, orig_inp_splits)
        # We do a .t() here because there is a rank-major to expert-major shuffle
        expected_out_splits = expected_out_splits.reshape(self.world_size, ne).t()
        torch.testing.assert_close(plan.out_splits, expected_out_splits.reshape(-1))

        # Check dst offsets
        out_split_list = expected_out_splits.tolist()
        for i in range(ne):
            expert_sum = 0
            for j in range(self.world_size):
                expert_sum += out_split_list[i][j]
            # # Align up expert_sum
            # expert_sum_aligned = (expert_sum + align - 1) // align * align
            # # If 0, make it at least `align` (bc cutlass currently does not support empty bins)
            # expert_sum_aligned = max(expert_sum_aligned, align)
            # # last element absorbs the padding
            # out_split_list[i][-1] += expert_sum_aligned - expert_sum

        out_splits_padded = torch.tensor(out_split_list, device=self.device).reshape(-1)
        out_offsets = torch.cumsum(out_splits_padded, dim=0)  # inclusive scan
        # Make it exclusive scan because that's what `all_to_all_vdev_2d` returns
        out_offsets = torch.cat(
            [torch.zeros(1, device=self.device), out_offsets[:-1]]
        ).to(torch.int64)
        expected_dst_offsets = torch.empty(
            nsplits, dtype=torch.int64, device=self.device
        )
        dist.all_to_all_single(
            expected_dst_offsets,
            out_offsets.reshape(ne, self.world_size).t().contiguous(),
        )
        torch.testing.assert_close(
            expected_dst_offsets,
            plan.dst_offsets,
            msg=f"""
            Expecting
            {expected_dst_offsets}
            Got
            {plan.dst_offsets}""",
        )

    @skipIfRocm
    def test_all_to_all_v_2d_index_push(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        # Number of experts per rank
        ne = 4
        tot_experts = ne * self.world_size

        # Create topk indices of shape (n_tokens, topk)
        topk = 2
        n_tokens = 128
        topk_indices = torch.randint(
            tot_experts, (n_tokens, topk), dtype=torch.int64, device=self.device
        )

        # Convert indices to splits
        orig_inp_splits = torch.histc(
            topk_indices,
            bins=tot_experts,
        )

        # Create symm_mem tensors
        in_splits = symm_mem.empty(
            tot_experts, dtype=torch.int64, device=self.device
        ).copy_(orig_inp_splits)
        src_offsets = symm_mem.empty(tot_experts, dtype=torch.int64, device=self.device)
        out_splits = symm_mem.empty(
            tot_experts, dtype=torch.int64, device=self.device
        ).fill_(0)
        dst_offsets = symm_mem.empty(
            tot_experts, dtype=torch.int64, device=self.device
        ).fill_(0)

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        plan = symm_mem.make_a2a_2d_exchange_plan(
            in_splits, src_offsets, out_splits, dst_offsets, group_name
        )

        # Create data
        max_out_tokens = n_tokens * self.world_size
        dtype = torch.float
        hid_dim = 1024
        inp = symm_mem.empty(n_tokens, hid_dim, dtype=dtype, device=self.device).copy_(
            torch.randn(n_tokens, hid_dim, dtype=dtype, device=self.device)
        )
        out = symm_mem.empty(
            max_out_tokens, hid_dim, dtype=dtype, device=self.device
        ).fill_(-1)

        # Figure out rank of each token in its expert chunk
        occurrences = get_occurrence_numbers(topk_indices.view(-1))

        # Number of CUDA blocks (random choice)
        n_blocks = 2
        # Evenly spread token to CUDA blocks
        tokens_per_block = n_tokens // n_blocks
        # Start offset of each CUDA block
        b_start = torch.arange(
            0, n_tokens, tokens_per_block, dtype=torch.int64, device=self.device
        )
        # Number of tokens for each CUDA block
        b_len = torch.full(
            (n_blocks,), tokens_per_block, dtype=torch.int64, device=self.device
        )
        # Ready signal for each CUDA block. In this test we set all tokens as ready in one shot
        b_head = b_start + b_len

        dist.barrier()

        torch.ops.symm_mem._all_to_all_v_2d_index_push(
            inp,
            out,
            topk_indices,
            occurrences,
            plan.dst_offsets,
            group_name,
            b_start,
            b_len,
            b_head,
        )

        # Check data using all_to_all_vdev_2d
        # Token sequence is inflated topk times
        expanded_seqlen = n_tokens * topk
        sorted_indices = torch.argsort(topk_indices.view(-1))
        expanded_inp = symm_mem.empty(
            expanded_seqlen, hid_dim, dtype=dtype, device=self.device
        ).copy_(inp[sorted_indices // topk])
        overflow = 2
        expected_out = symm_mem.empty(
            expanded_seqlen * overflow, hid_dim, dtype=dtype, device=self.device
        )
        out_splits_offsets = symm_mem.empty(
            (2, tot_experts), dtype=torch.int64, device=self.device
        )
        dist.barrier()
        torch.ops.symm_mem.all_to_all_vdev_2d(
            expanded_inp, expected_out, in_splits, out_splits_offsets, group_name
        )

        # Check data
        out_len = out_splits_offsets[1][-1] + out_splits_offsets[0][-1]
        torch.testing.assert_close(out[:out_len], expected_out[:out_len])


# Help function used by multiple tests
def dispatch_then_combine(device, align: int, group) -> None:
    """
    Shuffle the tokens, then combine them, and check if the combined data is
    exactly the same as the original input data
    """
    group_name = group.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    dtype = torch.float
    # Number of experts per rank
    ne = 8
    nsplits = ne * group.size()

    # Number of elements for an expert is random between [0, k)
    k = 10
    inp_splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=device)

    # Actual number of input elements
    inp_numel = inp_splits.sum().item()
    # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
    max_inp_numel = k * nsplits
    # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
    overflow_factor = group.size()  # worst case: one rank receives all data
    max_out_numel = max_inp_numel * overflow_factor

    # Buffers for shuffle
    inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=device).copy_(
        torch.randn(max_inp_numel, dtype=dtype, device=device)
    )
    out = symm_mem.empty(max_out_numel, dtype=dtype, device=device).fill_(-1)
    in_splits = symm_mem.empty(nsplits, dtype=torch.int64, device=device).copy_(
        inp_splits
    )
    # 2 rows: output splits, output offsets
    # Initiallizing all values to -1 to check if they are updated
    out_splits_offsets = symm_mem.empty(
        (2, nsplits), dtype=torch.int64, device=device
    ).fill_(-1)

    # Buffers for combine
    combine_out = symm_mem.empty(max_out_numel, dtype=dtype, device=device).fill_(-1)
    # 2 rows: output splits, output offsets
    # Initiallizing all values to -1 to check if they are updated
    combine_out_splits_offsets = symm_mem.empty(
        (2, nsplits), dtype=torch.int64, device=device
    ).fill_(-1)

    # Wait for all ranks to finish tensor allocation before accessing them
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    # Shuffle the tokens
    torch.ops.symm_mem.all_to_all_vdev_2d(
        inp, out, in_splits, out_splits_offsets, group_name, major_align=align
    )

    # Combine the tokens
    # `out_splits_offsets` from shuffle is exactly the `input_splits_offsets` for combine
    # `out` data from shuffle is exactly the `input` data for combine
    torch.ops.symm_mem.all_to_all_vdev_2d_offset(
        out, combine_out, out_splits_offsets, combine_out_splits_offsets, group_name
    )

    # Assert the combined data is exactly the same as the original input data
    torch.testing.assert_close(combine_out[:inp_numel], inp[:inp_numel])

    # Assert the combined out splits are exactly the same as the original input splits
    torch.testing.assert_close(combine_out_splits_offsets[0], inp_splits)

    # Assert the combined out offsets are exactly the same as the original input offsets
    inp_offsets = torch.cumsum(inp_splits, dim=0)  # inclusive scan
    # Make it exclusive scan because that's what `all_to_all_vdev_2d_offset` returns
    inp_offsets = torch.cat([torch.zeros(1, device=device), inp_offsets[:-1]]).to(
        torch.int64
    )
    torch.testing.assert_close(combine_out_splits_offsets[1], inp_offsets)

    # Wait for all ranks to finish accessing tensors before freeing them
    dist.barrier(group=group)
    torch.cuda.synchronize(device)


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class DispatchCombineTest(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    @parametrize("align", [1, 8, 16])  # `major_align` of output
    def test_dispatch_combine(self, align: int) -> None:
        """
        Test dispatch-and-combine over World group
        """
        torch.manual_seed(42 + self.rank)
        self._init_device()
        dispatch_then_combine(self.device, align, dist.group.WORLD)


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class DispatchCombineInSubgroups(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    # TODO: FIXIT. Currently, `MultiProcContinuousTest` treats the skip code as a
    # failure
    @skip_if_lt_x_gpu(4)
    def test_dispatch_combine_subgroup(self) -> None:
        """
        Test dispatch-and-combine over concurrent subgroups
        """
        torch.manual_seed(42 + self.rank)
        self._init_device()
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        # Test on two concurrent subgroups
        ngroups = 2
        subgroup_size = self.world_size // ngroups
        dm = init_device_mesh(
            device_type, (ngroups, subgroup_size), mesh_dim_names=("dp", "ep")
        )
        subgroup = dm.get_group("ep")
        dispatch_then_combine(self.device, align=8, group=subgroup)


@requires_nvshmem()
@requires_cuda_p2p_access()
class HierarchicalTest(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    def init_mesh(self) -> None:
        # Arrange gpus into [nnodes, ranks_per_node] mesh
        ranks_per_node = 2
        nnodes = self.world_size // ranks_per_node
        self.dm = init_device_mesh(
            device_type, (nnodes, ranks_per_node), mesh_dim_names=("inter", "intra")
        )
        self.inter_group = self.dm.get_group("inter")
        self.intra_group = self.dm.get_group("intra")
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        symm_mem.enable_symm_mem_for_group(self.inter_group.group_name)
        symm_mem.enable_symm_mem_for_group(self.intra_group.group_name)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def test_rail_dispatch(self) -> None:
        """
        Test rail-wise dispatch
        """
        self._init_device()
        self.init_mesh()
        torch.manual_seed(42)
        dtype = torch.float

        seqlen = 512
        hid_dim = 1024
        inp = torch.randn((seqlen, hid_dim), dtype=dtype, device=self.device)

        nnodes = self.inter_group.size()
        # Limit token routing to half of the nodes
        topk_nodes = nnodes // 2
        # Create some synthetic token choices for sending to which nodes
        topk_node_idx = torch.randint(
            nnodes, (seqlen, topk_nodes), dtype=torch.int64, device=self.device
        )
        # Convert indices to splits
        splits = torch.histc(topk_node_idx, bins=nnodes, min=0, max=nnodes - 1)
        sorted_indices = torch.argsort(topk_node_idx.view(-1))
        expanded_inp = inp[sorted_indices // topk_nodes]
        expanded_seqlen = seqlen * topk_nodes

        # Max number of output tokens (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = nnodes  # worst case: one rank receives all data
        max_out_len = expanded_seqlen * overflow_factor

        inp = symm_mem.empty(
            (expanded_seqlen, hid_dim), dtype=dtype, device=self.device
        ).copy_(expanded_inp)
        out = symm_mem.empty(
            (max_out_len, hid_dim), dtype=dtype, device=self.device
        ).fill_(-1)
        in_splits = symm_mem.empty(nnodes, dtype=torch.int64, device=self.device).copy_(
            splits
        )
        out_splits_offsets = symm_mem.empty(
            (2, nnodes), dtype=torch.int64, device=self.device
        )

        # Sync all ranks to ensure remote tensors are allocated
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev(
            inp, out, in_splits, out_splits_offsets, self.inter_group.group_name
        )


if __name__ == "__main__":
    run_tests()
