# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_nvshmem.py


import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcContinousTest
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
class NVSHMEMSymmetricMemoryTest(MultiProcContinousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # NOTE: required for nvshmem allocation
        torch.empty(1, device=self.device)
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
    def test_nvshmem_put(self) -> None:
        self._init_device()
        group_name = dist.group.WORLD.group_name
        symm_mem.enable_symm_mem_for_group(group_name)

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(tensor, group=group_name)

        if self.rank == 0:
            torch.ops.symm_mem.nvshmem_put(tensor, 1)
            # TODO: remove after we have wait_signal
            dist.barrier()
        elif self.rank == 1:
            # handle.wait_signal(src_rank=0)
            # TODO: remove after we have wait_signal
            dist.barrier()
            torch.testing.assert_close(
                tensor, torch.zeros(numel, dtype=dtype, device=self.device)
            )
        else:
            dist.barrier()

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


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMAll2AllTest(MultiProcContinousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # NOTE: required for nvshmem allocation
        torch.empty(1, device=self.device)
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
        in_out_splits = symm_mem.empty(
            (3, self.world_size), dtype=torch.int64, device=self.device
        )
        # Row 0 is input splits
        in_out_splits[0].copy_(inp_splits)

        torch.ops.symm_mem.all_to_all_vdev(inp, out, in_out_splits, group_name)

        # Check input splits (row 0) -- should not change
        torch.testing.assert_close(in_out_splits[0], inp_splits)

        # Check output splits (row 1)
        torch.testing.assert_close(in_out_splits[1], out_splits)

        # Check output offsets (row 2)
        out_offsets = torch.cumsum(out_splits, dim=0)  # inclusive scan
        # output offsets from `all_to_all_vdev` is exclusive scan
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

        # Exchange input splits to get output splits
        out_splits = torch.zeros_like(inp_splits)
        # First need to transpose the input splits
        inp_splits_t = inp_splits.reshape(ne, self.world_size).t().contiguous()
        dist.all_to_all_single(out_splits, inp_splits_t)

        # Actual number of output elements
        out_numel = out_splits.sum().item()
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

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            inp, out, in_splits_offsets, out_splits_offsets, group_name
        )
        received_out_splits = out_splits_offsets[0]
        received_out_offsets = out_splits_offsets[1]

        # Check input splits and offsets -- should not change
        torch.testing.assert_close(in_splits_offsets[0], inp_splits)
        torch.testing.assert_close(in_splits_offsets[1], inp_offsets)

        # Check output splits (row 1)
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
        self.assertEqual(out_expected.shape[0], out_numel)

        # Check data
        torch.testing.assert_close(out_expected, out[:out_numel])

    @skipIfRocm
    @parametrize("align", [1, 8, 16])  # `major_align` of output
    def test_shuffle_combine(self, align: int) -> None:
        """
        Shuffle the tokens, then combine them, and check if the combined data is
        exactly the same as the original input data
        """
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

        # Actual number of input elements
        inp_numel = inp_splits.sum().item()
        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * nsplits
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        # Buffers for shuffle
        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device).fill_(
            self.rank
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

        # Shuffle the tokens
        torch.ops.symm_mem.all_to_all_vdev_2d(
            inp, out, in_splits, out_splits_offsets, group_name, major_align=align
        )

        # Buffers for combine
        combine_out = symm_mem.empty(
            max_out_numel, dtype=dtype, device=self.device
        ).fill_(-1)
        # 2 rows: output splits, output offsets
        # Initiallizing all values to -1 to check if they are updated
        combine_out_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        ).fill_(-1)

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
        inp_offsets = torch.cat(
            [torch.zeros(1, device=self.device), inp_offsets[:-1]]
        ).to(torch.int64)
        torch.testing.assert_close(combine_out_splits_offsets[1], inp_offsets)


@instantiate_parametrized_tests
@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMAutogradTest(MultiProcContinousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # NOTE: required for nvshmem allocation
        torch.empty(1, device=self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    @skipIfRocm
    def test_autograd_AllToAllVDev2d(self) -> None:
        from torch.distributed._symmetric_memory._autograd import AllToAllVDev2d

        # Mimics Group GEMM alignment
        align = 8
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

        # Max number of input elements (must be a constant across ranks for symmetric memory allocation)
        max_inp_numel = k * nsplits
        # Max number of output elements (must be a constant across ranks for symmetric memory allocation)
        overflow_factor = self.world_size  # worst case: one rank receives all data
        max_out_numel = max_inp_numel * overflow_factor

        inp = (
            symm_mem.empty(max_inp_numel, dtype=dtype, device=self.device)
            .copy_(torch.randn(max_inp_numel, dtype=dtype, device=self.device))
            .requires_grad_(True)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=self.device)

        in_splits = symm_mem.empty(
            nsplits, dtype=torch.int64, device=self.device
        ).copy_(inp_splits)
        # 2 rows: output splits, output offsets
        out_splits_offsets = symm_mem.empty(
            (2, nsplits), dtype=torch.int64, device=self.device
        )

        class TokenDispatcher(torch.nn.Module):
            def __init__(self, group_name: str, align: int, in_len, out_len, token_shape, nsplits, dtype, device) -> None:
                super().__init__()
                self.group_name = group_name
                self.align = align
                self.grad_out_buf = symm_mem.empty(out_len, *token_shape, dtype=dtype, device=device)
                self.grad_in_buf = symm_mem.empty(in_len, *token_shape, dtype=dtype, device=device)
                self.grad_in_splits_offsets = symm_mem.empty((2, nsplits), dtype=torch.int64, device=device)

            def forward(
                self,
                inp: torch.Tensor,
                out: torch.Tensor,
                in_splits: torch.Tensor,
                out_splits_offsets: torch.Tensor,
            ) -> torch.Tensor:
                return AllToAllVDev2d.apply(
                    inp, out, in_splits, out_splits_offsets, self.group_name, self.align,
                    self.grad_out_buf, self.grad_in_buf, self.grad_in_splits_offsets,
                )

        dispatcher = TokenDispatcher(group_name, align, max_inp_numel, max_out_numel, inp.shape[1:], nsplits, dtype, self.device)

        # dispatcher = torch.compile(
        #     dispatcher,
        #     # backend="aot_eager",
        # )

        # Perform a Dot product with output, these are the weights
        weight = torch.empty(max_out_numel, dtype=dtype, device=self.device).fill_(self.rank + 1)

        # Run a few iterations
        iters = 2
        for i in range(iters):
            inp.grad = None
            output = dispatcher(inp, out, in_splits, out_splits_offsets)
            p = torch.dot(output, weight)
            p.backward()

        # Check gradients
        start = 0
        for i, split in enumerate(in_splits.tolist()):
            grad_chunk = inp.grad[start : start + split]
            dst_rank = i // ne
            torch.testing.assert_close(
                grad_chunk,
                torch.empty(split, device=self.device).fill_(dst_rank + 1),
            )
            start += split


if __name__ == "__main__":
    run_tests()
