# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard import shard_parameter
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
    generate_chunk_sharding_specs_for_test,
    generate_local_weight_sharding_params_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedEmbeddingBag(ShardedTensorTestBase):
    def _run_sharded_embedding_bag(
        self,
        spec,
        input_size,
        num_embeddings,
        embedding_dim,
        mode,
        include_last_offset=False,
        offset_size=None,
        max_norm=None,
        norm_type=2.0,
        padding_idx=None,
    ):
        # Use same seed.
        torch.manual_seed(0)
        local_embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            mode=mode,
            max_norm=max_norm,
            norm_type=norm_type,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        ).cuda(self.rank)

        sharded_embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            mode=mode,
            max_norm=max_norm,
            norm_type=norm_type,
            include_last_offset=include_last_offset,
            padding_idx=padding_idx,
        )

        # Copy the weights from local embedding bag.
        sharded_embedding_bag.weight = clone_module_parameter(
            local_embedding_bag, "weight"
        )

        # Shard the parameter.
        shard_parameter(sharded_embedding_bag, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.randint(0, num_embeddings, tuple(input_size)).cuda(self.rank)
        per_sample_weights = None
        if mode == "sum":
            per_sample_weights = torch.rand(*input_size).cuda(self.rank)

        offsets = None
        if len(input_size) == 1:
            # We need to generate certain length offset for each rank.
            # The current implementation and dist API does not support
            # the case when the offset has different lengths.
            # input_size[0] >> offset_size, so the while loop will not
            # for too long.
            while offsets is None or (offsets.size(0) != offset_size):
                offsets = torch.randint(input_size[0], (offset_size,))
                offsets[0] = 0
                if include_last_offset:
                    offsets[-1] = input_size[0]
                offsets = (
                    torch.unique(offsets, sorted=True).contiguous().cuda(self.rank)
                )

        # If max_norm is set, we need to ensure that the renorm has been applied across
        # inputs from all ranks.
        if max_norm is not None:
            gathered_inputs = [torch.zeros_like(inp) for _ in range(TEST_GPU_NUM)]
            dist.all_gather(gathered_inputs, inp)
            unique_inp = torch.unique(torch.cat(gathered_inputs))
            offsets_dummy = torch.tensor([len(unique_inp) // 2]).cuda(self.rank)
            local_embedding_bag(unique_inp, offsets=offsets_dummy)

        sharded_output = sharded_embedding_bag(
            inp,
            offsets=offsets,
            per_sample_weights=per_sample_weights,
        )

        # Run local computation
        local_output = local_embedding_bag(
            inp,
            offsets=offsets,
            per_sample_weights=per_sample_weights,
        )

        # Compare local weight and shared one to ensure the renorm
        # as expected.
        if max_norm is not None:
            sharded_dim = spec.dim
            sharded_weight = sharded_embedding_bag.weight.local_shards()[0].tensor
            (start_pos, chunk_size) = generate_local_weight_sharding_params_for_test(
                local_embedding_bag.weight, sharded_dim, TEST_GPU_NUM, spec, self.rank
            )
            local_weight_narrowed = local_embedding_bag.weight.narrow(
                sharded_dim, start_pos, chunk_size
            )
            self.assertEqual(local_weight_narrowed, sharded_weight)

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Validate for torch.nn.functional.embedding_bag version.
        local_output = torch.nn.functional.embedding_bag(
            inp,
            local_embedding_bag.weight,
            offsets=offsets,
            mode=mode,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )
        sharded_output = torch.nn.functional.embedding_bag(
            inp,
            sharded_embedding_bag.weight,
            offsets=offsets,
            mode=mode,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )

        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_embedding_bag_colwise(self):
        for spec in generate_chunk_sharding_specs_for_test(1):
            self._test_sharded_embedding_bag_with_test_cases(spec, 1)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_embedding_bag_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            self._test_sharded_embedding_bag_with_test_cases(spec, 0)

    def _test_sharded_embedding_bag_with_test_cases(self, spec, sharded_dim):
        self._run_sharded_embedding_bag(spec, [5, 5], 17, 14, "sum")
        self._run_sharded_embedding_bag(spec, [5, 4], 17, 12, "mean")
        self._run_sharded_embedding_bag(spec, [6, 7], 21, 11, "max")
        self._run_sharded_embedding_bag(
            spec,
            [5, 5],
            17,
            14,
            "sum",
            max_norm=2.5,
        )
        self._run_sharded_embedding_bag(
            spec,
            [5, 4],
            17,
            12,
            "mean",
            max_norm=2.0,
            norm_type=1.0,
        )
        self._run_sharded_embedding_bag(
            spec,
            [6, 7],
            21,
            11,
            "max",
            max_norm=1.5,
            norm_type=1.0,
        )
        self._run_sharded_embedding_bag(spec, [5, 5], 17, 14, "sum", padding_idx=6)
        self._run_sharded_embedding_bag(spec, [8, 6], 24, 13, "sum")
        self._run_sharded_embedding_bag(spec, [4, 3], 16, 14, "max")
        self._run_sharded_embedding_bag(spec, [8], 23, 13, "sum", offset_size=3)
        self._run_sharded_embedding_bag(spec, [5], 17, 12, "mean", offset_size=2)
        self._run_sharded_embedding_bag(spec, [12], 16, 12, "max", offset_size=4)
        self._run_sharded_embedding_bag(
            spec, [8], 23, 13, "sum", offset_size=3, include_last_offset=True
        )
        self._run_sharded_embedding_bag(
            spec, [12], 16, 12, "max", offset_size=4, include_last_offset=True
        )
        self._run_sharded_embedding_bag(
            spec,
            [12],
            17,
            12,
            "sum",
            offset_size=3,
            max_norm=1.25,
        )
        self._run_sharded_embedding_bag(
            spec,
            [5],
            17,
            12,
            "mean",
            offset_size=2,
            max_norm=1.25,
        )
        self._run_sharded_embedding_bag(
            spec,
            [5],
            17,
            12,
            "max",
            offset_size=2,
            max_norm=1.15,
        )
        self._run_sharded_embedding_bag(spec, [4, 3], 16, 14, "sum", padding_idx=12)
        self._run_sharded_embedding_bag(spec, [4, 3], 16, 14, "mean", padding_idx=12)
        self._run_sharded_embedding_bag(spec, [4, 3], 16, 14, "max", padding_idx=12)
        self._run_sharded_embedding_bag(
            spec,
            [12],
            17,
            12,
            "sum",
            offset_size=3,
            max_norm=1.25,
            padding_idx=10,
        )
        self._run_sharded_embedding_bag(
            spec,
            [5],
            17,
            12,
            "mean",
            offset_size=2,
            max_norm=1.25,
            padding_idx=10,
        )
        self._run_sharded_embedding_bag(
            spec,
            [5],
            17,
            12,
            "max",
            offset_size=2,
            max_norm=1.15,
            padding_idx=10,
        )


if __name__ == "__main__":
    run_tests()
