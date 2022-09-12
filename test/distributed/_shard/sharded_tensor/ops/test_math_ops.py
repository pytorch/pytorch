# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._shard import _shard_tensor
import torch.distributed._shard.sharded_tensor as sharded_tensor
import torch.distributed as dist

from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)

from torch.testing._internal.distributed._shard.sharded_tensor import (
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)

from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    gen_binary_op_func,
    generate_chunk_sharding_specs_for_test,
)


class TestMathOps(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_basic_math_ops(self):
        ops = ["torch.add", "torch.sub", "torch.mul", "torch.div", "+", "-", "*", "/"]

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        sharded_lhs = sharded_tensor.rand(spec, (12, 3))
        sharded_rhs = sharded_tensor.rand(spec, (12, 3))
        current_rank = dist.get_rank()
        global_lhs = (
            torch.empty((12, 3), device=current_rank) if current_rank == 0 else None
        )
        global_rhs = (
            torch.empty((12, 3), device=current_rank) if current_rank == 0 else None
        )
        sharded_lhs.gather(dst=0, out=global_lhs)
        sharded_rhs.gather(dst=0, out=global_rhs)

        for op in ops:
            binary_op = gen_binary_op_func(op)
            binary_op_ = gen_binary_op_func(op, inplace=True)
            # test basic math ops between ShardedTensors
            sharded_output = binary_op(sharded_lhs, sharded_rhs)
            output = (
                torch.empty((12, 3), device=current_rank) if current_rank == 0 else None
            )
            sharded_output.gather(dst=0, out=output)

            if current_rank == 0:
                global_output = binary_op(global_lhs, global_rhs)

                self.assertEqual(output, global_output)

            # test basic math ops between ShardedTensor and scalar
            scalars = [3, 1.8]
            for scalar in scalars:
                sharded_output_lhs = binary_op(sharded_lhs, scalar)
                output_lhs = (
                    torch.empty((12, 3), device=current_rank)
                    if current_rank == 0
                    else None
                )
                sharded_output_lhs.gather(dst=0, out=output_lhs)

                sharded_output_rhs = binary_op(scalar, sharded_rhs)
                output_rhs = (
                    torch.empty((12, 3), device=current_rank)
                    if current_rank == 0
                    else None
                )
                sharded_output_rhs.gather(dst=0, out=output_rhs)

                if current_rank == 0:
                    global_output_lhs = binary_op(global_lhs, scalar)
                    global_output_rhs = binary_op(scalar, global_rhs)

                    self.assertEqual(output_lhs, global_output_lhs)
                    self.assertEqual(output_rhs, global_output_rhs)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_math_ops_errors(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        sharded_lhs = sharded_tensor.rand(spec, (20, 3))
        sharded_rhs = sharded_tensor.rand(spec, (12, 3))

        with self.assertRaisesRegex(
            RuntimeError, "Implicit broadcasting not supported"
        ):
            torch.add(sharded_lhs, sharded_rhs)

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.rand(spec, 10, 10)

        with self.assertRaisesRegex(RuntimeError, "not supported"):
            torch.add(st, sharded_rhs)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_bmm(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            lhs = torch.rand(15, 4, 5).cuda(self.rank)
            rhs = torch.rand(15, 5, 6).cuda(self.rank)
            tensor = lhs.bmm(rhs)
            st_lhs = _shard_tensor(lhs, spec)
            st_rhs = _shard_tensor(rhs, spec)
            st_expected = _shard_tensor(tensor, spec)
            self.assertTrue(torch.allclose(torch.bmm(st_lhs, st_rhs), st_expected))
            self.assertTrue(torch.allclose(st_lhs.bmm(st_rhs), st_expected))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_bmm_errors(self):
        specs = generate_chunk_sharding_specs_for_test(0)
        st_lhs = sharded_tensor.rand(specs[0], (15, 5, 6))
        st_rhs = sharded_tensor.rand(specs[1], (15, 5, 6))
        with self.assertRaisesRegex(
            NotImplementedError,
            "Both st and st2 need to have same placements for bmm",
        ):
            torch.bmm(st_lhs, st_rhs)
        for spec in specs:
            st_lhs = sharded_tensor.rand(spec, (20, 3))
            st_rhs = sharded_tensor.rand(spec, (20, 3))
            with self.assertRaisesRegex(
                TypeError,
                "both st and st2 need to be a 3D ShardedTensor",
            ):
                torch.bmm(st_lhs, st_rhs)
            rhs = torch.rand(15, 5, 6).cuda(self.rank)
            with self.assertRaisesRegex(
                TypeError,
                "st2 needs to be a ShardedTensor for torch.bmm",
            ):
                torch.bmm(st_lhs, rhs)
            spec.dim = 1
            st_lhs = sharded_tensor.rand(spec, (15, 5, 6))
            st_rhs = sharded_tensor.rand(spec, (15, 5, 6))
            with self.assertRaisesRegex(
                NotImplementedError,
                "Only support performing bmm on tensors sharded on dim 0 now",
            ):
                torch.bmm(st_lhs, st_rhs)
