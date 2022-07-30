# Owner(s): ["oncall: distributed"]
import io

import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed._shard import _shard_tensor
from torch.distributed._shard.replicated_tensor import ReplicatedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)

from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    gen_binary_op_func
)
from torch.testing._internal.distributed._shard.sharded_tensor import TEST_GPU_NUM


class TestReplicatedTensor(ShardedTensorTestBase):

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_replicated_tensor_basics(self):
        local_tensor = torch.ones(3, 3, device=f"cuda:{self.rank}") * 4
        replica_tensor = ReplicatedTensor(local_tensor)
        # validate it's a replicated tensor by checking values on all rank
        validated = replica_tensor.validate()
        self.assertEqual(validated, True)
        res = replica_tensor + 2
        self.assertIsInstance(res, torch.Tensor)
        self.assertNotIsInstance(res, ReplicatedTensor)
        self.assertEqual(res, torch.ones(3, 3) * 6)

        # modify local tensor on certain rank, and test if validation raise
        if self.rank == 2:
            local_tensor += 3

        with self.assertRaisesRegex(ValueError, 'have different values'):
            replica_tensor.validate()

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_replicated_tensor_inter_op_replicated_tensor(self):
        local_tensor = torch.ones(3, 3, device=f"cuda:{self.rank}")
        replica_tensor1 = ReplicatedTensor(local_tensor * 4)
        replica_tensor2 = ReplicatedTensor(local_tensor * 6)

        new_tensor = replica_tensor1 * replica_tensor2
        self.assertIsInstance(new_tensor, ReplicatedTensor)
        self.assertEqual(new_tensor, torch.ones(3, 3) * 24)

        # test replicated tensor inter-op with different pgs
        new_pg = dist.new_group(ranks=[1, 2, 3])
        replica_tensor_new_group = ReplicatedTensor(local_tensor * 3, process_group=new_pg)

        with self.assertRaisesRegex(RuntimeError, 'must be in the same'):
            replica_tensor_new_group * replica_tensor1


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_replicated_tensor_inter_op_tensor(self):
        local_tensor = torch.ones(3, 3, device=f"cuda:{self.rank}") * 4
        replica_tensor = ReplicatedTensor(local_tensor)

        local_rand_tensor = torch.randn(3, 3, device=f"cuda:{self.rank}")

        new_tensor = replica_tensor + local_rand_tensor
        self.assertIsInstance(new_tensor, torch.Tensor)
        self.assertNotIsInstance(new_tensor, ReplicatedTensor)

        self.assertEqual(new_tensor, local_tensor + local_rand_tensor)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_replicated_tensor_inter_op_sharded_tensor(self):
        torch.manual_seed(self.rank)

        local_tensor1 = torch.rand(12, 3, device=f"cuda:{self.rank}") * 4
        local_tensor2 = torch.ones(12, 3, device=f"cuda:{self.rank}") * 4

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st = _shard_tensor(local_tensor1, spec, src_rank=0)
        replica_tensor = ReplicatedTensor(local_tensor2)

        ops = ["torch.add", "torch.sub", "torch.mul", "torch.div", "+", "-", "*", "/"]

        for op in ops:
            binary_op = gen_binary_op_func(op)
            res = binary_op(st, replica_tensor)
            self.assertIsInstance(res, sharded_tensor.ShardedTensor)
            self.assertNotIsInstance(res, ReplicatedTensor)
            output = torch.empty((12, 3), device=self.rank) if self.rank == 0 else None
            res.gather(dst=0, out=output)

            if self.rank == 0:
                local_output = binary_op(local_tensor1, local_tensor2)
                self.assertEqual(output, local_output)

            # reflective
            reflect_res = binary_op(replica_tensor, st)
            self.assertIsInstance(reflect_res, sharded_tensor.ShardedTensor)
            self.assertNotIsInstance(reflect_res, ReplicatedTensor)
            reflect_output = torch.empty((12, 3), device=self.rank) if self.rank == 0 else None
            reflect_res.gather(dst=0, out=reflect_output)

            if self.rank == 0:
                reflect_local_output = binary_op(local_tensor2, local_tensor1)
                self.assertEqual(reflect_output, reflect_local_output)


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_replicated_tensor_implicit_broadcasting(self):
        #  use same seed
        torch.manual_seed(self.rank)

        # test implicit broadcasting
        local_tensor1 = torch.rand(12, 3, device=f"cuda:{self.rank}") * 4
        # we use size (3) to trigger the implicit broadcasting logic
        # and it will fail if implicit broadcasting not happen.
        local_tensor2 = torch.ones(3, device=f"cuda:{self.rank}")

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st = _shard_tensor(local_tensor1, spec, src_rank=0)
        replica_tensor = ReplicatedTensor(local_tensor2)

        ops = ["torch.add", "torch.sub", "torch.mul", "torch.div", "+", "-", "*", "/"]

        for op in ops:
            binary_op = gen_binary_op_func(op)
            # replicated tensor should automatically broadcasted
            res = binary_op(st, replica_tensor)

            self.assertIsInstance(res, sharded_tensor.ShardedTensor)
            output = torch.empty((12, 3), device=self.rank) if self.rank == 0 else None
            res.gather(dst=0, out=output)

            if self.rank == 0:
                local_output = binary_op(local_tensor1, local_tensor2)
                self.assertEqual(output, local_output)


    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_replicated_tensor_inter_op_sharded_tensor_errors(self):
        local_tensor = torch.ones(3, 3, device=f"cuda:{self.rank}") * 4
        replica_tensor = ReplicatedTensor(local_tensor)

        torch.manual_seed(self.rank)
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st1 = sharded_tensor.rand(spec, (20, 3, 3))
        st2 = sharded_tensor.rand(spec, (30, 3, 3))

        with self.assertRaisesRegex(RuntimeError, 'Implicit broadcasting'):
            st1 + st2

        with self.assertRaisesRegex(RuntimeError, 'not supported for ShardedTensor'):
            st1 % replica_tensor

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_with_ddp(self):
        # Test Replicated params for DDP
        replica_tensor = ReplicatedTensor(torch.rand(4, 8, device=self.rank))
        model = torch.nn.Linear(8, 2).cuda(self.rank)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        ddp = DDP(model)

        # Test module.parameters.
        params = list(ddp.parameters())
        self.assertEqual(2, len(params))
        self.assertEqual(ddp.module.weight, params[0])
        self.assertEqual(ddp.module.bias, params[1])

        params = list(model.parameters())
        self.assertEqual(2, len(params))
        self.assertEqual(model.weight, params[0])
        self.assertEqual(model.bias, params[1])

        # Validate output
        out = ddp(replica_tensor)
        self.assertIsInstance(out, ReplicatedTensor)

        # Test backward and optimizer.

        # Validate backward.
        out.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(model.bias.grad)
        self.assertIsNotNone(ddp.module.weight.grad)
        self.assertIsNotNone(ddp.module.bias.grad)

        original_params = []
        for param_group in optim.param_groups:
            for original_param in param_group['params']:
                self.assertIsNotNone(original_param.grad)
                original_params.append(original_param)

        self.assertEqual(model.weight.grad, original_params[0].grad)
        self.assertEqual(model.bias.grad, original_params[1].grad)
        self.assertEqual(model.weight.grad, ddp.module.weight.grad)
        self.assertEqual(model.bias.grad, ddp.module.bias.grad)

        # Validate optimizer.
        optim.step()
        self.assertEqual(model.weight, ddp.module.weight)
        self.assertEqual(model.weight, original_params[0])

        self.assertEqual(model.bias, ddp.module.bias)
        self.assertEqual(model.bias, original_params[1])

        # Validate zero_grad
        optim.zero_grad()
        self.assertEqual(model.weight.grad, torch.zeros_like(model.weight.grad))
        self.assertEqual(model.weight.grad, ddp.module.weight.grad)
        self.assertEqual(model.weight.grad, original_params[0].grad)

        self.assertEqual(model.bias.grad, torch.zeros_like(model.bias.grad))
        self.assertEqual(model.bias.grad, ddp.module.bias.grad)
        self.assertEqual(model.bias.grad, original_params[1].grad)

        # Validate zero_grad set_to_none
        optim.zero_grad(set_to_none=True)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(model.weight.grad, ddp.module.weight.grad)
        self.assertEqual(model.weight.grad, original_params[0].grad)

        self.assertIsNone(model.bias.grad)
        self.assertEqual(model.bias.grad, ddp.module.bias.grad)
        self.assertEqual(model.bias.grad, original_params[1].grad)

        # Multiple forward passes.
        for _ in range(5):
            out = ddp(replica_tensor)
            self.assertIsInstance(out, ReplicatedTensor)

        # Test with context manager.
        from torch.nn.parallel._replicated_tensor_ddp_utils import _ddp_replicated_tensor
        with _ddp_replicated_tensor(False):
            for _ in range(5):
                with _ddp_replicated_tensor(True):
                    ddp = DDP(model)
                    out = ddp(replica_tensor)
                self.assertIsInstance(out, ReplicatedTensor)

        # Test save and load.
        with _ddp_replicated_tensor(False):
            ddp = DDP(model)
            expected_state_dict = ddp.state_dict()
            buffer = io.BytesIO()
            torch.save(ddp, buffer)

            buffer.seek(0)
            obj = torch.load(buffer)
            self.assertEqual(expected_state_dict, obj.state_dict())

        with _ddp_replicated_tensor(True):
            ddp = DDP(model)
            buffer = io.BytesIO()
            torch.save(ddp, buffer)

            buffer.seek(0)
            obj = torch.load(buffer)
            self.assertEqual(expected_state_dict, obj.state_dict())

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_unsqueeze(self):
        local_tensor = torch.rand(3, 3, device=self.rank)
        replicated_tensor = ReplicatedTensor(local_tensor)

        unsqueezed_replicated_tensor = replicated_tensor.unsqueeze(0)
        unsqueezed_local_tensor = local_tensor.unsqueeze(0)

        self.assertIsInstance(unsqueezed_replicated_tensor, ReplicatedTensor)
        self.assertIsInstance(torch.unsqueeze(replicated_tensor, 0), ReplicatedTensor)
        self.assertEqual(unsqueezed_local_tensor, unsqueezed_replicated_tensor)
        self.assertEqual(torch.unsqueeze(replicated_tensor, 0), unsqueezed_replicated_tensor)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_getitem(self):
        local_tensor = torch.rand(3, 3, device=self.rank)
        replicated_tensor = ReplicatedTensor(local_tensor)

        replicated_tensor_view = replicated_tensor[0]
        local_tensor_view = local_tensor[0]

        self.assertIsInstance(replicated_tensor_view, ReplicatedTensor)
        self.assertEqual(local_tensor_view, replicated_tensor_view)
