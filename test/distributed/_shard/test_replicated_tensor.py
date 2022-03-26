# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed._shard.replicated_tensor import ReplicatedTensor, ReplicatedParameter
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)

from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


class TestReplicatedTensor(ShardedTensorTestBase):

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_replicated_tensor_basics(self):
        local_tensor = torch.ones(3, 3, device=f"cuda:{self.rank}") * 4
        replica_tensor = ReplicatedTensor(local_tensor)
        print(replica_tensor.process_group)
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
    @skip_if_lt_x_gpu(4)
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
    @skip_if_lt_x_gpu(4)
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
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_replicated_tensor_inter_op_sharded_tensor(self):
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

        st = sharded_tensor.rand(spec, (12, 3))

        # add op
        new_tensor = replica_tensor + st
        # replica + sharded = sharded
        self.assertTrue(
            isinstance(new_tensor, sharded_tensor.ShardedTensor)
            and not isinstance(new_tensor, ReplicatedTensor)
        )
        self.assertEqual(new_tensor.local_tensor(), st.local_tensor() + 4)
        # test torch.add
        new_tensor = torch.add(replica_tensor, st)
        self.assertEqual(new_tensor.local_tensor(), st.local_tensor() + 4)

        new_tensor = st + replica_tensor
        self.assertEqual(new_tensor.local_tensor(), st.local_tensor() + 4)

        # sub op
        new_tensor = replica_tensor - st
        self.assertTrue(
            isinstance(new_tensor, sharded_tensor.ShardedTensor)
            and not isinstance(new_tensor, ReplicatedTensor)
        )
        self.assertEqual(new_tensor.local_tensor(), 4 - st.local_tensor())
        # test torch.sub
        new_tensor = torch.sub(replica_tensor, st)
        self.assertEqual(new_tensor.local_tensor(), 4 - st.local_tensor())

        new_tensor = st - replica_tensor
        self.assertEqual(new_tensor.local_tensor(), st.local_tensor() - 4)

        # mul op
        new_tensor = replica_tensor * st
        self.assertTrue(
            isinstance(new_tensor, sharded_tensor.ShardedTensor)
            and not isinstance(new_tensor, ReplicatedTensor)
        )
        self.assertEqual(new_tensor.local_tensor(), 4 * st.local_tensor())
        # test torch.mul
        new_tensor = torch.mul(replica_tensor, st)
        self.assertEqual(new_tensor.local_tensor(), 4 * st.local_tensor())

        new_tensor = st * replica_tensor
        self.assertEqual(new_tensor.local_tensor(), 4 * st.local_tensor())

        # div op
        new_tensor = replica_tensor / st
        self.assertTrue(
            isinstance(new_tensor, sharded_tensor.ShardedTensor)
            and not isinstance(new_tensor, ReplicatedTensor)
        )
        self.assertEqual(new_tensor.local_tensor(), 4 / st.local_tensor())
        # test torch.div
        new_tensor = torch.div(replica_tensor, st)
        self.assertEqual(new_tensor.local_tensor(), 4 / st.local_tensor())

        new_tensor = st / replica_tensor
        self.assertEqual(new_tensor.local_tensor(), st.local_tensor() / 4)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
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

        st = sharded_tensor.rand(spec, (20, 3))

        with self.assertRaisesRegex(RuntimeError, 'Implicit broadcasting'):
            st + replica_tensor

        with self.assertRaisesRegex(TypeError, 'unsupported operand type'):
            st + 4

        with self.assertRaisesRegex(RuntimeError, 'not supported for ShardedTensor'):
            st % replica_tensor

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_with_ddp(self):
        # Test Replicated params for DDP
        replica_tensor = ReplicatedTensor(torch.rand(4, 8, device=self.rank))
        module = torch.nn.Linear(8, 2).cuda(self.rank)
        ddp = DDP(module)
        self.assertIsInstance(ddp.module.weight, ReplicatedParameter)
        self.assertIsInstance(ddp.module.bias, ReplicatedParameter)
        self.assertIsInstance(module.weight, ReplicatedParameter)
        self.assertIsInstance(module.bias, ReplicatedParameter)

        # Test module.parameters.
        params = list(ddp.parameters())
        self.assertEqual(2, len(params))
        self.assertEqual(ddp.module.weight, params[0])
        self.assertEqual(ddp.module.bias, params[1])

        params = list(module.parameters())
        self.assertEqual(2, len(params))
        self.assertEqual(module.weight, params[0])
        self.assertEqual(module.bias, params[1])

        # Validate output
        out = ddp(replica_tensor)
        self.assertIsInstance(out, ReplicatedTensor)

        # Validate after forward pass
        self.assertIsInstance(ddp.module.weight, ReplicatedParameter)
        self.assertIsInstance(ddp.module.bias, ReplicatedParameter)
        self.assertIsInstance(module.weight, ReplicatedParameter)
        self.assertIsInstance(module.bias, ReplicatedParameter)

        # Test buffers.
        module.register_buffer('foo', torch.rand(10).cuda(self.rank))
        ddp = DDP(module)
        self.assertIsInstance(ddp.module.foo, ReplicatedTensor)
        self.assertIsInstance(module.foo, ReplicatedTensor)

        # Validate during forward pass
        def hook(module, input):
            self.assertIsInstance(ddp.module.foo, ReplicatedTensor)
            self.assertIsInstance(module.foo, ReplicatedTensor)

        module.register_forward_pre_hook(hook)
        for _ in range(2):
            out = ddp(replica_tensor)
            self.assertIsInstance(out, ReplicatedTensor)
