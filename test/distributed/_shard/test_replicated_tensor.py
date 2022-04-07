# Owner(s): ["oncall: distributed"]

import torch

import torch.distributed as dist

from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)

from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.distributed._shard.replicated_tensor import ReplicatedTensor


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
