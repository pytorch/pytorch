# Owner(s): ["oncall: distributed"]

import gc
import weakref
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, loss_parallel
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase
)
from torch.testing._internal.distributed.fake_pg import FakeStore



def _plain(mesh):
    a = distribute_tensor(torch.randn(8, 8), mesh, [Shard(0)])
    (a * 2).sum()


def _redistribute(mesh):
    a = distribute_tensor(torch.randn(8, 8), mesh, [Shard(0)])
    b = distribute_tensor(torch.randn(8, 8), mesh, [Replicate()])
    (a + b).sum()


def _conv(mesh):
    # routed to _tp_conv.convolution_handler which reaches the Python
    # sharding propagation cache rather than the C++ fast path.
    x = distribute_tensor(torch.randn(2, 3, 8, 8), mesh, [Replicate()])
    w = distribute_tensor(torch.randn(4, 3, 3, 3), mesh, [Replicate()])
    F.conv2d(x, w)


def _loss_parallel(mesh):
    # the only path reaching _propagate_tensor_meta_cached
    with loss_parallel():
        logits = distribute_tensor(torch.randn(4, 8), mesh, [Shard(1)])
        F.cross_entropy(logits, torch.randint(0, 8, (4,)))


WORKLOADS = {"plain": _plain, "redistribute": _redistribute, "conv": _conv, "loss_parallel": _loss_parallel}


class TestShardingPropCache(TestCase):
    """No DTensor cache may keep a DeviceMesh alive.
    
    The sharding propagation caches key DTensorSpec by value but sotred the
    mesh object, so every Devicemesh ever used stayed reachable for the life
    of the process.
    """

    def setUp(self):
        super().setUp()
        dist.init_process_group(
            backend="fake", rank=0, world_size=1, store=FakeStore()
        )
        self.mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def test_eager_identity(self):
        for _ in range(3):
            tp = self.mesh["tp"]
            lin = nn.Linear(16, 16, bias=False)
            parallelize_module(lin, tp, ColwiseParallel())
            self.assertIs(lin.weight.device_mesh, tp)

    @parametrize("workload", list(WORKLOADS))
    def test_submesh_not_retained(self, workload):
        def use_and_drop():
            # Build the submesh in this frame and return only a weak
            # reference so no strong reference can outlive the call.
            submesh = self.mesh["tp"]
            WORKLOADS[workload](submesh)
            return weakref.ref(submesh)

        ref = use_and_drop()
        gc.collect()
        self.assertIsNone(ref(), f"{workload} workload retained the submesh")


instantiate_parametrized_tests(TestShardingPropCache)


if __name__ == "__main__":
    run_tests()
