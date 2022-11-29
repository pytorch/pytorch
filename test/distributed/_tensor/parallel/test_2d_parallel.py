# Owner(s): ["oncall: distributed"]

from typing import Any


import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor as DT,
    Replicate,
    Shard,
)
from torch.distributed._tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)

import torch.distributed.distributed_c10d as distributed_c10d

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.distributed._tensor.parallel.fsdp import is_available

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Tensor-Parallel degree
TP_DEGREE = 2
LR = 3e-5


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


def _shard_wrap_module(
    module, module_shard, fsdp_wrap, mesh_2d, fsdp_pg, use_orig_params
):  
    device_mesh_1d = None
    if module_shard:
        module = parallelize_module(module, mesh_2d, PairwiseParallel(), tp_mesh_dim=1)
        device_mesh_1d = module.net1.weight.device_mesh

    if fsdp_wrap and module_shard:
        return FSDP(
            module, process_group=fsdp_pg, use_orig_params=use_orig_params
        ), device_mesh_1d
    if fsdp_wrap:
        return FSDP(
            module,
            process_group=distributed_c10d._get_default_group(),
            use_orig_params=use_orig_params,
        ), device_mesh_1d
    return module, device_mesh_1d


def init_model(model_parallel_size=TP_DEGREE, use_orig_params=False):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()

    model = SimpleModel().cuda(rank)

    # 2-D mesh is [dp, tp]
    twod_mesh = DeviceMesh(
        device_type="cuda",
        mesh=torch.arange(0, world_size).view(model_parallel_size, -1),
    )

    fsdp_pg = twod_mesh.get_dim_groups()[0]

    # Create Input
    model, device_mesh_1d = _shard_wrap_module(
        model, True, True, twod_mesh, fsdp_pg, use_orig_params
    )
    return model, fsdp_pg, device_mesh_1d


def is_nested_tensor(val: Any) -> bool:
    if isinstance(val, ShardedTensor):
        if len(val.local_shards()) == 0:
            return False
        if isinstance(val.local_shards()[0].tensor, ShardedTensor):
            return True
        if isinstance(val.local_shards()[0].tensor, DT):
            raise ValueError("Cannot handle DT nested insided ST")
    # Safety valve for when this eventually happen
    elif isinstance(val, DT) and isinstance(
        val._local_tensor, (DT, ShardedTensor)
    ):
        raise ValueError("Cannot handle nested DT")
    return False


class Test2dParallelIntegration(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_functionality(self) -> None:
        if not is_available():
            self.skipTest("FSDP 2d parallel integration not available")

        model_tp = init_model()[0]

        with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
            state_dict = model_tp.state_dict()
            # TODO once 2D is out, validate the nesting
            self.assertTrue(is_nested_tensor(state_dict["net1.weight"]))
            self.assertFalse(is_nested_tensor(state_dict["net3.bias"]))

        optim = torch.optim.Adam(model_tp.parameters(), lr=0.0001)

        # Create Input
        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        input = torch.rand(4, 5).cuda(self.rank)

        model_tp(input).sum().backward()
        optim.step()

        optim_state = FSDP.sharded_optim_state_dict(model_tp, optim)
        # TODO once 2D is out, validate the nesting
        self.assertTrue(
            is_nested_tensor(optim_state["state"]["net1.weight"]["exp_avg"])
        )
        self.assertFalse(
            is_nested_tensor(optim_state["state"]["net3.bias"]["exp_avg"])
        )

    def _compare_params(self, m1, m2, device_mesh_1d, use_orig_params):
        with FSDP.summon_full_params(m1):
            with FSDP.summon_full_params(m2):
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    self.assertEqual(n_p1[0], n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if use_orig_params:
                        p1 = p1.data
                        p2 = p2.data
                    if "net1" in name:
                        spec = [Shard(0)]
                    elif name == "net2.weight":
                        spec = [Shard(1)]
                    else:
                        spec = [Replicate()]
                    if type(p2) is DT:
                        p2 = p2.redistribute(
                            device_mesh_1d, [Replicate()]
                        ).to_local()
                    # elif name != "net2.bias":
                    #     p2 = DT.from_local(p2, device_mesh_1d, spec).redistribute(
                    #        device_mesh_1d, [Replicate()] 
                    #     ).to_local()
                    # print(name, p1.size(), p2.size(), spec)
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    def _test_2d_e2e_flow(self, use_orig_params=False) -> None:
        if not is_available():
            self.skipTest("FSDP 2d parallel integration not available")
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        model = FSDP(model, use_orig_params=use_orig_params)
        torch.manual_seed(0)
        model_2d, dp_pg, device_mesh_1d = init_model(use_orig_params=use_orig_params)
        # Check named parameters are returning the same name at least.
        param_names_2d = [name for name, _ in model_2d.named_parameters()]
        for name, p in model.named_parameters():
            self.assertTrue(name in param_names_2d)
        self._compare_params(model, model_2d, device_mesh_1d, use_orig_params)

        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.0001)
        return

        for i in range(5):
            # Ensure all input across TP ranks are same.
            torch.manual_seed(i + dist.get_rank(dp_pg))
            input = torch.rand(4, 5).cuda(self.rank)
            output = model(input)
            output_2d = model_2d(input)
            self.assertEqual(output, output_2d)
            output.sum().backward()
            output_2d.sum().backward()
            optim.step()
            optim_2d.step()
            self.assertEqual(model(input), model_2d(input))
        
        # Ensure all params are still the same after optimizer update.
        self._compare_params(model, model_2d, device_mesh_1d, use_orig_params)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_correctness(self) -> None:
        self._test_2d_e2e_flow()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_use_orig_params(self) -> None:
        self._test_2d_e2e_flow(use_orig_params=True)


if __name__ == "__main__":
    run_tests()
