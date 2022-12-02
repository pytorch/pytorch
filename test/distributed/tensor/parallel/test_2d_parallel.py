# Owner(s): ["oncall: distributed"]

from typing import Any

import torch
import torch.distributed as dist

import torch.distributed.distributed_c10d as distributed_c10d
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DeviceMesh, DTensor as DT, Replicate
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.fsdp import is_available
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from torch.testing._internal.common_utils import run_tests

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


def _distribute_and_fsdp_wrap_module(
    module, module_shard, mesh_2d, fsdp_pg, use_orig_params, fsdp_nested
):
    if module_shard:
        module = parallelize_module(module, mesh_2d, PairwiseParallel(), tp_mesh_dim=1)
    pg = fsdp_pg if module_shard else distributed_c10d._get_default_group()

    if fsdp_nested:
        module.net1 = FSDP(
            module.net1, process_group=pg, use_orig_params=use_orig_params
        )
        module.net2 = FSDP(
            module.net2, process_group=pg, use_orig_params=use_orig_params
        )
    return FSDP(module, process_group=pg, use_orig_params=use_orig_params)


def init_model(model_parallel_size=TP_DEGREE, use_orig_params=False, fsdp_nested=False):
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
    model = _distribute_and_fsdp_wrap_module(
        model, True, twod_mesh, fsdp_pg, use_orig_params, fsdp_nested
    )
    return model, fsdp_pg


def is_nested_tensor(val: Any) -> bool:
    if isinstance(val, ShardedTensor):
        if len(val.local_shards()) == 0:
            return False
        if isinstance(val.local_shards()[0].tensor, ShardedTensor):
            return True
        if isinstance(val.local_shards()[0].tensor, DT):
            raise ValueError("Cannot handle DT nested insided ST")
    # Safety valve for when this eventually happen
    elif isinstance(val, DT) and isinstance(val._local_tensor, (DT, ShardedTensor)):
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
        self.assertFalse(is_nested_tensor(optim_state["state"]["net3.bias"]["exp_avg"]))

    def _compare_params(self, m1, m2):
        with FSDP.summon_full_params(m1):
            with FSDP.summon_full_params(m2):
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    self.assertEqual(n_p1[0], n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    def _test_2d_e2e_flow(
        self, use_orig_params=False, fsdp_nested=False, multi_param_group=False
    ) -> None:
        if not is_available():
            self.skipTest("FSDP 2d parallel integration not available")
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        model = FSDP(model, use_orig_params=use_orig_params)
        torch.manual_seed(0)
        model_2d, dp_pg = init_model(
            use_orig_params=use_orig_params, fsdp_nested=fsdp_nested
        )
        # Check named parameters are returning the same name at least.
        param_names_2d = [name for name, _ in model_2d.named_parameters()]
        for name, _ in model.named_parameters():
            self.assertTrue(name in param_names_2d)
        self._compare_params(model, model_2d)

        if multi_param_group and use_orig_params:
            param_group = [
                {"params": model.net1.parameters(), "lr": 0.02},
                {"params": model.net2.parameters(), "lr": 0.15},
            ]
            optim = torch.optim.Adam(param_group, lr=0.01)
            param_group = [
                {"params": model_2d.net1.parameters(), "lr": 0.02},
                {"params": model_2d.net2.parameters(), "lr": 0.15},
            ]
            optim_2d = torch.optim.Adam(param_group, lr=0.01)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=0.01)
            optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

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
        self._compare_params(model, model_2d)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_correctness(self) -> None:
        self._test_2d_e2e_flow()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_use_orig_params(self) -> None:
        self._test_2d_e2e_flow(use_orig_params=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_fsdp_nested(self) -> None:
        self._test_2d_e2e_flow(fsdp_nested=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_fsdp_nested_param_groups(self) -> None:
        self._test_2d_e2e_flow(
            fsdp_nested=True, use_orig_params=True, multi_param_group=True
        )


if __name__ == "__main__":
    run_tests()
