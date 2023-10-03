# Owner(s): ["oncall: distributed"]

import functools
from typing import Any

import torch
import torch.distributed as dist

import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor as DT, init_device_mesh, Replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    clean_tensor_name,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.distributed.tensor.parallel.input_reshard import input_reshard
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
        super().__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


def _wrap_module(
    module,
    mesh_2d,
    use_orig_params,
    fsdp_nested,
    recompute_activation,
):
    fsdp_pg = mesh_2d.get_dim_groups()[0]
    module = parallelize_module(module, mesh_2d, PairwiseParallel(), tp_mesh_dim=1)

    fsdp_ctor = functools.partial(
        FSDP,
        process_group=fsdp_pg,
        use_orig_params=use_orig_params,
        device_id=torch.cuda.current_device(),
    )
    if fsdp_nested:
        module.net1 = fsdp_ctor(module.net1)
        module.net2 = fsdp_ctor(module.net2)

    if recompute_activation:
        module = checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    return fsdp_ctor(module)


def init_model(
    model_parallel_size=TP_DEGREE,
    use_orig_params=False,
    fsdp_nested=False,
    recompute_activation=False,
):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()

    model = SimpleModel()

    # 2-D mesh is [dp, tp]
    mesh_shape = (world_size // model_parallel_size, model_parallel_size)
    mesh_dim_names = ("DP", "TP")
    mesh_2d = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)

    # Create Input
    model = _wrap_module(
        model,
        mesh_2d,
        use_orig_params,
        fsdp_nested,
        recompute_activation,
    )
    return model, mesh_2d


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


def _apply_optim_in_backward(param_group):
    _apply_optimizer_in_backward(
        optimizer_class=torch.optim.Adam,
        params=param_group["params"],
        optimizer_kwargs={"lr": param_group["lr"]},
    )


class Test2dParallelIntegration(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_functionality(self) -> None:
        if not enable_2d_with_fsdp():
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
                    if n_p1[0] != n_p2[0]:
                        self.assertTrue(n_p1[0] in n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    def _test_2d_e2e_flow(
        self,
        use_orig_params=False,
        fsdp_nested=False,
        multi_param_group=False,
        recompute_activation=False,
        optim_in_backward=False,
    ) -> None:
        if not enable_2d_with_fsdp():
            self.skipTest("FSDP 2d parallel integration not available")
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        model = FSDP(model, use_orig_params=use_orig_params)
        torch.manual_seed(0)
        model_2d, mesh_2d = init_model(
            use_orig_params=use_orig_params,
            fsdp_nested=fsdp_nested,
            recompute_activation=recompute_activation,
        )
        dp_pg = mesh_2d.get_dim_groups()[0]
        if recompute_activation:
            model_2d = input_reshard(model_2d, mesh_2d["TP"], 0)
        # Check named parameters are returning the same name at least.
        param_names_2d = [
            clean_tensor_name(name) for name, _ in model_2d.named_parameters()
        ]
        for name, _ in model.named_parameters():
            name = clean_tensor_name(name)
            self.assertTrue(name in param_names_2d)
        self._compare_params(model, model_2d)
        if multi_param_group and use_orig_params:
            param_group = [
                {"params": model.net1.parameters(), "lr": 0.15},
                # TODO: Disable it for now and we need to further investigate
                # the reason why test_2d_fsdp_integration_fsdp_nested_param_groups failed.
                # {"params": model.net2.parameters(), "lr": 0.05},
            ]
            if optim_in_backward:
                for grp_idx in len(param_group):
                    _apply_optim_in_backward(param_group=param_group[grp_idx])
            else:
                optim = torch.optim.Adam(param_group, lr=0.01)
            param_group = [
                {"params": model_2d.net1.parameters(), "lr": 0.15},
                # {"params": model_2d.net2.parameters(), "lr": 0.05},
            ]
            if optim_in_backward:
                for grp_idx in len(param_group):
                    _apply_optim_in_backward(param_group=param_group[grp_idx])
            else:
                optim_2d = torch.optim.Adam(param_group, lr=0.01)
        else:
            if optim_in_backward:
                _apply_optimizer_in_backward(
                    optimizer_class=torch.optim.Adam,
                    params=model.parameters(),
                    optimizer_kwargs={"lr": 0.01},
                )
                _apply_optimizer_in_backward(
                    optimizer_class=torch.optim.Adam,
                    params=model_2d.parameters(),
                    optimizer_kwargs={"lr": 0.01},
                )
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
            self._compare_params(model, model_2d)
            if not optim_in_backward:
                optim.step()
                optim_2d.step()
            self._compare_params(model, model_2d)
            self.assertEqual(
                model(input), model_2d(input), f"results different at iter {i}"
            )

        # Ensure all params are still the same after optimizer update.
        self._compare_params(model, model_2d)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_correctness(self) -> None:
        self._test_2d_e2e_flow()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_correctness_w_recompute_activation(self) -> None:
        self._test_2d_e2e_flow(recompute_activation=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_use_orig_params(self) -> None:
        self._test_2d_e2e_flow(use_orig_params=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration_optim_in_backward(self) -> None:
        self._test_2d_e2e_flow(
            use_orig_params=True, fsdp_nested=True, optim_in_backward=True
        )

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


class TestNew2dParallelIntegration(DTensorTestBase):
    # TODO: this is duplicate code from above, but once we remove the enable_2d_with_fsdp(),
    # we will remove the above test class Test2dParallelIntegration.
    def _compare_params(self, m1, m2):
        with FSDP.summon_full_params(m1):
            with FSDP.summon_full_params(m2):
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    if n_p1[0] != n_p2[0]:
                        self.assertTrue(n_p1[0] in n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_state_enable_extension(self):
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        model = FSDP(
            SimpleModel().cuda(),
            device_mesh=mesh_2d["dp"],
        )
        fsdp_state = _get_module_fsdp_state(model)
        self.assertEqual(fsdp_state._enable_extension, True)

    def _test_2d_e2e_training(
        self,
        use_orig_params=False,
        recompute_activation=False,
    ) -> None:
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        model = FSDP(model, use_orig_params=use_orig_params)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        model_2d = parallelize_module(SimpleModel().cuda(), tp_mesh, PairwiseParallel())
        model_2d = FSDP(
            model_2d,
            device_mesh=dp_mesh,
            use_orig_params=use_orig_params,
        )
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        if recompute_activation:
            model_2d = input_reshard(model_2d, mesh_2d["tp"], 0)

        # Check named parameters are returning the same name at least.
        param_names_2d = [
            clean_tensor_name(name) for name, _ in model_2d.named_parameters()
        ]
        for name, _ in model.named_parameters():
            name = clean_tensor_name(name)
            if name not in param_names_2d:
                print(name, param_names_2d)
            self.assertTrue(name in param_names_2d)
        self._compare_params(model, model_2d)

        # TODO: add additional tests for multi_param_group and optim_in_backward.

        for i in range(5):
            # Ensure all input across TP ranks are same.
            # TODO: add a get_group_rank() to DeviceMesh.
            torch.manual_seed(i + dist.get_rank(dp_mesh.get_dim_groups()[0]))
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
    def test_2d_e2e_training_default(self):
        self._test_2d_e2e_training()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_use_orig_params(self):
        self._test_2d_e2e_training(use_orig_params=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_not_use_orig_params(self):
        self._test_2d_e2e_training(recompute_activation=True)


if __name__ == "__main__":
    run_tests()
