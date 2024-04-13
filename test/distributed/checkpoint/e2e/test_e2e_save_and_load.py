# Owner(s): ["oncall: distributed"]

import time
from dataclasses import dataclass, field
from enum import auto, Enum
from functools import partial
from io import BytesIO
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.distributed.checkpoint.state_dict_saver as saver
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    get_model_state_dict,
    get_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn.parallel import DistributedDataParallel

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


# Simple and boring model
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


class TestStatefulObj:
    def __init__(self):
        self.data = torch.rand(10, 10, device="cuda")

    def state_dict(self):
        return {"data": self.data}

    def load_state_dict(self, state_dict):
        self.data = state_dict["data"]

    def __eq__(self, other):
        return torch.equal(self.data, other.data)


class ModelType(Enum):
    FSDP = auto()
    HSDP = auto()
    FSDP_TP = auto()
    DDP = auto()
    NONE = auto()  # no parallelization


@dataclass
class TestTrainState:
    step: int = 0
    current_loss: float = -1
    losses: List[float] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        loss_bytes = BytesIO()
        torch.save(self.losses, loss_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "current_loss": torch.tensor(self.current_loss, dtype=torch.float32),
            "losses": loss_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.current_loss = state_dict["current_loss"].item()
        state_dict["losses"].seek(0)
        self.losses = torch.load(state_dict["losses"])

    def __eq__(self, other):
        return (
            self.step == other.step
            and self.current_loss == other.current_loss
            and self.losses == other.losses
        )


def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None

    train_state = TestTrainState()

    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()

        # We usually sync the loss across dp ranks in real training.
        # This is just simulating for testing purpose.
        train_state.step += 1
        train_state.current_loss = torch.rand(1).item()
        train_state.losses.append(train_state.current_loss)

        optim.step()
        optim.zero_grad()

    return loss, train_state


class TestE2ESaveAndLoad(DTensorTestBase, VerifyStateDictMixin):
    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"

    def _create_model(self, compile, model_type, state_dict_options=None):
        dummy_model = TestDummyModel().cuda()

        assert model_type in ModelType, f"{model_type} is not supported."
        if model_type == ModelType.FSDP:
            device_mesh = init_device_mesh(self.device_type, (self.world_size,))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
            )
        elif model_type == ModelType.HSDP:
            device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
            model = FSDP(
                dummy_model,
                device_mesh=device_mesh,
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        elif model_type == ModelType.FSDP_TP:
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            tp_mesh = mesh_2d["tp"]
            dp_mesh = mesh_2d["dp"]
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            model = parallelize_module(dummy_model, tp_mesh, parallelize_plan)
            model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)
        elif model_type == ModelType.DDP:
            model = DistributedDataParallel(dummy_model)
            model.get_input = partial(TestDummyModel.get_input, model)
        else:
            model = dummy_model

        if compile:
            # TODO: enable dynamic=True when dynamic shape support is enabled.
            # model = torch.compile(model)
            model = torch.compile(model, dynamic=False)

        optim = self._optim(model)
        if model_type is not ModelType.NONE:
            _patch_model_state_dict(model, options=state_dict_options)
            _patch_optimizer_state_dict(
                model, optimizers=optim, options=state_dict_options
            )

        return model, optim

    def _optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    @parametrize("compile", [True, False])
    # TODO: Previously PairwiseParallel does not shard properly, passing ModelType.FSDP_TP test where it
    # should have failed. Disabling the failed test temporarily to unblock the deprecation of PairwiseParallel.
    @parametrize("model_type", [ModelType.FSDP, ModelType.HSDP, ModelType.DDP])
    def test_e2e(self, compile, model_type):
        self._run_e2e_test(compile, model_type)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_e2e_async(self):
        self._run_e2e_test(compile=False, model_type=ModelType.FSDP, async_op=True)

    def _run_e2e_test(self, compile, model_type, async_op=False):
        model, optim = self._create_model(compile, ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(compile, model_type)
        _, original_train_state = _train(dist_model, dist_optim, train_steps=2)

        original_stateful_obj = TestStatefulObj()  # tests arbitrary saving/loading
        sd = {
            "model": dist_model,
            "optimizer": dist_optim,
            "s": original_stateful_obj,
            "train_state": original_train_state,
        }

        if async_op:
            f = saver.async_save(sd, checkpoint_id=self.temp_dir)
            t = time.monotonic()
            while not f.done():
                time.sleep(1)
                print(f"still waiting... {time.monotonic() - t}")

            f.result()
        else:
            DCP.save(sd, checkpoint_id=self.temp_dir)

        loaded_stateful_obj = TestStatefulObj()
        loaded_train_state = TestTrainState()
        dist_model, dist_optim = self._create_model(compile, model_type)

        DCP.load(
            state_dict={
                "model": dist_model,
                "optimizer": dist_optim,
                "s": loaded_stateful_obj,
                "train_state": loaded_train_state,
            },
            checkpoint_id=self.temp_dir,
        )

        self.assertEqual(original_stateful_obj, loaded_stateful_obj)
        self.assertEqual(original_train_state, loaded_train_state)

        # train one more step on both models
        loss, _ = _train(model, optim, train_steps=1)
        dist_loss, _ = _train(dist_model, dist_optim, train_steps=1)
        self.assertEqual(loss, dist_loss)

        dist_msd, dist_osd = get_state_dict(dist_model, optimizers=dist_optim)
        model_sd, optim_sd = get_state_dict(model, optimizers=optim)

        self._verify_msd(model_sd, dist_msd)
        self._verify_osd_by_load(model, optim, self._optim(model), dist_osd)

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_different_ordered_state_dict_keys(self):
        """Tests that the order of keys in the state dict does not matter when loading
        If order was not accounted for, the following test would cause a deadlock.
        """

        world_size = self.world_size

        class Foo:
            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                tl = [
                    torch.ones(2, dtype=torch.int64, device="cuda")
                    for _ in range(world_size)
                ]
                t = (
                    torch.arange(2, dtype=torch.int64, device="cuda")
                    + 1
                    + 2 * dist.get_rank()
                )
                dist.all_gather(tl, t, async_op=False)

        class Bar:
            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                tensor = (
                    torch.arange(2, dtype=torch.int64, device="cuda")
                    + 1
                    + 2 * dist.get_rank()
                )
                dist.all_reduce(tensor, op=ReduceOp.SUM)

        if self.rank == 0:
            sd = {
                "A": Foo(),
                "B": Bar(),
            }
        else:
            sd = {
                "B": Bar(),
                "A": Foo(),
            }

        DCP.save(sd, checkpoint_id=self.temp_dir)
        DCP.load(sd, checkpoint_id=self.temp_dir)

    @with_temp_dir
    def test_no_dist(self):
        # since comm's are not initialized in this method, `no_dist`
        # is assumed False
        DCP.save({}, checkpoint_id=self.temp_dir)
        DCP.load({}, checkpoint_id=self.temp_dir)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_partial_load(self):
        model, optim = self._create_model(compile=False, model_type=ModelType.NONE)
        _train(model, optim, train_steps=2)

        dist_model, dist_optim = self._create_model(
            compile=False, model_type=ModelType.FSDP
        )
        _train(dist_model, dist_optim, train_steps=2)

        DCP.save(
            {"model": dist_model, "optimizer": dist_optim}, checkpoint_id=self.temp_dir
        )

        dist_model, _ = self._create_model(compile=False, model_type=ModelType.FSDP)
        DCP.load({"model": dist_model}, checkpoint_id=self.temp_dir)

        dist_msd = get_model_state_dict(dist_model)
        model_sd = get_model_state_dict(model)
        self._verify_msd(model_sd, dist_msd)

        # another way
        loaded_model_sd = _load_state_dict_from_keys(
            "model", checkpoint_id=self.temp_dir
        )["model"]
        self._verify_msd(model_sd, loaded_model_sd, offload_to_cpu=True)

        loaded_optim_state = _load_state_dict_from_keys(
            "optimizer.state", checkpoint_id=self.temp_dir
        )["optimizer"]["state"]
        self.assertNotIn("param_groups", loaded_optim_state)
        for k, v in dist_optim.state_dict()["state"].items():
            for optim_key in ["exp_avg", "exp_avg_sq", "step"]:
                self._compare_tensor(
                    loaded_optim_state[k][optim_key], v[optim_key], offload_to_cpu=True
                )


class TestNoCPU(DTensorTestBase):
    @property
    def backend(self):
        return "nccl"

    @with_comms
    def test_no_cpu(self):
        with self.assertRaisesRegex(
            AssertionError, r"A CPU backend must be enabled for async save;.*?"
        ):
            f = saver.async_save({})
            f.result()


instantiate_parametrized_tests(TestE2ESaveAndLoad)
if __name__ == "__main__":
    run_tests()
