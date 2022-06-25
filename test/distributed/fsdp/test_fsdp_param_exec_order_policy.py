# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FlatParameter,
    OptimStateKeyType,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import HandleInitMode, ParamExecOrderPolicy
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


STATE_DICT_MAPPING = {
    "state_dict": StateDictType.FULL_STATE_DICT,
    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
}
HANDLE_INIT_MAPPING = {
    "module_level": HandleInitMode.MODULE_LEVEL,
    "param_level": HandleInitMode.PARAM_LEVEL,
}


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def get_inp_shape() -> torch.Size:
        return torch.Size([4, 3, 32, 32])  # (N, C, H, W)


class TestFSDPExecOrderPolicy(FSDPTest):
    @property
    def device(self):
        return torch.device("cuda")

    def _init_fsdp_ddp(self, model_class, optim_class, handle_init_mode, *model_args, **model_kwargs):
        torch.manual_seed(42)
        model = model_class(*model_args, **model_kwargs).cuda()
        group = dist.distributed_c10d._get_default_group()

        fsdp_model = FSDP(
            copy.deepcopy(model), group, auto_wrap_policy=ParamExecOrderPolicy(handle_init_mode),
        )
        fsdp_optim = optim_class(fsdp_model.parameters(), lr=1e-3)
        ddp_model = DDP(model, device_ids=[self.rank], process_group=group)
        ddp_optim = optim_class(ddp_model.parameters(), lr=1e-3)
        return fsdp_model, fsdp_optim, ddp_model, ddp_optim

    def _warmup_fsdp(self, fsdp_model, optim_class, inp_shape, num_warmup_iters: int = 1):
        for _ in range(num_warmup_iters):
            inp = torch.randn(inp_shape).to(self.rank)
            # No optimizer step to keep the model parameters unchanged
            self._step_model(fsdp_model, inp)
        # Reset gradients and construct new optimizer
        return optim_class(fsdp_model.parameters(), lr=1e-3)

    def _check_fsdp_param_parity(self, fsdp_model, ref_model):
        with FSDP.summon_full_params(fsdp_model):
            fsdp_params = list(fsdp_model.parameters())
            ref_params = list(ref_model.parameters())
            if len(fsdp_params) != len(ref_params):
                print(f"[Rank {self.rank}] expected len={len(ref_params)} got {len(fsdp_params)}")
                print(f"[Rank {self.rank}] fsdp names: {[n for n, _ in fsdp_model.named_parameters()]}")
            self.assertEqual(len(fsdp_params), len(ref_params))
            for p1, p2 in zip(fsdp_model.parameters(), ref_model.parameters()):
                torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-4)

    def _step_model(self, model, inp, optim=None):
        if optim is not None:
            optim.zero_grad()
        out = model(inp)
        loss = out.sum()
        loss.backward()
        if optim is not None:
            optim.step()
        return loss

    def _check_fsdp_train_parity(
        self,
        fsdp_model: FSDP,
        fsdp_optim: torch.optim.Optimizer,
        ref_model: nn.Module,
        ref_optim: torch.optim.Optimizer,
        inp_shape: torch.Size,
        num_iters: int,
    ):
        """Checks that FSDP model parameters match those of the reference,
        trains both for a few iterations, and checks that the losses and
        parameters match."""
        self._check_fsdp_param_parity(fsdp_model, ref_model)
        losses: List[List[torch.Tensor]] = []
        for _ in range(num_iters):
            inp = torch.randn(inp_shape).to(self.rank)
            iter_losses = [
                self._step_model(fsdp_model, inp, fsdp_optim),
                self._step_model(ref_model, inp, ref_optim),
            ]
            losses.append(iter_losses)        
        for l1, l2 in losses:
            torch.testing.assert_close(l1, l2)
        self._check_fsdp_param_parity(fsdp_model, ref_model)

    def _strip_module_prefix(self, optim_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips the prefix "module." from the parameter name keys in
        ``optim_state_dict``. The prefix arises from wrapping a model with DDP.
        Args:
            optim_state_dict (Dict[str, Any]): Optimizer state dict from an
                optimizer used for a DDP model that has been rekeyed to
                parameter names.
        """
        new_osd = {"state": {}, "param_groups": []}
        prefix = "module."
        for param_name, param_state in optim_state_dict["state"].items():
            assert param_name.startswith(prefix)
            new_osd["state"][param_name.lstrip(prefix)] = param_state
        for param_group in optim_state_dict["param_groups"]:
            new_params = []
            for param_name in param_group["params"]:
                assert param_name.startswith(prefix)
                new_params.append(param_name.lstrip(prefix))
            new_param_group = copy.copy(param_group)
            new_param_group["params"] = new_params
            new_osd["param_groups"].append(new_param_group)
        return new_osd

    @skip_if_lt_x_gpu(2)
    def test_pre_warmup(self):
        """Tests that on the FSDP model is constructed with one original
        parameter per ``FlatParameter``."""
        model_class = CNN
        optim_class = torch.optim.Adam
        fsdp_model, _, ddp_model, _ = self._init_fsdp_ddp(
            model_class, optim_class, HandleInitMode.PARAM_LEVEL,
        )
        num_ddp_params = len(list(ddp_model.parameters()))
        num_fsdp_params = 0
        for param in fsdp_model.parameters():
            self.assertTrue(isinstance(param, FlatParameter))
            self.assertEqual(param._num_params, 1)
            num_fsdp_params += 1
        self.assertEqual(num_ddp_params, num_fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("handle_init_mode", ["module_level", "param_level"])
    def test_train_flow(self, handle_init_mode: HandleInitMode):
        """Tests training parity with DDP."""
        model_class = CNN
        optim_class = torch.optim.Adam
        num_iters = 5
        fsdp_model, fsdp_optim, ddp_model, ddp_optim = self._init_fsdp_ddp(
            model_class, optim_class, HANDLE_INIT_MAPPING[handle_init_mode],
        )
        inp_shape = model_class.get_inp_shape()
        fsdp_optim = self._warmup_fsdp(fsdp_model, optim_class, inp_shape)
        
        self._check_fsdp_train_parity(
            fsdp_model, fsdp_optim, ddp_model, ddp_optim, inp_shape, num_iters,
        )

    # TODO (awgu): `_sharded_pre_load_state_dict_hook()` needs some work to
    # support non-recursive wrapping
    @skip_if_lt_x_gpu(2)
    @parametrize("model_state_dict_type", ["state_dict"])
    @parametrize("handle_init_mode", ["module_level", "param_level"])
    def test_train_with_full_state_dict(
        self,
        model_state_dict_type: str,
        handle_init_mode: HandleInitMode,
    ):
        """Tests training parity with DDP when loading FSDP from a full state
        dict (for both model and optimizer)."""
        num_iters = 5
        model_class = CNN
        optim_class = torch.optim.Adam
        fsdp_model, fsdp_optim, ddp_model, ddp_optim = self._init_fsdp_ddp(
            model_class,optim_class, HANDLE_INIT_MAPPING[handle_init_mode],
        )
        inp_shape = model_class.get_inp_shape()
        fsdp_optim = self._warmup_fsdp(fsdp_model, optim_class, inp_shape)
        # Run the DDP model for a few iterations
        for _ in range(num_iters):
            inp = torch.randn(inp_shape).to(self.rank)
            self._step_model(ddp_model, inp, ddp_optim)
        # Load model state dict from DDP to FSDP
        ddp_state_dict = ddp_model.module.state_dict()
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[model_state_dict_type]):
            fsdp_model.load_state_dict(ddp_state_dict)
        # Load optim state dict from DDP to FSDP
        ddp_optim_state_dict = self._strip_module_prefix(
            FSDP.rekey_optim_state_dict(
                ddp_optim.state_dict(), OptimStateKeyType.PARAM_NAME, ddp_model,
            )
        )
        fsdp_optim.load_state_dict(
            FSDP.shard_full_optim_state_dict(ddp_optim_state_dict, fsdp_model),
        )
        # Run both models and check parity
        self._check_fsdp_train_parity(
            fsdp_model, fsdp_optim, ddp_model, ddp_optim, inp_shape, num_iters,
        )

instantiate_parametrized_tests(TestFSDPExecOrderPolicy)

if __name__ == "__main__":
    run_tests()
