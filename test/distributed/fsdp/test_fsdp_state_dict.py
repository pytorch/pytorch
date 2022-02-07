# Owner(s): ["oncall: distributed"]

import sys
from typing import Any, Dict

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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

INNER_SHAPE = [4, 4]
OUTER_SHAPE = [4, 5]


class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()
        self.inner = Linear(*INNER_SHAPE)
        if wrap_fsdp:
            self.inner = FSDP(self.inner)
        self.outer = Linear(*OUTER_SHAPE)

    def forward(self, x):
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestFSDPStateDict(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _initialize_model(self, wrap_fsdp: bool):
        # keep everything deterministic for input data
        torch.manual_seed(0)

        model = Model(wrap_fsdp).cuda()
        if wrap_fsdp:
            model = FSDP(model)
        else:
            model = DistributedDataParallel(model, device_ids=[self.rank])
        return model

    @staticmethod
    def _state_dict(model: Module, state_dict_name: str):
        return getattr(model, state_dict_name)()

    @staticmethod
    def _load_state_dict(
        model: Module, state_dict_name: str, state_dict: Dict[str, Any]
    ):
        getattr(model, f"load_{state_dict_name}")(state_dict)

    def _dist_train(self, wrap_fsdp: bool, state_dict_name: str = ""):
        model = self._initialize_model(wrap_fsdp)
        optim = SGD(model.parameters(), lr=0.1)

        in_data = torch.rand(64, 4).cuda()
        in_data.requires_grad = True
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        if wrap_fsdp:
            state_dict = self._state_dict(model, state_dict_name)
            blank_model = FSDP(Model(True).cuda())
            self._load_state_dict(blank_model, state_dict_name, state_dict)
            get_full_params(blank_model)
            model = blank_model

        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_name", ["local_state_dict"])
    def test_state_dict_save_load_flow(self, state_dict_name):
        fsdp_states = self._dist_train(wrap_fsdp=True, state_dict_name=state_dict_name)
        ddp_states = self._dist_train(wrap_fsdp=False)
        self.assertEqual(ddp_states, fsdp_states)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_name", ["local_state_dict"])
    def test_fsdp_state_dict_keys(self, state_dict_name):
        state_dict = self._state_dict(self._initialize_model(True), state_dict_name)
        if state_dict_name != "local_state_dict":
            ddp_state_dict = self._initialize_model(False).state_dict(state_dict)
            self.assertEqual(
                set(dpp_state_dict.keys()),
                set(state_dict.keys()),
                "state_dict/sharded_state_dict should have the same keys as ddp.state_dict()",
            )
        else:
            self.assertEqual(set(["flat_param", "inner.flat_param"]), state_dict.keys())


instantiate_parametrized_tests(TestFSDPStateDict)

if __name__ == "__main__":
    run_tests()
