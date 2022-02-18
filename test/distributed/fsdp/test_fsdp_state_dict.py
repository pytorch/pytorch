# Owner(s): ["oncall: distributed"]

import sys
from typing import Any, Dict

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
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
    def _state_dict(model: Module, state_dict_type: str):
        return getattr(model, state_dict_type)()

    @staticmethod
    def _load_state_dict(
        model: Module, state_dict_type: str, state_dict: Dict[str, Any]
    ):
        getattr(model, f"load_{state_dict_type}")(state_dict)

    def _dist_train(
        self, wrap_fsdp: bool, state_dict_type: str = "", with_context: bool = False
    ):
        # TODO: Move this test to common_fsdp.
        model = self._initialize_model(wrap_fsdp)
        optim = SGD(model.parameters(), lr=0.1)

        in_data = torch.rand(64, 4, requires_grad=True, device=torch.device("cuda"))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()

        if wrap_fsdp:
            blank_model = FSDP(Model(True).cuda())
            if with_context:
                state_dict_type = {
                    "full_state_dict": StateDictType.FULL_STATE_DICT,
                    "local_state_dict": StateDictType.LOCAL_STATE_DICT,
                    "sharded_state_dict": StateDictType.SHARDED_STATE_DICT,
                }[state_dict_type]
                with model.state_dict_type(state_dict_type):
                    state_dict = model.state_dict()
                with blank_model.state_dict_type(state_dict_type):
                    blank_model.load_state_dict(state_dict)
            else:
                state_dict = self._state_dict(model, state_dict_type)
                self._load_state_dict(blank_model, state_dict_type, state_dict)
            get_full_params(blank_model)
            model = blank_model

        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", ["local_state_dict"])
    def test_state_dict_save_load_flow(self, state_dict_type):
        fsdp_params = self._dist_train(wrap_fsdp=True, state_dict_type=state_dict_type)
        fsdp_params_using_context = self._dist_train(
            wrap_fsdp=True, state_dict_type=state_dict_type, with_context=True
        )
        ddp_params = self._dist_train(wrap_fsdp=False)
        self.assertEqual(ddp_params, fsdp_params)
        self.assertEqual(ddp_params, fsdp_params_using_context)

    @skip_if_lt_x_gpu(2)
    @parametrize("state_dict_type", ["local_state_dict"])
    def test_fsdp_state_dict_keys(self, state_dict_type):
        state_dict = self._state_dict(self._initialize_model(True), state_dict_type)
        if state_dict_type == "local_state_dict":
            self.assertEqual(set(["flat_param", "inner.flat_param"]), state_dict.keys())


instantiate_parametrized_tests(TestFSDPStateDict)

if __name__ == "__main__":
    run_tests()
