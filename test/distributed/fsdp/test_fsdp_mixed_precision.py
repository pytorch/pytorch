# Owner(s): ["oncall: distributed"]

import sys
from copy import deepcopy
from functools import partial
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn import Linear, Module
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
    _get_full_detached_param,
    _zero_model,
    _get_state_dict,
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


class TestFSDPMixedPrecision(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _get_simple_nested_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(
            nn.Sequential(
                FSDP(nn.Linear(10, 10, bias=False), *fsdp_args, **fsdp_kwargs),
                nn.Linear(10, 10, bias=False),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(nn.Linear(10, 10, bias=False), *fsdp_args, **fsdp_kwargs)
        return model

    # @skip_if_lt_x_gpu(2)
    # @parametrize(
    #     "cpu_offload",
    #     [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    # )
    # @parametrize("fp16", [True, False])
    @skip_if_lt_x_gpu(2)
    def test_basic_mixed_precision_e2e(self):
        """
        tests
        """
        pass
