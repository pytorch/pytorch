# Owner(s): ["oncall: distributed"]

import functools
import itertools
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from expecttest import TestCase
from torch import distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleConfig
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
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


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p1 = nn.Parameter(torch.randn((3, 3)))
        self.p2 = nn.Parameter(torch.randn((4, 4)))
        self.lin = nn.Linear(2, 2)  # weight and bias
        self.p3 = self.lin.weight  # shared


class TestFlatParamHandle(FSDPTest):
    ...

    def device(self) -> torch.device:
        return torch.device("cuda")

    def _get_total_numel(self, params: List[nn.Parameter]) -> int:
        return sum(param.numel() for param in params)

    @skip_if_lt_x_gpu(1)
    def test_init_flat_param_all_params_no_grads(self):
        """"""
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_init_flat_param_all_params_no_grads,
        )

    def _test_init_flat_param_all_params_no_grads(
        self, sharding_strategy: ShardingStrategy, use_orig_params: bool
    ):
        model = Model()
        params = list(model.parameters())
        expected_numel = self._get_total_numel(params)
        handle = FlatParamHandle(
            params,
            model,
            self.device,
            HandleConfig(sharding_strategy, False, None, None, False),
            self.process_group,
            use_orig_params,
        )
        self.assertEqual(handle.flat_param.numel(), expected_numel)
        self.assertEqual(handle.flat_param.ndim, 1)

    def _test_init_flat_param_some_params_no_grads(
        self,
    ):
        ...

    def _test_init_flat_param_all_params_all_grads(
        self,
    ):
        ...

    def _test_init_flat_param_all_params_some_grads(
        self,
    ):
        ...


if __name__ == "__main__":
    run_tests()
