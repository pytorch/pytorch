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
    CPUOffload,
    MixedPrecision
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
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

class LinearMixedPrecision(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)

    def forward(self, tup):
        # Param and input should be the mixed precision type
        # cls.assertEqual(x.dtype, mp_config.param_dtype)
        #cls.assertEqual(next(self.lin.parameters()).dtype, mp_config.param_dtype)
        inp, cls, fsdp, mp_config = tup
        expected_param_type = mp_config.param_dtype
        cls.assertEqual(inp.dtype, expected_param_type)

        # In FSDP, self.params should point to the right type.
        for fsdp_module in FSDP.fsdp_modules(fsdp):
            fsdp_managed_params = fsdp_module.params
            # Single param assumption
            cls.assertEqual(1, len(fsdp_managed_params))
            for param in fsdp_managed_params:
                if param._full_param_padded.storage().size() > 0:
                    # This FSDP unit is active, verify param points to mixed
                    cls.assertEqual(param.dtype, expected_param_type)
                else:
                    # This FSDP unit is not active as full param has been
                    # freed. Ensure param points to full precision param.
                    cls.assertEqual(param.dtype, torch.float32)

        return (self.lin(inp), cls, fsdp, mp_config)


class TestFSDPMixedPrecision(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _get_simple_nested_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(
            nn.Sequential(
                FSDP(LinearMixedPrecision().cuda(), *fsdp_args, **fsdp_kwargs),
                LinearMixedPrecision().cuda(),
            ),
            *fsdp_args,
            **fsdp_kwargs,
        )
        return model

    def _get_simple_model(self, *fsdp_args, **fsdp_kwargs):
        model = FSDP(LinearMixedPrecision().cuda(), *fsdp_args, **fsdp_kwargs)
        return model

    @skip_if_lt_x_gpu(2)
    def test_basic_mixed_precision_e2e(self):
        torch.cuda.set_device(self.rank)
        mp_config = MixedPrecision()
        # mp_config = None
        cpu_offload = CPUOffload(offload_params=False)
        fsdp_models = [
            self._get_simple_model(cpu_offload=cpu_offload, mixed_precision=mp_config),
            self._get_simple_nested_model(cpu_offload=cpu_offload, mixed_precision=mp_config),
        ]
        for model in fsdp_models:
            if not cpu_offload.offload_params:
                model.cuda()

            inp = torch.randn(3, 10).cuda()
            act, *_ = model((inp, self, model, mp_config))
            # p._mp_shard should be freed.
            fsdp_units = FSDP.fsdp_modules(model)
            for fsdp in fsdp_units:
                for param in fsdp.params:
                    self.assertEqual(0, param._mp_shard.storage().size())

            loss = act.sum()
            self.assertEqual(loss.dtype, mp_config.param_dtype)
            loss.backward()
            # p._mp_shard should be freed.

            # Ensure params and grads are in full precision
            for param in model.parameters():
                self.assertEqual(param.dtype, torch.float32)
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float32)
