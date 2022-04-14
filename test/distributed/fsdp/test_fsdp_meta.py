# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.flatten_params_wrapper import FlattenParamsWrapper
from torch.distributed.fsdp.wrap import always_wrap_policy as always_wrap
from torch.distributed.fsdp.wrap import wrap, enable_wrap
from torch.testing._internal.common_fsdp import (
    FSDPTest,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
    sandcastle_skip_if,
)

_TORCHDISTX_AVAIL = True

try:
    from torchdistx import fake, deferred_init  # noqa: F401
except ImportError:
    _TORCHDISTX_AVAIL = False

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

class MyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.lin1 = nn.Linear(2, 2, bias=False, device=device)
        self.lin2 = nn.Linear(2, 2, bias=False, device=device)

    def forward(self, x):
        return self.lin2(self.lin1(x))

    def reset_parameters(self, *args, **kwargs):
        # Reset any parameters that are not FSDP
        for m in [self.lin1, self.lin2]:
            if not isinstance(m, FSDP):
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                m.reset_parameters()


class NestedModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.lin1 = nn.Linear(2, 2, bias=False, device=device)
        self.lin1 = wrap(self.lin1)
        self.lin2 = nn.Linear(2, 2, bias=False, device=device)
        self.l3 = MyModel(device=device)
        self.l3 = wrap(self.l3)

    def forward(self, x):
        return self.l3(self.lin2(self.lin1(x)))

    def reset_parameters(self):
        for m in [self.lin1, self.lin2, self.l3]:
            if not isinstance(m, FSDP):
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                m.reset_parameters()

def _init_with_reset_params(module):
    is_meta = any(t.is_meta for t in module.parameters())
    if is_meta:
        module.to_empty(device=torch.cuda.current_device())
    with torch.no_grad():
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        module.reset_parameters()

def _torchdistx_init(module):
    if not _TORCHDISTX_AVAIL:
        raise ValueError(
            "Should not call _torchdistx_init if torchdistx is not available."
        )

    def check_fn(k):
        return not isinstance(k, FSDP)
    deferred_init.materialize_module(module, check_fn=check_fn)


class TestFSDPWithMetaDevice(FSDPTest):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    def _compare_fsdp(self, fsdp1, fsdp2):
        with fsdp1.summon_full_params(fsdp1):
            with fsdp2.summon_full_params(fsdp2):
                for p1, p2 in zip(fsdp1.parameters(), fsdp2.parameters()):
                    self.assertTrue(torch.allclose(p1, p1), f"{p1} vs {p2}")

    def test_simple_model_with_meta_device_reset_params(self):
        model = MyModel(device="meta")
        assert next(model.lin1.parameters()).is_meta
        assert next(model.lin2.parameters()).is_meta

        fsdp_meta = FSDP(
            model,
            auto_wrap_policy=always_wrap,
            param_init_fns=_init_with_reset_params,
        )

        # Test to make sure it is the same model parameters as regular FSDP
        # approach.
        regular = MyModel(device="cuda")
        regular.reset_parameters()
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)

        self._compare_fsdp(fsdp_meta, fsdp_regular)

        # Test that meta init works if all submodules are contained in only a
        # single FSDP unit.
        fsdp_meta = FSDP(model, param_init_fns=_init_with_reset_params)
        regular = MyModel(device="cuda")
        regular.reset_parameters()
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)

        inp = torch.randn(10, 2).cuda()
        # Run a forward + backward pass to ensure things work
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        self._compare_fsdp(fsdp_meta, fsdp_regular)


    @parametrize("auto_wrap", [True, False])
    def test_nested_model_with_meta_device_reset_params(self, auto_wrap):
        if auto_wrap:
            module = NestedModel(device="meta")
            fsdp_meta = FSDP(
                module,
                auto_wrap_policy=always_wrap,
                param_init_fns=_init_with_reset_params,
            )
            module_regular = NestedModel(device="cuda")
            module_regular.reset_parameters()
            fsdp_regular = FSDP(
                module_regular,
                auto_wrap_policy=always_wrap,
            )
        else:
            with enable_wrap(
                wrapper_cls=FSDP, param_init_fns=_init_with_reset_params
            ):
                module = NestedModel(device="meta")
                # Users need to explicitly initialize non FSDP modules.
                module.lin2.to_empty(device=torch.cuda.current_device())
                module.lin2.reset_parameters()
                fsdp_meta = wrap(module)

            # Init and reset parameters before wrapping so that reset_params
            # matches up with meta device's initialization.
            module_regular = NestedModel(device="cuda")  # note: not wrapped now
            module_regular.reset_parameters()
            with enable_wrap(wrapper_cls=FSDP):
                module_regular = NestedModel(device="cuda")
                module_regular.lin1 = wrap(module_regular.lin1)
                module_regular.l3 = wrap(module_regular.l3)
                fsdp_regular = wrap(module_regular)

        inp = torch.randn(10, 2, device='cuda')
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        self._compare_fsdp(fsdp_meta, fsdp_regular)


instantiate_parametrized_tests(TestFSDPWithMetaDevice)

if __name__ == "__main__":
    run_tests()
