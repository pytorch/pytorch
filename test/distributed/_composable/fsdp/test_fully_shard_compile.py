# Owner(s): ["oncall: distributed"]

import inspect
from enum import auto, Enum

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests


class CompileEntry(Enum):
    MODULE_WRAPPER = auto()  # torch.compile(module)
    MODULE_METHOD = auto()  # module.compile()


class TestFullyShardSkipCompilingFSDPHooks(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_skip_compiling_fsdp_hooks(self):
        self.run_subtests(
            {
                "compile_entry": [
                    CompileEntry.MODULE_WRAPPER,
                    CompileEntry.MODULE_METHOD,
                ],
                "skip_fsdp_hooks": [True, False],
            },
            self._test_skip_compiling_fsdp_hooks,
        )

    def _test_skip_compiling_fsdp_hooks(
        self, compile_entry: CompileEntry, skip_fsdp_hooks: bool
    ):
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(3, torch.device("cpu")) for _ in range(3)])
        fully_shard(model)
        torch._dynamo.reset()

        torch.manual_seed(42 + self.rank + 1)
        with torch._dynamo.config.patch("skip_fsdp_hooks", skip_fsdp_hooks):
            if compile_entry == CompileEntry.MODULE_WRAPPER:
                model = torch.compile(model)
                inner_model = model._orig_mod
            elif compile_entry == CompileEntry.MODULE_METHOD:
                model.compile()
                inner_model = model
            fsdp_hook = next(iter(inner_model._forward_pre_hooks.values()))
            if skip_fsdp_hooks:
                self.assertTrue(hasattr(fsdp_hook, "_torchdynamo_disable"))
                dynamo_ctx = inspect.getclosurevars(fsdp_hook).nonlocals["self"]
                self.assertTrue(
                    isinstance(dynamo_ctx, torch._dynamo.eval_frame.DisableContext)
                )
            else:
                self.assertTrue(not hasattr(fsdp_hook, "_torchdynamo_disable"))

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
