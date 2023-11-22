from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest


class FSDPTests(torch._dynamo.test_case.TestCase, FSDPTest):
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._dynamo.config, "capture_dynamic_output_shape_ops", True)
    def test_fsdp_compiles(self):
        self.run_subtests(
            {
                "use_orig_params": [True],
                "auto_wrap_policy": [ModuleWrapPolicy({nn.Linear})],
            },
            self._test_fsdp_compiles,
        )

    def _test_fsdp_compiles(self, use_orig_params, auto_wrap_policy):
        fsdp_kwargs = {
            "use_orig_params": True,
            "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
        }
        model = nn.Sequential(
            nn.Linear(3, 3, device="cuda"), nn.ReLU(), nn.Linear(3, 3, device="cuda")
        )
        model = FSDP(
            model,
            **fsdp_kwargs,
        )
        # TODO: Add `model = torch.compile(model)` here if desired
        cnt = torch._dynamo.testing.CompileCounter()
        model = torch._dynamo.optimize(cnt, nopython=True)(model)
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        inp = torch.randn((2, 3), device="cuda")

        losses = []

        for _ in range(3):
            optim.zero_grad(set_to_none=True)
            inp = torch.randn((2, 3), device="cuda")
            out = model(inp)
            loss = out.sum()
            losses.append(loss)
            loss.backward()
            optim.step()
        self.assertEqual(len(losses), 3)
        self.assertEqual(cnt.frame_count, 1)
