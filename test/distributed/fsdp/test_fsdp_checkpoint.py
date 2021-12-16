# Owner(s): ["oncall: distributed"]

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributed._fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint._checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    _maybe_wrap_fsdp,
)
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)


class TestFSDPCheckpoint(FSDPTest):

    class SequentialModule(nn.Module):
        def __init__(self, checkpoint_layer=False, wrap_fsdp=False, *fsdp_args, **fsdp_kwargs):
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            super().__init__()
            l1 = nn.Linear(3, 3).cuda()
            l2 = nn.Linear(3, 3).cuda()
            l3 = nn.Linear(3, 3).cuda()

            if checkpoint_layer:
                l1 = checkpoint_wrapper(l1)
                l2 = checkpoint_wrapper(l2)
                l3 = checkpoint_wrapper(l3)

            fsdp_wrapper = partial(
                _maybe_wrap_fsdp,
                wrap_fsdp=wrap_fsdp,
                *fsdp_args,
                **fsdp_kwargs
            )
            self.ffn = nn.Sequential(
                fsdp_wrapper(l1),
                fsdp_wrapper(l2),
                fsdp_wrapper(l3),
            )

        def forward(self, x):
            return self.ffn(x)


    def _verify_parity(self, losses, outputs, models):
        assert losses
        assert outputs
        assert models

        for (l, o) in zip(losses[1:], outputs[1:]):
            self.assertEqual(losses[0], l)
            self.assertEqual(outputs[0], o)

        # Verify grads
        ref_model = models[0]
        ref_grads = [p.grad for p in ref_model.parameters()]
        for m in models[1:]:
            grads = [p.grad for p in m.parameters()]
            for ref_g, g in zip(ref_grads, grads):
                self.assertEqual(ref_g, g)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
    )
    def test_checkpoint_fsdp_wrapping(self, cpu_offload):
        # Test checkpoint(FSDP(layer1), FSDP(layer2), ....)
        ckpt_sequential_wrapped_fsdp = checkpoint_wrapper(
            TestFSDPCheckpoint.SequentialModule(
                wrap_fsdp=True, cpu_offload=cpu_offload
            )
        )
        # Test FSDP(checkpoint(layer1)), FSDP(checkpoint(layer2)), ....
        inner_ckpt = TestFSDPCheckpoint.SequentialModule(
            checkpoint_layer=True, wrap_fsdp=True, cpu_offload=cpu_offload
        )

        baseline = TestFSDPCheckpoint.SequentialModule(
            wrap_fsdp=True, cpu_offload=cpu_offload
        )

        # note that reentrant-based checkpointing requires inputs to have grad
        # flag set.
        inp = torch.randn(10, 3, device=torch.cuda.current_device(), requires_grad=True)

        models = [
            ckpt_sequential_wrapped_fsdp,
            inner_ckpt,
            baseline
        ]

        for _ in range(2):
            losses = []
            outputs = []
            for m in models:
                out = m(inp)
                loss = out.sum()
                loss.backward()
                losses.append(loss)
                outputs.append(out)

            self._verify_parity(losses, outputs, models)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
    )
    def test_basic_checkpoint_end_to_end(self, cpu_offload):
        seq = TestFSDPCheckpoint.SequentialModule().to(torch.cuda.current_device())
        # Runs FSDP with no checkpointing
        fsdp_only_seq = FSDP(deepcopy(seq), cpu_offload=cpu_offload)
        # Runs checkpoint-wrapped FSDP
        checkpointed_fsdp = checkpoint_wrapper(FSDP(deepcopy(seq), cpu_offload=cpu_offload))
        # Runs FSDP-wrapped checkpointed module
        fsdp_wrapped_checkpoint = FSDP(checkpoint_wrapper(deepcopy(seq)), cpu_offload=cpu_offload)
        # Runs FSDP with manual calls to checkpoint.
        fsdp_call_checkpoint = FSDP(deepcopy(seq), cpu_offload=cpu_offload)
        # note that reentrant-based checkpointing requires inputs to have grad
        # flag set.

        inp = torch.randn(10, 3, device=torch.cuda.current_device(), requires_grad=True)

        models = [
            fsdp_only_seq,
            checkpointed_fsdp,
            fsdp_wrapped_checkpoint,
            fsdp_call_checkpoint
        ]

        for _ in range(6):
            losses = []
            outputs = []
            for m in models:
                if m == fsdp_call_checkpoint:
                    out = checkpoint(m, inp)
                else:
                    out = m(inp)
                loss = out.sum()
                loss.backward()
                losses.append(loss)
                outputs.append(out)

            self._verify_parity(losses, outputs, models)

instantiate_parametrized_tests(TestFSDPCheckpoint)

if __name__ == "__main__":
    run_tests()
