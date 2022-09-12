# Owner(s): ["oncall: distributed"]

import contextlib
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
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
from torch.utils.checkpoint import checkpoint

_save_on_cpu_called = False
def get_patched_save_on_cpu():
    orig_save_on_cpu = torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu

    def patched_save_on_cpu(*args, **kwargs):
        global _save_on_cpu_called
        _save_on_cpu_called = True
        return orig_save_on_cpu(*args, **kwargs)

    return patched_save_on_cpu

@contextlib.contextmanager
def patch_save_on_cpu(new_save_on_cpu):
    orig_save_on_cpu = torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu
    torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu = new_save_on_cpu
    try:
        yield
    finally:
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu = orig_save_on_cpu

class TestFSDPCheckpoint(FSDPTest):
    class SequentialModule(nn.Module):
        def __init__(
            self,
            checkpoint_layer=False,
            offload_activations=False,
            wrap_fsdp=False,
            *fsdp_args,
            **fsdp_kwargs,
        ):
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            super().__init__()
            l1 = nn.Linear(3, 3).cuda()
            l2 = nn.Linear(3, 3).cuda()
            l3 = nn.Linear(3, 3).cuda()

            if checkpoint_layer:
                ckpt_wrapper = partial(
                    checkpoint_wrapper, offload_to_cpu=offload_activations
                )

                l1 = ckpt_wrapper(l1)
                l2 = ckpt_wrapper(l2)
                l3 = ckpt_wrapper(l3)

            fsdp_wrapper = partial(
                _maybe_wrap_fsdp, wrap_fsdp=wrap_fsdp, *fsdp_args, **fsdp_kwargs
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
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("offload_activations", [True, False])
    def test_checkpoint_fsdp_wrapping(self, cpu_offload, offload_activations):
        # Test checkpoint(FSDP(layer1), FSDP(layer2), ....)
        ckpt_sequential_wrapped_fsdp = checkpoint_wrapper(
            TestFSDPCheckpoint.SequentialModule(
                wrap_fsdp=True, cpu_offload=cpu_offload
            ),
            offload_to_cpu=offload_activations,
        )
        # Test FSDP(checkpoint(layer1)), FSDP(checkpoint(layer2)), ....
        inner_ckpt = TestFSDPCheckpoint.SequentialModule(
            checkpoint_layer=True,
            offload_activations=offload_activations,
            wrap_fsdp=True,
            cpu_offload=cpu_offload,
        )

        baseline = TestFSDPCheckpoint.SequentialModule(
            wrap_fsdp=True, cpu_offload=cpu_offload
        )

        # note that reentrant-based checkpointing requires inputs to have grad
        # flag set.
        inp = torch.randn(10, 3, device=torch.cuda.current_device(), requires_grad=True)

        global _save_on_cpu_called
        models = [ckpt_sequential_wrapped_fsdp, inner_ckpt, baseline]
        with patch_save_on_cpu(get_patched_save_on_cpu()):
            for i in range(2):
                losses = []
                outputs = []
                for m in models:
                    check_offload = m != baseline and i == 0 and offload_activations
                    if check_offload:
                        self.assertFalse(_save_on_cpu_called)
                    out = m(inp)
                    if check_offload:
                        self.assertTrue(_save_on_cpu_called)
                        _save_on_cpu_called = False
                    loss = out.sum()
                    loss.backward()
                    losses.append(loss)
                    outputs.append(out)

                self._verify_parity(losses, outputs, models)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("offload_activations", [True, False])
    def test_basic_checkpoint_end_to_end(self, cpu_offload, offload_activations):
        global _save_on_cpu_called
        with patch_save_on_cpu(get_patched_save_on_cpu()):
            seq = TestFSDPCheckpoint.SequentialModule().to(torch.cuda.current_device())
            # Runs FSDP with no checkpointing
            fsdp_only_seq = FSDP(deepcopy(seq), cpu_offload=cpu_offload)
            # Runs checkpoint-wrapped FSDP
            checkpointed_fsdp = checkpoint_wrapper(
                FSDP(deepcopy(seq), cpu_offload=cpu_offload),
                offload_to_cpu=offload_activations,
            )
            # Runs FSDP-wrapped checkpointed module
            fsdp_wrapped_checkpoint = FSDP(
                checkpoint_wrapper(deepcopy(seq), offload_to_cpu=offload_activations),
                cpu_offload=cpu_offload,
            )
            # Runs FSDP with manual calls to checkpoint.
            fsdp_call_checkpoint = FSDP(deepcopy(seq), cpu_offload=cpu_offload)
            # note that reentrant-based checkpointing requires inputs to have grad
            # flag set.

            inp = torch.randn(10, 3, device=torch.cuda.current_device(), requires_grad=True)

            models = [
                fsdp_only_seq,
                checkpointed_fsdp,
                fsdp_wrapped_checkpoint,
                fsdp_call_checkpoint,
            ]
            # Ensure _save_on_cpu is not yet called
            self.assertFalse(_save_on_cpu_called)
            for i in range(6):
                losses = []
                outputs = []
                for m in models:
                    check_offload = m != fsdp_only_seq and i == 0 and offload_activations
                    if m == fsdp_call_checkpoint:
                        # _save_on_cpu should not be called yet
                        self.assertFalse(_save_on_cpu_called)
                        offload_ctx = (
                            get_patched_save_on_cpu()(pin_memory=True)
                            if offload_activations
                            else contextlib.suppress()
                        )
                        with offload_ctx:
                            out = checkpoint(m, inp)
                    else:
                        # _save_on_cpu should not be called yet
                        self.assertFalse(_save_on_cpu_called)
                        out = m(inp)

                    if check_offload:
                        self.assertTrue(_save_on_cpu_called)
                    loss = out.sum()
                    loss.backward()
                    losses.append(loss)
                    outputs.append(out)
                    _save_on_cpu_called = False

                self._verify_parity(losses, outputs, models)

instantiate_parametrized_tests(TestFSDPCheckpoint)

if __name__ == "__main__":
    run_tests()
