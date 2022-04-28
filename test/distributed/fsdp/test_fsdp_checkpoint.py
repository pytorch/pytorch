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

        models = [ckpt_sequential_wrapped_fsdp, inner_ckpt, baseline]

        offload_to_cpu_event = "Memcpy DtoH" if torch.version.cuda else "CopyDeviceToHost"

        for i in range(2):
            losses = []
            outputs = []
            for m in models:
                check_offload = m != baseline and i == 0 and offload_activations
                profiler_ctx = (
                    torch.profiler.profile(use_cuda=True)
                    if check_offload
                    else contextlib.suppress()
                )
                with profiler_ctx as prof:
                    out = m(inp)

                if check_offload:
                    event_names = [event.name for event in prof.events()]
                    offload_occured = any(
                        offload_to_cpu_event in name for name in event_names
                    )
                    self.assertTrue(offload_occured)
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

        offload_to_cpu_event = "Memcpy DtoH" if torch.version.cuda else "CopyDeviceToHost"

        for i in range(6):
            losses = []
            outputs = []
            for m in models:
                check_offload = m != fsdp_only_seq and i == 0 and offload_activations
                profiler_ctx = (
                    torch.profiler.profile(use_cuda=True)
                    if check_offload
                    else contextlib.suppress()
                )
                with profiler_ctx as prof:
                    if m == fsdp_call_checkpoint:
                        offload_ctx = (
                            torch.autograd.graph.save_on_cpu(pin_memory=True)
                            if offload_activations
                            else contextlib.suppress()
                        )
                        with offload_ctx:
                            out = checkpoint(m, inp)
                    else:
                        out = m(inp)

                if check_offload:
                    event_names = [event.name for event in prof.events()]
                    offload_occured = any(
                        offload_to_cpu_event in name for name in event_names
                    )
                    self.assertTrue(offload_occured)
                loss = out.sum()
                loss.backward()
                losses.append(loss)
                outputs.append(out)

            self._verify_parity(losses, outputs, models)

instantiate_parametrized_tests(TestFSDPCheckpoint)

if __name__ == "__main__":
    run_tests()
