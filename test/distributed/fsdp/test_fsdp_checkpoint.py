# Owner(s): ["oncall: distributed"]
import contextlib
import sys
from copy import deepcopy
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import _maybe_wrap_fsdp, FSDPTest, get_devtype
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.utils.checkpoint import checkpoint


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)
_save_on_cpu_called = False


def get_patched_save_on_cpu():
    orig_save_on_cpu = (
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu
    )

    def patched_save_on_cpu(*args, **kwargs):
        global _save_on_cpu_called
        _save_on_cpu_called = True
        return orig_save_on_cpu(*args, **kwargs)

    return patched_save_on_cpu


@contextlib.contextmanager
def patch_save_on_cpu(new_save_on_cpu):
    orig_save_on_cpu = (
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu
    )
    torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu = (
        new_save_on_cpu
    )
    try:
        yield
    finally:
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.save_on_cpu = (
            orig_save_on_cpu
        )


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
            super().__init__()
            l1 = nn.Linear(3, 3).to(device_type.type)
            l2 = nn.Linear(3, 3).to(device_type.type)
            l3 = nn.Linear(3, 3).to(device_type.type)
            if checkpoint_layer:
                if offload_activations:
                    ckpt_wrapper = offload_wrapper
                else:
                    ckpt_wrapper = checkpoint_wrapper
                l1 = ckpt_wrapper(l1)
                l2 = ckpt_wrapper(l2)
                l3 = ckpt_wrapper(l3)
            fsdp_wrapper = partial(
                _maybe_wrap_fsdp, *fsdp_args, wrap_fsdp=wrap_fsdp, **fsdp_kwargs
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
        for l, o in zip(losses[1:], outputs[1:]):
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
    @parametrize("use_orig_params", [False, True])
    def test_checkpoint_fsdp_wrapping(
        self,
        cpu_offload: CPUOffload,
        offload_activations: bool,
        use_orig_params: bool,
    ):
        # Test checkpoint(FSDP(layer1), FSDP(layer2), ....)
        if offload_activations:
            wrapper_to_use = offload_wrapper
        else:
            wrapper_to_use = checkpoint_wrapper
        fsdp_kwargs = {"cpu_offload": cpu_offload, "use_orig_params": use_orig_params}
        ckpt_sequential_wrapped_fsdp = wrapper_to_use(
            TestFSDPCheckpoint.SequentialModule(
                wrap_fsdp=True,
                **fsdp_kwargs,
            ),
        )
        # Test FSDP(checkpoint(layer1)), FSDP(checkpoint(layer2)), ....
        inner_ckpt = TestFSDPCheckpoint.SequentialModule(
            checkpoint_layer=True,
            offload_activations=offload_activations,
            wrap_fsdp=True,
            **fsdp_kwargs,
        )
        baseline = TestFSDPCheckpoint.SequentialModule(
            wrap_fsdp=True,
            **fsdp_kwargs,
        )
        # note that reentrant-based checkpointing requires inputs to have grad
        # flag set.
        inp = torch.randn(10, 3, device=device_type.type, requires_grad=True)
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
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    @parametrize("offload_activations", [True, False])
    @parametrize("use_orig_params", [False, True])
    def test_basic_checkpoint_end_to_end(
        self,
        cpu_offload: CPUOffload,
        offload_activations: bool,
        use_orig_params: bool,
    ):
        fsdp_kwargs = {"cpu_offload": cpu_offload, "use_orig_params": use_orig_params}
        global _save_on_cpu_called
        with patch_save_on_cpu(get_patched_save_on_cpu()):
            seq = TestFSDPCheckpoint.SequentialModule().to(device_type.type)
            # Runs FSDP with no checkpointing
            fsdp_only_seq = FSDP(deepcopy(seq), **fsdp_kwargs)
            # Runs checkpoint-wrapped FSDP
            if offload_activations:
                wrapper_to_use = offload_wrapper
            else:
                wrapper_to_use = checkpoint_wrapper
            checkpointed_fsdp = wrapper_to_use(
                FSDP(deepcopy(seq), **fsdp_kwargs),
            )
            # Runs FSDP-wrapped checkpointed module
            fsdp_wrapped_checkpoint = FSDP(
                wrapper_to_use(deepcopy(seq)),
                **fsdp_kwargs,
            )
            # Runs FSDP with manual calls to checkpoint.
            fsdp_call_checkpoint = FSDP(deepcopy(seq), **fsdp_kwargs)
            # note that reentrant-based checkpointing requires inputs to have grad
            # flag set.
            inp = torch.randn(10, 3, device=device_type.type, requires_grad=True)
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
                    check_offload = (
                        m != fsdp_only_seq and i == 0 and offload_activations
                    )
                    if m == fsdp_call_checkpoint:
                        # _save_on_cpu should not be called yet
                        self.assertFalse(_save_on_cpu_called)
                        offload_ctx = (
                            get_patched_save_on_cpu()(pin_memory=True)
                            if offload_activations
                            else contextlib.nullcontext()
                        )
                        with offload_ctx:
                            out = checkpoint(m, inp, use_reentrant=True)
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
        dist.barrier()


instantiate_parametrized_tests(TestFSDPCheckpoint)


class CheckpointModule(nn.Module):
    def __init__(self, checkpoint: bool = False, use_reentrant: bool = True):
        super().__init__()
        self.seq = nn.Sequential(*[nn.Linear(100, 100) for _ in range(4)])
        self.checkpoint = checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, x):
        return (
            checkpoint(self.seq, x, use_reentrant=self.use_reentrant)
            if self.checkpoint
            else self.seq(x)
        )


class ModelWithCheckpointSubmodule(nn.Module):
    def __init__(self, checkpoint: bool = False, use_reentrant: bool = True):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.s1 = CheckpointModule(checkpoint, use_reentrant)
        self.s2 = CheckpointModule(checkpoint, use_reentrant)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.relu(self.s2(self.s1(self.l1(x)))))


class TestModel(nn.Module):
    def __init__(self, checkpoint: bool = False, use_reentrant: bool = True):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
        self.checkpoint1 = ModelWithCheckpointSubmodule(checkpoint, use_reentrant)
        self.checkpoint2 = ModelWithCheckpointSubmodule(checkpoint, use_reentrant)
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.relu(self.checkpoint2(self.checkpoint1(self.l1(x)))))


class TestFSDPCheckpointSubmodule(FSDPTest):
    # TODO: grad value checks occasionally fails when use_reentrant = True
    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [False])
    def test_checkpoint_submodule(self, device, use_reentrant: bool):
        model = TestModel(use_reentrant=use_reentrant).to(device_type.type)
        model_ac = deepcopy(model)
        for _, m in model_ac.named_modules():
            if isinstance(m, CheckpointModule):
                m.checkpoint = True
        self.assertTrue(model_ac.checkpoint1.s1.checkpoint)
        self.assertTrue(model_ac.checkpoint2.s2.checkpoint)
        fsdp_kwargs = {
            "device_id": device_type.type,
            "sharding_strategy": ShardingStrategy.NO_SHARD,
        }
        # Wrap no checkpointing model submodules with FSDP
        model.checkpoint1 = FSDP(module=model.checkpoint1, **fsdp_kwargs)
        model.checkpoint2 = FSDP(module=model.checkpoint2, **fsdp_kwargs)
        # Wrap checkpointing model submodules with FSDP
        model_ac.checkpoint1 = FSDP(module=model_ac.checkpoint1, **fsdp_kwargs)
        model_ac.checkpoint2 = FSDP(module=model_ac.checkpoint2, **fsdp_kwargs)
        x = torch.randn(2, 100, device=self.device_type)
        model(x).sum().backward()
        model_ac(x).sum().backward()
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model_ac.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(p1.grad.allclose(p2.grad))


devices = ("cuda", "hpu")
instantiate_device_type_tests(TestFSDPCheckpointSubmodule, globals(), only_for=devices)
if __name__ == "__main__":
    run_tests()
