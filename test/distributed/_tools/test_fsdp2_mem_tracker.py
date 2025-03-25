# Owner(s): ["module: unknown"]
import functools
import gc
from typing import Union

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.distributed._tensor import init_device_mesh
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointWrapper,
)
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


def _init_cublas_workspace(dev: torch.device):
    lin = torch.nn.Linear(768, 768, device=dev)
    inp = torch.randn(1, 768, device=dev)
    lin(inp).sum().backward()
    del lin
    del inp


def _reset_mem_stats(dev: torch.device):
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats(dev)
    torch.cuda.reset_peak_memory_stats(dev)


class TestTrackerFullyShard1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_tracker_multi_group_eager(self):
        """
        Tests tracker accuracy when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction) and different mixed precision policies.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=False),
                    OffloadPolicy(),
                ],
                "mp_policy": [
                    MixedPrecisionPolicy(
                        param_dtype=torch.float16, reduce_dtype=torch.float32
                    ),
                ],
            },
            self._test_tracker_multi_group,
        )

    def _test_tracker_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        mp_policy: MixedPrecisionPolicy,
    ):
        debug = False
        dev = torch.device(torch.cuda.current_device())
        _init_cublas_workspace(dev)
        gc.collect()
        _reset_mem_stats(dev)
        mem_stats = torch.cuda.memory_stats(dev)
        pre_cuda_active = mem_stats["active_bytes.all.current"]
        torch.manual_seed(42)
        lin_dim, bsz = 2048, 8192
        with torch.device(dev):
            model = nn.Sequential(*[MLP(dim=lin_dim, device=dev) for _ in range(4)])
        mesh = init_device_mesh("cuda", (self.world_size,))
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        inp = torch.randn((bsz, lin_dim), device=dev)
        fmt = FSDPMemTracker(model, optim)
        fmt.track_inputs((inp,))
        with fmt:
            for iter_idx in range(2):
                loss = model(inp).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    fmt.reset_mod_stats()
        mem_stats = torch.cuda.memory_stats()
        tracker_max = fmt.get_tracker_snapshot("peak")[dev]["Total"]
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        accuracy = tracker_max / cuda_max
        if self.rank == 0 and debug:
            print(f"Accuracy: {accuracy} Tracker Max:{tracker_max} CUDA Max:{cuda_max}")
        self.assertAlmostEqual(
            accuracy,
            1.0,
            delta=0.1,
            msg=f"Tracker Max:{tracker_max} CUDA Max:{cuda_max}",
        )
        del model
        del inp
        del optim

    @skip_if_lt_x_gpu(2)
    def test_tracker_non_root_forward_backward(self):
        """
        Tests tracker accracy when running forward/backward through a non-root.
        """
        debug = False
        dev = torch.device(torch.cuda.current_device())
        _init_cublas_workspace(dev)
        gc.collect()
        _reset_mem_stats(dev)
        mem_stats = torch.cuda.memory_stats(dev)
        pre_cuda_active = mem_stats["active_bytes.all.current"]
        torch.manual_seed(42)
        lin_dim, bsz = 2048, 8
        model = nn.Sequential(*[MLP(lin_dim, dev) for _ in range(3)])
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((bsz, lin_dim), device=dev)
        fmt = FSDPMemTracker(model, optim)
        fmt.track_inputs((inp,))
        with fmt:
            for iter_idx in range(2):
                nonroot_loss = model[0](inp).sum()
                nonroot_loss.backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    fmt.reset_mod_stats()
        mem_stats = torch.cuda.memory_stats()
        tracker_max = fmt.get_tracker_snapshot("peak")[dev]["Total"]
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        accuracy = tracker_max / cuda_max
        if self.rank == 0 and debug:
            print(f"Accuracy: {accuracy} Tracker Max:{tracker_max} CUDA Max:{cuda_max}")
        self.assertAlmostEqual(
            accuracy,
            1.0,
            delta=0.1,
            msg=f"Tracker Max:{tracker_max} CUDA Max:{cuda_max}",
        )
        del inp
        del model
        del optim


class TestTrackerFullyShard1DTrainingCompose(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    @skip_if_lt_x_gpu(2)
    def test_tracker_with_activation_checkpointing(self):
        """
        Tests tracker accuracy when composing with activation checkpointing.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "checkpoint_impl": ["composable", "wrapper"],
            },
            self._test_tracker_with_activation_checkpointing,
        )

    def _test_tracker_with_activation_checkpointing(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: str
    ):
        assert checkpoint_impl in ("composable", "wrapper")
        debug = False
        dev = torch.device(torch.cuda.current_device())
        _init_cublas_workspace(dev)
        gc.collect()
        _reset_mem_stats(dev)
        mem_stats = torch.cuda.memory_stats(dev)
        pre_cuda_active = mem_stats["active_bytes.all.current"]
        torch.manual_seed(42)
        vocab_size = 8192
        bsz, seq_len = 16, 512
        with torch.device(dev):
            model_args = ModelArgs(
                n_layers=4,
                n_heads=4,
                vocab_size=vocab_size,
                max_seq_len=seq_len,
                dropout_p=0.1,
            )
            model = Transformer(model_args)
        foreach = False
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
        )
        if checkpoint_impl == "wrapper":
            apply_activation_checkpointing(
                model, check_fn=lambda m: isinstance(m, TransformerBlock)
            )
            for module in model.modules():
                # Apply to `CheckpointWrapper`, which wraps `TransformerBlock`
                if isinstance(module, CheckpointWrapper):
                    fully_shard_fn(module)
        else:
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    if checkpoint_impl == "composable":
                        checkpoint(module)
                    fully_shard_fn(module)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        fmt = FSDPMemTracker(model, optim)
        fmt.track_inputs((inp,))
        with fmt:
            for iter_idx in range(2):
                loss = model(inp).sum()
                loss.backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    fmt.reset_mod_stats()
        mem_stats = torch.cuda.memory_stats()
        tracker_max = fmt.get_tracker_snapshot("peak")[dev]["Total"]
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        accuracy = tracker_max / cuda_max
        if self.rank == 0 and debug:
            print(f"Accuracy: {accuracy} Tracker Max:{tracker_max} CUDA Max:{cuda_max}")
        self.assertAlmostEqual(
            accuracy,
            1.0,
            delta=0.1,
            msg=f"Tracker Max:{tracker_max} CUDA Max:{cuda_max}",
        )
        del inp
        del model
        del optim


if __name__ == "__main__":
    run_tests()
