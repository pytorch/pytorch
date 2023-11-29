# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest

from typing import List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from _test_fully_shard_common import (
    Block,
    check_sharded_grad_parity,
    check_sharded_param_parity,
    GPT,
    GPTConfig,
    MLP,
    patch_all_gather,
    patch_reduce_scatter,
)

from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import fully_shard, OffloadPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests
from torch.utils._triton import has_triton


class TestFullyShardTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(6, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_single_group(self):
        """
        Tests train parity against DDP when using a single parameter group for
        communication.
        """
        self.run_subtests(
            {
                "lin_shapes": [[(16, 15), (15, 8)], [(7, 15), (15, 3)]],
                "reshard_after_forward": [True, False, 1, self.world_size, 2],
            },
            self._test_train_parity_single_group,
        )

    def _test_train_parity_single_group(
        self, lin_shapes: List[Tuple[int, int]], reshard_after_forward: Union[bool, int]
    ):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0], bias=True),
            nn.ReLU(),
            nn.Linear(*lin_shapes[1], bias=True),
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((4, lin_shapes[0][0]), device=device)
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "offload_policy": [OffloadPolicy()],
                "device": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group_cpu_offload(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication and CPU offloading.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "offload_policy": [OffloadPolicy("cpu")],
                "device": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group_on_cpu(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication and using CPU device.
        """
        # TODO: For CPU training even with CUDA available, we are blocked on
        # device mesh support: https://github.com/pytorch/pytorch/issues/114629
        return
        # TODO: To support CPU device *without CUDA available*, we need to
        # refactor the implementation to avoid calling stream APIs if using
        # CPU device. This test still assumes CUDA is available.
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "offload_policy": [OffloadPolicy()],
                "device": ["cpu"],
                "delay_after_forward": [False],
                "delay_before_all_gather": [False],
                "delay_before_reduce_scatter": [False],
                "delay_before_optim": [False],
            },
            self._test_train_parity_multi_group,
        )

    def _test_train_parity_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        device: str,
        delay_after_forward: bool,
        delay_before_all_gather: bool,
        delay_before_reduce_scatter: bool,
        delay_before_optim: bool,
    ):
        # Only test individual delays or all four delays to save test time
        if (
            delay_after_forward
            + delay_before_all_gather
            + delay_before_reduce_scatter
            + delay_before_optim
            in (2, 3)
        ):
            return
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model)
        if device == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            device=device,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        delay_in_ms = 100
        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def delayed_all_gather(*args, **kwargs):
            if delay_before_all_gather:
                torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_all_gather(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            if delay_before_reduce_scatter:
                torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
        with patch_all_gather(delayed_all_gather), patch_reduce_scatter(
            delayed_reduce_scatter
        ):
            for iter_idx in range(10):
                inp = torch.randn((8, lin_dim), device=torch.device(device))
                losses: List[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    losses.append(_model(inp).sum())
                    if _model is model and delay_after_forward:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    losses[-1].backward()
                    if _model is model and delay_before_optim:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    _optim.step()
                self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_multi_forward_module(self):
        """
        Tests parity with DDP when running a module that participates multiple
        times in forward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_multi_forward_module,
        )

    def _test_multi_forward_module(self, reshard_after_forward: Union[bool, int]):
        class MultiForwardModule(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                self.inner = nn.Linear(4, 4, device=device)
                self.outer = nn.Linear(4, 5, device=device)

            def forward(self, x):
                i = self.inner(x)
                j = self.inner(x)
                return self.outer(i + j)

        torch.manual_seed(42)
        model = MultiForwardModule(device="cuda")
        ref_model = copy.deepcopy(model)
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model.inner)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        inp = torch.randn((32, 4), device="cuda")
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


class TestFullyShardTrainingCompose(FSDPTest):
    @property
    def world_size(self) -> int:
        # Since these tests run with a larger GPT model, they see some numeric
        # drift with >2 GPUs
        return 2

    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_activation_checkpointing(self):
        """
        Tests train parity against DDP when composing with activation
        checkpointing.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "checkpoint_impl": ["utils", "composable"],
            },
            self._test_train_parity_with_activation_checkpointing,
        )

    def _test_train_parity_with_activation_checkpointing(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: str
    ):
        assert checkpoint_impl in ("composable", "utils")
        torch.manual_seed(42)
        vocab_size = 1024
        with torch.device(torch.device("cuda")):
            config = GPTConfig(
                n_layer=3,
                n_head=4,
                vocab_size=vocab_size,
                checkpoint_activations=(checkpoint_impl == "utils"),
            )
            model = GPT(config)
        ref_model = replicate(copy.deepcopy(model), device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
        )
        for module in model.modules():
            if isinstance(module, Block):
                if checkpoint_impl == "compoasble":
                    checkpoint(module)
                fully_shard_fn(module)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        torch.manual_seed(42 + self.rank)
        # Reuse the same input across iterations to avoid loss explosion from
        # trying to learn from random inputs
        src = torch.randint(0, vocab_size, (3, 64), device="cuda")
        tgt = torch.randint(0, vocab_size, (3, 64), device="cuda")
        inp = (src, tgt)
        check_sharded_param_parity(self, ref_model, model)
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(*inp).sum())
                losses[-1].backward()
            check_sharded_param_parity(self, ref_model, model)
            check_sharded_grad_parity(self, ref_model, model)
            self.assertEqual(losses[0], losses[1])
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            model.zero_grad()
            check_sharded_param_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not has_triton(), "Inductor on GPU needs Triton and recent GPU arch"
    )
    def test_train_parity_with_compile(self):
        # Increase the cache size limit since it is shared across subtests
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 16
        self.run_subtests(
            {"reshard_after_forward": [True], "backend": ["aot_eager", "inductor"]},
            self._test_train_parity_with_compile,
        )

    def _test_train_parity_with_compile(
        self,
        reshard_after_forward: bool,
        backend: str,
    ):
        torch.manual_seed(42)
        vocab_size = 1024
        config = GPTConfig(n_layer=3, n_head=4, vocab_size=vocab_size)
        model = GPT(config)
        ref_model = copy.deepcopy(model).cuda()
        for block in ref_model.transformer.h:
            block.forward = torch.compile(block.forward, backend=backend)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for block in model.transformer.h:
            block.forward = torch.compile(block.forward, backend=backend)
            fully_shard(block, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        if self.rank == 0:
            print(model)

        # Reuse the same input across iterations to avoid loss explosion from
        # trying to learn from random inputs
        torch.manual_seed(42 + self.rank)
        src = torch.randint(0, vocab_size, (4, 32), device="cuda")
        tgt = torch.randint(0, vocab_size, (4, 32), device="cuda")
        inp = (src, tgt)
        # Gradients drift apart on the 5th iteration for inductor backend
        num_iters = 8 if backend == "aot_eager" else 4
        for iter_idx in range(num_iters):
            if self.rank == 0:
                print(f"Iter index: {iter_idx}")
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(*inp).sum())
                losses[-1].backward()
            self.assertEqual(losses[0], losses[1])
            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)
            check_sharded_grad_parity(self, ref_model, model)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))


if __name__ == "__main__":
    run_tests()
