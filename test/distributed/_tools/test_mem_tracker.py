# Owner(s): ["module: unknown"]
import gc
import unittest
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributed._tools.mem_tracker import MemTracker
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils.checkpoint import checkpoint


class TestMemTracker(TestCase):
    def _init_cublas_workspace(self, dev: torch.device):
        lin = torch.nn.Linear(768, 768, device=dev)
        inp = torch.randn(1, 768, device=dev)
        lin(inp).sum().backward()
        del lin
        del inp

    def _reset_mem_stats(self, dev: torch.device):
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_cuda_tracker_equivalence(
        self,
    ):
        """
        Tests that the tracker correctly calculates the peak memory.
        """
        dev = torch.device(torch.cuda.current_device())
        self._init_cublas_workspace(dev)
        gc.collect(1)
        self._reset_mem_stats(dev)
        mem_stats = torch.cuda.memory_stats(dev)
        pre_cuda_active = mem_stats["active_bytes.all.current"]
        bsz, n_layers, dim, dtype = 16, 4, 512, torch.bfloat16

        class DummyModel(nn.Module):
            def __init__(self, n_layers: int, dim: int, dtype: torch.dtype):
                super().__init__()
                self.linears = nn.ModuleList()
                for _ in range(n_layers):
                    self.linears.append(nn.Linear(dim, dim, dtype=dtype))
                    self.linears.append(nn.ReLU())

            def forward(self, x):
                for layer in self.linears:
                    x = layer(x)
                return x

        with torch.device(dev):
            model = DummyModel(n_layers, dim, dtype=dtype)
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        input_batch = torch.randn(bsz, dim, device=dev, dtype=dtype)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim, input_batch)
        with mem_tracker as mt:
            for iter_idx in range(2):
                model(input_batch).sum().backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()
        # Check for accuracy of peak memory

        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
        mem_stats = torch.cuda.memory_stats(dev)
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        accuracy = tracker_max / cuda_max
        self.assertAlmostEqual(accuracy, 1.0, delta=0.1)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_tracker_with_activation_checkpointing(
        self,
    ):
        """
        Tests that the tracker correctly computes the peak memory during activation checkpointing.
        """
        dev = torch.device(torch.cuda.current_device())
        self._init_cublas_workspace(dev)
        gc.collect(1)
        self._reset_mem_stats(dev)
        mem_stats = torch.cuda.memory_stats(dev)
        pre_cuda_active = mem_stats["active_bytes.all.current"]

        bsz, n_layers, dim, dtype = 128, 4, 1024, torch.float16

        class MLPBlock(nn.Module):
            def __init__(self, dim: int, dtype: torch.dtype):
                super().__init__()
                self.mlp_block = nn.Sequential(
                    nn.Linear(dim, 2 * dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(2 * dim, dim, dtype=dtype),
                )

            def forward(self, x):
                return self.mlp_block(x)

        class MyModule(nn.Module):
            def __init__(
                self, n_layers: int, dim: int, dtype: torch.dtype, use_ac: bool = False
            ):
                super().__init__()
                self.mlp_blocks = nn.ModuleList()
                self.use_ac = use_ac
                for _ in range(n_layers):
                    self.mlp_blocks.append(MLPBlock(dim, dtype=dtype))

            def forward(self, x):
                for i, block in enumerate(self.mlp_blocks):
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                return x

        with torch.device(dev):
            model = MyModule(n_layers, dim, dtype, True)
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim)
        with mem_tracker as mt:
            input_batch = torch.randn(bsz, dim, dim, device=dev, dtype=dtype)
            for iter_idx in range(2):
                model(input_batch).sum().backward()
                optim.step()
                optim.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()

        # Check for accuracy of peak memory
        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
        mem_stats = torch.cuda.memory_stats(dev)
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        accuracy = tracker_max / cuda_max
        self.assertAlmostEqual(accuracy, 1.0, delta=0.1)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    def test_tracker_attribution(self):
        """
        Tests that the tracker correctly categorizes params, gradients, and optimizer states.
        """
        dev = torch.device(torch.get_default_device())
        gc.collect(1)
        bsz, n_layers, dim, dtype = 16, 3, 128, torch.float32

        def get_param_grad_optstate_actual_bytes(
            model: nn.Module, opt: torch.optim.Optimizer
        ) -> Tuple[int, int, int]:
            param_bytes = 0
            grad_bytes = 0
            opt_state_bytes = 0
            for param in model.parameters():
                if param.device == dev:
                    param_bytes += param.numel() * param.element_size()
                if param.grad is not None and param.grad.device == dev:
                    grad_bytes += param.grad.numel() * param.grad.element_size()

            for state in opt.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor) and v.device == dev:
                        opt_state_bytes += v.numel() * v.element_size()
            return param_bytes, grad_bytes, opt_state_bytes

        def get_param_grad_optstate_bytes_from_tracker(
            tracker: MemTracker,
        ) -> Tuple[int, int, int]:
            snapshot = tracker.get_tracker_snapshot()
            param_bytes = snapshot[dev]["Parameter"]
            grad_bytes = snapshot[dev]["Gradient"]
            opt_state_bytes = snapshot[dev]["Optstate"]
            return param_bytes, grad_bytes, opt_state_bytes

        def test_attribution_equivalence(
            mt: MemTracker,
            model: nn.Module,
            opt: torch.optim.Optimizer,
        ) -> None:
            actual = get_param_grad_optstate_actual_bytes(model, opt)
            tracker = get_param_grad_optstate_bytes_from_tracker(mt)
            for a, b in zip(actual, tracker):
                if a == 0:
                    self.assertEqual(b, 0)
                else:
                    self.assertAlmostEqual(b / a, 1.0, delta=0.1)

        class DummyModel(nn.Module):
            def __init__(self, n_layers: int, dim: int, dtype: torch.dtype):
                super().__init__()
                self.MLP_layers = nn.ModuleList()
                for _ in range(n_layers):
                    self.MLP_layers.extend(
                        [nn.Linear(dim, 2 * dim, dtype=dtype), nn.GELU()]
                    )
                    self.MLP_layers.extend(
                        [nn.Linear(2 * dim, dim, dtype=dtype), nn.GELU()]
                    )

            def forward(self, x):
                for layer in self.MLP_layers:
                    x = layer(x)
                return x

        with torch.device(dev):
            model = DummyModel(n_layers, dim, dtype=dtype)
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optim)
        with mem_tracker as mt:
            input_batch = torch.randn(bsz, dim, device=dev, dtype=dtype)
            # Before forward: Only parameters and input are allocated
            test_attribution_equivalence(mt, model, optim)
            output = model(input_batch)
            output.sum().backward()
            # After backward: Gradients are allocated
            test_attribution_equivalence(mt, model, optim)
            output = None
            optim.step()
            # After step: Optimizer state is allocated
            test_attribution_equivalence(mt, model, optim)
            optim.zero_grad()
            # After zero_grad: Gradients are deallocated
            test_attribution_equivalence(mt, model, optim)


if __name__ == "__main__":
    run_tests()
