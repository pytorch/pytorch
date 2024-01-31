# Owner(s): ["oncall: distributed"]

import copy
import unittest
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import run_tests


class TestFullyShardForwardInputs(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_root_move_forward_input_to_device(self):
        device = torch.device("cuda", 0)

        class ParamlessModule(nn.Module):
            def forward(self, x: torch.Tensor, ys: Tuple[torch.Tensor, ...]):
                # Check that FSDP moved the inputs to GPU, including recursing
                # into the tuple data structure
                assert x.device == device, f"Expects {device} but got {x.device}"
                assert (
                    ys[0].device == device
                ), f"Expects {device} but got {ys[0].device}"
                assert (
                    ys[1].device == device
                ), f"Expects {device} but got {ys[1].device}"
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule()
        fully_shard(model)
        x = torch.randn((3,))
        ys = (torch.randn((3,)), torch.randn((3,)))
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(ys[0].device, torch.device("cpu"))
        self.assertEqual(ys[1].device, torch.device("cpu"))
        model(x, ys)


class TestFullyShardRegisteredParams(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_sharded_params_registered_after_forward(self):
        """Tests that the sharded parameters are registered after forward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        model = MLP(8, device)
        fully_shard(model)
        inp = torch.randn((2, 8), device="cuda")
        self._assert_dtensor_params(model)
        model(inp)
        self._assert_dtensor_params(model)

        # Multiple FSDP groups
        model = MLP(8, device)
        fully_shard(model.in_proj)
        fully_shard(model.out_proj)
        fully_shard(model)
        self._assert_dtensor_params(model)
        model(inp)
        self._assert_dtensor_params(model)

    def _assert_dtensor_params(self, module: nn.Module):
        for param in module.parameters():
            self.assertIsInstance(param, DTensor)


class TestFullyShard1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_single_group(self):
        """Tests train parity with DDP for a single FSDP group."""
        self.run_subtests(
            {
                "lin_shapes": [[(16, 15), (15, 8)], [(7, 15), (15, 3)]],
            },
            self._test_train_parity_single_group,
        )

    def _test_train_parity_single_group(self, lin_shapes: List[Tuple[int, int]]):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0], bias=True),
            nn.ReLU(),
            nn.Linear(*lin_shapes[1], bias=True),
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        inp = (torch.randn((4, lin_shapes[0][0]), device="cuda"),)
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                # TODO: Test forward only for now
                losses.append(_model(*inp).sum())
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
