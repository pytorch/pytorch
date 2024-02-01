# Owner(s): ["oncall: distributed"]

import copy
import unittest
from typing import Iterable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import FSDP, fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


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
        return 4

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_forward(self):
        """Tests the parameter registration after forward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = MLP(3, device)
            # Since seed is per process, not per thread, we broadcast to ensure
            # the same parameters across ranks
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 3), device="cuda")
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)  # root does not reshard after forward
            self._assert_tensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = nn.Sequential(MLP(3, device), MLP(3, device))
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            fully_shard(model[0].in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model[0].out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)

            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)
            non_root_params = list(model[0].in_proj.parameters()) + list(
                model[0].out_proj.parameters()
            )
            root_params = list(set(model.parameters()) - set(non_root_params))
            if reshard_after_forward is False:
                self._assert_tensor_params(non_root_params)
            else:
                self._assert_dtensor_params(non_root_params)
            self._assert_tensor_params(root_params)
            self._assert_same_params(model.parameters(), ref_model.parameters())
            for module in model.modules():
                if isinstance(module, FSDP):
                    module.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_backward(self):
        """Tests the parameter registration after backward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            model = MLP(8, device)
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 8), device="cuda")
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            model = MLP(8, device)
            fully_shard(model.in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model.out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

    def _assert_tensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            self.assertNotIsInstance(param, DTensor)
            self.assertIsInstance(param, torch.Tensor)

    def _assert_dtensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            self.assertIsInstance(param, DTensor)

    def _assert_same_params(
        self, params: Iterable[nn.Parameter], ref_params: Iterable[nn.Parameter]
    ):
        params, ref_params = list(params), list(ref_params)
        self.assertEqual(len(params), len(ref_params))
        for param, ref_param in zip(params, ref_params):
            if isinstance(param, DTensor):
                param = param.full_tensor()
            self.assertEqual(param.shape, ref_param.shape)
            self.assertEqual(param, ref_param)


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
            nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
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
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(*inp).sum())
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
                "device_type": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    def _test_train_parity_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        device_type: str,
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
        assert device_type in ("cuda", "cpu"), f"{device_type}"
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model)
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mesh = init_device_mesh(device_type, (self.world_size,))
        for mlp in model:
            fully_shard(mlp, mesh=mesh, reshard_after_forward=reshard_after_forward)
        fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
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
                inp = torch.randn((8, lin_dim), device=torch.device(device_type))
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
    def test_non_root_forward_backward(self):
        """
        Tests running forward/backward through the root and then through a
        non-root. The non-root needs to synchronize streams/queue the callback.
        """
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((8, lin_dim), device=torch.device("cuda"))

        ref_root_loss = ref_model(inp).sum()
        ref_root_loss.backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad)
            param.grad.detach().div_(self.world_size)
        ref_optim.step()
        ref_optim.zero_grad()
        ref_nonroot_loss = ref_model[0](inp).sum()
        ref_nonroot_loss.backward()
        for param in ref_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.detach().div_(self.world_size)
        ref_optim.step()

        root_loss = model(inp).sum()
        root_loss.backward()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        optim.step()
        optim.zero_grad()
        nonroot_loss = model[0](inp).sum()
        nonroot_loss.backward()
        optim.step()

        self.assertEqual(ref_root_loss, root_loss)
        self.assertEqual(ref_nonroot_loss, nonroot_loss)
        self.assertEqual(ref_model(inp).sum(), model(inp).sum())

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


class TestFullyShardSharedParams(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_shared_params(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
            },
            self._test_train_shared_params,
        )

    def _test_train_shared_params(
        self,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
    ):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0, weight_tying=True)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if use_activation_checkpointing:
                    checkpoint(module)
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(10):
            inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
