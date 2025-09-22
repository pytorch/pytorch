# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import itertools
import unittest
from collections.abc import Iterable
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, OffloadPolicy
from torch.distributed.tensor import DTensor, init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    compiled_fsdp_test,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TEST_HPU,
    wrapSwapTensorsTest,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional

from torch.testing._internal.common_fsdp import get_devtype


device_type = torch.device(get_devtype())


class TestReplicateForwardInputs(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(1)
    def test_root_move_forward_input_to_device(self):
        device = torch.device(device_type.type, 0)

        class ParamlessModule(nn.Module):
            def forward(self, x: torch.Tensor, ys: tuple[torch.Tensor, ...]):
                # Check that Replicate moved the inputs to GPU, including recursing
                # into the tuple data structure
                assert x.device == device, f"Expects {device} but got {x.device}"
                assert ys[0].device == device, (
                    f"Expects {device} but got {ys[0].device}"
                )
                assert ys[1].device == device, (
                    f"Expects {device} but got {ys[1].device}"
                )
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule().to(device)
        replicate(model).to(device)
        x = torch.randn((3,))
        ys = (torch.randn((3,)), torch.randn((3,)))
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(ys[0].device, torch.device("cpu"))
        self.assertEqual(ys[1].device, torch.device("cpu"))
        model(x, ys)


class TestReplicateRegisteredParams(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(1)
    def test_param_registration_after_forward(self):
        """Tests the parameter registration after forward."""
        device = torch.device(device_type.type, 0)
        # Single Replicate group
        for reshard_after_forward in (True, False, None):
            torch.manual_seed(42)
            model = MLP(3, device)
            # Since seed is per process, not per thread, we broadcast to ensure
            # the same parameters across ranks
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            replicate(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 3), device=device_type.type)
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)
            if reshard_after_forward:
                self._assert_dtensor_params(model.parameters())
            else:
                self._assert_tensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

        # Multiple Replicate groups
        for reshard_after_forward in (True, False, None):
            torch.manual_seed(42)
            model = nn.Sequential(MLP(3, device), MLP(3, device))
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            replicate(model[0].in_proj, reshard_after_forward=reshard_after_forward)
            replicate(model[0].out_proj, reshard_after_forward=reshard_after_forward)
            replicate(model, reshard_after_forward=reshard_after_forward)

            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)
            non_root_params = list(model[0].in_proj.parameters()) + list(
                model[0].out_proj.parameters()
            )
            root_params = list(set(model.parameters()) - set(non_root_params))
            if reshard_after_forward is None:
                self._assert_dtensor_params(non_root_params)
                self._assert_tensor_params(root_params)
            elif reshard_after_forward:
                self._assert_dtensor_params(non_root_params)
                self._assert_dtensor_params(root_params)
            else:
                self._assert_tensor_params(non_root_params)
                self._assert_tensor_params(root_params)
            self._assert_same_params(model.parameters(), ref_model.parameters())
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

    @skip_if_lt_x_gpu(1)
    def test_param_registration_after_backward(self):
        """Tests the parameter registration after backward."""
        device = torch.device(device_type.type, 0)
        # Single Replicate group
        for reshard_after_forward in (True, False):
            model = MLP(8, device)
            replicate(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 8), device=device_type.type)
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

        # Multiple Replicate groups
        for reshard_after_forward in (True, False):
            model = MLP(8, device)
            replicate(model.in_proj, reshard_after_forward=reshard_after_forward)
            replicate(model.out_proj, reshard_after_forward=reshard_after_forward)
            replicate(model, reshard_after_forward=reshard_after_forward)
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

    def _assert_tensor_params(self, params: Iterable[nn.Parameter]):
        # need to iterate over the list multiple times
        params = list(params)
        self.assertGreater(len(params), 0)
        for param in params:
            self.assertNotIsInstance(param, DTensor)
            self.assertIsInstance(param, torch.Tensor)

    def _assert_dtensor_params(self, params: Iterable[nn.Parameter]):
        params = list(params)
        self.assertGreater(len(params), 0)
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


class TestReplicateCastAfterInit(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(1)
    @wrapSwapTensorsTest(True)
    def test_to_float64_after_init(self):
        """Tests that the user can cast the module to float64 after init."""
        # NOTE: Test fp64 instead of a lower precision dtype like bf16 for
        # better numerics. The important part is changing the dtype.

        torch.manual_seed(42)
        mlp_dim, device, dtype = 4, device_type, torch.float64
        model = MLP(mlp_dim, device=device)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model).to(dtype)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in (model.in_proj, model.out_proj, model):
            replicate(module)
        model.to(dtype)
        for param in model.parameters():
            self.assertEqual(param.dtype, dtype)
            self.assertEqual(param.to_local().dtype, dtype)
            self.assertEqual(param._spec.tensor_meta.dtype, dtype)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, mlp_dim), device=device_type.type, dtype=dtype)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for param in model.parameters():
                self.assertEqual(param.dtype, dtype)
                self.assertEqual(param.to_local().dtype, dtype)
                self.assertEqual(param._spec.tensor_meta.dtype, dtype)
                self.assertEqual(param.grad.dtype, dtype)
                self.assertEqual(param.grad.to_local().dtype, dtype)
                self.assertEqual(param.grad._spec.tensor_meta.dtype, dtype)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))


class TestReplicate1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_single_group(self):
        """
        Tests train parity with DDP for a single FSDP group when sharding
        parameters on dim-0.
        """
        self.run_subtests(
            {
                "lin_shapes": [
                    [(16, 15), (15, 8)],
                    [(7, 15), (15, 3)],
                    [(16, 17), (17, 8)],
                ],
                "use_shard_placement_fn": [False],
            },
            self._test_train_parity_single_group,
        )

    def _test_train_parity_single_group(
        self, lin_shapes: list[tuple[int, int]], use_shard_placement_fn: bool
    ):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
        )
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        inp = (torch.randn((4, lin_shapes[0][0]), device=device_type.type),)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(*inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep kernel not supported for HPU")
    @compiled_fsdp_test(compile_compute_on_module=Transformer)
    def test_train_parity_multi_groups(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "test_device_type": [device_type.type],
                "offload_policy": [OffloadPolicy()],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
                "unshard_async_op": [False],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "sleep kernel not supported on HPU")
    def test_train_parity_multi_group_cpu_offload_eager(self):
        """
        Tests train parity when using multiple parameter groups for
        communication and CPU offloading.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True],  # save CI time
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
                "test_device_type": [device_type.type],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
                "unshard_async_op": [False],
            },
            self._test_train_parity_multi_group,
        )

    def _test_train_parity_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        test_device_type: str,
        delay_after_forward: bool,
        delay_before_all_gather: bool,
        delay_before_reduce_scatter: bool,
        delay_before_optim: bool,
        unshard_async_op: bool,
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
        assert test_device_type in ("cuda", "hpu", "xpu", "cpu"), f"{test_device_type}"
        torch.manual_seed(42)
        vocab_size = 1024
        model_args = ModelArgs(
            n_layers=3,
            n_heads=4,
            vocab_size=vocab_size,
            max_seq_len=64,
            dropout_p=0,
        )
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mesh = init_device_mesh(
            test_device_type,
            (self.world_size, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        fully_shard_fn = functools.partial(
            replicate,
            device_mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
        fully_shard_fn(model)
        if unshard_async_op:
            model._set_unshard_async_op(unshard_async_op)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        delay_in_ms = 100
        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.get_device_module(device_type)._sleep(
                int(delay_in_ms * get_cycles_per_ms())
            )
            return orig_all_gather(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            torch.get_device_module(device_type)._sleep(
                int(delay_in_ms * get_cycles_per_ms())
            )
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
        patch_all_gather_ctx = (
            patch_all_gather(delayed_all_gather)
            if delay_before_all_gather
            else contextlib.nullcontext()
        )
        patch_reduce_scatter_ctx = (
            patch_reduce_scatter(delayed_reduce_scatter)
            if delay_before_reduce_scatter
            else contextlib.nullcontext()
        )
        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                inp = torch.randint(0, vocab_size, (3, 64), device=device_type)
                losses: list[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    losses.append(_model(inp).sum())
                    if _model is model and delay_after_forward:
                        torch.get_device_module(device_type)._sleep(
                            int(delay_in_ms * get_cycles_per_ms())
                        )
                    losses[-1].backward()
                    if _model is model and delay_before_optim:
                        torch.get_device_module(device_type)._sleep(
                            int(delay_in_ms * get_cycles_per_ms())
                        )

                for param in ref_model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad)
                        param.grad.div_(self.world_size)

                for _optim in (ref_optim, optim):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
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
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            replicate(mlp)
        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((8, lin_dim), device=device_type)

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
        torch.get_device_module(device_type)._sleep(int(100 * get_cycles_per_ms()))
        optim.step()
        optim.zero_grad()
        nonroot_loss = model[0](inp).sum()
        nonroot_loss.backward()
        optim.step()

        self.assertEqual(ref_root_loss, root_loss)
        self.assertEqual(ref_nonroot_loss, nonroot_loss)
        self.assertEqual(ref_model(inp).sum(), model(inp).sum())

    @skip_if_lt_x_gpu(2)
    def test_multi_forward_module(self):
        """
        Tests parity when running a module that participates multiple
        times in forward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False]},
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
        model = MultiForwardModule(device=device_type.type)
        ref_model = copy.deepcopy(model).to(device_type)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        replicate(model.inner)
        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        inp = torch.randn((32, 4), device=device_type.type)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                _optim.step()

            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_explicit_prefetching(self):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=8, dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)

        for layer in itertools.chain(model.layers, [model]):
            replicate(layer)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        num_to_forward_prefetch = num_to_backward_prefetch = 2
        for i, layer in enumerate(model.layers):
            if i >= len(model.layers) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [
                model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
            ]
            layer.set_modules_to_forward_prefetch(layers_to_prefetch)
        for i, layer in enumerate(model.layers):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [
                model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
            ]
            layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device=device_type.type)
        for _ in range(10):
            losses: list[torch.Tensor] = []

            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad()
                _optim.step()

            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_post_optim_event(self):
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type.type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for layer in itertools.chain(model.layers, [model]):
            replicate(layer)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        def step_post_hook(
            fsdp_module: FSDPModule, opt: torch.optim.Optimizer, args, kwargs
        ) -> None:
            post_optim_event = (
                torch.get_device_module(device_type).current_stream().record_event()
            )
            fsdp_module.set_post_optim_event(post_optim_event)

        optim.register_step_post_hook(functools.partial(step_post_hook, model))

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device=device_type.type)
        # Track all losses and check for equality at the end to avoid a CPU
        # sync point after each iteration
        ref_losses: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for _ in range(10):
            ref_optim.zero_grad()
            ref_losses.append(ref_model(inp).sum())
            ref_losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            ref_optim.step()
        for _ in range(10):
            optim.zero_grad()
            losses.append(model(inp).sum())
            losses[-1].backward()
            optim.step()
            # Sleep after the optimizer step to allow CPU to run ahead into the
            # next iteration's forward, exercising the post-optim stream sync
            torch.get_device_module(device_type)._sleep(int(25 * get_cycles_per_ms()))
        for ref_loss, loss in zip(ref_losses, losses):
            self.assertEqual(ref_loss, loss)


if __name__ == "__main__":
    run_tests()
