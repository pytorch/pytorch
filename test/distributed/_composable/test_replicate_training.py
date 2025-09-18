# Owner(s): ["oncall: distributed"]

import copy
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
)
from torch.testing._internal.common_utils import run_tests, wrapSwapTensorsTest


c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional


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


if __name__ == "__main__":
    run_tests()
