# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import threading
import unittest
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.two_tensor import TwoTensor


def two_tensor_fsdp_pre_all_gather(
    self, mesh: DeviceMesh
) -> Tuple[Tuple[torch.Tensor, ...], Any]:
    all_gather_inputs = (self.a, self.b)
    metadata = None
    return all_gather_inputs, metadata


def two_tensor_fsdp_post_all_gather(
    self,
    all_gather_outputs: Tuple[torch.Tensor, ...],
    metadata: Any,
    param_dtype: torch.dtype,
    *,
    out: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], None]:
    assert metadata is None, f"{metadata}"
    a, b = all_gather_outputs
    if out is not None:
        assert isinstance(out, TwoTensor), f"{type(out)}"
        if a.dtype == param_dtype:
            assert a.untyped_storage().data_ptr() == out.a.untyped_storage().data_ptr()
            assert b.untyped_storage().data_ptr() == out.b.untyped_storage().data_ptr()
        else:
            assert out.a.dtype == param_dtype, f"{out.a.dtype} {param_dtype}"
            assert out.b.dtype == param_dtype, f"{out.b.dtype} {param_dtype}"
            out.a.copy_(a)
            out.b.copy_(b)
        return
    tensors_to_free = (a, b)
    # If the cast is real, then the all-gather outputs will not alias the
    # returned `TwoTensor`'s `a` and `b`
    two_tensor = TwoTensor(a, b).to(param_dtype)
    return two_tensor, tensors_to_free


class TestFullyShardAllGatherExtensionsCommon:
    @property
    def world_size(self) -> int:
        return 2

    @contextlib.contextmanager
    def _patch_two_tensor_fsdp_all_gather(self):
        lock = threading.Lock()
        TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather
        TwoTensor.fsdp_post_all_gather = two_tensor_fsdp_post_all_gather
        dist.barrier()
        try:
            yield
        finally:
            dist.barrier()
            with lock:  # only one thread needs to delete
                if hasattr(TwoTensor, "fsdp_pre_all_gather"):
                    delattr(TwoTensor, "fsdp_pre_all_gather")
                if hasattr(TwoTensor, "fsdp_post_all_gather"):
                    delattr(TwoTensor, "fsdp_post_all_gather")

    def _init_two_tensor_mlp(self) -> nn.Module:
        # Disable bias because the reference model will end up with a bias
        # gradient that is a `TwoTensor`, whereas the FSDP model does not
        model = nn.Sequential(*[MLP(8, bias=False) for _ in range(3)])
        for mlp in model:
            mlp.in_proj.weight = nn.Parameter(
                TwoTensor(mlp.in_proj.weight, mlp.in_proj.weight.clone())
            )
            mlp.out_proj.weight = nn.Parameter(
                TwoTensor(mlp.out_proj.weight, mlp.out_proj.weight.clone())
            )
        return model


class TestFullyShardAllGatherExtensionsMultiProcess(
    TestFullyShardAllGatherExtensionsCommon, FSDPTest
):
    @skip_if_lt_x_gpu(2)
    def test_all_gather_extensions_train_parity(self):
        with self._patch_two_tensor_fsdp_all_gather():
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_train_parity,
            )

    def _test_all_gather_extensions_train_parity(self, reshard_after_forward: bool):
        torch.manual_seed(42)
        model = self._init_two_tensor_mlp()
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=True)
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device="cuda")
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()
                if _model is ref_model:
                    for param_name, param in _model.named_parameters():
                        dist.all_reduce(param.grad)
                        param.grad.detach().div_(self.world_size)
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            check_sharded_parity(self, ref_model, model)


class TestFullyShardAllGatherExtensionsMultiThread(
    TestFullyShardAllGatherExtensionsCommon, FSDPTestMultiThread
):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_extensions_end_to_end(self):
        with self._patch_two_tensor_fsdp_all_gather():
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_end_to_end,
            )

    def _test_all_gather_extensions_end_to_end(self, reshard_after_forward: bool):
        # Check that we can run the meta-device initialization flow
        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("meta"))
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        model.to_empty(device=self.device)
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # Run a few iterations to check for errors
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device="cuda")
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_extensions_monkey_patch(self):
        # Define a pre/post-all-gather pair that quantizes to bf16 for the
        # all-gather and de-quantizes back to the parameter dtype
        def fsdp_pre_all_gather(self) -> Tuple[Tuple[torch.Tensor, ...], Any]:
            return (self.to(torch.bfloat16),), None

        def fsdp_post_all_gather(
            self,
            all_gather_outputs: Tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
        ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], None]:
            (tensor,) = all_gather_outputs
            assert metadata is None, f"{metadata}"
            assert tensor.dtype == torch.bfloat16, f"{tensor.dtype}"
            if out is not None:
                out.copy_(tensor)
                return
            return tensor.to(param_dtype), (tensor,)

        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        model.to_empty(device=self.device)
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        # Monkey patch the pre/post-all-gather functions *after* `to_empty()`
        # since the local tensor objects change from materialization
        self.assertGreater(sum("weight" in n for n, _ in model.named_parameters()), 0)
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                local_param = param.to_local()
                # Monkey patch on the `torch.Tensor` to show that the extension
                # can work even without a subclass
                local_param.fsdp_pre_all_gather = fsdp_pre_all_gather
                local_param.fsdp_post_all_gather = fsdp_post_all_gather
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # Run a few iterations to check for errors
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device="cuda")
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()


if __name__ == "__main__":
    run_tests()
