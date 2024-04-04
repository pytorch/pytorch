# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import threading
import unittest
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTestMultiThread,
    MLP,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.two_tensor import TwoTensor


def two_tensor_fsdp_pre_all_gather(self):
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
):
    assert metadata is None, f"{metadata}"
    a, b = all_gather_outputs
    if out is not None:
        assert isinstance(out, TwoTensor), f"{type(out)}"
        assert a.untyped_storage().data_ptr() == out.a.untyped_storage().data_ptr()
        assert b.untyped_storage().data_ptr() == out.b.untyped_storage().data_ptr()
        return
    tensors_to_free = (a, b)
    return TwoTensor(a, b), tensors_to_free


class TestFullyShardAllGatherExtensions(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")

    @contextlib.contextmanager
    def patch_two_tensor_fsdp_all_gather(self):
        lock = threading.Lock()
        TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather
        TwoTensor.fsdp_post_all_gather = two_tensor_fsdp_post_all_gather
        dist.barrier()
        torch.cuda.synchronize()
        try:
            yield
        finally:
            with lock:  # only one thread needs to delete
                if hasattr(TwoTensor, "fsdp_pre_all_gather"):
                    delattr(TwoTensor, "fsdp_pre_all_gather")
                if hasattr(TwoTensor, "fsdp_post_all_gather"):
                    delattr(TwoTensor, "fsdp_post_all_gather")

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_extensions(self):
        with self.patch_two_tensor_fsdp_all_gather():
            self._test_all_gather_extensions()

    def _test_all_gather_extensions(self):
        torch.manual_seed(42)
        model = MLP(8)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        model.in_proj.weight = nn.Parameter(
            TwoTensor(model.in_proj.weight, model.in_proj.weight.clone())
        )
        model.out_proj.weight = nn.Parameter(
            TwoTensor(model.out_proj.weight, model.out_proj.weight.clone())
        )
        self.assertTrue(model.in_proj.weight.requires_grad)
        self.assertTrue(model.out_proj.weight.requires_grad)
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=True)
        fully_shard(model.in_proj)
        fully_shard(model.out_proj)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

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


if __name__ == "__main__":
    run_tests()
