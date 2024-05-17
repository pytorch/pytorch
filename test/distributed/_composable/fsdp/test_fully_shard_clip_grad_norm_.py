# Owner(s): ["oncall: distributed"]

import copy
import functools
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import Shard
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class _TestClipGradNormBase(FSDPTest):
    def _test_clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        ref_model: nn.Module,
        ref_optim: torch.optim.Optimizer,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        dp_mesh: Optional[DeviceMesh] = None,
    ):
        vector_norm_fn = functools.partial(torch.linalg.vector_norm, ord=norm_type)
        dp_mesh = dp_mesh or init_device_mesh("cuda", (self.world_size,))
        torch.manual_seed(42 + dp_mesh.get_local_rank() + 1)
        inp = torch.randint(0, model.model_args.vocab_size, (3, 16), device="cuda")
        for iter_idx in range(10):
            ref_optim.zero_grad()
            ref_model(inp).sum().backward()
            optim.zero_grad()
            model(inp).sum().backward()

            ref_grads = [p.grad.detach().clone() for p in ref_model.parameters()]
            local_grads = [
                p.grad.to_local().detach().clone() for p in model.parameters()
            ]
            for ref_grad, param in zip(ref_grads, model.parameters()):
                # TODO: Skip the check for the parameters since FSDP needs
                # strided sharding for it to work with `full_tensor`
                if tuple(param.placements) == (Shard(0), Shard(0)):
                    continue
                self.assertEqual(ref_grad, param.grad.full_tensor())

            # Check that all gradients have norm greater than the max norm
            # before clipping to ensure the clipping is not vacuous
            self.assertTrue(all(vector_norm_fn(g).item() > max_norm for g in ref_grads))
            self.assertTrue(
                all(vector_norm_fn(g).item() > max_norm for g in local_grads)
            )

            # Check gradient norm clipping via total norm and individual
            # gradient norms post-clipping
            ref_total_norm = torch.nn.utils.clip_grad_norm_(
                ref_model.parameters(), max_norm=max_norm, norm_type=norm_type
            )
            comm_mode = CommDebugMode()
            with comm_mode:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_norm,
                    norm_type=norm_type,
                    foreach=True,
                )
            self.assertEqual(ref_total_norm, total_norm)
            # Expect one all-reduce per mesh dim for partial -> replicate
            expected_all_reduces = len(total_norm.placements)
            self.assertEqual(
                comm_mode.get_comm_counts()[torch.ops.c10d_functional.all_reduce],
                expected_all_reduces,
            )
            # For zero gradients, clipping has no effect
            for param, grad in zip(ref_model.parameters(), ref_grads):
                self.assertTrue(vector_norm_fn(param.grad).item() <= max_norm)
                if torch.count_nonzero(grad):
                    self.assertFalse(torch.equal(param.grad, grad))
            for param, grad in zip(model.parameters(), local_grads):
                self.assertTrue(
                    vector_norm_fn(param.grad.to_local()).item() <= max_norm
                )
                if torch.count_nonzero(grad):
                    self.assertFalse(torch.equal(param.grad.to_local(), grad))


class TestClipGradNormWorldSize2(_TestClipGradNormBase):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_clip_grad_norm_1d(self):
        for norm_type in (2, 1, float("inf")):
            torch.manual_seed(42)
            model_args = ModelArgs(dropout_p=0.0)
            model = Transformer(model_args)
            ref_model = replicate(copy.deepcopy(model).cuda())
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    fully_shard(module)
            fully_shard(model)
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            self._test_clip_grad_norm(1, norm_type, ref_model, ref_optim, model, optim)


class TestClipGradNormWorldSize4(_TestClipGradNormBase):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    @skip_if_lt_x_gpu(4)
    def test_clip_grad_norm_2d(self):
        for norm_type in (2, 1, 3, float("inf")):
            dp_size = 2
            global_mesh = init_device_mesh(
                "cuda",
                (dp_size, self.world_size // dp_size),
                mesh_dim_names=("dp", "tp"),
            )
            dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
            torch.manual_seed(42)
            model_args = ModelArgs(dropout_p=0.0)
            model = Transformer(model_args)
            ref_model = replicate(
                copy.deepcopy(model).cuda(), process_group=dp_mesh.get_group()
            )
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            model = Transformer.parallelize(model, tp_mesh, use_seq_parallel=True)
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    fully_shard(module, mesh=dp_mesh)
            fully_shard(model, mesh=dp_mesh)
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            self._test_clip_grad_norm(
                1, norm_type, ref_model, ref_optim, model, optim, dp_mesh
            )


if __name__ == "__main__":
    run_tests()
