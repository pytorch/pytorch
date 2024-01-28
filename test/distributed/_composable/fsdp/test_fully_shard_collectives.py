# Owner(s): ["oncall: distributed"]

import itertools
import unittest
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp._fsdp_collectives import (
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce_scatter,
)
from torch.distributed._composable.fsdp._fsdp_common import (
    _chunk_with_empty,
    FSDPMeshInfo,
)
from torch.distributed._composable.fsdp._fsdp_init import _init_default_fully_shard_mesh
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests


class TestFullyShardCollectives(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 128

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0")

    def _get_param_sizes(self) -> List[torch.Size]:
        # For world size 128, the fp32 all-gather and reduce-scatter testing
        # requires ~0.22 GB
        return [
            torch.Size([17, 257]),
            torch.Size([17]),
            torch.Size([64, 312]),
            torch.Size([64]),
            torch.Size([64, 64]),
            torch.Size([512, 64]),
            torch.Size([256]),
            torch.Size([64, 297]),
        ]

    def _init_params(self, param_sizes: List[torch.Size]) -> List[nn.Parameter]:
        torch.manual_seed(42)
        orig_params = [
            nn.Parameter(torch.randn(size, device=self.device)) for size in param_sizes
        ]
        # Since seed is per process, not per thread, we broadcast to ensure the
        # same original parameters across ranks
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        return orig_params

    def _init_fsdp_param_group(self, params: List[nn.Parameter]):
        module = nn.ParameterList([param.detach().clone() for param in params])
        mesh_info = FSDPMeshInfo(_init_default_fully_shard_mesh(), shard_mesh_dim=0)
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()), module, mesh_info, mesh_info, self.device
        )
        return fsdp_param_group

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.cuda.current_stream()
        stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
        for async_op, all_gather_copy_in_stream, all_gather_stream in itertools.product(
            (False, True),
            (default_stream, stream1),
            (default_stream, stream2),
        ):
            if (
                all_gather_copy_in_stream is default_stream
                and all_gather_stream is not default_stream
            ) or (
                all_gather_stream is default_stream
                and all_gather_copy_in_stream is not default_stream
            ):
                # Only test both default or both non-default to save test time
                continue
            self._test_all_gather(
                param_sizes,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
                all_gather_dtype=torch.float32,
            )

    def _test_all_gather(
        self,
        param_sizes: List[torch.Size],
        async_op: bool,
        all_gather_copy_in_stream: torch.cuda.Stream,
        all_gather_stream: torch.cuda.Stream,
        all_gather_dtype: torch.dtype,
    ):
        # - Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params)
        fsdp_params = fsdp_param_group.fsdp_params
        module = fsdp_param_group.module

        # - Sanity check that the parameter sharding is as expected
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, DTensor))
            orig_param_chunk = _chunk_with_empty(orig_param, self.world_size, dim=0)[
                self.rank
            ]
            if param._local_tensor.numel() > 0:
                self.assertEqual(param._local_tensor, orig_param_chunk)
            else:
                self.assertEqual(orig_param_chunk.numel(), 0)

        # - Run the foreach all-gather (including copy-in and copy-out)
        all_gather_result = foreach_all_gather(
            fsdp_params,
            fsdp_param_group.mesh_info.shard_process_group,
            async_op=async_op,
            all_gather_copy_in_stream=all_gather_copy_in_stream,
            all_gather_stream=all_gather_stream,
            device=self.device,
            dtype=torch.float32,
        )
        foreach_all_gather_copy_out(
            all_gather_result,
            fsdp_params,
            fsdp_param_group.mesh_info.shard_process_group,
        )
        # Transition to unsharded state to register unsharded parameters
        for fsdp_param in fsdp_params:
            fsdp_param.init_unsharded_param()
            fsdp_param.to_unsharded()

        # - Check all-gather correctness
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, torch.Tensor))
            self.assertTrue(isinstance(param, nn.Parameter))
            self.assertEqual(param, orig_param.to(param.dtype))

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_reduce_scatter_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        for reduce_scatter_stream in (default_stream, stream):
            self._test_reduce_scatter(
                param_sizes,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_dtype=torch.float32,
            )

    def _test_reduce_scatter(
        self,
        param_sizes: List[torch.Size],
        reduce_scatter_stream: torch.cuda.Stream,
        reduce_scatter_dtype: torch.dtype,
    ):
        # - Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params)
        fsdp_params = fsdp_param_group.fsdp_params
        fsdp_param_group.comm_ctx.init()

        # - Run one unshard to initialize metadata
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        fsdp_param_group.reshard()

        # - Run the foreach reduce-scatter (including copy-in and view-out)
        torch.manual_seed(42)
        unsharded_grads = [torch.ones_like(param) * self.rank for param in orig_params]
        group = fsdp_param_group.mesh_info.shard_process_group
        self.assertEqual(group.size(), self.world_size)
        view_out_event = foreach_reduce_scatter(
            fsdp_params,
            unsharded_grads,
            group,
            reduce_scatter_stream,
            orig_dtype=orig_params[0].dtype,
            reduce_dtype=reduce_scatter_dtype,
            device=self.device,
            predivide_factor=fsdp_param_group._grad_predivide_factor,
            postdivide_factor=fsdp_param_group._grad_postdivide_factor,
        )
        torch.cuda.current_stream().wait_event(view_out_event)

        # - Check reduce-scatter correctness
        reduced_grads = [grad.detach().clone() for grad in unsharded_grads]
        for grad in reduced_grads:
            dist.all_reduce(grad, group=group)
            grad /= self.world_size
        for fsdp_param, reduced_grad in zip(fsdp_params, reduced_grads):
            sharded_grad = fsdp_param.sharded_param.grad
            reduced_grad_chunk = _chunk_with_empty(
                reduced_grad, self.world_size, dim=0
            )[self.rank]
            if reduced_grad_chunk.numel() > 0:
                self.assertIsInstance(sharded_grad, DTensor)
                self.assertEqual(sharded_grad._local_tensor, reduced_grad_chunk)
            else:  # pure padding
                self.assertIsNone(sharded_grad)


if __name__ == "__main__":
    run_tests()
