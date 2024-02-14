# Owner(s): ["oncall: distributed"]

import itertools
import unittest
from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp._fsdp_collectives import (
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce_scatter,
)
from torch.distributed._composable.fsdp._fsdp_common import FSDPMeshInfo
from torch.distributed._composable.fsdp._fsdp_init import (
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
)
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

    def _init_fsdp_param_group(
        self, params: List[nn.Parameter], reshard_after_forward: Union[bool, int]
    ):
        module = nn.ParameterList([param.detach().clone() for param in params])
        mesh_info = FSDPMeshInfo(_init_default_fully_shard_mesh(), shard_mesh_dim=0)
        post_forward_mesh_info = _get_post_forward_mesh_info(
            reshard_after_forward, mesh_info
        )
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()),
            module,
            mesh_info,
            post_forward_mesh_info,
            self.device,
            MixedPrecisionPolicy(),
        )
        return fsdp_param_group

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_all_gather_fp32(self):
        param_sizes = self._get_param_sizes()
        default_stream = torch.cuda.current_stream()
        stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
        for async_op, streams, reshard_after_forward in itertools.product(
            (False, True),
            ((default_stream, default_stream), (stream1, stream2)),
            (True, 8),
        ):
            all_gather_copy_in_stream, all_gather_stream = streams
            # Save test time by only testing reshard after forward as an int
            # for non-async and non-default streams (like in pre-backward)
            if type(reshard_after_forward) is int and (
                async_op or all_gather_stream is default_stream
            ):
                continue
            self._test_all_gather(
                param_sizes,
                reshard_after_forward=reshard_after_forward,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
            )

    def _test_all_gather(
        self,
        param_sizes: List[torch.Size],
        reshard_after_forward: Union[bool, int],
        async_op: bool,
        all_gather_copy_in_stream: torch.cuda.Stream,
        all_gather_stream: torch.cuda.Stream,
    ):
        def all_gather(fsdp_param_group: FSDPParamGroup, group: dist.ProcessGroup):
            all_gather_result = foreach_all_gather(
                fsdp_param_group.fsdp_params,
                group,
                async_op=async_op,
                all_gather_copy_in_stream=all_gather_copy_in_stream,
                all_gather_stream=all_gather_stream,
                device=self.device,
            )
            foreach_all_gather_copy_out(all_gather_result, fsdp_params, group)
            # Transition to unsharded state to register unsharded parameters
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.init_unsharded_param()
            fsdp_param_group._to_unsharded()

        def check_all_gathered_params(
            orig_params: List[nn.Parameter], module: nn.Module
        ):
            for orig_param, param in zip(orig_params, module.parameters()):
                self.assertIsInstance(param, torch.Tensor)
                self.assertIsInstance(param, nn.Parameter)
                self.assertEqual(param, orig_param.to(param.dtype))

        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(
            orig_params, reshard_after_forward
        )
        fsdp_params = fsdp_param_group.fsdp_params
        module = fsdp_param_group.module

        # Sanity check that the parameter sharding is as expected
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, DTensor))
            self.assertEqual(param.full_tensor(), orig_param)

        # Run the foreach all-gather (including copy-in and copy-out)
        all_gather(fsdp_param_group, fsdp_param_group.mesh_info.shard_process_group)

        # Check all-gather correctness
        check_all_gathered_params(orig_params, module)

        # For reshard after after forward as an int, further test emulating the
        # pre-backward all-gather
        if type(reshard_after_forward) is not int:
            return
        fsdp_param_group._to_sharded_post_forward()
        all_gather(
            fsdp_param_group,
            fsdp_param_group.post_forward_mesh_info.shard_process_group,
        )
        check_all_gathered_params(orig_params, module)

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
        # Set up the reference parameters and construct the FSDP group
        orig_params = self._init_params(param_sizes)
        fsdp_param_group = self._init_fsdp_param_group(orig_params, True)
        fsdp_params = fsdp_param_group.fsdp_params
        fsdp_param_group.comm_ctx.init()

        # Run one unshard to initialize metadata
        fsdp_param_group.unshard()
        fsdp_param_group.wait_for_unshard()
        fsdp_param_group.reshard()

        # Run the foreach reduce-scatter (including copy-in and view-out)
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

        # Check reduce-scatter correctness
        reduced_grads = [grad.detach().clone() for grad in unsharded_grads]
        for grad in reduced_grads:
            dist.all_reduce(grad, group=group)
            grad /= self.world_size
        for fsdp_param, reduced_grad in zip(fsdp_params, reduced_grads):
            sharded_grad = fsdp_param.sharded_param.grad
            self.assertIsInstance(sharded_grad, DTensor)
            self.assertEqual(sharded_grad.full_tensor(), reduced_grad)


if __name__ == "__main__":
    run_tests()
