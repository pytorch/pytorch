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

    def _get_param_sizes(self) -> List[torch.Size]:
        # For world size 128, the all-gather testing requires <0.22 GB
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

    def _test_all_gather(
        self,
        param_sizes: List[torch.Size],
        async_op: bool,
        all_gather_copy_in_stream: torch.cuda.Stream,
        all_gather_stream: torch.cuda.Stream,
        all_gather_dtype: torch.dtype,
    ):
        # - Set up the reference parameters
        torch.manual_seed(42)
        orig_params = [
            nn.Parameter(torch.randn(size, device="cuda")) for size in param_sizes
        ]
        # Since seed is per process, not per thread, we broadcast to ensure the
        # same original parameters across ranks
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        module = nn.ParameterList([param.detach().clone() for param in orig_params])

        # - Construct the FSDP parameter group (which shards parameters)
        device = torch.device("cuda:0")
        mesh_info = FSDPMeshInfo(
            _init_default_fully_shard_mesh(device.type), shard_mesh_dim=0
        )
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()), module, mesh_info, device
        )
        fsdp_params = fsdp_param_group.fsdp_params

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
            device=device,
            dtype=torch.float32,
        )
        if (event := all_gather_result.all_gather_event) is not None:
            torch.cuda.current_stream().wait_event(event)
        if async_op:
            all_gather_result.all_gather_work.wait()
        foreach_all_gather_copy_out(
            all_gather_result.all_gather_output,
            all_gather_result.all_gather_input_numels,
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


if __name__ == "__main__":
    run_tests()
