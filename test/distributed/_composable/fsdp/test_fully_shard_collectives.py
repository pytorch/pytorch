# Owner(s): ["oncall: distributed"]

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed._composable.fsdp._fsdp_collectives import (
    foreach_all_gather,
    foreach_all_gather_copy_out,
)
from torch.distributed._composable.fsdp._fsdp_common import (
    chunk_with_empty,
    FSDPMeshInfo,
)
from torch.distributed._composable.fsdp._fsdp_init import _init_default_fully_shard_mesh
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests


class TestFullyShardCollectives(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 8  # cannot make too large or else CI machines will OOM
        return 128

    @skip_if_lt_x_gpu(1)
    def test_all_gather(self):
        param_sizes = [
            torch.Size([32, 257]),
            torch.Size([32]),
            torch.Size([256, 5120]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([256, 256]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([8224, 256]),
            torch.Size([8224]),
            torch.Size([256, 5120]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([256, 256]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([256]),
            torch.Size([5120, 256]),
            torch.Size([5120]),
            torch.Size([1024, 1686]),
            torch.Size([1024]),
            torch.Size([1024]),
            torch.Size([1024]),
            torch.Size([8224, 1024]),
            torch.Size([8224]),
            torch.Size([2048, 8224]),
            torch.Size([2048]),
            torch.Size([2048]),
            torch.Size([2048]),
            torch.Size([5952]),
            torch.Size([5952]),
            torch.Size([1024, 5952]),
            torch.Size([1024]),
            torch.Size([5952, 1024]),
            torch.Size([5952]),
            torch.Size([3228, 5952]),
            torch.Size([3228]),
            torch.Size([3228]),
            torch.Size([3228]),
            torch.Size([1536, 3228]),
            torch.Size([1536]),
            torch.Size([1536]),
            torch.Size([1536]),
            torch.Size([3228, 1536]),
            torch.Size([3228]),
            torch.Size([3228]),
            torch.Size([3228]),
        ]
        self._test_all_gather(param_sizes, torch.bfloat16)

    def _test_all_gather(
        self, param_sizes: List[torch.Size], param_dtype: Optional[torch.dtype]
    ):
        # - Set up the reference parameters
        torch.manual_seed(42)
        orig_params = [
            nn.Parameter(torch.randn(size, device="cuda")) for size in param_sizes
        ]
        if self.rank == 0:
            total_numel = sum(p.numel() for p in orig_params)
            print(
                f"Original parameters: {total_numel} numel "
                f"({total_numel * 4 / 1e9:.3f} GB in fp32) ({total_numel * 2 / 1e9:.3f} GB in bf16)"
            )
        # Since seed is per process, not per thread, we broadcast to ensure the
        # same original parameters across ranks
        for orig_param in orig_params:
            dist.broadcast(orig_param, src=0)
        module = nn.ParameterList([param.detach().clone() for param in orig_params])

        # - Construct the FSDP parameter group, which shards parameters
        device = torch.device("cuda:0")
        mesh_info = FSDPMeshInfo(
            _init_default_fully_shard_mesh(device.type), shard_mesh_dim=0
        )
        fsdp_param_group = FSDPParamGroup(
            list(module.parameters()),
            module,
            mesh_info,
            mesh_info,
            device,
            MixedPrecisionPolicy(param_dtype=param_dtype),
            OffloadPolicy(),
        )
        fsdp_params = fsdp_param_group.fsdp_params
        current_stream = torch.cuda.current_stream()
        group = dist.distributed_c10d._get_default_group()

        # - Sanity check that the parameter sharding is as expected
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, DTensor))
            orig_param_chunk = chunk_with_empty(orig_param, self.world_size, dim=0)[
                self.rank
            ]
            if param._local_tensor.numel() > 0:
                self.assertEqual(param._local_tensor, orig_param_chunk)
            else:
                self.assertEqual(orig_param_chunk.numel(), 0)

        # - Run the overall all-gather (including copy-in and copy-out) and
        # transition the FSDP parameters to unsharded state
        all_gather_result = foreach_all_gather(
            fsdp_params,
            group,
            async_op=False,
            all_gather_copy_in_stream=current_stream,
            all_gather_stream=current_stream,
            use_uint8=False,
            device=device,
        )
        foreach_all_gather_copy_out(
            all_gather_result.all_gather_output,
            fsdp_params,
            group,
            use_uint8=False,
        )
        for fsdp_param in fsdp_params:
            fsdp_param.to_unsharded()

        # - Check that the all-gather was correct
        for orig_param, param in zip(orig_params, module.parameters()):
            self.assertTrue(isinstance(param, torch.Tensor))
            self.assertEqual(param, orig_param.to(param.dtype))

        if self.rank == 0:
            mem_stats = torch.cuda.memory_stats()
            peak_active_gb = mem_stats["active_bytes.all.peak"] / (1000**3)
            peak_reserved_gb = mem_stats["reserved_bytes.all.peak"] / (1000**3)
            print(
                f"peak active: {peak_active_gb} GB | peak reserved: {peak_reserved_gb} GB"
            )


if __name__ == "__main__":
    run_tests()
