# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, get_devtype
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


class ModelWithBuffer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lin = nn.Linear(10, 10, device=device)
        self.bn = nn.BatchNorm1d(10, device=device)

    def forward(self, x):
        return self.bn(self.lin(x))


class TestHSDPSyncModuleStates(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(1)
    def test_hsdp_buffer_sync_from_meta_device(self):
        """Test that HSDP sync_module_states correctly broadcasts buffers
        when only rank 0 has real weights and other ranks use meta device.

        _sync_module_params_and_buffers marks buffers with FSDP_SYNCED=True
        to avoid redundant syncs in nested wrapping. With two-phase broadcast
        (inter-node then intra-node), this flag must be reset between phases,
        otherwise the intra-node broadcast skips buffers. Non-persistent
        buffers (e.g. RoPE inv_freq) that are materialized from meta device
        via to_empty() and not restored by reset_parameters() would remain
        as uninitialized values on local ranks 1..N.

        Parameters are unaffected by broadcast order because they are
        unconditionally included in every sync call. This test specifically
        targets buffers, which are the only tensors gated by FSDP_SYNCED.
        """
        mesh_2d = init_device_mesh(device_type.type, (2, self.world_size // 2))

        if self.rank == 0:
            model = ModelWithBuffer(device=device_type)
            model.bn.running_mean.fill_(42.0)
            model.bn.running_var.fill_(7.0)
        else:
            model = ModelWithBuffer(device="meta")

        def param_init_fn(module):
            if any(p.is_meta for p in module.parameters(recurse=False)) or any(
                b.is_meta for b in module.buffers(recurse=False)
            ):
                module.to_empty(device=device_type, recurse=False)

        model = FSDP(
            model,
            device_mesh=mesh_2d,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            param_init_fn=param_init_fn,
        )

        # Buffers are not sharded by FSDP, so each rank has a local copy.
        # If the FSDP_SYNCED reset is missing, ranks on non-source nodes
        # will have uninitialized buffer values instead of rank 0's values.
        self.assertEqual(model.bn.running_mean, torch.full((10,), 42.0))
        self.assertEqual(model.bn.running_var, torch.full((10,), 7.0))


if __name__ == "__main__":
    run_tests()
