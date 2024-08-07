# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.device_mesh import _mesh_resources, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin


class Dummymodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class EPModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class SecondTier(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.ep_layers = nn.ModuleList(
            [EPModel(rank) if rank % 4 == i else Dummymodel() for i in range(4)]
        )
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class TopModel(nn.Module):
    def __init__(self, rank):
        super().__init__()
        torch.manual_seed(0)

        self.second = SecondTier(rank)
        self.net = nn.Sequential(nn.Linear(16, 16), nn.ReLU())

    def forward(self, x):
        raise NotImplementedError


class TestFSDPWithEP(DTensorTestBase, VerifyStateDictMixin):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @with_comms
    @skip_if_lt_x_gpu(8)
    @with_temp_dir
    def test_e2e(self):
        model = TopModel(self.rank).cuda()

        mesh_fsdp_tp = init_device_mesh(
            self.device_type, (2, 4), mesh_dim_names=("dp", "tp")
        )
        # TODO: we are using an internal API atm. Change to a publich API once it is ready.
        mesh_fsdp_ep = _mesh_resources.create_child_mesh(mesh_fsdp_tp, ("dp",))
        del _mesh_resources.child_to_parent_mapping[mesh_fsdp_ep]

        mesh_fsdp = init_device_mesh(self.device_type, (8,))
        for i, l in enumerate(model.second.ep_layers):
            model.second.ep_layers[i] = FSDP(
                l, use_orig_params=True, device_mesh=mesh_fsdp_ep
            )
        model.second = FSDP(model.second, use_orig_params=True, device_mesh=mesh_fsdp)
        model = FSDP(model, use_orig_params=True, device_mesh=mesh_fsdp)
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        msd, osd = get_state_dict(model, optim)

        # FSDP only params
        for key in (
            "net.0.weight",
            "net.0.bias",
            "second.net.0.weight",
            "second.net.0.bias",
        ):
            msd_v = msd[key]
            osd_v = osd["state"][key]["exp_avg"]
            for v in (msd_v, osd_v):
                self.assertTrue(isinstance(v, DTensor))
                self.assertEqual(tuple(v.device_mesh.mesh), tuple(range(8)))

        # FSDP/EP params
        layer = self.rank % 4
        ranks = (layer, layer + 4)
        for i in range(4):
            for key in (
                f"second.ep_layers.{i}.net1.0.weight",
                f"second.ep_layers.{i}.net1.0.bias",
                f"second.ep_layers.{i}.net2.0.weight",
                f"second.ep_layers.{i}.net2.0.bias",
            ):
                if layer != i:
                    self.assertTrue(key not in msd)
                else:
                    msd_v = msd[key]
                    osd_v = osd["state"][key]["exp_avg"]
                    for v in (msd_v, osd_v):
                        self.assertTrue(isinstance(v, DTensor))
                        self.assertEqual(tuple(v.device_mesh.mesh), ranks)

        self.assertEqual(set(osd["state"].keys()), set(msd.keys()))


if __name__ == "__main__":
    run_tests()
