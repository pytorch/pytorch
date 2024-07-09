# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.utils import _sync_module_states_with_mesh
from torch.testing._internal.common_dist_composable import CompositeModel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

c10d_functional = torch.ops.c10d_functional


class UtilTest(DTensorTestBase):
    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    @staticmethod
    def _compare_msd(state_dict, mesh) -> bool:
        keys = list(state_dict.keys())
        objects = [None for _ in range(mesh.size())]
        dist.all_gather_object(objects, keys, mesh.get_group())
        if not all(x == objects[0] for x in objects):
            return False
        for t in state_dict.values():
            objects = [torch.empty_like(t) for _ in objects]
            dist.all_gather(objects, t, mesh.get_group())
            if not all(torch.allclose(x, objects[0]) for x in objects):
                return False

        return True

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_sync_module_states_with_mesh(self):
        torch.manual_seed(self.rank)
        model = CompositeModel("cuda")
        model.register_buffer(
            "test_buf", torch.ones((10, 10), device="cuda") * self.rank
        )
        state_dict = model.state_dict()
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("DP", "TP")
        )
        mesh_1d = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("DP",)
        )

        self.assertFalse(self._compare_msd(state_dict, mesh_1d))
        _sync_module_states_with_mesh(model, mesh_2d["DP"])
        self.assertFalse(self._compare_msd(state_dict, mesh_1d))
        self.assertTrue(self._compare_msd(state_dict, mesh_2d["DP"]))
        _sync_module_states_with_mesh(model, mesh_1d)
        self.assertTrue(self._compare_msd(state_dict, mesh_1d))


if __name__ == "__main__":
    run_tests()
