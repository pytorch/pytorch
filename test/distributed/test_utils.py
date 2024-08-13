# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
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
    def _compare_msd(state_dict, mesh, ignored_names=None) -> bool:
        ignored_names = ignored_names if ignored_names is not None else []
        keys = list(state_dict.keys())
        objects = [None for _ in range(mesh.size())]
        dist.all_gather_object(objects, keys, mesh.get_group())
        if not all(x == objects[0] for x in objects):
            return False
        for name, v in state_dict.items():
            if name in ignored_names:
                continue
            if isinstance(v, DTensor):
                local_t = v.to_local()
                objects = [torch.empty_like(local_t) for _ in objects]
                dist.all_gather(objects, local_t, mesh.get_group())
            else:
                objects = [torch.empty_like(v) for _ in objects]
                dist.all_gather(objects, v, mesh.get_group())

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
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        mesh_1d = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp",)
        )

        state_dict = model.state_dict()
        self.assertFalse(self._compare_msd(state_dict, mesh_1d))
        _sync_module_states_with_mesh(model, mesh_2d["dp"])
        self.assertFalse(self._compare_msd(state_dict, mesh_1d))
        self.assertTrue(self._compare_msd(state_dict, mesh_2d["dp"]))
        _sync_module_states_with_mesh(model, mesh_1d)
        self.assertTrue(self._compare_msd(state_dict, mesh_1d))

        # Test with DTensor case
        model.register_buffer(
            "test_dtensor_1d",
            DTensor.from_local(
                torch.ones((10, 10), device="cuda") * self.rank,
                device_mesh=mesh_1d,
                placements=[Replicate()],
            ),
        )

        model.register_buffer(
            "test_dtensor_2d_replicate_shard",
            DTensor.from_local(
                torch.ones((10, 10), device="cuda") * self.rank,
                device_mesh=mesh_2d,
                placements=[Replicate(), Shard(1)],
            ),
        )

        model.register_buffer(
            "test_dtensor_2d_shard_only",
            DTensor.from_local(
                torch.ones((10, 10), device="cuda") * self.rank,
                device_mesh=mesh_2d,
                placements=[Shard(1), Shard(1)],
            ),
        )

        state_dict = model.state_dict()
        self.assertFalse(self._compare_msd(state_dict, mesh_1d))

        # 2D DTensor won't be broadcasted but 1D Dtensor will be brocasted
        _sync_module_states_with_mesh(model, mesh_1d)
        self.assertFalse(self._compare_msd(state_dict, mesh_1d))
        self.assertTrue(
            self._compare_msd(
                state_dict,
                mesh_1d,
                ignored_names=[
                    "test_dtensor_2d_replicate_shard",
                    "test_dtensor_2d_shard_only",
                ],
            )
        )

        # Only broadcast 2D DTensor along the dp dimension if it is replicated.
        # test_dtensor_2d_replicate_shard is replicated on the dp dimension, so
        # the local_tensor will be the same for the same dp group.
        # test_dtensor_2d_shard_only is sharded on both dp and tp dimensions, so
        # the local_tensor won't be the same regardless which dimension.
        _sync_module_states_with_mesh(model, mesh_2d["dp"])
        self.assertFalse(
            self._compare_msd(
                state_dict, mesh_1d, ignored_names=["test_dtensor_2d_shard_only"]
            )
        )
        self.assertTrue(
            self._compare_msd(
                state_dict,
                mesh_2d["dp"],
                ignored_names=["test_dtensor_2d_shard_only"],
            )
        )
        self.assertFalse(
            self._compare_msd(
                state_dict,
                mesh_2d["dp"],
                ignored_names=["test_dtensor_2d_replicate_shard"],
            )
        )
        self.assertFalse(
            self._compare_msd(
                state_dict,
                mesh_2d["tp"],
                ignored_names=["test_dtensor_2d_replicate_shard"],
            )
        )


if __name__ == "__main__":
    run_tests()
