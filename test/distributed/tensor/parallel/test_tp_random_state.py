# Owner(s): ["oncall: distributed"]
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.random as random

from torch.distributed._tensor import init_device_mesh, Replicate
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import ColwiseParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)


class TensorParallelRandomStateTests(DTensorTestBase):
    def get_tensor_slice(self, idx, n, large_tensor):
        shape = large_tensor.shape
        assert shape[0] % n == 0
        local_shape = [shape[0] // n, shape[1]]

        slice_idx = [
            slice(idx * local_shape[0], (idx + 1) * local_shape[0]),
            slice(local_shape[1]),
        ]
        return large_tensor[slice_idx]

    def check_gathered_tensors(self, self_rank, size, gathered_tensors, assertFunc):
        for other_rank in range(size):
            if self_rank != other_rank:
                assertFunc(
                    self.get_tensor_slice(self_rank, size, gathered_tensors),
                    self.get_tensor_slice(other_rank, size, gathered_tensors),
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_model_init(self):
        dp_size = 2
        tp_size = self.world_size // dp_size
        mesh_2d = init_device_mesh(
            self.device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh = mesh_2d["dp"]
        tp_mesh = mesh_2d["tp"]
        dp_rank = dp_mesh.get_coordinate()[0]
        tp_rank = tp_mesh.get_coordinate()[0]
        self.assertEqual(dp_rank, self.rank // tp_size)
        self.assertEqual(tp_rank, self.rank % tp_size)

        for enable_distribute_flag in [False, True]:
            # a local model on meta device
            model = MLPModule(device="meta")
            # the col-wise parallel style shards the weight over tensor dim 0
            model_tp = parallelize_module(
                model,
                tp_mesh,
                {
                    "net1": ColwiseParallel(output_layouts=Replicate()),
                    "net2": ColwiseParallel(output_layouts=Replicate()),
                },
            )
            # in most cases, the random number generator states is set by data loader
            # in the following way:
            #   - within a tensor parallel group, the RNG is set with the same seed
            #   - across data parallel groups, the RNG is set with different seeds
            torch.cuda.manual_seed(dp_rank)

            # disable/enable parallel RNG feature
            random._rng_tracker.distribute_region_enabled = enable_distribute_flag
            self.assertTrue(model_tp.net1.weight.is_meta)
            # initialize the model's local shard
            model_tp.to_empty(device=self.device_type)
            model_tp.reset_parameters()
            # examine that the weights are initialized adhere to DP/TP
            for dtensor in [model_tp.net1.weight, model_tp.net2.weight]:
                # check within the TP group
                # the 1d mesh represents the TP group
                _1d_mesh = dtensor.device_mesh
                assert _1d_mesh.ndim == 1
                self.assertEqual(_1d_mesh, tp_mesh)

                tensor_local = dtensor.to_local()

                # all-gather local shards
                tensor_gather = funcol.all_gather_tensor(
                    tensor_local,
                    gather_dim=0,
                    group=_1d_mesh,
                )
                self.assertEqual(_1d_mesh.get_coordinate()[0], tp_rank)

                # compare local shards within the TP group
                def tp_weights_assert(tensor1, tensor2):
                    if enable_distribute_flag:
                        # each rank within a TP group shall initialize local weights differently
                        self.assertNotEqual(tensor1, tensor2)
                    else:
                        # without the parallel RNG, weight initialization violates the TP setup:
                        # each rank within a TP group has the same initial weights
                        self.assertEqual(tensor1, tensor2)

                self.check_gathered_tensors(
                    tp_rank, tp_size, tensor_gather, tp_weights_assert
                )

                # check across TP groups
                # all-gather local shards
                tensor_gather = funcol.all_gather_tensor(
                    tensor_local,
                    gather_dim=0,
                    group=dp_mesh,
                )

                # compare local shards across TP groups
                def dp_weights_assert(tensor1, tensor2):
                    if enable_distribute_flag:
                        # local weights shall be initialized the same across TP groups
                        self.assertEqual(tensor1, tensor2)
                    else:
                        # without the parallel RNG, weight initialization violates the TP setup:
                        # local weights are initialized differently across TP groups due to different
                        # random seeds set in data loading.
                        self.assertNotEqual(tensor1, tensor2)

                self.check_gathered_tensors(
                    dp_rank, dp_size, tensor_gather, dp_weights_assert
                )


if __name__ == "__main__":
    run_tests()
