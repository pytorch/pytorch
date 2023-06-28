# Owner(s): ["oncall: distributed"]
import torch
import torch.distributed._tensor.random as random

from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    make_input_replicate_1d,
    make_input_shard_1d_last_dim,
    make_output_replicate_1d,
    make_output_tensor,
    make_sharded_output_tensor,
    PairwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)


class TensorParallelRandomStateTests(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_model_init(self):
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        tp_rank = device_mesh.get_coordinate()[0]  # the tensor parallel dimension is 0
        dp_rank = device_mesh.get_coordinate()[1]  # the data parallel dimension is 1

        for enable_distribute_flag in [False, True]:
            # a local model on meta device
            model = MLPModule(device="meta")
            # the col-wise parallel style shards the weight over tensor dim 0
            model_tp = parallelize_module(
                model,
                device_mesh,
                {
                    "net1": ColwiseParallel(
                        make_input_replicate_1d, make_output_replicate_1d
                    ),
                    "net2": ColwiseParallel(
                        make_input_replicate_1d, make_output_replicate_1d
                    ),
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

                tensor_local = dtensor.to_local()
                local_shape = tensor_local.shape

                # all-gather local shards
                tensor_gather = _1d_mesh.all_gather(tensor_local, gather_dim=0)

                # compare local shards within the TP group
                self.assertEqual(_1d_mesh.get_coordinate()[0], tp_rank)
                for other_rank in range(_1d_mesh.size(dim=0)):
                    if tp_rank != other_rank:
                        slice_idx = [
                            slice(other_rank * local_shape[0], (other_rank + 1) * local_shape[0]),
                            slice(local_shape[1]),
                        ]
                        if enable_distribute_flag:
                            # each rank within a TP group shall initialize local weights differently
                            self.assertNotEqual(tensor_gather[slice_idx], tensor_local)
                        else:
                            # without the parallel RNG, weight initialization violates the TP setup:
                            # each rank within a TP group has the same initial weights
                            self.assertEqual(tensor_gather[slice_idx], tensor_local)

                # check across TP groups
                # all-gather local shards
                tensor_gather = device_mesh.all_gather(tensor_local, mesh_dim=1, gather_dim=0)
                for other_rank in range(device_mesh.size(dim=1)):
                    if dp_rank != other_rank:
                        slice_idx = [
                            slice(other_rank * local_shape[0], (other_rank + 1) * local_shape[0]),
                            slice(local_shape[1]),
                        ]
                        if enable_distribute_flag:
                            # local weights shall be initialized the same acorss TP groups
                            self.assertEqual(tensor_gather[slice_idx], tensor_local)
                        else:
                            # without the parallel RNG, weight initialization violates the TP setup:
                            # local weights are initialized differently acorss TP groups due to different
                            # random seeds set in data loading.
                            self.assertNotEqual(tensor_gather[slice_idx], tensor_local)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_dropout_layer(self):
        local_shape = [16, 6]

        # in self-attention layer, the activation needs to go through 2 dropout layers:
        #   1. attention dropout which belongs to a tensor-parallel region
        #   2. residual dropout which belongs to a non-tensor-parallel region
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        tp_rank = device_mesh.get_coordinate()[0]  # the tensor parallel dimension is 0
        dp_rank = device_mesh.get_coordinate()[1]  # the data parallel dimension is 1

        attn_dropout = torch.nn.Dropout(p=0.2)
        attn_dropout_tp = parallelize_module(
            attn_dropout,
            device_mesh,
            PairwiseParallel(make_input_shard_1d_last_dim, make_sharded_output_tensor),
        )
        resid_dropout = torch.nn.Dropout(p=0.2)
        resid_dropout_tp = parallelize_module(
            resid_dropout,
            device_mesh,
            PairwiseParallel(make_input_replicate_1d, make_output_tensor),
        )

        for enable_distribute_flag in [False, True]:
            # disable/enable parallel RNG feature
            random._rng_tracker.distribute_region_enabled = enable_distribute_flag

            inp = torch.ones(*local_shape, device=self.device_type)
            out = attn_dropout_tp(inp)
            # examine the output
            # check within TP groups
            # all-gather local shards within the TP group
            assert isinstance(out, torch.Tensor)
            # TODO: absract out this logic and reuse
            tensor_local = out
            tensor_gather = device_mesh.all_gather(tensor_local, mesh_dim=0, gather_dim=0)
            for other_rank in range(device_mesh.size(dim=0)):
                if tp_rank != other_rank:
                    slice_idx = [
                        slice(other_rank * local_shape[0], (other_rank + 1) * local_shape[0]),
                        slice(local_shape[1]),
                    ]
                    if enable_distribute_flag:
                        # tensor-parallel dropout results should be different
                        # within a TP group
                        self.assertNotEqual(tensor_gather[slice_idx], tensor_local)
                    else:
                        # without the parallel RNG, dropout in a tensor-parallel region
                        # (i.e. on sharded DTensor) will have the same behavior with
                        # that in a non-tensor-parallel region (i.e. dropout on a 
                        # replicate DTensor).
                        self.assertEqual(tensor_gather[slice_idx], tensor_local)

            inp = torch.ones(*local_shape, device=self.device_type)
            out = resid_dropout_tp(inp)
            # examine the output
            # check across TP groups
            # all-gather local shards across TP groups
            assert isinstance(out, torch.Tensor)
            tensor_local = out
            tensor_gather = device_mesh.all_gather(tensor_local, mesh_dim=1, gather_dim=0)
            for other_rank in range(device_mesh.size(dim=1)):
                if dp_rank != other_rank:
                    slice_idx = [
                        slice(other_rank * local_shape[0], (other_rank + 1) * local_shape[0]),
                        slice(local_shape[1]),
                    ]
                    # in a non-tensor-parallel region, dropout results should be the same
                    # across TP groups
                    self.assertEqual(tensor_gather[slice_idx], tensor_local)


if __name__ == "__main__":
    run_tests()
