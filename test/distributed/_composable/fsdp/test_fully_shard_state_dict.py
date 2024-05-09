# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest

from typing import Dict

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFullyShardStateDictMultiProcess(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_1d_state_dict_save_load(self):
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5]},
            self._test_1d_state_dict_save_load,
        )

    def _test_1d_state_dict_save_load(self, mlp_dim: int):
        torch.manual_seed(42)
        base_model = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )
        # Check basic `reshard_after_forward=True`
        model1 = copy.deepcopy(base_model)
        for module in model1:
            fully_shard(module)
        fully_shard(model1)
        self._test_state_dict_save_load(model1)

        # Check `reshard_after_forward=False` before and after a forward
        model2 = copy.deepcopy(base_model)
        for module in model2:
            fully_shard(module, reshard_after_forward=False)
        fully_shard(model2, reshard_after_forward=False)
        self._test_state_dict_save_load(model2)
        ref_sharded_sd = model2.state_dict()
        inp = torch.randn((2, mlp_dim), device="cuda")
        model2(inp)  # parameters are not resharded after this forward
        # Check that state dict hooks reshard
        sharded_sd = model2.state_dict()
        self.assertEqual(set(ref_sharded_sd.keys()), set(sharded_sd.keys()))
        for key, value in ref_sharded_sd.items():
            self.assertEqual(value, sharded_sd[key])

    @skip_if_lt_x_gpu(2)
    def test_2d_state_dict_save_load(self):
        dp_size = 2
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5]},
            functools.partial(self._test_2d_state_dict_save_load, global_mesh),
        )

    def _test_2d_state_dict_save_load(self, global_mesh: DeviceMesh, mlp_dim: int):
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(mlp_dim) for _ in range(3)])
        model = parallelize_module(
            model,
            device_mesh=tp_mesh,
            parallelize_plan={
                "0.in_proj": ColwiseParallel(),
                "0.out_proj": RowwiseParallel(),
                "1.in_proj": ColwiseParallel(),
                "1.out_proj": RowwiseParallel(),
                "2.in_proj": ColwiseParallel(),
                "2.out_proj": RowwiseParallel(),
            },
        )
        for mlp in model:
            fully_shard(mlp, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)
        self._test_state_dict_save_load(model)

    def _test_state_dict_save_load(self, model: nn.Module):
        for param_name, param in model.named_parameters():
            self.assertIsInstance(
                param,
                DTensor,
                f"Expects parameters to be sharded as DTensors but got {param_name} "
                f"as {type(param)}: {param}",
            )
        old_fill_value = 1
        new_fill_value = 42 + self.rank
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(old_fill_value)
        # Use that the parameters are currently sharded, meaning that their
        # data pointers correspond to the sharded parameter data
        param_name_to_data_ptr = {
            n: p.to_local().data_ptr() for n, p in model.named_parameters()
        }
        ref_sharded_sizes = [p.size() for p in model.parameters()]
        state_dict = model.state_dict()
        for param, ref_sharded_size in zip(model.parameters(), ref_sharded_sizes):
            self.assertEqual(param.size(), ref_sharded_size)
            self.assertTrue(isinstance(param, nn.Parameter))

        # Verify that keys match, values are DTensors, and values share the
        # same storage as the existing sharded parameter data
        self.assertEqual(set(state_dict.keys()), set(param_name_to_data_ptr.keys()))
        for param_name, tensor in state_dict.items():
            self.assertTrue(isinstance(tensor, DTensor))
            if param_name_to_data_ptr[param_name] == 0:
                # Check that this is padding (added by DTensor)
                self.assertGreater(self.rank, 0)
                self.assertEqual(torch.count_nonzero(tensor.to_local()).item(), 0)
            else:
                self.assertEqual(
                    tensor.to_local().data_ptr(), param_name_to_data_ptr[param_name]
                )

        # Verify that we can load a new state dict that contains DTensors with
        # storages different from the current model parameters
        new_state_dict: Dict[str, DTensor] = {}
        for param_name, dtensor in state_dict.items():
            # Construct new DTensors to exercise load state dict writeback
            new_state_dict[param_name] = dtensor.detach().clone().fill_(new_fill_value)
        for param in model.parameters():
            self.assertEqual(
                param.to_local(),
                torch.ones_like(param.to_local()) * old_fill_value,
            )
        model.load_state_dict(new_state_dict)
        for param_name, param in model.named_parameters():
            self.assertEqual(
                param.to_local(),
                torch.ones_like(param.to_local()) * new_fill_value,
            )
            local_param = param.to_local()
            # Only guarantee that the local tensor's data pointer does not
            # change if the sharding was even (i.e. no padding); otherwise,
            # FSDP may re-pad the local tensor, changing its data pointer
            if local_param.size(0) * param.device_mesh.size() == param.size(0):
                self.assertEqual(
                    local_param.data_ptr(), param_name_to_data_ptr[param_name]
                )


class TestFullyShardStateDictMultiThread(FSDPTestMultiThread):
    @property
    def world_size(self):
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_rank0_offload_full_state_dict(self):
        # Construct a reference unsharded model on all ranks
        model_args = ModelArgs(dropout_p=0.0)
        torch.manual_seed(42)
        ref_model = Transformer(model_args).cuda()
        for param in ref_model.parameters():
            torch.distributed.broadcast(param.detach(), src=0)

        # Construct a sharded model and sharded state dict on all ranks
        model = copy.deepcopy(ref_model)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)
        sharded_sd = model.state_dict()

        # Save a reference CPU full state dict on rank 0 and delete the
        # reference model otherwise
        if self.rank != 0:
            del ref_model
        else:
            ref_gpu_full_sd = ref_model.state_dict()
            ref_full_sd = {k: v.cpu() for k, v in ref_gpu_full_sd.items()}
            del ref_gpu_full_sd

        # Reshard the GPU sharded state dict to a CPU full state dict on rank 0
        full_sd = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if self.rank == 0:
                full_sd[param_name] = full_param.cpu()
            else:
                del full_param

        # Check that we have a CPU full state dict only on rank 0
        if self.rank == 0:
            self.assertEqual(len(full_sd), len(ref_full_sd))
            self.assertEqual(list(full_sd.keys()), list(ref_full_sd.keys()))
            for (param_name, param), ref_param in zip(
                full_sd.items(), ref_full_sd.values()
            ):
                self.assertEqual(param.device, torch.device("cpu"))
                self.assertEqual(param.device, ref_param.device)
                self.assertEqual(param, ref_param)
        else:
            self.assertEqual(len(full_sd), 0)


if __name__ == "__main__":
    run_tests()
