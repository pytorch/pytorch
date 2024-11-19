# Owner(s): ["oncall: distributed"]

import copy
import functools
import unittest
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
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
        return min(8, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_dp_state_dict_save_load(self):
        fsdp_mesh = init_device_mesh("cuda", (self.world_size,))
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5], "mesh": [fsdp_mesh]},
            self._test_dp_state_dict_save_load,
        )
        self.run_subtests(
            {"mlp_dim": [16], "mesh": [fsdp_mesh], "use_shard_placement_fn": [True]},
            self._test_dp_state_dict_save_load,
        )
        if self.world_size % 2 != 0:
            return
        hsdp_mesh = init_device_mesh(
            "cuda",
            (self.world_size // 2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        self.run_subtests(
            {"mlp_dim": [2, 3, 4, 5], "mesh": [hsdp_mesh]},
            self._test_dp_state_dict_save_load,
        )
        self.run_subtests(
            {"mlp_dim": [16], "mesh": [hsdp_mesh], "use_shard_placement_fn": [True]},
            self._test_dp_state_dict_save_load,
        )

    def _test_dp_state_dict_save_load(
        self, mlp_dim: int, mesh: DeviceMesh, use_shard_placement_fn: bool = False
    ):
        torch.manual_seed(42)
        base_model = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )

        def _shard_placement_fn(param: nn.Parameter) -> Optional[Shard]:
            largest_dim = largest_dim_size = -1
            for dim, dim_size in enumerate(param.shape):
                if dim_size > largest_dim_size:
                    largest_dim = dim
                    largest_dim_size = dim_size
            return Shard(largest_dim)

        shard_placement_fn = _shard_placement_fn if use_shard_placement_fn else None
        fully_shard_fn = functools.partial(
            fully_shard, mesh=mesh, shard_placement_fn=shard_placement_fn
        )

        # Check basic `reshard_after_forward=True`
        model1 = copy.deepcopy(base_model)
        for module in model1:
            fully_shard_fn(module)
        fully_shard_fn(model1)
        self._test_state_dict_save_load(model1)

        # Check `reshard_after_forward=False` before and after a forward
        model2 = copy.deepcopy(base_model)
        for module in model2:
            fully_shard_fn(module, reshard_after_forward=False)
        fully_shard_fn(model2, reshard_after_forward=False)
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
    def test_dp_state_dict_cpu_offload(self):
        self.run_subtests(
            {
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
                "cpu_state_dict": [True, False],
            },
            self._test_dp_state_dict_cpu_offload,
        )

    def _test_dp_state_dict_cpu_offload(
        self, offload_policy: CPUOffloadPolicy, cpu_state_dict: bool
    ):
        mlp_dim = 4
        torch.manual_seed(42)
        with torch.device("meta"):
            model = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim, bias=False),
                nn.Linear(mlp_dim, mlp_dim, bias=False),
            )
        for module in model:
            fully_shard(module, offload_policy=offload_policy)
        fully_shard(model, offload_policy=offload_policy)

        # split full sd into multiple pieces
        # to test loading with `strict=False`
        state_dicts = []
        for name, dtensor in model.named_parameters():
            full_tensor = torch.randn(dtensor.size())
            sharded_tensor = distribute_tensor(
                full_tensor, dtensor.device_mesh, dtensor.placements
            )
            if cpu_state_dict:
                sharded_tensor = sharded_tensor.cpu()
            state_dicts.append({name: sharded_tensor})

        # check that we can load with some parameters still on meta device
        for sd in state_dicts:
            model.load_state_dict(sd, assign=True, strict=False)

        # lazy init without error
        inp = torch.rand((mlp_dim, mlp_dim), device="cuda")

        context = (
            self.assertRaisesRegex(
                RuntimeError,
                r"Found following parameters on non-CPU device: \[\('0.weight', device\(type='cuda'",
            )
            if not cpu_state_dict
            else nullcontext()
        )
        with context:
            model(inp).sum()
            state_dict = model.state_dict()
            for name, dtensor in state_dict.items():
                self.assertEqual(dtensor.device.type, "cpu")

    def test_2d_state_dict_correctness(self):
        dp_size = 2
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        torch.manual_seed(42)
        mlp_dim = 4

        # model init
        model = nn.Sequential(*[MLP(mlp_dim) for _ in range(3)])
        model_2d = copy.deepcopy(model)

        # FSDP + TP
        model_2d = parallelize_module(
            model_2d,
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
        for mlp in model_2d:
            fully_shard(mlp, mesh=dp_mesh)
        fully_shard(model_2d, mesh=dp_mesh)

        # state_dict parity check
        model_state_dict = model.state_dict()
        model_2d_state_dict = model_2d.state_dict()
        for tensor, dtensor in zip(
            model_state_dict.values(), model_2d_state_dict.values()
        ):
            self.assertTrue(isinstance(dtensor, DTensor))
            self.assertEqual(tensor, dtensor.full_tensor())

    @skip_if_lt_x_gpu(2)
    def test_dp_tp_state_dict_save_load(self):
        dp_size = 2
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        self.run_subtests(
            {"mlp_dim": [4, 6, 8, 10]},
            functools.partial(self._test_dp_tp_state_dict_save_load, global_mesh),
        )

    def _test_dp_tp_state_dict_save_load(self, global_mesh: DeviceMesh, mlp_dim: int):
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

    @skip_if_lt_x_gpu(4)
    def test_hsdp_tp_state_dict_save_load(self):
        global_mesh = init_device_mesh(
            "cuda",
            (2, 2, self.world_size // 4),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
        self.run_subtests(
            {"mlp_dim": [4, 6, 8, 10]},
            functools.partial(self._test_hsdp_tp_state_dict_save_load, global_mesh),
        )

    def _test_hsdp_tp_state_dict_save_load(self, global_mesh: DeviceMesh, mlp_dim: int):
        dp_mesh, tp_mesh = global_mesh["dp_replicate", "dp_shard"], global_mesh["tp"]
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
