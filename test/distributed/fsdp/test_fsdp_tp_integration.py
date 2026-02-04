# Owner(s): ["oncall: distributed"]
import copy
import sys
from collections import OrderedDict
from typing import Optional

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    MLPModule,
    RMSNormPython,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))

    @staticmethod
    def get_sharded_param_names() -> list[str]:
        return ["net1.weight", "net1.bias", "net2.weight"]

    @staticmethod
    def get_non_sharded_param_names() -> list[str]:
        return ["net3.weight", "net3.bias"]


def distribute_rmsnorm(module, device_mesh):
    def prepare_input_fn(mod, inputs, device_mesh):
        shard_tensor = DTensor.from_local(inputs[0], device_mesh, [Shard(0)])
        return shard_tensor

    def prepare_output_fn(mod, outputs, device_mesh):
        return outputs.to_local()

    return distribute_module(
        module, device_mesh, input_fn=prepare_input_fn, output_fn=prepare_output_fn
    )


class TestTPFSDPIntegration(FSDPTestContinuous):
    def _get_params_and_sharding_info(
        self,
        model: SimpleModel,
        sharded_param_names: list[str],
        tensor_parallel_size: int,
    ) -> tuple[dict[str, int], dict[str, tuple[torch.Size, int]]]:
        """ """
        if type(model) is not SimpleModel:
            raise AssertionError(
                "Expects a `SimpleModel` since the sharding cases on the model definition"
            )
        param_name_to_numel = OrderedDict()
        param_name_to_sharding_info = OrderedDict()
        for param_name, param in model.named_parameters():
            if param_name not in sharded_param_names:
                param_name_to_numel[param_name] = param.numel()
            else:
                param_name_to_numel[param_name] = param.numel() // tensor_parallel_size
                param_name_to_sharding_info[param_name] = (
                    param.size(),
                    0 if "net1" in param_name else 1,
                )
        return param_name_to_numel, param_name_to_sharding_info

    def _get_sub_pgs(self, tensor_parallel_size: int):
        """
        Generates TP and FSDP subprocess groups. ``tensor_parallel_size`` gives
        the TP process group size.

        For example, if the global world size is 8 and the tensor parallel size
        is 2, then this creates:
        - 4 TP subprocess groups: [0, 1], [2, 3], [4, 5], [6, 7]
        - 2 FSDP subprocess groups: [0, 2, 4, 6], [1, 3, 5, 7]
        """
        # 2-D mesh is [dp, tp]
        twod_mesh = DeviceMesh(
            device_type=device_type,
            mesh=torch.arange(0, self.world_size).view(-1, tensor_parallel_size),
        )

        fsdp_pg = twod_mesh.get_group(mesh_dim=0)
        tp_pg = twod_mesh.get_group(mesh_dim=1)
        return twod_mesh, fsdp_pg, tp_pg

    def _sync_tp_grads(
        self,
        tp_fsdp_model: FSDP,
        tp_pg: dist.ProcessGroup,
        param_name_to_numel: dict[str, int],
        non_sharded_param_names: list[str],
    ) -> None:
        """
        Syncs the tensor parallel parameters' gradients following the data
        parallel paradigm where gradients are averaged over ranks (in this
        case, the ones in the tensor parallel process group).
        """
        tp_world_size = tp_pg.size()
        fsdp_world_size = self.world_size // tp_world_size
        if not (
            type(tp_fsdp_model) is FSDP
            and len([m for m in tp_fsdp_model.modules() if type(m) is FSDP]) == 1
        ):
            raise AssertionError(
                "The following logic assumes a single top-level-only FSDP wrapping "
                "the model with TP already applied"
            )
        for flat_param in tp_fsdp_model.params:
            splits = tuple(param_name_to_numel.values())
            # Create a mask over the gradient elements to manually reduce
            unsharded_size = torch.Size([flat_param.numel() * fsdp_world_size])
            unsharded_zeros = torch.zeros(unsharded_size, device=flat_param.device)
            per_param_masks = unsharded_zeros.split(splits)
            for param_idx, param_name in enumerate(
                param_name_to_numel.keys()
            ):  # assumes fixed order
                if param_name not in non_sharded_param_names:
                    per_param_masks[param_idx][:] = 1
            unsharded_mask = (
                torch.cat(per_param_masks).contiguous().type(torch.BoolTensor)
            )
            sharded_mask = unsharded_mask.chunk(fsdp_world_size)[
                self.rank // tp_world_size
            ]
            grad_device = flat_param.grad.device
            grad = flat_param.grad.detach().clone().to(self.rank)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_pg)
            grad = grad.to(grad_device)
            flat_param.grad[~sharded_mask] = grad[~sharded_mask]
            # Average *all* gradient elements to match the FSDP only semantics
            flat_param.grad /= tp_world_size

    def _get_grads_as_flattened(
        self,
        model: FSDP,
        uses_tp: bool,
        param_name_to_numel: dict[str, int],
        param_name_to_sharding_info: dict[str, tuple[torch.Size, int]],
        tp_pg: Optional[dist.ProcessGroup],
        fsdp_pg: Optional[dist.ProcessGroup],
        sharded_param_names: Optional[list[str]],
    ) -> torch.Tensor:
        """
        Returns all unsharded gradients as a single flattened tensor. This
        returns the same value on all ranks.
        """
        local_grads_as_flattened = (
            torch.cat(
                [
                    (
                        torch.flatten(param.grad)
                        if param.grad is not None
                        else torch.zeros_like(torch.flatten(param))
                    )
                    for param in model.parameters()
                ]
            )
            .contiguous()
            .to(self.rank)
        )
        all_grads_as_flattened = torch.cat(
            [torch.empty_like(local_grads_as_flattened) for _ in range(fsdp_pg.size())]
        ).contiguous()
        dist.all_gather_into_tensor(
            all_grads_as_flattened, local_grads_as_flattened, group=fsdp_pg
        )
        if not uses_tp:
            return all_grads_as_flattened
        splits = tuple(param_name_to_numel.values())
        all_grads_per_param = list(all_grads_as_flattened.split(splits))
        for param_idx, param_name in enumerate(
            param_name_to_numel.keys()
        ):  # assumes fixed order
            if param_name in sharded_param_names:
                local_tensor_size = list(param_name_to_sharding_info[param_name][0])
                sharding_dim = param_name_to_sharding_info[param_name][1]
                local_tensor_size[sharding_dim] //= tp_pg.size()
                local_tensor = all_grads_per_param[param_idx].view(*local_tensor_size)
                local_tensors = [
                    torch.empty_like(local_tensor) for _ in range(tp_pg.size())
                ]
                dist.all_gather(local_tensors, local_tensor, group=tp_pg)
                all_grads_per_param[param_idx] = torch.cat(
                    local_tensors, dim=sharding_dim
                ).reshape(-1)
        return torch.cat(all_grads_per_param).contiguous()

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_integration(self):
        self.run_subtests(
            {
                "cpu_offload": [
                    CPUOffload(offload_params=False),
                    CPUOffload(offload_params=True),
                ],
                "sharding_strategy": [None, ShardingStrategy.SHARD_GRAD_OP],
                "use_orig_params": [False, True],
            },
            self._test_fsdp_tp_integration,
        )

    def _test_fsdp_tp_integration(
        self, cpu_offload, sharding_strategy, use_orig_params
    ):
        """
        Tests training for TP + FSDP integration by comparing an FSDP-only
        model with a TP + FSDP model.
        """
        tensor_parallel_size = 2
        LR = 3e-5
        torch.manual_seed(0)
        model = SimpleModel().to(self.rank)
        tp_fsdp_model = copy.deepcopy(model)
        sharded_param_names = SimpleModel.get_sharded_param_names()
        non_sharded_param_names = SimpleModel.get_non_sharded_param_names()
        (
            param_name_to_numel,
            param_name_to_sharding_info,
        ) = self._get_params_and_sharding_info(
            model,
            sharded_param_names,
            tensor_parallel_size,
        )

        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        inp_size = [2, 3, 5]
        inp = torch.rand(*inp_size).to(self.rank)
        self.assertEqual(model(inp), tp_fsdp_model(inp))  # sanity check

        mesh_1d = init_device_mesh(device_type, (self.world_size,))
        fsdp_model = FSDP(
            model,
            cpu_offload=cpu_offload,
            device_mesh=mesh_1d,
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
        )
        mesh_2d = init_device_mesh(
            device_type,
            (self.world_size // tensor_parallel_size, tensor_parallel_size),
            mesh_dim_names=["dp", "tp"],
        )
        # Shard with TP and then wrap with FSDP
        sequence_parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0)),
            "net2": RowwiseParallel(output_layouts=Shard(0)),
        }
        tp_fsdp_model = parallelize_module(
            tp_fsdp_model,
            mesh_2d["tp"],
            sequence_parallelize_plan,
        )
        tp_pg = mesh_2d["tp"].get_group(mesh_dim=0)
        if not isinstance(tp_fsdp_model.net1.weight, DTensor):
            raise AssertionError(
                f"Expected DTensor, got {type(tp_fsdp_model.net1.weight)}"
            )
        if not isinstance(tp_fsdp_model.net2.weight, DTensor):
            raise AssertionError(
                f"Expected DTensor, got {type(tp_fsdp_model.net2.weight)}"
            )
        tp_fsdp_model = FSDP(
            tp_fsdp_model,
            cpu_offload=cpu_offload,
            device_mesh=mesh_2d["dp"],
            sharding_strategy=sharding_strategy,
            use_orig_params=use_orig_params,
        )
        fsdp_pg = mesh_2d["dp"].get_group(mesh_dim=0)

        # Check the forward by checking output equality
        fsdp_out = fsdp_model(inp)
        tp_fsdp_out = tp_fsdp_model(inp)
        self.assertEqual(fsdp_out, tp_fsdp_out)

        # Check the backward by checking gradient equality
        fsdp_out.sum().backward()
        tp_fsdp_out.sum().backward()
        self._sync_tp_grads(
            tp_fsdp_model,
            tp_pg,
            param_name_to_numel,
            non_sharded_param_names,
        )
        model_grads = self._get_grads_as_flattened(
            fsdp_model,
            False,
            param_name_to_numel,
            param_name_to_sharding_info,
            None,
            self.process_group,
            None,
        )
        model_tp_grads = self._get_grads_as_flattened(
            tp_fsdp_model,
            True,
            param_name_to_numel,
            param_name_to_sharding_info,
            tp_pg,
            fsdp_pg,
            sharded_param_names,
        )
        self.assertEqual(model_grads, model_tp_grads)

        # Check the optimizer step by performing a second forward pass
        fsdp_optim = torch.optim.SGD(fsdp_model.parameters(), lr=LR)
        tp_fsdp_optim = torch.optim.SGD(tp_fsdp_model.parameters(), lr=LR)
        fsdp_optim.step()
        tp_fsdp_optim.step()
        torch.manual_seed(input_seed + 16)
        inp = torch.rand(*inp_size).to(self.rank)
        fsdp_out = fsdp_model(inp)
        tp_fsdp_out = tp_fsdp_model(inp)
        self.assertEqual(fsdp_out, tp_fsdp_out)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_extension_grad(self):
        """
        Tests TP + FSDP extension with correct gradient (i.e. no ACT)
        """
        mesh_2d = init_device_mesh(
            device_type, (self.world_size // 2, 2), mesh_dim_names=["dp", "tp"]
        )

        class TestModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = MLPModule(device_type)
                self.mlp_norm = RMSNormPython(10)

            def forward(self, x):
                return self.mlp(self.mlp_norm(x))

        model = TestModel().to(self.rank)

        # Shard with TP and test gradient
        tp_mesh = mesh_2d["tp"]
        tp_model = parallelize_module(
            model,
            tp_mesh,
            {
                "mlp.net1": ColwiseParallel(input_layouts=Shard(0)),
                "mlp.net2": RowwiseParallel(output_layouts=Shard(0)),
            },
        )
        distribute_rmsnorm(tp_model.mlp_norm, tp_mesh)

        fsdp_2d_model = FSDP(tp_model, device_mesh=mesh_2d["dp"])
        comm_mode = CommDebugMode()

        with comm_mode:
            fsdp_2d_model(torch.rand(2, 10).to(self.rank)).sum().backward()

        funcol = torch.ops.c10d_functional
        c10d_ops = torch.ops.c10d
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 7)
        # TP comms
        self.assertEqual(comm_counts[funcol.reduce_scatter_tensor], 2)
        self.assertEqual(comm_counts[funcol.all_gather_into_tensor], 2)
        self.assertEqual(comm_counts[funcol.all_reduce], 1)
        # FSDP comms
        self.assertEqual(comm_counts[c10d_ops._allgather_base_], 1)
        self.assertEqual(comm_counts[c10d_ops._reduce_scatter_base_], 1)

        grads = [p.grad for p in fsdp_2d_model.parameters() if p.grad is not None]

        for grad in grads:
            self.assertFalse(grad.isnan().any().item())

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_sync_module_state(self):
        mesh_2d = init_device_mesh(
            device_type, (self.world_size // 2, 2), mesh_dim_names=["dp", "tp"]
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]

        # set random seed for each rank
        torch.manual_seed(mesh_2d.get_rank())

        class TestModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                replicated_dt = DTensor.from_local(
                    torch.randn(8, 8), tp_mesh, [Replicate()], run_check=False
                )
                replicated_buffer_dt = DTensor.from_local(
                    torch.randn(8, 8), tp_mesh, [Replicate()], run_check=False
                )
                self.param = torch.nn.Parameter(replicated_dt)
                self.buf = torch.nn.Buffer(replicated_buffer_dt)

            def forward(self, x):
                return self.param + self.buffer + 1

        model = TestModel()

        def assert_local_shard_across_ranks(local_tensor, group, check_equal=True):
            gathered_tensors = [
                torch.empty_like(local_tensor) for _ in range(group.size())
            ]
            dist.all_gather(gathered_tensors, local_tensor, group=group)
            # on dp mesh dim local tensor does not equal
            tensor_to_compare = gathered_tensors[0]
            for tensor in gathered_tensors[1:]:
                if check_equal:
                    self.assertTrue(torch.equal(tensor, tensor_to_compare))
                else:
                    self.assertFalse(torch.equal(tensor, tensor_to_compare))

        dp_group = dp_mesh.get_group()

        # check on dp mesh dim param local tensor does not equal
        local_param = model.param.to_local()
        assert_local_shard_across_ranks(local_param, dp_group, check_equal=False)
        # check on dp mesh dim buffer local tensor does not equal
        local_buf = model.buf.to_local()
        assert_local_shard_across_ranks(local_buf, dp_group, check_equal=False)

        # wrap with fsdp sync param should sync dp mesh dim
        fsdp_mod = FSDP(model, device_mesh=dp_mesh, sync_module_states=True)
        with fsdp_mod.summon_full_params(fsdp_mod):
            # on dp mesh dim local param does equal after sync_module_states
            local_param = fsdp_mod.param.to_local()
            assert_local_shard_across_ranks(local_param, dp_group, check_equal=True)

            # on dp mesh dim local buf does equal after sync_module_states
            local_buf = fsdp_mod.buf.to_local()
            assert_local_shard_across_ranks(local_buf, dp_group, check_equal=True)


instantiate_parametrized_tests(TestTPFSDPIntegration)

if __name__ == "__main__":
    run_tests()
