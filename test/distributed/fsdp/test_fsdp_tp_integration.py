# Owner(s): ["oncall: distributed"]
import copy
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, DTensor as DT
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
    SequenceParallel,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


def _is_nested_tensor(val: Any) -> bool:
    if type(val) is ShardedTensor:
        if len(val.local_shards()) == 0:
            return False
        if type(val.local_shards()[0].tensor) is ShardedTensor:
            return True
    return False


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))

    @staticmethod
    def get_sharded_param_names() -> List[str]:
        return ["net1.weight", "net1.bias", "net2.weight"]

    @staticmethod
    def get_non_sharded_param_names() -> List[str]:
        return ["net3.weight", "net3.bias"]


class TestTPFSDPIntegration(FSDPTest):
    def _get_params_and_sharding_info(
        self,
        model: SimpleModel,
        sharded_param_names: List[str],
        tensor_parallel_size: int,
    ) -> Tuple[Dict[str, int], Dict[str, Tuple[torch.Size, int]]]:
        """ """
        assert (
            type(model) is SimpleModel
        ), "Expects a `SimpleModel` since the sharding cases on the model definition"
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
            device_type="cuda",
            mesh=torch.arange(0, self.world_size).view(-1, tensor_parallel_size),
        )

        fsdp_pg = twod_mesh.get_dim_groups()[0]
        tp_pg = twod_mesh.get_dim_groups()[1]
        return twod_mesh, fsdp_pg, tp_pg

    def _get_chunk_sharding_spec(self, tp_world_size: int, tp_pg: dist.ProcessGroup):
        placements = [
            f"rank:{idx}/cuda:{dist.distributed_c10d.get_global_rank(tp_pg, idx) % torch.cuda.device_count()}"
            for idx in range(tp_world_size)
        ]
        # Rowwise and colwise sharding are specified with respect to the
        # transposed linear weight
        colwise_spec = ChunkShardingSpec(dim=0, placements=placements)
        rowwise_spec = ChunkShardingSpec(dim=1, placements=placements)
        return colwise_spec, rowwise_spec

    def _sync_tp_grads(
        self,
        tp_fsdp_model: FSDP,
        tp_pg: dist.ProcessGroup,
        param_name_to_numel: Dict[str, int],
        non_sharded_param_names: List[str],
    ) -> None:
        """
        Syncs the tensor parallel parameters' gradients following the data
        parallel paradigm where gradients are averaged over ranks (in this
        case, the ones in the tensor parallel process group).
        """
        tp_world_size = tp_pg.size()
        fsdp_world_size = self.world_size // tp_world_size
        assert (
            type(tp_fsdp_model) is FSDP and len(list(tp_fsdp_model.parameters())) == 1
        ), (
            "The following logic assumes a single top-level-only FSDP wrapping "
            "the model with TP already applied"
        )
        flat_param = tp_fsdp_model.params[0]
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
        unsharded_mask = torch.cat(per_param_masks).contiguous().type(torch.BoolTensor)
        sharded_mask = unsharded_mask.chunk(fsdp_world_size)[self.rank // tp_world_size]
        grad_device = flat_param.grad.device
        grad = flat_param.grad.detach().clone().cuda(self.rank)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_pg)
        grad = grad.to(grad_device)
        flat_param.grad[~sharded_mask] = grad[~sharded_mask]
        # Average *all* gradient elements to match the FSDP only semantics
        flat_param.grad /= tp_world_size

    def _get_grads_as_flattened(
        self,
        model: FSDP,
        uses_tp: bool,
        param_name_to_numel: Dict[str, int],
        param_name_to_sharding_info: Dict[str, Tuple[torch.Size, int]],
        tp_pg: Optional[dist.ProcessGroup],
        fsdp_pg: Optional[dist.ProcessGroup],
        sharded_param_names: Optional[List[str]],
    ) -> torch.Tensor:
        """
        Returns all unsharded gradients as a single flattened tensor. This
        returns the same value on all ranks.
        """
        local_grads_as_flattened = (
            torch.cat([torch.flatten(param.grad) for param in model.parameters()])
            .contiguous()
            .cuda(self.rank)
        )
        all_grads_as_flattened = torch.cat(
            [torch.empty_like(local_grads_as_flattened) for _ in range(fsdp_pg.size())]
        ).contiguous()
        dist._all_gather_base(
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
    @parametrize("tensor_parallel_size", [2, 4])
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    def test_fsdp_tp_integration(self, tensor_parallel_size, cpu_offload):
        """
        Tests training for TP + FSDP integration by comparing an FSDP-only
        model with a TP + FSDP model.
        """
        self.assertTrue(
            enable_2d_with_fsdp(), "FSDP 2d parallel integration not available"
        )
        LR = 3e-5
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
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
        inp = torch.rand(*inp_size).cuda(self.rank)
        self.assertEqual(model(inp), tp_fsdp_model(inp))  # sanity check

        mesh_2d, fsdp_pg, tp_pg = self._get_sub_pgs(tensor_parallel_size)
        fsdp_model = FSDP(
            model, process_group=self.process_group, cpu_offload=cpu_offload
        )
        # Shard with TP and then wrap with FSDP
        tp_fsdp_model = parallelize_module(
            tp_fsdp_model, mesh_2d, SequenceParallel(), tp_mesh_dim=1
        )
        assert isinstance(tp_fsdp_model.net1.weight, DT)
        assert isinstance(tp_fsdp_model.net2.weight, DT)
        tp_fsdp_model = FSDP(
            tp_fsdp_model, process_group=fsdp_pg, cpu_offload=cpu_offload
        )

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
        inp = torch.rand(*inp_size).cuda(self.rank)
        fsdp_out = fsdp_model(inp)
        tp_fsdp_out = tp_fsdp_model(inp)
        self.assertEqual(fsdp_out, tp_fsdp_out)

    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_checkpoint_integration(self):
        """Tests checkpointing for TP + FSDP integration."""
        self.assertTrue(
            enable_2d_with_fsdp(), "FSDP 2d parallel integration not available"
        )
        tensor_parallel_size = 2
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        mesh_2d, fsdp_pg, _ = self._get_sub_pgs(tensor_parallel_size)
        # Shard with TP and then wrap with FSDP
        tp_fsdp_model = parallelize_module(
            model, mesh_2d, PairwiseParallel(), tp_mesh_dim=1
        )
        tp_fsdp_model = FSDP(model, process_group=fsdp_pg)

        # Check that we produce a nested ST from model state dict
        with FSDP.state_dict_type(tp_fsdp_model, StateDictType.SHARDED_STATE_DICT):
            state_dict = tp_fsdp_model.state_dict()
            # TODO once 2D is out, validate the nesting
            self.assertTrue(_is_nested_tensor(state_dict["net1.weight"]))
            self.assertFalse(_is_nested_tensor(state_dict["net3.bias"]))
            tp_fsdp_model.load_state_dict(state_dict)

        tp_fsdp_optim = torch.optim.Adam(tp_fsdp_model.parameters(), lr=0.0001)

        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        inp_size = [2, 3, 5]
        inp = torch.rand(*inp_size).cuda(self.rank)

        tp_fsdp_model(inp).sum().backward()
        tp_fsdp_optim.step()

        # Check that we produce a nested ST from optim state dict
        optim_state = FSDP.sharded_optim_state_dict(tp_fsdp_model, tp_fsdp_optim)
        # TODO once 2D is out, validate the nesting
        self.assertTrue(
            _is_nested_tensor(optim_state["state"]["net1.weight"]["exp_avg"])
        )
        self.assertFalse(
            _is_nested_tensor(optim_state["state"]["net3.bias"]["exp_avg"])
        )


instantiate_parametrized_tests(TestTPFSDPIntegration)

if __name__ == "__main__":
    run_tests()
