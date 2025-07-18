# Owner(s): ["oncall: distributed"]
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MLPModuleEven(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = nn.Linear(d_hid, d_hid)
        self.net2 = nn.Linear(d_hid, d_hid)
        self.net3 = nn.Linear(d_hid, d_hid * 2)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


class ComposabilityTest(MultiProcessTestCase):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 4

    @property
    def device(self):
        return self.rank

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    def test_pp_and_dcp(self):
        """
        Test that pipeline parallelism and distributed checkpointing can be used together and
        with saved correct FQNs
        """

        class AppState(Stateful):
            def __init__(self, model, optimizer):
                self.model = model
                self.optimizer = optimizer

            def state_dict(self):
                # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
                model_state_dict, optimizer_state_dict = get_state_dict(
                    self.model, self.optimizer
                )
                return {"model": model_state_dict, "optim": optimizer_state_dict}

            def load_state_dict(self, state_dict):
                # sets our state dicts on the model and optimizer, now that we've loaded
                set_state_dict(
                    self.model,
                    self.optimizer,
                    model_state_dict=state_dict["model"],
                    optim_state_dict=state_dict["optim"],
                )

        class PPModelChunk(nn.Module):
            def __init__(self, layers: nn.ModuleDict, start_index: int, end_index: int):
                super().__init__()
                # Filter layers based on start_index and end_index
                self.layers = nn.ModuleDict(
                    {str(i): layers[str(i)] for i in range(start_index, end_index)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        device = torch.device("cuda", self.device)
        torch.cuda.set_device(self.device)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        # create "entire model"
        total_layers = 8
        dim = 10
        full_model = nn.ModuleDict(
            {f"{i}": MLPModule(dim) for i in range(total_layers)}
        )
        # Calculate start and end indices based on rank
        start_index = self.rank * 2
        end_index = start_index + 2
        pp_model = PPModelChunk(full_model, start_index, end_index)

        pp_model.to(self.device)
        opt = torch.optim.Adam(pp_model.parameters(), lr=0.1)

        # perform work in a temp dir that is cleaned up after the test
        @with_temp_dir
        def _dcp_test(self):
            state_dict = {"app": AppState(pp_model, opt)}
            dcp.save(state_dict, checkpoint_id=self.temp_dir)
            # temp checkpoint
            sd: STATE_DICT_TYPE = {}
            _load_state_dict(
                sd,
                storage_reader=FileSystemReader(self.temp_dir),
                planner=_EmptyStateDictLoadPlanner(),
            )
            # Check parameter names in sd and compare with pp_model
            pp_model_param_names = set(pp_model.state_dict().keys())
            sd_param_names = set(sd["app"]["model"].keys())
            # Verify each parameter name in pp_model is contained in sd
            for param_name in pp_model_param_names:
                self.assertIn(
                    param_name,
                    sd_param_names,
                    f"Parameter name '{param_name}' not found in state_dict.",
                )

        _dcp_test(self)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 8+ GPUs")
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    @parametrize(
        "MixedPrecisionParam",
        [
            torch.bfloat16,
            torch.float32,
        ],
    )
    def test_3d_with_tp_dp_pp(self, ScheduleClass, MixedPrecisionParam):
        _device_raii = torch.device("cuda", self.device)
        torch.cuda.set_device(self.device)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        dim = 8
        tp_size = 2
        pp_size = 2
        num_microbatches = 8
        dp_size = self.world_size // (tp_size * pp_size)
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dp_size, pp_size, tp_size),
            mesh_dim_names=("dp", "pp", "tp"),
        )
        dp_mesh = device_mesh["dp"]
        tp_mesh = device_mesh["tp"]
        pp_mesh = device_mesh["pp"]
        pp_group = device_mesh["pp"].get_group()

        # create "entire model"
        total_layers = 8
        full_model = nn.ModuleList([MLPModuleEven(dim) for _ in range(total_layers)])

        # dummy loss needed just to force backwards to run in schedule step
        def loss_fn(y, target):
            return y.sum()

        # Apply DP to stage module
        def apply_fsdp(partial_model):
            # apply FSDP
            mp_policy = MixedPrecisionPolicy(
                param_dtype=MixedPrecisionParam,
                reduce_dtype=torch.float32,
            )
            fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
            for layer_id in range(len(partial_model)):
                fully_shard(
                    partial_model[layer_id],
                    **fsdp_config,
                    reshard_after_forward=False,
                )
            dp_model = fully_shard(partial_model, **fsdp_config)
            return dp_model

        def apply_tp(
            model: nn.Module,
            tp_mesh: DeviceMesh,
        ):
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
                "net3": ColwiseParallel(),
            }
            for layer in model:
                parallelize_module(layer, tp_mesh, parallelize_plan)
            return model

        # Attach to a schedule
        if issubclass(ScheduleClass, PipelineScheduleSingle):
            stage_idx = pp_group.rank()
            partial_model = nn.Sequential(
                *full_model[stage_idx * 2 : stage_idx * 2 + 2]
            )
            partial_model.to(self.device)

            tp_model = apply_tp(partial_model, tp_mesh)
            dp_model = apply_fsdp(tp_model)
            pipeline_stage = PipelineStage(
                dp_model,
                stage_idx,
                pp_group.size(),
                self.device,
                group=pp_group,
            )
            partial_models = [pipeline_stage.submod]
            pipeline_schedule = ScheduleClass(
                pipeline_stage,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )
        else:
            n_virtual = 2
            num_stages = pp_group.size() * n_virtual
            stages = []
            for i in range(n_virtual):
                stage_idx = pp_group.rank() + n_virtual * i
                # divide the model layers by the number of stages
                partial_model = nn.Sequential(*full_model[stage_idx : stage_idx + 1])
                partial_model.to(self.device)

                tp_model = apply_tp(partial_model, tp_mesh)
                dp_model = apply_fsdp(tp_model)
                stage = PipelineStage(
                    dp_model,
                    stage_idx,
                    num_stages,
                    self.device,
                    group=pp_group,
                )

                stages.append(stage)
                partial_models = [pipeline_stage.submod for pipeline_stage in stages]
            pipeline_schedule = ScheduleClass(
                stages,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )

        optimizer_kwargs = {
            "lr": 0.01,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": False,
            "foreach": True,
        }
        optimizers = [
            torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            for model in partial_models
        ]

        for train_step in range(5):
            for optimizer in optimizers:
                optimizer.zero_grad()
            inputs = torch.rand((num_microbatches, dim), device=self.device)
            labels = torch.rand((num_microbatches, dim), device=self.device)
            is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1
            if pp_mesh.get_local_rank() == 0:
                pipeline_schedule.step(inputs)
            elif is_last_stage:
                losses = []
                pipeline_schedule.step(target=labels, losses=losses)
            else:
                pipeline_schedule.step()

            for optimizer in optimizers:
                optimizer.step()

        torch.distributed.destroy_process_group()


instantiate_parametrized_tests(ComposabilityTest)

if __name__ == "__main__":
    run_tests()
