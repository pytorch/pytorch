# Owner(s): ["oncall: distributed"]
import copy
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.replicate_with_fsdp import replicate
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
from torch.testing._internal.common_distributed import (
    at_least_x_gpu,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
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


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = torch.distributed.get_default_backend_for_device(device_type)


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
        return backend

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
        return 8

    @property
    def device(self):
        return self.rank

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(8)
    @skip_but_pass_in_sandcastle_if(not at_least_x_gpu(8), "Test requires 8+ GPUs")
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

        device = torch.device(device_type, self.device)
        torch.accelerator.set_device_index(self.device)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            backend=backend,
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
        start_index = self.rank
        end_index = start_index + 1
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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(8)
    @skip_but_pass_in_sandcastle_if(not at_least_x_gpu(8), "Test requires 8+ GPUs")
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
        torch.accelerator.set_device_index(self.device)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            backend=backend,
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
            device_type,
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

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            n_virtual = 1
        else:
            n_virtual = 2

        num_stages = pp_group.size() * n_virtual
        layers_per_stage = total_layers // num_stages
        stages = []
        for i in range(n_virtual):
            stage_idx = pp_group.rank() + pp_group.size() * i
            start_layer = stage_idx * layers_per_stage
            end_layer = start_layer + layers_per_stage
            # divide the model layers by the number of stages
            partial_model = nn.Sequential(*full_model[start_layer:end_layer])
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

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            stages = stages[0]

        pipeline_schedule = ScheduleClass(
            stages,
            n_microbatches=num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
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

        for _train_step in range(5):
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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(8)
    @skip_but_pass_in_sandcastle_if(not at_least_x_gpu(8), "Test requires 8+ GPUs")
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
    def test_replicate_pp(self, ScheduleClass, MixedPrecisionParam):
        torch.accelerator.set_device_index(self.device)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        dim = 8
        pp_size = 2
        num_microbatches = 8
        replicate_size = self.world_size // (pp_size)
        device_mesh = init_device_mesh(
            device_type,
            mesh_shape=(replicate_size, pp_size),
            mesh_dim_names=("replicate", "pp"),
        )
        torch.manual_seed(42)
        dp_mesh = device_mesh["replicate"]
        pp_mesh = device_mesh["pp"]
        pp_group = device_mesh["pp"].get_group()

        # create "entire model"
        total_layers = 8
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_full_model = copy.deepcopy(full_model)

        # dummy loss needed just to force backwards to run in schedule step
        def loss_fn(y, target):
            return y.sum()

        # Apply DP to stage module
        def apply_replicate(partial_model):
            # apply replicate
            mp_policy = MixedPrecisionPolicy(
                param_dtype=MixedPrecisionParam,
                reduce_dtype=torch.float32,
            )
            replicate_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
            for layer_id in range(len(partial_model)):
                replicate(
                    partial_model[layer_id],
                    **replicate_config,
                )
            dp_model = replicate(partial_model, **replicate_config)
            return dp_model

        # Apply same precision to reference model (without replicate)
        def apply_same_precision(partial_model):
            if MixedPrecisionParam != torch.float32:
                # Cast to same precision as pipeline model
                partial_model = partial_model.to(dtype=MixedPrecisionParam)
            return partial_model

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            n_virtual = 1
        else:
            n_virtual = 2

        num_stages = pp_group.size() * n_virtual
        layers_per_stage = total_layers // num_stages
        stages = []
        ref_stages = []
        for i in range(n_virtual):
            stage_idx = pp_group.rank() + pp_group.size() * i
            start_layer = stage_idx * layers_per_stage
            end_layer = start_layer + layers_per_stage
            # divide the model layers by the number of stages
            partial_model = nn.Sequential(*full_model[start_layer:end_layer])
            partial_model.to(self.device)

            ref_partial_model = nn.Sequential(*ref_full_model[start_layer:end_layer])
            ref_partial_model.to(self.device)

            dp_model = apply_replicate(partial_model)
            ref_dp_model = apply_same_precision(ref_partial_model)

            stage = PipelineStage(
                dp_model,
                stage_idx,
                num_stages,
                self.device,
                group=pp_group,
            )

            ref_stage = PipelineStage(
                ref_dp_model,
                stage_idx,
                num_stages,
                self.device,
                group=pp_group,
            )

            stages.append(stage)
            ref_stages.append(ref_stage)

            partial_models = [pipeline_stage.submod for pipeline_stage in stages]
            ref_partial_models = [
                pipeline_stage.submod for pipeline_stage in ref_stages
            ]

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            stages = stages[0]
            ref_stages = ref_stages[0]

        pipeline_schedule = ScheduleClass(
            stages,
            n_microbatches=num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
        )

        ref_pipeline_schedule = ScheduleClass(
            ref_stages,
            n_microbatches=num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
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

        ref_optimizers = [
            torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            for model in ref_partial_models
        ]

        for _train_step in range(5):
            for optimizer in optimizers:
                optimizer.zero_grad()
            for ref_optimizer in ref_optimizers:
                ref_optimizer.zero_grad()

            inputs = torch.rand(
                (num_microbatches, dim), device=self.device, dtype=MixedPrecisionParam
            )
            labels = torch.rand(
                (num_microbatches, dim), device=self.device, dtype=MixedPrecisionParam
            )
            is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1
            if pp_mesh.get_local_rank() == 0:
                pipeline_schedule.step(inputs)
                ref_pipeline_schedule.step(inputs)
            elif is_last_stage:
                losses = []
                ref_losses = []
                pipeline_schedule.step(target=labels, losses=losses)
                ref_pipeline_schedule.step(target=labels, losses=ref_losses)

                for loss, ref_loss in zip(losses, ref_losses):
                    self.assertEqual(loss, ref_loss)
            else:
                pipeline_schedule.step()
                ref_pipeline_schedule.step()

            for optimizer in optimizers:
                optimizer.step()
            for ref_optimizer in ref_optimizers:
                ref_optimizer.step()

        torch.distributed.destroy_process_group()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(8)
    @skip_but_pass_in_sandcastle_if(not at_least_x_gpu(8), "Test requires 8+ GPUs")
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
    def test_replicate_pp_grads(self, ScheduleClass):
        torch.accelerator.set_device_index(self.device)
        store = torch.distributed.FileStore(self.file_name, self.world_size)
        torch.distributed.init_process_group(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        dim = 8
        pp_size = 2
        num_microbatches = 8
        replicate_size = self.world_size // (pp_size)
        device_mesh = init_device_mesh(
            device_type,
            mesh_shape=(replicate_size, pp_size),
            mesh_dim_names=("replicate", "pp"),
        )
        torch.manual_seed(42)
        dp_mesh = device_mesh["replicate"]
        pp_mesh = device_mesh["pp"]
        pp_group = device_mesh["pp"].get_group()
        dp_group = device_mesh["replicate"].get_group()

        # create "entire model"
        total_layers = 8
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model)).to(self.device)

        # dummy loss needed just to force backwards to run in schedule step
        def loss_fn(y, target):
            return y.sum()

        # Simulate microbatch processing for reference model
        def simulate_stage_forward_backward(model, inputs, labels):
            """Simulate forward and backward passes through stages for microbatch processing"""
            batch_size, _ = inputs.shape
            total_loss = 0

            # Split inputs into microbatches
            microbatch_size = batch_size // num_microbatches

            for mb_idx in range(num_microbatches):
                start_idx = mb_idx * microbatch_size
                end_idx = start_idx + microbatch_size
                mb_input = inputs[start_idx:end_idx]
                mb_label = labels[start_idx:end_idx] if labels is not None else None

                # Simulate stage-by-stage processing
                if issubclass(ScheduleClass, PipelineScheduleSingle):
                    num_stages = pp_group.size()
                    layers_per_stage = total_layers // pp_group.size()  # 8 // 2 = 4
                else:
                    n_virtual = 2
                    num_stages = pp_group.size() * n_virtual
                    layers_per_stage = total_layers // num_stages

                # Forward pass through all stages
                x = mb_input

                for stage in range(num_stages):
                    start_layer = stage * layers_per_stage
                    end_layer = start_layer + layers_per_stage

                    # Process layers for this stage
                    for layer_idx in range(start_layer, min(end_layer, len(model))):
                        x = model[layer_idx](x)

                mb_loss = loss_fn(x, mb_label)
                total_loss += mb_loss

                # Backward pass
                mb_loss.backward()

            return total_loss / num_microbatches

        # Apply replicate to stage module
        def apply_replicate(partial_model):
            for layer_id in range(len(partial_model)):
                replicate(
                    partial_model[layer_id],
                    mesh=dp_mesh,
                )
            dp_model = replicate(partial_model, mesh=dp_mesh)
            return dp_model

        def pipelined_models_parameters(start_layer, model):
            layer_idx = start_layer

            for layer in model.children():
                for name, param in layer.named_parameters():
                    updated_param_name = f"{layer_idx}.{name}"
                    pipeline_model_parameter_dict[updated_param_name] = param
                layer_idx += 1

        def check_gradient_parity(
            pipeline_model_parameter_dict, ref_model_parameter_dict
        ):
            for parameter in pipeline_model_parameter_dict:
                assert parameter in ref_model_parameter_dict

                pipeline_parameter = pipeline_model_parameter_dict[parameter]
                if pipeline_parameter.grad is not None:
                    pipeline_parameter_grad = pipeline_parameter.grad.to_local()
                    ref_parameter = ref_model_parameter_dict[parameter]
                    if ref_parameter.grad is not None:
                        torch.testing.assert_close(
                            pipeline_parameter_grad,
                            ref_parameter.grad,
                            rtol=1e-4,
                            atol=1e-5,
                        )
                    else:
                        assert pipeline_parameter.grad is None

        pipeline_model_parameter_dict = {}

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            n_virtual = 1
        else:
            n_virtual = 2

        num_stages = pp_group.size() * n_virtual
        layers_per_stage = total_layers // num_stages
        stages = []
        for i in range(n_virtual):
            stage_idx = pp_group.rank() + pp_group.size() * i
            start_layer = stage_idx * layers_per_stage
            end_layer = start_layer + layers_per_stage
            # divide the model layers by the number of stages
            partial_model = nn.Sequential(*full_model[start_layer:end_layer])
            partial_model.to(self.device)

            dp_model = apply_replicate(partial_model)
            pipelined_models_parameters(start_layer, dp_model)
            stage = PipelineStage(
                dp_model,
                stage_idx,
                num_stages,
                self.device,
                group=pp_group,
            )

            stages.append(stage)
            partial_models = [pipeline_stage.submod for pipeline_stage in stages]

        if issubclass(ScheduleClass, PipelineScheduleSingle):
            stages = stages[0]

        pipeline_schedule = ScheduleClass(
            stages,
            n_microbatches=num_microbatches,
            loss_fn=loss_fn,
            scale_grads=False,
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

        ref_optimizer = torch.optim.AdamW(ref_model.parameters(), **optimizer_kwargs)

        # Helper function to simulate all-reduce for reference model gradients
        def simulate_all_reduce_grads(model, group):
            """Simulate all-reduce operation on gradients like replicate does"""
            for param in model.parameters():
                if param.grad is not None:
                    # Scale by the number of replicas (like replicate does)
                    param.grad.div_(group.size())
                    # Simulate all-reduce
                    torch.distributed.all_reduce(param.grad, group=group)

        ref_model_parameter_dict = {}
        ref_model_parameter_dict = dict(ref_model.named_parameters())

        torch.manual_seed(42 + self.rank)
        for _ in range(5):
            for optimizer in optimizers:
                optimizer.zero_grad()
            ref_optimizer.zero_grad()

            inputs = torch.rand((num_microbatches, dim), device=self.device)
            labels = torch.rand((num_microbatches, dim), device=self.device)

            # Ensure all ranks use the same inputs/labels for comparison
            torch.distributed.broadcast(inputs, 0)
            torch.distributed.broadcast(labels, 0)

            is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

            # Run pipeline schedule
            if pp_mesh.get_local_rank() == 0:
                pipeline_schedule.step(inputs)
            elif is_last_stage:
                losses = []
                pipeline_schedule.step(target=labels, losses=losses)
            else:
                pipeline_schedule.step()

            # Run reference model simulation
            if is_last_stage:
                ref_loss = simulate_stage_forward_backward(ref_model, inputs, labels)
                # Simulate all-reduce on reference model gradients
                simulate_all_reduce_grads(ref_model, dp_group)

                # Compare losses - only check on last stage where we have losses
                if "losses" in locals() and len(losses) > 0:
                    # Average the microbatch losses to match ref_loss
                    avg_pipeline_loss = sum(losses) / len(losses)
                    torch.testing.assert_close(
                        avg_pipeline_loss, ref_loss, rtol=1e-4, atol=1e-5
                    )
            else:
                # For non-last stages, still run ref model to generate gradients
                simulate_stage_forward_backward(ref_model, inputs, None)
                simulate_all_reduce_grads(ref_model, dp_group)

            # Step optimizers
            for optimizer in optimizers:
                optimizer.step()
            ref_optimizer.step()

            check_gradient_parity(
                pipeline_model_parameter_dict, ref_model_parameter_dict
            )
        torch.distributed.destroy_process_group()


instantiate_parametrized_tests(ComposabilityTest)

if __name__ == "__main__":
    run_tests()
