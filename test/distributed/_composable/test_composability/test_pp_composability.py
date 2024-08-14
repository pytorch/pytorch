# Owner(s): ["oncall: distributed"]
import copy
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineScheduleRuntime,
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleFlexibleInterleaved1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)


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


class ComposabilityTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.cuda.device_count()
        cls.device = torch.device(f"cuda:{dev_id}")
        # TODO: investigate why this is needed to prevent multiple NCCL ranks from hitting the same device
        torch.cuda.set_device(cls.device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize("dp_type", ["DDP", "FSDP"])
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleFlexibleInterleaved1F1B,
        ],
    )
    @parametrize("use_new_runtime", [False, True])
    def test_manual_with_data_parallel(self, dp_type, ScheduleClass, use_new_runtime):
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(2, 2), mesh_dim_names=("dp", "pp")
        )
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # create "entire model"
        total_layers = 8
        dim = 10
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(self.device)

        # Prepare inputs
        num_microbatches = 8
        inputs = [
            torch.rand((num_microbatches, dim), device=self.device)
            for _ in range(dp_mesh.size())
        ]
        input = inputs[dp_mesh.get_local_rank()]
        input_mb = [[input[i].reshape((1, dim))] for i in range(num_microbatches)]

        # dummy loss needed just to force backwards to run in schedule step
        def loss_fn(y, target):
            return y.sum()

        # Get stage module i from the entire model
        def get_stage_module(stage_idx, num_stages):
            # divide the model (8 layers) by the number of stages
            layers_per_stage = total_layers // num_stages
            assert layers_per_stage * num_stages == total_layers
            # return offset so validation code can match partial layer back to orig model
            offset = stage_idx * layers_per_stage
            partial_model = nn.Sequential(
                *full_model[offset : (stage_idx + 1) * layers_per_stage]
            )
            partial_model.to(self.device)
            return partial_model, offset

        # Apply DP to stage module
        def apply_dp(partial_model, dp_type):
            if dp_type == "FSDP":
                # apply FSDP
                mp_policy = MixedPrecisionPolicy(
                    # TODO(whc) need to fix PP + FSDP-mixed-precision
                    # tracer for PP assumes f32 and is caught off guard when runtime FSDP interacts using bf16 inputs
                    # param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
                for layer in partial_model.children():
                    fully_shard(
                        layer,
                        **fsdp_config,
                        reshard_after_forward=False,
                    )
                dp_model = fully_shard(partial_model, **fsdp_config)
            elif dp_type == "DDP":
                dp_model = DDP(partial_model, process_group=dp_mesh.get_group())
            else:
                raise RuntimeError(f"unsupported dp type {dp_type}")
            return dp_model

        # Create pipeline stage
        def build_stage(stage_idx, num_stages):
            partial_model, offset = get_stage_module(stage_idx, num_stages)
            dp_model = apply_dp(partial_model, dp_type)
            stage = PipelineStage(
                dp_model,
                stage_idx,
                num_stages,
                self.device,
                group=pp_group,
                input_args=input_mb[0],
            )
            return stage, offset

        # Attach to a schedule
        if issubclass(ScheduleClass, PipelineScheduleSingle):
            if use_new_runtime:
                # Can't test PipelineScheduleSingle classes using new runtime
                # return should still clean up this test instance correctly
                return
            pipeline_stage, offset = build_stage(pp_group.rank(), pp_group.size())
            partial_models = [pipeline_stage.submod]
            offsets = [offset]
            pipeline_schedule = ScheduleClass(
                pipeline_stage,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )
        else:
            n_virtual = 2
            num_stages = pp_group.size() * n_virtual
            stages = []
            offsets = []
            for i in range(n_virtual):
                stage, offset = build_stage(pp_group.rank() + n_virtual * i, num_stages)
                stages.append(stage)
                offsets.append(offset)
                partial_models = [pipeline_stage.submod for pipeline_stage in stages]
            pipeline_schedule = ScheduleClass(
                stages,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )
            if use_new_runtime:
                old_sch = pipeline_schedule
                pipeline_schedule = _PipelineScheduleRuntime(
                    stages,
                    num_microbatches,
                    loss_fn=loss_fn,
                    stage_index_to_group_rank=old_sch.stage_index_to_group_rank,
                )
                pipeline_schedule._load_actions(old_sch.pipeline_order)

        # Run
        pipeline_schedule._step_microbatches(arg_mbs=input_mb, target_mbs=input_mb)

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the FSDP reduce becomes a no-op.
        # (in fsdp case, we use one of these inputs on each DP rank)
        (ref_model(inputs[0]).sum()).backward()
        (ref_model(inputs[1]).sum()).backward()

        # simulate the built-in averaging done by FSDP
        for p in ref_model.parameters():
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        # (we force FSDP's grads to be all-gathered (.full_tensor) to make it simpler)
        ref_parameters = dict(ref_model.named_parameters())
        if dp_type == "FSDP":
            for partial_model, offset in zip(partial_models, offsets):
                for name, p in partial_model.named_parameters():
                    parts = name.split(".")
                    parts[0] = str(int(parts[0]) + offset)
                    name = ".".join(parts)
                    ref_p = ref_parameters[name]
                    self.assertTrue(isinstance(p.grad, DTensor))
                    self.assertEqual(ref_p.grad, p.grad.full_tensor())
        elif dp_type == "DDP":
            for partial_model, offset in zip(partial_models, offsets):
                for name, p in partial_model.named_parameters():
                    parts = name.split(".")[1:]  # remove the "module." prefix
                    parts[0] = str(int(parts[0]) + offset)
                    name = ".".join(parts)
                    ref_p = ref_parameters[name]
                    self.assertEqual(ref_p.grad, p.grad)


instantiate_parametrized_tests(ComposabilityTest)

if __name__ == "__main__":
    # Check if GPU and NCCL are available
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() >= 4
    ):
        print(
            "Composability test requires at least 4 GPUs, but not enough found, skipping",
            file=sys.stderr,
        )
        sys.exit(0)

    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 4))

    if rank != -1:
        # Launched with torchrun or other multi-proc launchers. Directly run the test.
        ComposabilityTest.run_rank(rank, world_size)
    else:
        # Launched as a single process. Spawn subprocess to run the tests.
        # Also need a rendezvous file for `init_process_group` purpose.
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        torch.multiprocessing.spawn(
            ComposabilityTest.run_rank,
            nprocs=world_size,
            args=(world_size, rdvz_file),
        )
