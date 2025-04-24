# Owner(s): ["oncall: distributed"]
import copy
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
)


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)
        self.init_weights()

    def init_weights(self):
        # ensure a proper init otherwise gradient tests will be more likely to get zero grad values
        torch.nn.init.kaiming_uniform_(
            self.net1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_uniform_(
            self.net2.weight, mode="fan_in", nonlinearity="relu"
        )

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
        self.init_weights()

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(
            self.net1.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_uniform_(
            self.net2.weight, mode="fan_in", nonlinearity="relu"
        )
        torch.nn.init.kaiming_uniform_(
            self.net3.weight, mode="fan_in", nonlinearity="relu"
        )

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


def loss_fn(y, target, scale=1e-4):
    # Scale the loss to simulate a small learning rate and avoid exploding grads
    return torch.nn.functional.cross_entropy(y, target) * scale


class ComposabilityTest(MultiProcContinousTest):
    world_size = 4

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
        torch.cuda.set_device(cls.device)

    def _build_mesh(self, mesh_shape=(2, 2), mesh_dim_names=("dp", "pp")):
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
        )
        return device_mesh

    def _rand_microbatches(self, dp_mesh, num_microbatches, dim, dtype=torch.float32):
        full = [
            torch.rand((num_microbatches, dim), device=self.device, dtype=dtype)
            for _ in range(dp_mesh.size())
        ]
        local = full[dp_mesh.get_local_rank()]
        local_mb = [[local[i].reshape((1, dim))] for i in range(num_microbatches)]
        return full, local, local_mb

    # build a pipeline stage
    def _build_pp_stage(
        self, pp_group, full_model, total_layers, apply_dp, stage_idx, num_stages
    ):
        # divide the model (e.g. 8 layers) by the number of stages
        layers_per_stage = total_layers // num_stages
        assert layers_per_stage * num_stages == total_layers
        # return offset so validation code can match partial layer back to orig model
        offset = stage_idx * layers_per_stage
        partial_model = nn.Sequential(
            *full_model[offset : (stage_idx + 1) * layers_per_stage]
        )
        partial_model.to(self.device)
        dp_model = apply_dp(partial_model)
        stage = PipelineStage(
            dp_model,
            stage_idx,
            num_stages,
            self.device,
            group=pp_group,
        )
        return stage, offset

    def _build_pp_schedule(
        self,
        ScheduleClass,
        num_microbatches,
        pp_group,
        full_model,
        total_layers,
        apply_dp,
        loss_fn,
    ):
        if issubclass(ScheduleClass, PipelineScheduleSingle):
            pipeline_stage, offset = self._build_pp_stage(
                pp_group,
                full_model,
                total_layers,
                apply_dp,
                pp_group.rank(),
                pp_group.size(),
            )

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
                stage, offset = self._build_pp_stage(
                    pp_group,
                    full_model,
                    total_layers,
                    apply_dp,
                    pp_group.rank() + n_virtual * i,
                    num_stages,
                )
                stages.append(stage)
                offsets.append(offset)
            partial_models = [pipeline_stage.submod for pipeline_stage in stages]
            pipeline_schedule = ScheduleClass(
                stages,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )
        return pipeline_schedule, partial_models, offsets

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize(
        "ScheduleClass",
        [
            ScheduleGPipe,
            ScheduleInterleaved1F1B,
            ScheduleInterleavedZeroBubble,
        ],
    )
    def test_pp_ddp(self, ScheduleClass):
        if ScheduleClass == ScheduleInterleavedZeroBubble:
            # TODO: DDP + InterleavedZeroBubble is not currently supported due to issue with DDP reducer not triggering
            # https://github.com/pytorch/pytorch/issues/144530
            return

        device_mesh = self._build_mesh((2, 2), ("dp", "pp"))
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # create "entire model"
        total_layers = 8
        num_microbatches = 8
        dim = 10
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(self.device)

        # Prepare inputs
        inputs, input_local, _ = self._rand_microbatches(dp_mesh, num_microbatches, dim)
        targets, target_local, _ = self._rand_microbatches(
            dp_mesh, num_microbatches, dim
        )

        def apply_dp(partial_model):
            return DDP(partial_model, process_group=dp_mesh.get_group())

        # Build pipeline stages, apply data parallelism and attach to a schedule
        pipeline_schedule, partial_models, offsets = self._build_pp_schedule(
            ScheduleClass,
            num_microbatches,
            pp_group,
            full_model,
            total_layers,
            apply_dp,
            loss_fn,
        )

        # Run the pipeline
        if pp_group.rank() == 0:
            pipeline_schedule.step(input_local)
        else:
            pipeline_schedule.step(target=target_local)

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the DDP all-reduce becomes a no-op.
        for sim_dp_rank in range(dp_mesh.size()):
            loss_fn(ref_model(inputs[sim_dp_rank]), targets[sim_dp_rank]).backward()
        ref_model.to(torch.float32)
        for p in ref_model.parameters():
            p.grad = p.grad.to(torch.float32)
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        ref_parameters = dict(ref_model.named_parameters())
        for partial_model, offset in zip(partial_models, offsets):
            for name, p in partial_model.named_parameters():
                parts = name.split(".")[
                    1:
                ]  # remove the DDP module. prefix (FSDP2 doesn't have one)
                parts[0] = str(int(parts[0]) + offset)
                name = ".".join(parts)
                ref_p = ref_parameters[name]
                torch.testing.assert_close(p.grad, ref_p.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize("dp_type", ["FSDP", "FSDP_MP"])
    @parametrize(
        "ScheduleClass",
        [
            Schedule1F1B,
            ScheduleInterleaved1F1B,
            ScheduleLoopedBFS,
            ScheduleInterleavedZeroBubble,
        ],
    )
    def test_pp_fsdp(self, dp_type, ScheduleClass):
        if TEST_WITH_ROCM:
            return

        device_mesh = self._build_mesh((2, 2), ("dp", "pp"))
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # fsdp_mixed-precision dtype
        mp_dtype = torch.bfloat16 if dp_type == "FSDP_MP" else torch.float32

        # create "entire model"
        total_layers = 8
        num_microbatches = 8
        dim = 10
        full_model = nn.ModuleList([MLPModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(self.device)
        if dp_type == "FSDP_MP":
            ref_model.to(dtype=mp_dtype)

        # Prepare inputs
        inputs, input_local, _ = self._rand_microbatches(
            dp_mesh, num_microbatches, dim, dtype=mp_dtype
        )
        targets, target_local, _ = self._rand_microbatches(
            dp_mesh, num_microbatches, dim, dtype=mp_dtype
        )

        # Apply FSDP to stage module
        def apply_dp(partial_model):
            mp_policy = MixedPrecisionPolicy(
                param_dtype=mp_dtype,
                reduce_dtype=torch.float32,
            )
            fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
            for layer in partial_model.children():
                fully_shard(
                    layer,
                    **fsdp_config,
                    reshard_after_forward=False,
                )
            return fully_shard(partial_model, **fsdp_config)

        # Build pipeline stages, apply data parallelism and attach to a schedule
        pipeline_schedule, partial_models, offsets = self._build_pp_schedule(
            ScheduleClass,
            num_microbatches,
            pp_group,
            full_model,
            total_layers,
            apply_dp,
            loss_fn,
        )

        # Run the pipeline
        if pp_group.rank() == 0:
            pipeline_schedule.step(input_local)
        else:
            pipeline_schedule.step(target=target_local)
        for m in partial_models:
            for p in m.parameters():
                assert p.grad is not None
                # introduce a race condition for FSDP's reduce-scatter which could corrupt gradients if pipelining
                # does not properly synchronize with FSDP
                p.grad.div_(2.0)
                p.grad.mul_(2.0)

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the FSDP reduce becomes a no-op.
        # (in fsdp case, we use one of these inputs on each DP rank)
        for sim_dp_rank in range(dp_mesh.size()):
            loss_fn(ref_model(inputs[sim_dp_rank]), targets[sim_dp_rank]).backward()
        ref_model.to(torch.float32)
        for p in ref_model.parameters():
            p.grad = p.grad.to(torch.float32)
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        # (we force FSDP's grads to be all-gathered (.full_tensor) to make it simpler)
        ref_parameters = dict(ref_model.named_parameters())
        for partial_model, offset in zip(partial_models, offsets):
            for name, p in partial_model.named_parameters():
                parts = name.split(".")
                parts[0] = str(int(parts[0]) + offset)
                name = ".".join(parts)
                ref_p = ref_parameters[name]
                self.assertTrue(isinstance(p.grad, DTensor))
                torch.testing.assert_close(p.grad.full_tensor(), ref_p.grad)


instantiate_parametrized_tests(ComposabilityTest)
if __name__ == "__main__":
    # Check if GPU and NCCL are available
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() > 1
    ):
        print(
            "c10d NCCL not available or not enough GPUs, skipping tests",
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
