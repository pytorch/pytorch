# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A FlexShard bucket whose forward gets one input computed from another, as an
MoE's routed experts get its hidden states and the routing scores computed from
them, must not keep the forward's graph alive after backward. Under pipeline
parallelism that graph starts from the stage's received activation."""

import gc
import weakref

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import BucketSpec, flex_shard
from torch.distributed.fsdp._flex_shard.placements.shard import per_param_placements
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining._recv_buffers import _RecvInfo
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import HardwareClassification, run_tests


device_type = torch.device(get_devtype())

_DIM = 64
_NUM_EXPERTS = 4
_TOKENS = 32


class _Experts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(_NUM_EXPERTS, _DIM, _DIM) * 0.02)

    def forward(self, h: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("td,edk->tek", h, self.w)
        return (out * scores.unsqueeze(-1)).sum(1)


class _MoEBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(_DIM)
        self.router = torch.nn.Linear(_DIM, _NUM_EXPERTS, bias=False)
        self.experts = _Experts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scores = self.router(h).softmax(-1)
        return x + self.experts(h, scores)


def _sharded_block(mesh) -> _MoEBlock:
    torch.manual_seed(0)
    block = _MoEBlock().to(device_type)
    # The experts' bucket hooks the experts module, whose inputs are the
    # hidden states and the scores computed from them.
    flex_shard(
        block,
        buckets=[
            BucketSpec(
                ["norm.*", "router.*"], placement_fn=per_param_placements, mesh=mesh
            ),
            BucketSpec(["experts.*"], placement_fn=per_param_placements, mesh=mesh),
        ],
    )
    return block


class TestFlexShardPipelining(FSDPTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_dependent_inputs_free_graph_after_backward(self):
        block = _sharded_block(init_device_mesh(device_type.type, (self.world_size,)))
        gc.disable()
        try:
            for _ in range(4):
                # A leaf that requires grad, as a pipeline stage's received
                # activation is.
                x = torch.randn(_TOKENS, _DIM, device=device_type, requires_grad=True)
                block(x).sum().backward()
                x_ref = weakref.ref(x)
                del x
                self.assertIsNone(x_ref(), msg="the forward's graph outlived backward")
        finally:
            gc.enable()

    @skip_if_lt_x_gpu(2)
    def test_pipeline_frees_received_activations(self):
        # Two stages, one per rank, each sharded over its own rank.
        mesh = init_device_mesh(
            device_type.type, (self.world_size, 1), mesh_dim_names=("pp", "dp")
        )
        stage_index = self.rank
        stage = PipelineStage(
            _sharded_block(mesh["dp"]),
            stage_index,
            self.world_size,
            device_type,
            group=mesh.get_group("pp"),
        )
        num_microbatches = 8
        schedule = ScheduleGPipe(
            stage,
            n_microbatches=num_microbatches,
            loss_fn=lambda out, target: out.sum(),
        )
        received: list[weakref.ref] = []
        original_allocate = _RecvInfo.allocate_buffer

        def allocate_and_track(info, device):
            buffer = original_allocate(info, device)
            received.append(weakref.ref(buffer))
            return buffer

        _RecvInfo.allocate_buffer = allocate_and_track
        gc.disable()
        try:
            for step in range(3):
                torch.manual_seed(step)
                x = torch.randn(num_microbatches * _TOKENS, _DIM, device=device_type)
                if stage_index == 0:
                    schedule.step(x)
                else:
                    schedule.step(target=x)
                torch.cuda.synchronize()
                alive = sum(ref() is not None for ref in received)
                self.assertGreater(len(received), 0)
                self.assertEqual(
                    alive, 0, msg=f"step {step}: {alive} received buffers still alive"
                )
                received.clear()
        finally:
            gc.enable()
            _RecvInfo.allocate_buffer = original_allocate
        dist.barrier()


if __name__ == "__main__":
    run_tests()
