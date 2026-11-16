# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard training steps captured in a CUDA graph and replayed, as
torchtitan's CUDA-graph wrapper runs them, bitwise against eager steps."""

from collections.abc import Callable

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import bucket_runtime, BucketSpec, flex_shard
from torch.distributed.fsdp._flex_shard.placements.shard import per_param_placements
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import HardwareClassification, run_tests
from torch.testing._internal.distributed.common_flex_shard import make_test_sgd


device_type = torch.device(get_devtype())

_STEPS = 4


def _build(mesh, *, reshard_after_forward: bool) -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16)
    ).to(device_type)
    flex_shard(
        model,
        buckets=[
            BucketSpec(
                [pattern],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
            for pattern in ("0.*", "2.*")
        ],
    )
    return model


def _sync_step(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    loss = model(x).sum()
    loss.backward()
    return loss


def _accumulate_step(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Two microbatches in one step: the first without gradient sync.
    losses = []
    for microbatch, chunk in enumerate(x.chunk(2)):
        model.set_requires_gradient_sync(microbatch == 1)
        loss = model(chunk).sum()
        loss.backward()
        losses.append(loss)
    return losses[0] + losses[1]


def _manual_finalization_step(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # As a pipeline stage: backwards without sync, then finalize_backward with
    # an asynchronous handle that the step waits on.
    model.set_manual_backward_finalization(True)
    model.set_requires_gradient_sync(False)
    losses = []
    for chunk in x.chunk(2):
        loss = model(chunk).sum()
        loss.backward()
        losses.append(loss)
    model.set_requires_gradient_sync(True)
    model.finalize_backward(async_op=True).wait()
    model.set_manual_backward_finalization(False)
    return losses[0] + losses[1]


def _async_unshard_step(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # As a multi-stage pipeline schedule: unshard ahead of the forward.
    model.unshard(async_op=True)
    return _sync_step(model, x)


class TestFlexShardCUDAGraph(FSDPTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 2

    def _assert_idle(self, model: torch.nn.Module, context: str) -> None:
        """Nothing a step started is left for the next one, as with FSDP2."""
        contexts = getattr(model, bucket_runtime._EAGER_COMM_CONTEXTS_ATTR).values()
        for comm_context in contexts:
            for bucket in comm_context.buckets:
                self.assertFalse(bucket.is_unsharded, msg=context)
            self.assertEqual(comm_context.pending_unshards, [], msg=context)
            self.assertEqual(comm_context.reduce_grad_states, [], msg=context)
            self.assertEqual(comm_context.retired_reduce_grad_states, [], msg=context)
            self.assertIsNone(comm_context.pending_finalization, msg=context)
            self.assertFalse(comm_context.post_backward_callback_queued, msg=context)

    def _train(
        self,
        model: torch.nn.Module,
        step: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        inputs: list[torch.Tensor],
        *,
        graph: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Train ``_STEPS`` steps. With ``graph``, as torchtitan's wrapper: one
        eager step on the capture stream, then capture the forward-backward
        once and replay it, restoring the captured gradients after each
        replay; the optimizer runs eagerly."""
        params = [param for param in model.parameters() if param.requires_grad]
        optim = make_test_sgd(params, lr=0.1)
        losses = []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cuda_graph, static_x, static_loss, grads = None, None, None, []
        with torch.cuda.stream(stream):
            for index, x in enumerate(inputs):
                if not graph or index == 0:
                    losses.append(step(model, x).detach().clone())
                else:
                    if cuda_graph is None:
                        self._assert_idle(model, "before capture")
                        static_x = x.clone()
                        cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cuda_graph, stream=stream):
                            static_loss = step(model, static_x)
                        self._assert_idle(model, "after capture")
                        grads = [(param, param.grad) for param in params]
                    static_x.copy_(x)
                    cuda_graph.replay()
                    for param, grad in grads:
                        param.grad = grad
                    losses.append(static_loss.detach().clone())
                self._assert_idle(model, f"after step {index}")
                optim.step()
                optim.zero_grad(set_to_none=True)
        torch.cuda.current_stream().wait_stream(stream)
        return losses, [param.detach().clone() for param in params]

    def _check(self, step, *, reshard_after_forward: bool = True) -> None:
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        generator = torch.Generator().manual_seed(self.rank)
        inputs = [
            torch.randn(8, 16, generator=generator).to(device_type)
            for _ in range(_STEPS)
        ]
        eager = self._train(
            _build(mesh, reshard_after_forward=reshard_after_forward),
            step,
            inputs,
            graph=False,
        )
        replayed = self._train(
            _build(mesh, reshard_after_forward=reshard_after_forward),
            step,
            inputs,
            graph=True,
        )
        for kind, expected, actual in zip(("losses", "params"), eager, replayed):
            for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
                self.assertTrue(torch.equal(want, got), msg=f"{kind}[{index}]")

    @skip_if_lt_x_gpu(2)
    def test_sync_step(self):
        for reshard_after_forward in (True, False):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                self._check(_sync_step, reshard_after_forward=reshard_after_forward)

    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation_step(self):
        self._check(_accumulate_step)

    @skip_if_lt_x_gpu(2)
    def test_manual_finalization_step(self):
        self._check(_manual_finalization_step, reshard_after_forward=False)

    @skip_if_lt_x_gpu(2)
    def test_async_unshard_step(self):
        self._check(_async_unshard_step, reshard_after_forward=False)

    @skip_if_lt_x_gpu(2)
    def test_explicit_prefetch_step(self):
        mesh = init_device_mesh(device_type.type, (self.world_size,))

        def build(mesh, *, reshard_after_forward):
            model = _build(mesh, reshard_after_forward=reshard_after_forward)
            first, second = model.sharded_bucket_storages
            first.set_buckets_to_forward_prefetch([second])
            second.set_buckets_to_backward_prefetch([first])
            return model

        generator = torch.Generator().manual_seed(self.rank)
        inputs = [
            torch.randn(8, 16, generator=generator).to(device_type)
            for _ in range(_STEPS)
        ]
        eager = self._train(
            build(mesh, reshard_after_forward=True), _sync_step, inputs, graph=False
        )
        replayed = self._train(
            build(mesh, reshard_after_forward=True), _sync_step, inputs, graph=True
        )
        for kind, expected, actual in zip(("losses", "params"), eager, replayed):
            for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
                self.assertTrue(torch.equal(want, got), msg=f"{kind}[{index}]")


if __name__ == "__main__":
    run_tests()
