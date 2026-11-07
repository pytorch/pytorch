# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard ``Shard`` buckets against FSDP2's ``fully_shard``, bit for bit."""

import contextlib
import copy
import functools
import itertools
from collections import defaultdict
from collections.abc import Callable, Iterator

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy as FSDPMixedPrecisionPolicy,
)
from torch.distributed.fsdp._flex_shard import (
    BucketSpec,
    flex_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.fsdp._flex_shard.placements.shard import Shard
from torch.distributed.tensor import Shard as DTensorShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import HardwareClassification, run_tests


device_type = torch.device(get_devtype())

_VOCAB = 11
# Odd, so every Shard(0) parameter below pads its last chunks.
_DIM = 13
# Divisible by the world size, as FSDP2 requires for Shard(1).
_HIDDEN = 8
_BATCH = 6
# Param/reduce dtypes, gradient divide factors (None averages over the mesh;
# torchtitan sets 1.0 and scales the loss), and (microbatches, no_sync).
_DTYPE_CONFIGS = [
    (None, None),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, None),
]
_DIVIDE_FACTORS = [None, 1.0]
_ACCUMULATIONS = [(1, False), (2, False), (2, True)]
# Chunks of the batch for the chunked loss.
_NUM_LOSS_CHUNKS = 3


def _configs() -> Iterator[tuple[tuple, str]]:
    """Yield every combination of the above, with a description."""
    for (param_dtype, reduce_dtype), divide_factor, (
        microbatches,
        no_sync,
    ) in itertools.product(_DTYPE_CONFIGS, _DIVIDE_FACTORS, _ACCUMULATIONS):
        config = (param_dtype, reduce_dtype, divide_factor, microbatches, no_sync)
        context = (
            f"param_dtype={param_dtype} reduce_dtype={reduce_dtype} "
            f"divide_factor={divide_factor} microbatches={microbatches} "
            f"no_sync={no_sync}"
        )
        yield config, context


class _StackedLinear(nn.Module):
    """Two projections stacked as torchtitan stores them: weight ``[N, F, D]``
    and bias ``[N, F]``, sharded on their matrix rows, ``Shard(1)``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, _HIDDEN, _DIM))
        self.bias = nn.Parameter(torch.randn(2, _HIDDEN))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.einsum("bd,nfd->bnf", x, self.weight) + self.bias).sum(1)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Registered before the children, but FSDP2's post-order walk takes it
        # after their parameters.
        self.scale = nn.Parameter(torch.randn(_DIM))
        self.stacked = _StackedLinear()
        self.proj = nn.Linear(_HIDDEN, _DIM)
        # Never gets a grad, so FSDP2 leaves it out of the reduce-scatter. Its
        # three rows leave the last of four ranks an empty shard.
        self.unused = nn.Linear(_DIM, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale is used twice, so its grad accumulates two uses in a backward.
        return x + self.proj(torch.relu(self.stacked(x * self.scale))) * self.scale


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(_VOCAB, _DIM)
        self.layers = nn.ModuleList(_Block() for _ in range(2))
        self.norm = nn.LayerNorm(_DIM)
        self.lm_head = nn.Linear(_DIM, _VOCAB, bias=False)
        self.norm.bias.requires_grad_(False)
        # As torchtitan's decoder under its chunked loss: return the norm's
        # output and leave lm_head to the loss.
        self.skip_lm_head = False
        # As DeepSeek V3's multi-token prediction, which runs the norm and
        # lm_head again on shifted hidden states: call the head group twice.
        self.second_head_call = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        if self.skip_lm_head:
            return self.norm(h)
        logits = self.lm_head(self.norm(h))
        if self.second_head_call:
            logits = logits + self.lm_head(self.norm(h.roll(1, 0)))
        return logits


def _shard_dim(fqn: str) -> int:
    return 1 if ".stacked." in fqn else 0


def _apply_fsdp2(
    model: _Decoder,
    mesh,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    divide_factor: float | None,
    head_reshard_after_forward: bool = False,
) -> None:
    """Group the decoder as torchtitan's ``apply_fsdp_to_decoder`` does."""
    dims = {param: _shard_dim(fqn) for fqn, param in model.named_parameters()}
    fsdp_config = {
        "mesh": mesh,
        "mp_policy": FSDPMixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        ),
    }
    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    fully_shard(
        [model.norm, model.lm_head],
        **fsdp_config,
        reshard_after_forward=head_reshard_after_forward,
    )
    for layer in model.layers:
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=True,
            shard_placement_fn=lambda param: DTensorShard(dims[param]),
        )
    fully_shard(model, **fsdp_config)
    if divide_factor is not None:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.set_gradient_divide_factor(divide_factor)


def _apply_flex_shard(
    model: _Decoder,
    mesh,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    divide_factor: float | None,
    fsdp2_compatible: bool = False,
    head_reshard_after_forward: bool = False,
    fsdp2_scheduling: bool = False,
) -> None:
    def placement_fn(named_params, mesh):
        del mesh
        return {fqn: (Shard(_shard_dim(fqn)),) for fqn, _ in named_params}

    def bucket(patterns: list[str], reshard_after_forward: bool) -> BucketSpec:
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
            gradient_divide_factor=divide_factor,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=fsdp2_compatible,
        )

    flex_shard(
        model,
        buckets=[
            bucket(["tok_embeddings"], True),
            *(bucket([f"layers.{i}"], True) for i in range(len(model.layers))),
            bucket(["norm", "lm_head"], head_reshard_after_forward),
        ],
    )
    if fsdp2_scheduling:
        model.set_post_backward_per_call(True)
        model.set_keep_prefetches_until_used(True)


@contextlib.contextmanager
def _record_collective_inputs() -> Iterator[dict[str, list[torch.Tensor]]]:
    """Record every all-gather and reduce-scatter input, by collective.

    Both call ``dist.reduce_scatter_single``.
    """
    recorded: dict[str, list[torch.Tensor]] = defaultdict(list)
    all_gather_single = dist.all_gather_single
    reduce_scatter_single = c10d.reduce_scatter_single

    def record_all_gather(output, input, *args, **kwargs):
        recorded["all_gather"].append(input.detach().clone())
        return all_gather_single(output, input, *args, **kwargs)

    def record_reduce_scatter(output, input, *args, **kwargs):
        recorded["reduce_scatter"].append(input.detach().clone())
        return reduce_scatter_single(output, input, *args, **kwargs)

    patches = [
        (dist, "all_gather_single", record_all_gather),
        (dist, "reduce_scatter_single", record_reduce_scatter),
        (c10d, "reduce_scatter_single", record_reduce_scatter),
    ]
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
    try:
        for owner, name, patched in patches:
            setattr(owner, name, patched)
        yield recorded
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _train(
    model: nn.Module,
    *,
    rank: int,
    steps: int,
    microbatches: int,
    no_sync: bool,
) -> list[dict[str, object]]:
    """Train with AdamW; return per-step losses, local grads and params."""
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=True)
    history: list[dict[str, object]] = []
    for step in range(steps):
        losses = []
        for microbatch in range(microbatches):
            if no_sync:
                model.set_requires_gradient_sync(microbatch == microbatches - 1)
            generator = torch.Generator().manual_seed(
                10_000 * step + 100 * microbatch + rank
            )
            tokens, targets = (
                torch.randint(_VOCAB, (_BATCH,), generator=generator).to(device_type)
                for _ in range(2)
            )
            loss = F.cross_entropy(model(tokens).float(), targets)
            loss.backward()
            losses.append(loss.detach())
        grads = {
            fqn: None if param.grad is None else _local(param.grad).detach().clone()
            for fqn, param in model.named_parameters()
        }
        optim.step()
        optim.zero_grad()
        params = {
            fqn: _local(param).detach().clone()
            for fqn, param in model.named_parameters()
        }
        history.append({"losses": losses, "grads": grads, "params": params})
    return history


def _set_head_requires_gradient_sync(head, requires_gradient_sync: bool) -> None:
    if isinstance(head, FSDPModule):
        head.set_requires_gradient_sync(requires_gradient_sync, recurse=False)
    else:
        head.set_requires_gradient_sync(requires_gradient_sync)


def _head_reshard_settings(head) -> tuple[bool, bool]:
    if isinstance(head, FSDPModule):
        # FSDPModule has setters for these but no getters.
        (group,) = head._get_fsdp_state()._fsdp_param_groups
        return group._reshard_after_forward, group.reshard_after_backward
    return head.reshard_after_forward, head.reshard_after_backward


def _chunked_loss(
    model: _Decoder, head, tokens: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Forward and backward as torchtitan's ChunkedLossWrapper runs them.

    ``head`` is lm_head's fully_shard group or its bucket's storage. It stays
    unsharded across the chunks, without gradient sync until the last one,
    and gets its reshard settings back after them; the gradients of the
    hidden states then go through the model in one backward.
    """
    hidden = model(tokens)
    reshard_after_forward, reshard_after_backward = _head_reshard_settings(head)
    head.set_reshard_after_forward(False)
    head.set_reshard_after_backward(False)
    _set_head_requires_gradient_sync(head, False)
    head.unshard()
    chunks = [
        chunk.detach().requires_grad_() for chunk in hidden.chunk(_NUM_LOSS_CHUNKS)
    ]
    loss = torch.zeros((), device=device_type)
    for idx, (chunk, target) in enumerate(
        zip(chunks, targets.chunk(_NUM_LOSS_CHUNKS), strict=True)
    ):
        if idx == _NUM_LOSS_CHUNKS - 1:
            _set_head_requires_gradient_sync(head, True)
        chunk_loss = (
            F.cross_entropy(model.lm_head(chunk).float(), target, reduction="sum")
            / _BATCH
        )
        chunk_loss.backward()
        loss += chunk_loss.detach()
    head.set_reshard_after_forward(reshard_after_forward)
    head.set_reshard_after_backward(reshard_after_backward)
    head.reshard()
    hidden.backward(torch.cat([chunk.grad for chunk in chunks]))
    return loss


def _train_chunked_loss(
    model: _Decoder,
    *,
    rank: int,
    steps: int,
    microbatches: int,
    no_sync: bool,
) -> list[dict[str, object]]:
    """``_train`` with ``_chunked_loss``; without sync, every microbatch but
    the last also keeps the parameters unsharded after backward, as
    torchtitan's gradient accumulation does."""
    model.skip_lm_head = True
    head = (
        model.lm_head
        if isinstance(model.lm_head, FSDPModule)
        else model.bucket_storage_of(model.lm_head.weight)
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=True)
    history: list[dict[str, object]] = []
    for step in range(steps):
        losses = []
        for microbatch in range(microbatches):
            if no_sync:
                is_last = microbatch == microbatches - 1
                model.set_reshard_after_backward(is_last)
                model.set_requires_gradient_sync(is_last)
            generator = torch.Generator().manual_seed(
                10_000 * step + 100 * microbatch + rank
            )
            tokens, targets = (
                torch.randint(_VOCAB, (_BATCH,), generator=generator).to(device_type)
                for _ in range(2)
            )
            losses.append(_chunked_loss(model, head, tokens, targets))
        grads = {
            fqn: None if param.grad is None else _local(param.grad).detach().clone()
            for fqn, param in model.named_parameters()
        }
        optim.step()
        optim.zero_grad()
        params = {
            fqn: _local(param).detach().clone()
            for fqn, param in model.named_parameters()
        }
        history.append({"losses": losses, "grads": grads, "params": params})
    return history


class _FullyShardParity(FSDPTest):
    def _assert_bitwise_equal(
        self, expected: torch.Tensor | None, actual: torch.Tensor | None, context: str
    ) -> None:
        if expected is None or actual is None:
            self.assertIs(expected, actual, msg=context)
            return
        self.assertEqual(expected.dtype, actual.dtype, msg=context)
        self.assertEqual(expected.shape, actual.shape, msg=context)
        self.assertTrue(torch.equal(expected, actual), msg=context)

    def _assert_same_inputs(
        self,
        expected: list[torch.Tensor],
        actual: list[torch.Tensor],
        context: str,
    ) -> None:
        # Pair the collectives by size, in issue order, in case the backends
        # prefetch in different orders.
        def by_numel(inputs):
            grouped = defaultdict(list)
            for tensor in inputs:
                grouped[tensor.numel()].append(tensor)
            return grouped

        expected_by_numel, actual_by_numel = by_numel(expected), by_numel(actual)
        self.assertEqual(
            sorted(expected_by_numel), sorted(actual_by_numel), msg=context
        )
        for numel, tensors in expected_by_numel.items():
            self.assertEqual(len(tensors), len(actual_by_numel[numel]), msg=context)
            for expected_tensor, actual_tensor in zip(
                tensors, actual_by_numel[numel], strict=True
            ):
                self.assertEqual(
                    expected_tensor.dtype, actual_tensor.dtype, msg=context
                )
                self.assertTrue(
                    torch.equal(expected_tensor, actual_tensor),
                    msg=f"{context}: {numel}-element inputs differ",
                )

    def _assert_same_history(
        self,
        expected_history: list[dict[str, object]],
        actual_history: list[dict[str, object]],
        context: str,
    ) -> None:
        for step, (expected, actual) in enumerate(
            zip(expected_history, actual_history, strict=True)
        ):
            for expected_loss, actual_loss in zip(
                expected["losses"], actual["losses"], strict=True
            ):
                self._assert_bitwise_equal(
                    expected_loss, actual_loss, f"{context} step {step} loss"
                )
            for kind in ("grads", "params"):
                self.assertEqual(list(expected[kind]), list(actual[kind]))
                for fqn in expected[kind]:
                    self._assert_bitwise_equal(
                        expected[kind][fqn],
                        actual[kind][fqn],
                        f"{context} step {step} {kind} {fqn}",
                    )

    def _run_backend(
        self,
        apply: Callable[..., None],
        reference: _Decoder,
        mesh,
        config: tuple,
    ) -> list[dict[str, object]]:
        param_dtype, reduce_dtype, divide_factor, microbatches, no_sync = config
        model = copy.deepcopy(reference).to(device_type)
        apply(model, mesh, param_dtype, reduce_dtype, divide_factor)
        return _train(
            model,
            rank=self.rank,
            steps=3,
            microbatches=microbatches,
            no_sync=no_sync,
        )


class TestFlexShardMatchesFullyShard(_FullyShardParity):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        # A sum of two values does not depend on their order, so buckets match
        # fully_shard bit for bit at two ranks whatever their parameter order.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_matches_fully_shard(self):
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        reference = _Decoder()
        # Without fsdp2_compatible, buckets reduce zeros for params without a
        # grad, where FSDP2 leaves them out; frozen, both skip them.
        for layer in reference.layers:
            layer.unused.requires_grad_(False)
        for config, context in _configs():
            fsdp2_history = self._run_backend(_apply_fsdp2, reference, mesh, config)
            flex_history = self._run_backend(_apply_flex_shard, reference, mesh, config)
            self._assert_same_history(fsdp2_history, flex_history, context)


class TestFlexShardFSDP2Parity(_FullyShardParity):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        # Sums of four values make the reduction order observable.
        return 4

    def _check_matches_fully_shard(
        self,
        reference: _Decoder,
        collectives: tuple[str, ...],
        fsdp2_scheduling: bool = False,
    ) -> None:
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        apply_fsdp2_compatible = functools.partial(
            _apply_flex_shard,
            fsdp2_compatible=True,
            fsdp2_scheduling=fsdp2_scheduling,
        )
        for config, context in _configs():
            with _record_collective_inputs() as fsdp2_recorded:
                fsdp2_history = self._run_backend(_apply_fsdp2, reference, mesh, config)
            with _record_collective_inputs() as flex_recorded:
                flex_history = self._run_backend(
                    apply_fsdp2_compatible, reference, mesh, config
                )
            for collective in collectives:
                self.assertTrue(fsdp2_recorded[collective], msg=context)
                self._assert_same_inputs(
                    fsdp2_recorded[collective],
                    flex_recorded[collective],
                    f"{context} {collective}",
                )
            self._assert_same_history(fsdp2_history, flex_history, context)

    @skip_if_lt_x_gpu(4)
    def test_matches_fully_shard(self):
        torch.manual_seed(0)
        self._check_matches_fully_shard(_Decoder(), ("all_gather", "reduce_scatter"))

    @skip_if_lt_x_gpu(4)
    def test_matches_fully_shard_with_head_called_twice(self):
        # FSDP2 runs a post-backward per call: each call's backward
        # reduce-scatters its grads and reshards the group, which the other
        # call's backward gathers again. With FSDP2's scheduling options, the
        # buckets issue the same collectives.
        torch.manual_seed(0)
        reference = _Decoder()
        reference.second_head_call = True
        self._check_matches_fully_shard(
            reference, ("all_gather", "reduce_scatter"), fsdp2_scheduling=True
        )


class TestFlexShardChunkedLossFSDP2Parity(_FullyShardParity):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_matches_fully_shard(self):
        # torchtitan's chunked loss with the norm and lm_head in one group.
        # FSDP2 reduce-scatters lm_head's gradient alone at the last chunk and
        # the norm's alone in the final backward; the bucket must issue the
        # same reduce-scatters, with or without reshard after forward and
        # with gradient accumulation without sync, whose microbatches still
        # reduce-scatter the head group (the chunk loop turns its sync on).
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        reference = _Decoder()
        for (
            (param_dtype, reduce_dtype),
            (microbatches, no_sync),
            head_reshard_after_forward,
        ) in itertools.product(
            [(None, None), (torch.bfloat16, torch.float32)],
            [(1, False), (2, True)],
            [False, True],
        ):
            context = (
                f"param_dtype={param_dtype} reduce_dtype={reduce_dtype} "
                f"microbatches={microbatches} no_sync={no_sync} "
                f"head_reshard_after_forward={head_reshard_after_forward}"
            )
            histories, recorded = [], []
            for apply in (
                _apply_fsdp2,
                functools.partial(_apply_flex_shard, fsdp2_compatible=True),
            ):
                model = copy.deepcopy(reference).to(device_type)
                apply(
                    model,
                    mesh,
                    param_dtype,
                    reduce_dtype,
                    1.0,
                    head_reshard_after_forward=head_reshard_after_forward,
                )
                with _record_collective_inputs() as inputs:
                    histories.append(
                        _train_chunked_loss(
                            model,
                            rank=self.rank,
                            steps=3,
                            microbatches=microbatches,
                            no_sync=no_sync,
                        )
                    )
                recorded.append(inputs["reduce_scatter"])
            # Their all-gathers may differ in number: a syncing backward always
            # reshards a bucket.
            self.assertTrue(recorded[0], msg=context)
            self._assert_same_inputs(
                recorded[0], recorded[1], f"{context} reduce_scatter"
            )
            self._assert_same_history(histories[0], histories[1], context)


if __name__ == "__main__":
    run_tests()
