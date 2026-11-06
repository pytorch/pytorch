# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import gc
import warnings
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp._flex_shard import (
    bucket_runtime,
    BucketSpec,
    flex_shard,
    is_flex_shard_param,
    MixedPrecisionPolicy,
)
from torch.distributed.fsdp._flex_shard.placements.block_shard import (
    BucketedBlockShard,
    make_bucketed_block_placement_fn,
)
from torch.distributed.fsdp._flex_shard.placements.mixed_bucket import (
    MixedBucketPlacement,
)
from torch.distributed.fsdp._flex_shard.placements.shard import per_param_placements
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.common_flex_shard import (
    alloc_grads_in_param_layout,
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cuda_mesh,
    transformer_inputs,
)


class _TEStyleScaleFn(torch.autograd.Function):
    """Like TransformerEngine: ``module.parameters()`` are the autograd inputs,
    and backward reads ``module.weight`` instead of saving it."""

    @staticmethod
    def forward(ctx, x, module, *params):
        assert params[0] is module.weight
        ctx.save_for_backward(x)
        ctx.module = module
        ctx.param = params[0]
        return x * module.weight

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        weight = ctx.module.weight
        # With one rank the local shard equals the full param, so check identity.
        assert weight is ctx.param and weight.untyped_storage().size() > 0
        return grad_output * weight, None, (grad_output * x).sum(0)


@dataclasses.dataclass
class _Output:
    value: torch.Tensor


class _TEStyleScale(nn.Module):
    def __init__(self, dim: int, dataclass_output: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.dataclass_output = dataclass_output

    def forward(self, x: torch.Tensor) -> torch.Tensor | _Output:
        out = _TEStyleScaleFn.apply(x, self, *self.parameters())
        return _Output(out) if self.dataclass_output else out


class _ViewHead(nn.Module):
    """Returns a view, and leaves one linear unused like an unrouted expert."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.used = nn.Linear(dim, dim)
        self.unused = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x).view(-1)


class _PersistentParamsNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TEStyleScale(dim, dataclass_output=True)]
            + [_TEStyleScale(dim) for _ in range(2)]
        )
        self.head = _ViewHead(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x).value
        x = self.layers[2](self.layers[1](x))
        # The head runs in two parallel branches, and the later branch's view
        # output is modified in place.
        out = self.head(x)
        later = self.head(2 * x)
        later.mul_(2)
        return out + later


class _TwoStageBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(dim, dim)
        self.second = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


class _SkipNet(nn.Module):
    """Three linears; ``skip`` leaves the middle one out of the forward."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.skip = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        if not self.skip:
            x = self.layers[1](x)
        return self.layers[2](x)


class _ChunkedHeadNet(nn.Module):
    """A layer, then a norm and an output projection; with ``skip_head``, the
    forward returns the norm's output, as torchtitan's decoder does under its
    chunked loss."""

    def __init__(self, dim: int, vocab: int) -> None:
        super().__init__()
        self.layer = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.skip_head = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.layer(x))
        return h if self.skip_head else self.head(h)


def _chunked_loss_backward(
    model: _ChunkedHeadNet,
    x: torch.Tensor,
    targets: torch.Tensor,
    num_chunks: int,
    head_storage=None,
    after_chunk_backward=None,
) -> None:
    """torchtitan's ChunkedLossWrapper sequence: the head on each detached
    chunk of the hidden states with a backward per chunk, then one backward
    through the model. With ``head_storage``, the head's bucket stays
    unsharded across the chunks and reduce-scatters at the last one."""
    model.skip_head = True
    hidden = model(x)
    if head_storage is not None:
        reshard_after_forward = head_storage.reshard_after_forward
        reshard_after_backward = head_storage.reshard_after_backward
        head_storage.set_reshard_after_forward(False)
        head_storage.set_reshard_after_backward(False)
        head_storage.set_requires_gradient_sync(False)
        head_storage.unshard()
    chunks = [chunk.detach().requires_grad_() for chunk in hidden.chunk(num_chunks)]
    for idx, (chunk, target) in enumerate(
        zip(chunks, targets.chunk(num_chunks), strict=True)
    ):
        if head_storage is not None and idx == num_chunks - 1:
            head_storage.set_requires_gradient_sync(True)
        F.cross_entropy(model.head(chunk), target, reduction="sum").backward()
        if after_chunk_backward is not None:
            after_chunk_backward(idx)
    if head_storage is not None:
        head_storage.set_reshard_after_forward(reshard_after_forward)
        head_storage.set_reshard_after_backward(reshard_after_backward)
        head_storage.reshard()
    hidden.backward(torch.cat([chunk.grad for chunk in chunks]))


class _TiedNet(nn.Module):
    """The head shares the embedding's weight, so one bucket holds both names."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=False)
        self.mid = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.mid(self.embed(x))))


class _ReentrantCheckpointNet(nn.Module):
    """A block under reentrant activation checkpointing: its forward runs
    without grad, and backward recomputes it, as Megatron's recompute does."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = _TwoStageBlock(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=True)
        return self.head(x)


def _bucket(patterns, mesh, reshard_after_forward, placement_fn=per_param_placements):
    return BucketSpec(
        patterns,
        placement_fn=placement_fn,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
    )


def _buckets(model: nn.Module):
    return [
        bucket
        for context in model._flex_shard_eager_comm_contexts.values()
        for bucket in context.buckets
    ]


class TestFlexShardEagerRuntime(TestCase):
    def test_local_shard_shares_parameter_version_counter(self):
        module = nn.Linear(2, 2, bias=False)
        runtime = bucket_runtime.BucketRuntime(
            bucket_storage=Mock(),
            bucket_params=[
                bucket_runtime.BucketParam(
                    param_owner=bucket_runtime.ParamOwnerRef(module, "weight"),
                    param_info=Mock(),
                    sharded_param=module.weight,
                )
            ],
            context=Mock(),
            debug_fqn=None,
        )

        (local_shard,) = runtime._local_shards(use_autograd=False)
        original_version = local_shard._version
        with torch.no_grad():
            module.weight.add_(1)

        self.assertFalse(local_shard.requires_grad)
        self.assertEqual(local_shard._version, original_version + 1)

    def test_meta_to_empty_materializes_bucket_storage_and_runtime(self):
        with single_rank_cuda_mesh() as mesh:
            with torch.device("meta"):
                args, model = make_transformer_model()
            model.output.weight.grad_dtype = torch.bfloat16

            flex_shard_cuda(model, mesh)
            for storage in model.sharded_bucket_storages:
                self.assertEqual(storage.byte_storage.device.type, "meta")

            # The second to_empty() replaces the params the runtime captured.
            for _ in range(2):
                model.to_empty(device="cuda")
                for storage in model.sharded_bucket_storages:
                    self.assertEqual(storage.byte_storage.device.type, "cuda")
                for param in model.parameters():
                    self.assertTrue(is_flex_shard_param(param))
                    nn.init.uniform_(param, -0.1, 0.1)
                    param.grad = None
                output_weight = model.output._parameters["weight"]
                self.assertEqual(output_weight.grad_dtype, torch.bfloat16)

                loss = model(transformer_inputs(args, device="cuda")).sum()
                loss.backward()

                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                self.assertEqual(output_weight.grad.dtype, torch.bfloat16)

    def test_meta_to_empty_keeps_tied_weights_shared(self):
        # to_empty() from meta gives each slot of a shared parameter its own new
        # parameter; FlexShard re-installs one sharded parameter in every slot.
        with single_rank_cuda_mesh() as mesh:
            with torch.device("meta"):
                args, model = make_transformer_model(weight_tying=True)
            flex_shard_cuda(model, mesh)
            for _ in range(2):
                model.to_empty(device="cuda")
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                for param in model.parameters():
                    nn.init.uniform_(param, -0.1, 0.1)
                    param.grad = None
                model(transformer_inputs(args, device="cuda")).sum().backward()
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                self.assertIsNotNone(model.tok_embeddings.weight.grad)

    def test_torch_compile_tied_weights(self):
        # Under compile, _BucketUnshard's outputs swap into every slot of a
        # shared parameter, so the tied output layer reads the gathered weight.
        # The tied bucket is hooked on the root, around the per-layer buckets.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            args, model = make_transformer_model(
                device="cuda", n_layers=2, weight_tying=True
            )
            reference = copy.deepcopy(model)
            spec = dict(
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=False,
            )
            flex_shard_cuda(
                model,
                mesh,
                buckets=[
                    BucketSpec(
                        [
                            "tok_embeddings.*",
                            "pos_embeddings.*",
                            "norm.*",
                            "output.*",
                        ],
                        **spec,
                    ),
                    *(
                        BucketSpec([f"layers.{idx}.*"], **spec)
                        for idx in range(args.n_layers)
                    ),
                ],
            )
            compiled_model = torch.compile(model, backend="eager", fullgraph=True)
            x = transformer_inputs(args, device="cuda")
            loss = compiled_model(x).sum()
            ref_loss = reference(x).sum()
            self.assertEqual(loss, ref_loss)
            loss.backward()
            ref_loss.backward()
            self.assertIs(model.output.weight, model.tok_embeddings.weight)
            self.assertEqual(
                model.tok_embeddings.weight.grad, reference.tok_embeddings.weight.grad
            )

    def test_module_patterns(self):
        # Patterns naming sibling modules make a bucket of their params, hooked
        # on them rather than on their parent: it unshards in the first one's
        # forward and reshards after the last one's, before the next module
        # runs. Backward re-gathers it from the last one's outputs and reduces
        # once the first one's input grads are computed; training matches an
        # unsharded reference. Patterns may not mix modules with parameters or
        # name a container without a forward.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)
            )
            reference = copy.deepcopy(model).cuda()
            for invalid, bucket_patterns, error in (
                (copy.deepcopy(model), (["0", "2.*"], ["4.*"]), "mix module FQNs"),
                (
                    nn.Sequential(nn.ModuleList([nn.Linear(8, 8)])),
                    (["0"],),
                    "no forward",
                ),
            ):
                with self.assertRaisesRegex(ValueError, error):
                    flex_shard(
                        invalid,
                        buckets=[
                            _bucket(patterns, mesh, reshard_after_forward=True)
                            for patterns in bucket_patterns
                        ],
                    )
            flex_shard(
                model,
                buckets=[
                    _bucket(["0", "2"], mesh, reshard_after_forward=True),
                    _bucket(["4"], mesh, reshard_after_forward=True),
                ],
            )
            layers, _ = _buckets(model)
            unsharded_in = {}
            for idx in (2, 4):
                model[idx].register_forward_pre_hook(
                    lambda module, args, idx=idx: unsharded_in.update(
                        {idx: layers.is_unsharded}
                    )
                )
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
            for _ in range(2):
                optim.zero_grad()
                ref_optim.zero_grad()
                x = torch.randn(4, 8, device="cuda", requires_grad=True)
                model(x * 2).sum().backward()
                reference(x * 2).sum().backward()
                self.assertEqual(unsharded_in, {2: True, 4: False})
                for param, ref_param in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(param.grad, ref_param.grad)
                optim.step()
                ref_optim.step()

    def test_persistent_unsharded_params(self):
        for reshard_after_forward in (False, True):
            with (
                self.subTest(reshard_after_forward=reshard_after_forward),
                single_rank_cuda_mesh() as mesh,
            ):
                self._check_persistent_unsharded_params(mesh, reshard_after_forward)

    def _check_persistent_unsharded_params(self, mesh, reshard_after_forward):
        torch.manual_seed(0)
        model = _PersistentParamsNet(8)
        reference = copy.deepcopy(model).cuda()
        model.layers[0].weight.custom_tag = "kept"
        seen_weights = []
        # Registered before flex_shard(), whose pre-forward hook still runs first.
        model.layers[0].register_forward_pre_hook(
            lambda module, args: seen_weights.append(module.weight)
        )
        mixed = MixedBucketPlacement({})

        def head_placements(named_params, mesh):
            return {
                fqn: (
                    # BlockShard refills with torch.cat(out=), which bumps
                    # version counters of weights autograd saved.
                    mixed.block_shard(blocks_per_rank=(1,))
                    if fqn.endswith("weight")
                    else mixed.shard0,
                )
                for fqn, _ in named_params
            }

        flex_shard(
            model,
            buckets=[
                # One module at a dotted path, and a bucket spanning a
                # ModuleList, which has no forward of its own.
                _bucket(["layers.0.*"], mesh, reshard_after_forward),
                _bucket(
                    ["layers.1.*", "layers.2.*"],
                    mesh,
                    reshard_after_forward,
                    make_bucketed_block_placement_fn(dims=(0,), blocks_per_rank=(1,)),
                ),
                _bucket(["head.*"], mesh, reshard_after_forward, head_placements),
            ],
        )
        buckets = _buckets(model)
        for bucket, hook_module in zip(
            buckets, (model.layers[0], model, model.head), strict=True
        ):
            self.assertIs(bucket.forward_hook_module(), hook_module)
        head_resharded = []

        def probe(module, args, output):
            # Runs after the head's and layers[2]'s backward, before layers[1]'s.
            if output.requires_grad:
                output.register_hook(
                    lambda grad: head_resharded.append(not buckets[2].is_unsharded)
                )

        model.layers[1].register_forward_hook(probe)

        x = torch.randn(4, 8, device="cuda")
        with torch.inference_mode():  # the first unshard runs in inference mode
            model(x)
        persistent = [param for bucket in buckets for param in bucket.unsharded_params]
        buffers = [buffer for bucket in buckets for buffer in bucket.persistent_buffers]
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
        for step in range(3):
            if step == 1:
                # Freezing a local shard after flex_shard() takes effect. The
                # frozen weight is read in backward after its bucket's other
                # grads have landed.
                model.layers[1].weight.requires_grad_(False)
                reference.layers[1].weight.requires_grad_(False)
            optim.zero_grad()
            ref_optim.zero_grad()
            loss = model(x).sum()
            self.assertTrue(
                all(
                    (buffer.untyped_storage().size() == 0) == reshard_after_forward
                    for buffer in buffers
                )
            )
            loss.backward()
            reference(x).sum().backward()
            for param, ref_param in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                # Trainable params unused in forward get zero grads, as on main.
                ref_grad = ref_param.grad
                if ref_grad is None and ref_param.requires_grad:
                    ref_grad = torch.zeros_like(ref_param)
                torch.testing.assert_close(param.grad, ref_grad)
            optim.step()
            ref_optim.step()
            # The same persistent params, their storage freed after backward.
            self.assertTrue(
                all(
                    a is b
                    for a, b in zip(
                        (p for bucket in buckets for p in bucket.unsharded_params),
                        persistent,
                        strict=True,
                    )
                )
            )
            self.assertTrue(all(b.untyped_storage().size() == 0 for b in buffers))
        self.assertEqual(head_resharded, [True] * 3)
        self.assertTrue(all(weight is persistent[0] for weight in seen_weights))
        self.assertEqual(persistent[0].custom_tag, "kept")
        with torch.inference_mode():
            torch.testing.assert_close(model(x), reference(x))
        model(x)  # a forward whose backward will not run
        model.reshard()
        self.assertTrue(all(b.untyped_storage().size() == 0 for b in buffers))

        # FlexShard does not recover from errors: after a backward that raises
        # partway, the next forward and reshard() raise too.
        def fail(grad):
            raise RuntimeError("injected backward failure")

        model.layers[1].register_forward_hook(
            lambda module, args, output: output.register_hook(fail) and None
        )
        with self.assertRaisesRegex(RuntimeError, "injected backward failure"):
            model(x).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "a previous backward raised"):
            model(x)
        with self.assertRaisesRegex(RuntimeError, "a previous backward raised"):
            model.reshard()

    def test_reshard_after_forward_holds_one_bucket(self):
        # Each bucket is freed before the next one is re-gathered, in forward
        # and in backward, as in FSDP2.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(3)))
            flex_shard(
                model,
                buckets=[
                    _bucket([f"{idx}.*"], mesh, reshard_after_forward=True)
                    for idx in range(3)
                ],
            )
            buckets = _buckets(model)
            others_unsharded = []
            finish_unshard = bucket_runtime.BucketRuntime.finish_unshard

            def counting_finish_unshard(bucket, result):
                others_unsharded.append(
                    sum(b.is_unsharded for b in buckets if b is not bucket)
                )
                return finish_unshard(bucket, result)

            with patch.object(
                bucket_runtime.BucketRuntime, "finish_unshard", counting_finish_unshard
            ):
                model(torch.randn(4, 8, device="cuda")).sum().backward()
            self.assertEqual(others_unsharded, [0] * 6)

    def test_prefetch_follows_learned_order(self):
        # BucketSpec lists the buckets in reverse execution order. From the
        # second step, prefetch follows the order learned in the first, and the
        # recompute inside backward consumes the checkpointed block's backward
        # prefetch: one unshard per bucket in forward and backward, none wasted.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(
                nn.Linear(8, 8), checkpoint_wrapper(_TwoStageBlock(8))
            )
            block = "1._checkpoint_wrapped_module"
            flex_shard(
                model,
                buckets=[
                    _bucket([pattern], mesh, reshard_after_forward=True)
                    for pattern in (f"{block}.second.*", f"{block}.first.*", "0.*")
                ],
            )
            x = torch.randn(4, 8, device="cuda")
            model(x).sum().backward()
            begin_unshard = bucket_runtime.BucketRuntime.begin_unshard
            take = bucket_runtime.BucketCommContext.take_pending_unshard
            hits = []

            def counting_take(context, bucket):
                result = take(context, bucket)
                hits.append(result is not None)
                return result

            with (
                patch.object(
                    bucket_runtime.BucketRuntime,
                    "begin_unshard",
                    autospec=True,
                    side_effect=begin_unshard,
                ) as unshards,
                patch.object(
                    bucket_runtime.BucketCommContext,
                    "take_pending_unshard",
                    counting_take,
                ),
            ):
                model(x).sum().backward()
            self.assertEqual(unshards.call_count, 6)
            # All but the first unshard of forward and of backward were prefetched.
            self.assertEqual(sum(hits), 4)

    def test_explicit_prefetch_replaces_learned_order(self):
        # As torchtitan sets FSDP2's explicit prefetch: in forward, the first
        # bucket prefetches the next two; in backward, the last bucket the
        # previous two; the buckets in between prefetch nothing. Every unshard
        # after the first of each pass is then prefetched, two in flight at a
        # time, none released, and training matches a plain model.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(4)))
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=[
                    _bucket([f"{idx}.*"], mesh, reshard_after_forward=True)
                    for idx in range(4)
                ],
            )
            storages = model.sharded_bucket_storages
            storages[0].set_buckets_to_forward_prefetch(storages[1:3])
            storages[2].set_buckets_to_forward_prefetch([storages[3]])
            storages[3].set_buckets_to_backward_prefetch([storages[2], storages[1]])
            storages[1].set_buckets_to_backward_prefetch([storages[0]])
            for idx in (1, 3):
                storages[idx].set_buckets_to_forward_prefetch([])
            for idx in (0, 2):
                storages[idx].set_buckets_to_backward_prefetch([])
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
            x = torch.randn(4, 8, device="cuda")
            begin_unshard = bucket_runtime.BucketRuntime.begin_unshard
            finish_unshard = bucket_runtime.BucketRuntime.finish_unshard
            take = bucket_runtime.BucketCommContext.take_pending_unshard
            hits = []
            in_flight = []

            def counting_take(context, bucket):
                result = take(context, bucket)
                hits.append(result is not None)
                return result

            def counting_finish_unshard(bucket, result):
                in_flight.append(len(bucket.context.pending_unshards))
                return finish_unshard(bucket, result)

            for _ in range(2):
                optim.zero_grad()
                ref_optim.zero_grad()
                with (
                    patch.object(
                        bucket_runtime.BucketRuntime,
                        "begin_unshard",
                        autospec=True,
                        side_effect=begin_unshard,
                    ) as unshards,
                    patch.object(
                        bucket_runtime.BucketRuntime,
                        "finish_unshard",
                        counting_finish_unshard,
                    ),
                    patch.object(
                        bucket_runtime.BucketCommContext,
                        "take_pending_unshard",
                        counting_take,
                    ),
                ):
                    output = model(x)
                    output.sum().backward()
                reference_output = reference(x)
                torch.testing.assert_close(output, reference_output)
                reference_output.sum().backward()
                optim.step()
                ref_optim.step()
                self.assertEqual(unshards.call_count, 8)
            # Each step, every unshard but the first of forward and of backward
            # was prefetched, and the second bucket of each pass finished while
            # the third's prefetch was in flight, which the learned order's one
            # prefetch at a time does not do.
            self.assertEqual(sum(hits), 12)
            self.assertEqual(in_flight, [0, 1, 0, 0] * 4)
            torch.testing.assert_close(model(x).detach(), reference(x).detach())

    def test_prefetch_targets_must_be_bucket_storages(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            flex_shard(
                model,
                buckets=[_bucket([f"{idx}.*"], mesh, True) for idx in range(2)],
            )
            with self.assertRaisesRegex(TypeError, "sharded_bucket_storages"):
                model.sharded_bucket_storages[0].set_buckets_to_forward_prefetch(
                    [model[1]]
                )

    def test_bucket_unshard_reshard(self):
        # A bucket storage takes FSDPModule's unshard(), reshard() and
        # set_reshard_after_forward() for its group, and bucket_storage_of()
        # finds it from its local shard or unsharded param.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
            flex_shard(
                model,
                buckets=[_bucket([idx], mesh, True) for idx in ("0", "2")],
            )
            first, second = model.sharded_bucket_storages
            first_bucket, second_bucket = _buckets(model)
            local_shard = model[2].weight
            self.assertIs(model.bucket_storage_of(local_shard), second)
            with self.assertRaisesRegex(ValueError, "not in this module's buckets"):
                model.bucket_storage_of(nn.Parameter(torch.zeros(1)))

            second.unshard()
            self.assertTrue(second_bucket.is_unsharded)
            self.assertFalse(first_bucket.is_unsharded)
            self.assertIsNot(model[2].weight, local_shard)
            self.assertIs(model.bucket_storage_of(model[2].weight), second)
            second.reshard()
            self.assertFalse(second_bucket.is_unsharded)
            self.assertIs(model[2].weight, local_shard)

            # reshard() releases the bucket's prefetched unshard.
            first_bucket.context.prefetch(second_bucket)
            second.reshard()
            self.assertEqual(first_bucket.context.pending_unshards, [])
            self.assertFalse(second_bucket.is_unsharded)

            # Without reshard after forward, only that bucket keeps its params
            # after the forward, until its backward.
            self.assertTrue(second.reshard_after_forward)
            self.assertTrue(second.reshard_after_backward)
            second.set_reshard_after_forward(False)
            self.assertFalse(second.reshard_after_forward)
            output = model(torch.randn(4, 8, device="cuda", requires_grad=True))
            self.assertFalse(first_bucket.is_unsharded)
            self.assertTrue(second_bucket.is_unsharded)
            output.register_hook(lambda grad: second.unshard())
            with self.assertRaisesRegex(RuntimeError, "cannot run in backward"):
                output.sum().backward()

    def test_chunked_head(self):
        # torchtitan's chunked loss with the norm and the head in one bucket:
        # the forward skips the head, so the root's post-forward completes the
        # bucket, whose re-gather in the final backward serves the norm. Each
        # call of the head on a detached (leaf) chunk triggers its own
        # post-backward, keeping the head's grads until the last chunk's
        # reduce-scatter. The chunk backwards, which re-gather nothing, leave
        # the layer bucket to the final backward. Training matches an
        # unsharded reference.
        num_chunks = 3
        for head_reshard_after_forward in (False, True):
            with (
                self.subTest(head_reshard_after_forward=head_reshard_after_forward),
                single_rank_cuda_mesh() as mesh,
            ):
                torch.manual_seed(0)
                model = _ChunkedHeadNet(8, 16)
                reference = copy.deepcopy(model).cuda()
                flex_shard(
                    model,
                    buckets=[
                        _bucket(["layer"], mesh, reshard_after_forward=False),
                        # As torchtitan's: params without a grad stay out of
                        # the reduce-scatter.
                        dataclasses.replace(
                            _bucket(
                                ["norm", "head"],
                                mesh,
                                reshard_after_forward=head_reshard_after_forward,
                            ),
                            fsdp2_compatible=True,
                        ),
                    ],
                )
                head_storage = model.bucket_storage_of(model.head.weight)
                head_shard, norm_shard = model.head.weight, model.norm.weight
                layer_bucket, head_bucket = _buckets(model)
                optim = torch.optim.SGD(model.parameters(), lr=0.1)
                ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
                for _ in range(2):
                    optim.zero_grad()
                    ref_optim.zero_grad()
                    x = torch.randn(6, 8, device="cuda")
                    targets = torch.randint(16, (6,), device="cuda")
                    observed = []

                    def after_chunk_backward(idx):
                        observed.append(
                            (
                                layer_bucket.is_unsharded,
                                head_bucket.is_unsharded,
                                head_shard.grad is not None,
                                norm_shard.grad is not None,
                            )
                        )

                    _chunked_loss_backward(
                        model,
                        x,
                        targets,
                        num_chunks,
                        head_storage,
                        after_chunk_backward,
                    )
                    _chunked_loss_backward(reference, x, targets, num_chunks)
                    # The layer stays gathered through the chunk backwards. The
                    # head reduce-scatters its grads alone at the last chunk;
                    # the norm's come with the final backward.
                    self.assertEqual(
                        observed,
                        [(True, True, False, False)] * (num_chunks - 1)
                        + [(True, False, True, False)],
                    )
                    self.assertFalse(layer_bucket.is_unsharded)
                    self.assertFalse(head_bucket.is_unsharded)
                    self.assertEqual(
                        head_storage.reshard_after_forward, head_reshard_after_forward
                    )
                    for param, ref_param in zip(
                        model.parameters(), reference.parameters(), strict=True
                    ):
                        torch.testing.assert_close(param.grad, ref_param.grad)
                    optim.step()
                    ref_optim.step()

    def test_chunked_head_releases_hidden_states(self):
        # The post-backward trigger on the head's leaf chunk inputs must not
        # keep them, or the hidden states their storage is part of, alive
        # after the step. torchtitan turns automatic garbage collection off,
        # so a reference cycle through the trigger would keep one hidden state
        # per step until the next collection.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = _ChunkedHeadNet(64, 16)
            flex_shard(
                model,
                buckets=[
                    _bucket(["layer"], mesh, reshard_after_forward=True),
                    dataclasses.replace(
                        _bucket(["norm", "head"], mesh, reshard_after_forward=False),
                        fsdp2_compatible=True,
                    ),
                ],
            )
            head_storage = model.bucket_storage_of(model.head.weight)
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            x = torch.randn(1024, 64, device="cuda")
            targets = torch.randint(16, (1024,), device="cuda")
            allocated = []
            gc.collect()
            gc.disable()
            try:
                for _ in range(3):
                    _chunked_loss_backward(model, x, targets, 4, head_storage)
                    optim.step()
                    optim.zero_grad()
                    allocated.append(torch.cuda.memory_allocated())
            finally:
                gc.enable()
            self.assertEqual(allocated[1], allocated[2])

    def test_shard_unshard_buffers(self):
        # On a 1-rank mesh Shard skips the all-gather, as FSDP2 does: the only
        # buffer is the bucket's local storage, which finish casts into the
        # unsharded params. A refill allocates the persistent storage only at
        # copy-out, as FSDP2 allocates its unsharded parameters. The in-place
        # all-gather on more ranks is checked in test_flex_shard_training.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=[
                    BucketSpec(
                        [idx],
                        placement_fn=per_param_placements,
                        mesh=mesh,
                        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                        reshard_after_forward=True,
                    )
                    for idx in ("0", "2")
                ],
            )
            _, bucket = _buckets(model)
            # The first unshard creates the persistent buffers.
            x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
            model(x).sum().backward()
            handle = bucket.begin_unshard()
            self.assertEqual(len(handle.prepared.buffers), 1)
            self.assertTrue(handle.prepared.placement_state.local_copy_out)
            self.assertTrue(
                all(b.untyped_storage().size() == 0 for b in bucket.persistent_buffers)
            )
            bucket.finish_unshard(handle)
            self.assertTrue(
                all(b.untyped_storage().size() > 0 for b in bucket.persistent_buffers)
            )
            for bucket_param, unsharded in zip(
                bucket.bucket_params, bucket.unsharded_params, strict=True
            ):
                expected = reference.get_parameter(bucket_param.param_info.fqn)
                torch.testing.assert_close(
                    unsharded, expected.to(torch.bfloat16), atol=0, rtol=0
                )
            bucket.reshard()

    def test_released_refill_frees_storage(self):
        # A prefetch that refills a bucket's persistent storage, which begin
        # re-allocates, and is then released unused frees that storage; later
        # steps still match.
        with single_rank_cuda_mesh() as mesh:
            model = _SkipNet(8)
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=[_bucket([f"layers.{idx}.*"], mesh, True) for idx in range(3)],
            )
            buckets = _buckets(model)
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
            x = torch.randn(4, 8, device="cuda")
            release = bucket_runtime.BucketRuntime.free_persistent_storage
            released = []

            def recording_release(bucket):
                released.append(bucket)
                return release(bucket)

            for skip in (False, True, False):
                model.skip = reference.skip = skip
                optim.zero_grad()
                ref_optim.zero_grad()
                with patch.object(
                    bucket_runtime.BucketRuntime,
                    "free_persistent_storage",
                    recording_release,
                ):
                    output = model(x)
                    reference_output = reference(x)
                    torch.testing.assert_close(output, reference_output)
                    output.sum().backward()
                reference_output.sum().backward()
                optim.step()
                ref_optim.step()
                self.assertTrue(
                    all(
                        buffer.untyped_storage().size() == 0
                        for bucket in buckets
                        for buffer in bucket.persistent_buffers
                    )
                )
            # Skipping layers.1, layers.0's prefetch of it is released.
            self.assertIn(buckets[1], released)
            # Printing resharded buckets reads no freed storage.
            for bucket in buckets:
                repr(bucket)
            torch.cuda.synchronize()

    def test_gradient_bucket_matches_reference(self):
        # With grads allocated in the parameter layout by a pre-backward hook,
        # the reduce-scatter reads that gradient bucket as is, and grads match
        # an unsharded reference: with a param or a whole bucket frozen after
        # the first step, a shared (tied) param, and a reentrant activation
        # checkpoint, whose forward runs without grad so only its recompute
        # allocates the gradient bucket.
        placement_fn = make_bucketed_block_placement_fn(dims=(0,), blocks_per_rank=(1,))
        cases = {
            "plain": (lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)), ["*"]),
            "frozen": (lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)), ["*"]),
            # A whole bucket frozen: its post-backward reduces no grads.
            "frozen_bucket": (
                lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)),
                ["0.*", "1.*"],
            ),
            "tied": (lambda: _TiedNet(8), ["*"]),
            "reentrant_checkpoint": (
                lambda: _ReentrantCheckpointNet(8),
                ["block.*", "head.*"],
            ),
        }
        gradient_bucket_of = BucketedBlockShard._gradient_bucket_of
        with single_rank_cuda_mesh() as mesh:
            for case, (make_model, patterns) in cases.items():
                for reshard_after_forward in (False, True):
                    with self.subTest(
                        case=case, reshard_after_forward=reshard_after_forward
                    ):
                        torch.manual_seed(0)
                        model = make_model()
                        reference = copy.deepcopy(model).cuda()
                        flex_shard(
                            model,
                            buckets=[
                                BucketSpec(
                                    [pattern],
                                    placement_fn=placement_fn,
                                    mesh=mesh,
                                    reshard_after_forward=reshard_after_forward,
                                    pre_backward_hook=alloc_grads_in_param_layout,
                                )
                                for pattern in patterns
                            ],
                        )
                        optim = torch.optim.SGD(model.parameters(), lr=0.1)
                        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
                        reduced_bucket = []

                        def recording(placement, *args):
                            bucket = gradient_bucket_of(placement, *args)
                            reduced_bucket.append(bucket is not None)
                            return bucket

                        with (
                            patch.object(
                                BucketedBlockShard, "_gradient_bucket_of", recording
                            ),
                            warnings.catch_warnings(),
                        ):
                            warnings.simplefilter("error", UserWarning)
                            for step in range(3):
                                if case == "frozen" and step == 1:
                                    model[0].weight.requires_grad_(False)
                                    reference[0].weight.requires_grad_(False)
                                if case == "frozen_bucket" and step == 1:
                                    for param in (
                                        *model[0].parameters(),
                                        *reference[0].parameters(),
                                    ):
                                        param.requires_grad_(False)
                                optim.zero_grad()
                                ref_optim.zero_grad()
                                x = torch.randn(4, 8, device="cuda", requires_grad=True)
                                output = model(x)
                                reference_output = reference(x)
                                torch.testing.assert_close(output, reference_output)
                                output.square().sum().backward()
                                reference_output.square().sum().backward()
                                for param, ref_param in zip(
                                    model.parameters(),
                                    reference.parameters(),
                                    strict=True,
                                ):
                                    torch.testing.assert_close(
                                        param.grad, ref_param.grad
                                    )
                                optim.step()
                                ref_optim.step()
                        self.assertTrue(reduced_bucket)
                        self.assertTrue(all(reduced_bucket), reduced_bucket)

    def test_torch_compile_forward_backward_on_cuda_mesh(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            compiled_model = torch.compile(model, backend="eager", fullgraph=True)

            loss = compiled_model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)

            model.set_requires_gradient_sync(False)
            with self.assertRaisesRegex(Exception, "eager-only"):
                compiled_model(transformer_inputs(args, device="cuda"))

    def test_torch_compile_traces_per_bucket_unshards(self):
        # The multi-rank collectives are checked in test_flex_shard_training.
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            graphs = []

            def capture_backend(gm, example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled_model = torch.compile(
                model, backend=capture_backend, fullgraph=True
            )
            compiled_model(transformer_inputs(args, device="cuda")).sum().backward()

            # fullgraph=True must produce exactly one graph with no breaks.
            self.assertEqual(1, len(graphs))
            root = graphs[0]
            top_targets = [str(node.target) for node in root.graph.nodes]

            # Dynamo traces into the bucket forward pre-hook rather than treating
            # it as opaque, so bucket member params are lifted as graph inputs.
            self.assertTrue(
                any("bucket_params" in target for target in top_targets),
                f"no bucket params lifted as graph inputs: {top_targets}",
            )

            # One unshard autograd function per bucket, not one for the whole model.
            unshards = [t for t in top_targets if t == "autograd_function_apply"]
            self.assertEqual(len(model.sharded_bucket_storages), len(unshards))

            # On a 1-rank mesh Shard runs no collectives, as FSDP2 does.
            subgraph_targets = set()
            for _, submodule in root.named_modules():
                if isinstance(submodule, torch.fx.GraphModule):
                    subgraph_targets.update(
                        str(node.target) for node in submodule.graph.nodes
                    )
            self.assertNotIn(
                "_c10d_functional.all_gather_into_tensor", subgraph_targets
            )
            self.assertNotIn("_c10d_functional.reduce_scatter_tensor", subgraph_targets)

    def test_torch_compile_frozen_params_do_not_widen_reduce_dtype(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(
                nn.Linear(8, 8, bias=False),
                nn.Linear(8, 8, bias=False),
            ).to(device="cuda", dtype=torch.bfloat16)
            # Explicit None stores the reduced grad in the reduce dtype.
            model[0].weight.grad_dtype = None
            model[1].weight.grad_dtype = torch.float32
            model[1].weight.requires_grad_(False)
            flex_shard_cuda(model, mesh)
            compiled_model = torch.compile(model, backend="eager", fullgraph=True)

            x = torch.ones(2, 8, dtype=torch.bfloat16, device="cuda")
            compiled_model(x).float().sum().backward()

            # Compile backward gets the frozen param's zero grad; like eager,
            # its fp32 grad_dtype does not widen the bucket's bf16.
            params = dict(model.named_parameters())
            self.assertIsNone(params["1.weight"].grad)
            self.assertEqual(params["0.weight"].grad.dtype, torch.bfloat16)


if __name__ == "__main__":
    run_tests()
