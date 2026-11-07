# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import weakref
from collections import Counter
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import (
    bucket_runtime,
    BucketSpec,
    flex_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.fsdp._flex_shard.checkpoint import get_flex_shard_global_layouts
from torch.distributed.fsdp._flex_shard.placements.block_shard import (
    BlockShard,
    BucketedBlockShard,
    make_bucketed_block_placement_fn,
)
from torch.distributed.fsdp._flex_shard.placements.mixed_bucket import (
    make_mixed_placement_fn,
    MixedBucketPlacement,
)
from torch.distributed.fsdp._flex_shard.placements.owned import (
    make_bucketed_owned_full_param_placement_fn,
)
from torch.distributed.fsdp._flex_shard.placements.shard import (
    per_param_placements,
    Shard,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import HardwareClassification, run_tests
from torch.testing._internal.distributed.common_flex_shard import (
    alloc_grads_in_param_layout,
    check_flex_shard_parity,
    expected_shard,
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_test_sgd,
    make_transformer_model,
    transformer_bucket_specs,
    transformer_inputs,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


device_type = torch.device(get_devtype())


class _UnevenMLP(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.out_activation = torch.nn.PReLU(device=device)
        self.in_proj = torch.nn.Linear(3, 5, device=device)
        self.out_proj = torch.nn.Linear(5, 3, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_activation(self.out_proj(torch.tanh(self.in_proj(x))))


class _UnevenMLPStack(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _UnevenMLP(device=device),
                _UnevenMLP(device=device),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _MixedPrecisionWeights(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.low_weight = torch.nn.Parameter(torch.randn(8, 8, device=device))
        # Interleaved so the bf16 params are not adjacent in declaration order.
        self.full_weight = torch.nn.Parameter(torch.randn(8, 8, device=device))
        self.low_weight2 = torch.nn.Parameter(torch.randn(8, 8, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # matmul rejects mismatched dtypes, so this checks each unsharded dtype.
        low_x = x.bfloat16()
        low = low_x @ self.low_weight + 2 * (low_x @ self.low_weight2)
        return low.float() + x @ self.full_weight


class _ParallelLinears(torch.nn.Module):
    def __init__(self, width: int, *, device: torch.device) -> None:
        super().__init__()
        self.first = torch.nn.Linear(width, width, bias=False, device=device)
        self.second = torch.nn.Linear(width, width, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.first(x) + self.second(x)


class _AccumulatingWeight(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 8, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Grads of 256 and then 1 accumulate to 257 in fp32 but 256 in bf16."""
        return (x @ self.weight).float().sum()


class _AccumulatingWeights(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.a = _AccumulatingWeight(device=device, dtype=dtype)
        self.b = _AccumulatingWeight(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)


class _ReusedWeight(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 8, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-use grads 256 and 1 sum to 257 in fp32 but round to 256 in bf16."""
        return ((256 * x) @ self.weight).float().sum() + (x @ self.weight).float().sum()


class _ReusedWeights(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.a = _ReusedWeight(device=device, dtype=dtype)
        self.b = _ReusedWeight(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)


class _GatedBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(dim, dim)
        self.mlp = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(x)
        # The gate's backward unpacks saved activations, so a checkpointed
        # block recomputes before either linear's pre-backward hook runs.
        return h + torch.sigmoid(h) * self.mlp(h)


class _FusedWgradLinearFn(torch.autograd.Function):
    """``x @ weight.t()`` whose backward adds the weight grad into
    ``weight.main_grad`` in place and gives autograd none, like
    TransformerEngine's ``fuse_wgrad_accumulation``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        # The parameter object carrying main_grad, as TransformerEngine keeps it.
        ctx.weight_ref = weakref.ref(weight)
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        main_grad = ctx.weight_ref().main_grad
        main_grad.add_(grad_output.t().to(main_grad.dtype) @ x.to(main_grad.dtype))
        return grad_output @ weight, None


class _FusedWgradLinear(torch.nn.Module):
    """A linear layer with fused weight-grad accumulation and an ordinary bias."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.randn(8, device=device, dtype=dtype))
        self.weights_seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weights_seen.append(self.weight)
        return _FusedWgradLinearFn.apply(x, self.weight) + self.bias


class _DelayedWgradLinearFn(torch.autograd.Function):
    """``x @ weight.t()`` whose backward leaves the weight grad to
    ``backward_dw()``, like TransformerEngine's ``delay_wgrad_compute``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, pending: list):
        ctx.save_for_backward(x, weight)
        ctx.weight_ref = weakref.ref(weight)
        ctx.pending = pending
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        ctx.pending.append((ctx.weight_ref, x, grad_output))
        return grad_output @ weight, None, None


class _RunDelayedWgrad(torch.autograd.Function):
    """Identity whose backward runs after the next layer's backward and calls
    ``callback``, as Megatron-LM's MoE layer runs the experts' delayed weight
    grads in the token dispatch's backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, callback) -> torch.Tensor:
        ctx.callback = callback
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.callback()
        return grad_output, None


class _DelayedWgradLinear(torch.nn.Module):
    """A linear layer whose weight grad waits for ``backward_dw()``, which adds
    it to the weight's grad, and an ordinary bias."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8, device=device))
        self.bias = torch.nn.Parameter(torch.randn(8, device=device))
        self.pending: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DelayedWgradLinearFn.apply(x, self.weight, self.pending) + self.bias

    def backward_dw(self) -> None:
        for weight_ref, x, grad_output in self.pending:
            weight, wgrad = weight_ref(), grad_output.t() @ x
            weight.grad = wgrad if weight.grad is None else weight.grad + wgrad
        self.pending.clear()


class _DelayedWgradModel(torch.nn.Module):
    """``out(relu(layer(inp(x))))``: ``layer``'s delayed weight grad is computed
    after ``layer``'s backward, which then calls ``on_wgrad``."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.inp = torch.nn.Linear(8, 8, device=device)
        self.layer = _DelayedWgradLinear(device=device)
        self.out = torch.nn.Linear(8, 6, device=device)
        self.on_wgrad = lambda: None

    def _backward_dw(self) -> None:
        self.layer.backward_dw()
        self.on_wgrad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _RunDelayedWgrad.apply(self.inp(x), self._backward_dw)
        return self.out(torch.relu(self.layer(x)))


def _init_params_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(),
                dtype=param.dtype,
                device=param.device,
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


class _RankRoutedExperts(torch.nn.Module):
    """Each rank routes to one expert, like experts that got tokens on one rank.

    The side branch runs on every rank, but only rank 0's output uses it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(2))
        self.side = torch.nn.Linear(8, 8)

    def forward(
        self, x: torch.Tensor, use_side: bool = True, shift: int = 0
    ) -> torch.Tensor:
        out = self.experts[(dist.get_rank() + shift) % 2](x)
        if use_side:
            side = self.side(x)
            if dist.get_rank() == 0:
                out = out + side
        return out


def _average_reference_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def _prefer_recompute_context_fn():
    def prefer_recompute_policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(prefer_recompute_policy)


def _checkpoint_transformer_execution_units(model: torch.nn.Module) -> None:
    model.tok_embeddings = checkpoint_wrapper(
        model.tok_embeddings,
        context_fn=_prefer_recompute_context_fn,
    )
    model.pos_embeddings = checkpoint_wrapper(
        model.pos_embeddings,
        context_fn=_prefer_recompute_context_fn,
    )
    for idx, layer in enumerate(model.layers):
        model.layers[idx] = checkpoint_wrapper(
            layer,
            context_fn=_prefer_recompute_context_fn,
        )
    model.norm = checkpoint_wrapper(
        model.norm,
        context_fn=_prefer_recompute_context_fn,
    )
    model.output = checkpoint_wrapper(
        model.output,
        context_fn=_prefer_recompute_context_fn,
    )


class _RecomputeAtOutput(torch.autograd.Function):
    """Identity that returns a view of the region output and saves an empty
    tensor, so the region's recompute starts in this backward, as
    torch_remat's ``_TriggerCheckpointRecompute`` does."""

    @staticmethod
    def forward(ctx, output: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(output.new_empty(0))
        return output.view_as(output)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Unpacking the saved tensor runs the recompute.
        (_,) = ctx.saved_tensors
        return grad_output


def _checkpoint_like_torch_remat(module: torch.nn.Module) -> None:
    """Checkpoint ``module`` as torchtitan's ``RegionAC`` does with
    ``torch_remat.checkpoint``: replace its forward, keeping its hooks and
    parameter FQNs, with a non-reentrant checkpoint whose recompute replays
    the original forward without the module's hooks."""
    forward = module.forward

    def region(*args: object) -> torch.Tensor:
        return _RecomputeAtOutput.apply(forward(*args))

    def checkpointed_forward(*args: object) -> torch.Tensor:
        return checkpoint(region, *args, use_reentrant=False)

    module.forward = checkpointed_forward


class TestFlexShardTraining(FSDPTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_input_lifetime(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        width = 512
        model = torch.nn.ModuleList(
            [_ParallelLinears(width, device=device_type) for _ in range(2)]
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        # 1 MiB FP32 grads stay in the caching allocator's small pool, so the
        # 2 MiB packed buckets are the only large-pool blocks. With the default
        # of one pending reduce-scatter, as in FSDP2, the second pack reuses the
        # first pack's freed block.
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [f"{index}.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                )
                for index in range(2)
            ],
        )

        # Grad-requiring inputs let each layer's input-grad trigger reduce
        # mid-backward, one bucket after the other.
        x = torch.ones(
            (2, width),
            dtype=torch.bfloat16,
            device=device_type.type,
            requires_grad=True,
        )
        output = sum(layer((index + 1) * x) for index, layer in enumerate(model))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        packed_ptrs: list[int] = []
        original_pack = Shard._pack_reduce_scatter_grad
        original_reduce_scatter = dist.reduce_scatter_single

        def record_packed_buffer(placement, tensors, infos, world_size, *args):
            packed, layout = original_pack(placement, tensors, infos, world_size, *args)
            packed_ptrs.append(packed.data_ptr())
            if len(packed_ptrs) == 2:
                # The fix orders this producer stream after the first consumer.
                torch.cuda.current_stream().synchronize()
            return packed, layout

        def delayed_reduce_scatter(output, input, *args, **kwargs):
            if len(packed_ptrs) == 1:
                torch.cuda._sleep(500_000_000)
            # Stage before the real collective so the original packed buffer's
            # cross-stream lifetime remains FlexShard's responsibility.
            staged_input = input.clone()
            return original_reduce_scatter(output, staged_input, *args, **kwargs)

        with (
            mock.patch.object(
                Shard,
                "_pack_reduce_scatter_grad",
                record_packed_buffer,
            ),
            mock.patch.object(
                dist,
                "reduce_scatter_single",
                delayed_reduce_scatter,
            ),
        ):
            output.float().sum().backward()

        for index, layer in enumerate(model):
            for param in layer.parameters():
                self.assertIsNotNone(param.grad)
                self.assertEqual(
                    param.grad, torch.full_like(param.grad, 2 * (index + 1))
                )
        self.assertEqual(len(packed_ptrs), len(model))
        self.assertEqual(packed_ptrs[0], packed_ptrs[1])

    def _check_unshard_buffer_lifetime(
        self,
        mesh,
        placement_fn,
        placement_type,
        run_backward,
    ):
        width = 1024
        x = torch.ones(
            (2, width),
            dtype=torch.bfloat16,
            device=device_type.type,
            requires_grad=run_backward,
        )
        model = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    width,
                    width,
                    bias=False,
                    device=device_type.type,
                    dtype=torch.bfloat16,
                )
            ]
        )
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].weight.diagonal().fill_(1)
        model[0].weight.requires_grad_(False)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["0.*"],
                    placement_fn=placement_fn,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )
        comm_context = next(iter(model._flex_shard_eager_comm_contexts.values()))
        torch.cuda.empty_cache()

        original_finish_prepared_unshard = placement_type.finish_prepared_unshard
        gathered_buffer_metadata = None
        reused_before_forward = None
        scratch_keepalive: list[torch.Tensor] = []

        def record_gathered_buffer(placement, prepared):
            nonlocal gathered_buffer_metadata
            gathered_buffer = prepared.buffers[1]
            gathered_buffer_metadata = (
                tuple(gathered_buffer.shape),
                gathered_buffer.dtype,
                gathered_buffer.device,
                gathered_buffer.data_ptr(),
            )
            if not run_backward:
                torch.cuda._sleep(500_000_000)
            return original_finish_prepared_unshard(placement, prepared)

        def try_reuse_gathered_buffer():
            buffer_metadata = gathered_buffer_metadata
            if buffer_metadata is None:
                raise AssertionError("Expected the all-gather output buffer.")
            shape, dtype, device, data_ptr = buffer_metadata
            with torch.cuda.stream(comm_context.unshard_stream):
                scratch_buffer = torch.zeros(
                    shape,
                    dtype=dtype,
                    device=device,
                )
            comm_context.unshard_stream.synchronize()
            scratch_keepalive.append(scratch_buffer)
            return scratch_buffer.data_ptr() == data_ptr

        def probe_before_forward(module, args):
            nonlocal reused_before_forward
            reused_before_forward = try_reuse_gathered_buffer()

        before_forward_hook = model[0].register_forward_pre_hook(probe_before_forward)
        with mock.patch.object(
            placement_type,
            "finish_prepared_unshard",
            record_gathered_buffer,
        ):
            if run_backward:
                output = model[0](x)
            else:
                with torch.no_grad():
                    output = model[0](x)
        before_forward_hook.remove()

        # The unshard copied the gathered bucket into the persistent buffers, so
        # the all-gather output is reusable before module compute and backward.
        self.assertTrue(reused_before_forward)
        if not run_backward:
            self.assertEqual(output, x)
            return

        torch.cuda._sleep(500_000_000)
        backward_delay_done = torch.cuda.Event()
        backward_delay_done.record()
        output.sum().backward()
        self.assertTrue(backward_delay_done.query())
        self.assertEqual(x.grad, torch.ones_like(x))

    @skip_if_lt_x_gpu(2)
    def test_unshard_buffer_lifetime(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        bucketed_block_placements = make_bucketed_block_placement_fn(
            dims=(0,),
            blocks_per_rank=(1, 1),
        )
        cases = (
            ("shard", per_param_placements, Shard, False),
            ("bucketed_block", bucketed_block_placements, BucketedBlockShard, False),
            (
                "bucketed_block_backward",
                bucketed_block_placements,
                BucketedBlockShard,
                True,
            ),
        )
        for case in cases:
            with self.subTest(placement=case[0]):
                self._check_unshard_buffer_lifetime(mesh, *case[1:])

    @skip_if_lt_x_gpu(2)
    def test_uneven_shard_bucket_matches_unsharded_reference(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(42)
        model = _UnevenMLPStack(device=device_type)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [f"layers.{layer_idx}.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=True,
                )
                for layer_idx in range(len(model.layers))
            ],
        )

        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        element_size = infos[0].dtype.itemsize
        self.assertEqual(
            {
                fqn: info.storage_nbytes // element_size
                for fqn, info in storage.param_infos.items()
            },
            {
                "layers.0.out_activation.weight": 1,
                "layers.0.in_proj.weight": 9,
                "layers.0.in_proj.bias": 3,
                "layers.0.out_proj.weight": 10,
                "layers.0.out_proj.bias": 2,
            },
        )
        self.assertEqual(storage.total_bytes, 25 * element_size)
        self.assertEqual(
            sum(info.local_numel for info in infos),
            25 if self.rank == 0 else 14,
        )
        self.assertEqual(infos[0].fqn, "layers.0.out_activation.weight")
        self.assertEqual(infos[0].local_numel, 1 if self.rank == 0 else 0)
        local_params = [storage.get_local_view(info.fqn) for info in infos]
        prepared_unshard = infos[0].placement.prepare_unshard_bucket(
            local_params,
            infos,
            mesh,
            None,
        )
        send_buf = prepared_unshard.buffers[0]
        self.assertEqual(send_buf.numel(), 25)
        self.assertEqual(
            send_buf.untyped_storage().data_ptr(),
            storage.byte_storage.untyped_storage().data_ptr(),
        )
        del prepared_unshard

        optimizer = make_test_sgd(model.parameters(), lr=0.1)
        reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
        torch.manual_seed(43 + self.rank)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(4, 3, device=device_type)
            output = model(inputs)
            reference_output = reference(inputs)
            self.assertEqual(output, reference_output)

            output.square().sum().backward()
            reference_output.square().sum().backward()
            _average_reference_grads(reference)
            check_flex_shard_parity(
                self,
                reference,
                model,
                self.rank,
                self.world_size,
            )

            optimizer.step()
            reference_optimizer.step()
            check_flex_shard_parity(
                self,
                reference,
                model,
                self.rank,
                self.world_size,
            )

    @skip_if_lt_x_gpu(2)
    def test_copy_free_matches_reference(self):
        # The all-gather sends the local storage as is, refills gather straight
        # into the persistent buckets, and with grads allocated in the parameter
        # layout (gradient_bucket, by a pre-backward hook) the reduce-scatter
        # reads that bucket as is; training matches an unsharded reference, with
        # and without
        # reshard-after-forward and no-sync. Whole-parameter BucketedOwned
        # buckets (Muon's) do so in their owners' rows of the gathered bucket,
        # and get their grads allocated in that layout even without the hook.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        placement_fns = {
            "bucketed_block": make_bucketed_block_placement_fn(
                dims=(0,), blocks_per_rank=(1,) * self.world_size
            ),
            "owned": make_bucketed_owned_full_param_placement_fn(),
        }
        # Where each placement's reduce looks for the gradient bucket.
        gradient_bucket_lookups = {
            "bucketed_block": (BucketedBlockShard, "_gradient_bucket_of"),
            "owned": (MixedBucketPlacement, "_prepare_owner_rows_reduce_grad"),
        }
        begin_bucket_unshard = bucket_runtime.begin_bucket_unshard
        for placement, reshard_after_forward, gradient_bucket in (
            ("bucketed_block", False, True),
            ("bucketed_block", True, True),
            ("bucketed_block", True, False),
            ("owned", False, True),
            ("owned", True, True),
            ("owned", True, False),
        ):
            placement_fn = placement_fns[placement]
            lookup = gradient_bucket_lookups[placement]
            find_gradient_bucket = getattr(*lookup)
            with self.subTest(
                placement=placement,
                reshard_after_forward=reshard_after_forward,
                gradient_bucket=gradient_bucket,
            ):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(16, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 8),
                ).to(device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [f"{idx}.*"],
                            placement_fn=placement_fn,
                            mesh=mesh,
                            reshard_after_forward=reshard_after_forward,
                            pre_backward_hook=(
                                alloc_grads_in_param_layout if gradient_bucket else None
                            ),
                        )
                        for idx in (0, 2, 4)
                    ],
                )
                optimizer = make_test_sgd(model.parameters(), lr=0.1)
                reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
                in_place: list[bool] = []
                sent_storage: list[bool] = []
                reduced_bucket: list[bool] = []

                def recording_begin(local_shards, *args, **kwargs):
                    in_place.append(kwargs.get("persistent_buffers") is not None)
                    handle = begin_bucket_unshard(local_shards, *args, **kwargs)
                    send = handle.prepared.buffers[0]
                    sent_storage.append(
                        send.untyped_storage().data_ptr()
                        in {
                            shard.untyped_storage().data_ptr() for shard in local_shards
                        }
                    )
                    return handle

                def recording_find_gradient_bucket(*args):
                    found = find_gradient_bucket(*args)
                    reduced_bucket.append(found is not None)
                    return found

                torch.manual_seed(1 + self.rank)
                with (
                    mock.patch.object(
                        bucket_runtime, "begin_bucket_unshard", recording_begin
                    ),
                    mock.patch.object(*lookup, recording_find_gradient_bucket),
                ):
                    for step in range(4):
                        optimizer.zero_grad(set_to_none=True)
                        reference_optimizer.zero_grad(set_to_none=True)
                        # The last step accumulates two microbatches without
                        # sync into the same grads.
                        microbatches = 2 if step == 3 else 1
                        for microbatch in range(microbatches):
                            model.set_requires_gradient_sync(
                                microbatch == microbatches - 1
                            )
                            x = torch.randn(4, 16, device=device_type)
                            output = model(x)
                            reference_output = reference(x)
                            self.assertEqual(output, reference_output)
                            output.square().sum().backward()
                            reference_output.square().sum().backward()
                        _average_reference_grads(reference)
                        optimizer.step()
                        reference_optimizer.step()
                # The first unshard of each bucket copies; refills are in place.
                self.assertEqual(in_place[:3], [False] * 3)
                self.assertTrue(all(in_place[3:]))
                self.assertGreater(len(in_place), 3)
                self.assertTrue(all(sent_storage), sent_storage)
                self.assertTrue(reduced_bucket)
                expect_bucket = gradient_bucket or placement == "owned"
                self.assertTrue(all(found == expect_bucket for found in reduced_bucket))

    @skip_if_lt_x_gpu(2)
    def test_owner_rows_hold_no_padding(self):
        # Whole-parameter owners keep rows of their own lengths: each rank
        # stores its row alone, the gathered bucket holds the rows back to
        # back, and uneven rows take one broadcast and one reduce per owner
        # (the list all_gather and reduce_scatter), while equal rows keep the
        # single collectives. Training matches an unsharded reference.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        names = (
            "all_gather",
            "all_gather_into_tensor",
            "reduce_scatter",
            "reduce_scatter_single",
        )
        for out_features, gradient_bucket in ((8, False), (8, True), (16, False)):
            # Linear(16, 16) then Linear(16, out_features) on 2 ranks: largest
            # first to the least-loaded rank gives rows of 256 and 152 elements
            # for 8 output features, and 272 each for 16.
            rows = (256, 152) if out_features == 8 else (272, 272)
            with self.subTest(
                out_features=out_features, gradient_bucket=gradient_bucket
            ):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(16, 16),
                    torch.nn.Linear(16, out_features),
                ).to(device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["0.*", "1.*"],
                            placement_fn=make_bucketed_owned_full_param_placement_fn(),
                            mesh=mesh,
                            reshard_after_forward=False,
                            pre_backward_hook=(
                                alloc_grads_in_param_layout if gradient_bucket else None
                            ),
                        )
                    ],
                )
                element_size = 4
                self.assertEqual(
                    model.sharded_bucket_storages[0].total_bytes,
                    rows[self.rank] * element_size,
                )
                optimizer = make_test_sgd(model.parameters(), lr=0.1)
                reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
                calls: Counter[str] = Counter()

                def counting(name, original):
                    def call(*args, **kwargs):
                        calls[name] += 1
                        return original(*args, **kwargs)

                    return call

                patches = [
                    mock.patch.object(dist, name, counting(name, getattr(dist, name)))
                    for name in names
                ]
                torch.manual_seed(1 + self.rank)
                for patch in patches:
                    patch.start()
                try:
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        reference_optimizer.zero_grad(set_to_none=True)
                        x = torch.randn(4, 16, device=device_type)
                        output = model(x)
                        # Unsharded until backward: the params view the
                        # gathered rows.
                        self.assertEqual(
                            model[0].weight.untyped_storage().nbytes(),
                            sum(rows) * element_size,
                        )
                        reference_output = reference(x)
                        self.assertEqual(output, reference_output)
                        output.square().sum().backward()
                        reference_output.square().sum().backward()
                        _average_reference_grads(reference)
                        optimizer.step()
                        reference_optimizer.step()
                finally:
                    for patch in patches:
                        patch.stop()
                uneven = rows[0] != rows[1]
                self.assertEqual(calls["all_gather"] > 0, uneven, calls)
                self.assertEqual(calls["reduce_scatter"] > 0, uneven, calls)
                self.assertEqual(calls["all_gather_into_tensor"] > 0, not uneven, calls)
                self.assertEqual(calls["reduce_scatter_single"] > 0, not uneven, calls)

    @skip_if_lt_x_gpu(2)
    def test_owner_rows_reduce_grads_in_param_layout(self):
        # Owner rows reduce grads allocated in the parameter layout as is, so
        # the bucket runtime allocates them there when the unsharded params are
        # in the dtype their grads accumulate and reduce in, not when the grads
        # accumulate in a wider dtype. Shard buckets copy theirs in.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        bf16 = torch.bfloat16
        owned = make_bucketed_owned_full_param_placement_fn()
        for placement_fn, mp_policy, expected in (
            (owned, None, True),
            (owned, MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16), True),
            (owned, MixedPrecisionPolicy(param_dtype=bf16), False),
            (per_param_placements, None, False),
        ):
            with self.subTest(placement_fn=placement_fn, mp_policy=mp_policy):
                model = torch.nn.Sequential(
                    torch.nn.Linear(16, 16), torch.nn.Linear(16, 8)
                ).to(device_type)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["0.*", "1.*"],
                            placement_fn=placement_fn,
                            mesh=mesh,
                            mp_policy=mp_policy,
                        )
                    ],
                )
                infos = list(model.sharded_bucket_storages[0].param_infos.values())
                self.assertEqual(
                    infos[0].placement.reduces_grads_in_param_layout(
                        infos, self.world_size
                    ),
                    expected,
                )

    @skip_if_lt_x_gpu(2)
    def test_gradient_divide_factor(self):
        # The reduced gradient is the sum over the mesh divided by the factor,
        # e.g. the dense data-parallel size for an expert bucket on an expert
        # data-parallel mesh.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        cases = (
            # (dtype, factor in BucketSpec, factor set later, no-sync)
            (torch.float32, 3.0, None, True),  # PREMUL_SUM
            (torch.bfloat16, None, 8, False),
            (torch.float16, 6, None, False),  # divides before and after the sum
        )
        for dtype, factor, later_factor, no_sync in cases:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
                ).to(device_type, dtype)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                            gradient_divide_factor=factor,
                        )
                    ],
                )
                if later_factor is not None:
                    model.set_gradient_divide_factor(later_factor)
                torch.manual_seed(1 + self.rank)
                for step in range(2 if no_sync else 1):
                    model.set_requires_gradient_sync(step == int(no_sync))
                    x = torch.randn(4, 8, device=device_type, dtype=dtype)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                tolerance = {} if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.float()
                    dist.all_reduce(expected)
                    expected = expected_shard(
                        expected / (later_factor or factor),
                        rank=self.rank,
                        world_size=self.world_size,
                    )
                    self.assertEqual(param.grad.float(), expected, **tolerance)

    @skip_if_lt_x_gpu(2)
    def test_separate_reduce_scatter_group(self):
        # Reduce-scatters move to one dedicated process group over the mesh's
        # ranks, shared by the buckets, while all-gathers stay on the mesh's
        # group; the gradients stay bitwise equal, and False goes back.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
        ).to(device_type)
        shared = copy.deepcopy(model)
        for module in (model, shared):
            flex_shard(
                module,
                buckets=[
                    BucketSpec(
                        [f"{index}.*"],
                        placement_fn=per_param_placements,
                        mesh=mesh,
                    )
                    for index in (0, 2)
                ],
            )
        with mock.patch.object(dist, "new_group", wraps=dist.new_group) as new_group:
            model.set_separate_reduce_scatter_group()
        self.assertEqual(new_group.call_count, 1)
        storages = model.sharded_bucket_storages
        (reduce_mesh,) = {
            id(storage.reduce_mesh): storage.reduce_mesh for storage in storages
        }.values()
        self.assertIsNot(reduce_mesh.get_group(), mesh.get_group())
        self.assertEqual(reduce_mesh.mesh.tolist(), mesh.mesh.tolist())

        torch.manual_seed(1 + self.rank)
        x = torch.randn(4, 8, device=device_type)
        with (
            mock.patch.object(
                dist, "all_gather_single", wraps=dist.all_gather_single
            ) as all_gather,
            mock.patch.object(
                dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single
            ) as reduce_scatter,
        ):
            model(x).sum().backward()
        self.assertEqual(
            {call.kwargs["group"] for call in all_gather.call_args_list},
            {mesh.get_group()},
        )
        self.assertEqual(
            {call.kwargs["group"] for call in reduce_scatter.call_args_list},
            {reduce_mesh.get_group()},
        )
        shared(x).sum().backward()
        for param, shared_param in zip(model.parameters(), shared.parameters()):
            self.assertTrue(torch.equal(param.grad, shared_param.grad))

        # Enabling again keeps the group; False goes back to the mesh's.
        with mock.patch.object(dist, "new_group", wraps=dist.new_group) as new_group:
            model.set_separate_reduce_scatter_group()
        self.assertEqual(new_group.call_count, 0)
        self.assertTrue(all(storage.reduce_mesh is reduce_mesh for storage in storages))
        model.set_separate_reduce_scatter_group(False)
        self.assertTrue(all(storage.reduce_mesh is mesh for storage in storages))

    @skip_if_lt_x_gpu(2)
    def test_finalize_backward(self):
        # Grads accumulated without sync reduce-scatter outside backward, as a
        # pipeline stage does after its last backward: after automatic
        # finalization at each backward's end, and in manual mode, where
        # backward finishes nothing itself.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        for manual, async_op in ((False, True), (True, False)):
            with self.subTest(manual=manual, async_op=async_op):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
                ).to(device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )
                model.set_manual_backward_finalization(manual)
                model.set_requires_gradient_sync(False)
                model.set_reshard_after_backward(False)
                torch.manual_seed(1 + self.rank)
                for _ in range(2):
                    x = torch.randn(4, 8, device=device_type)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                model.set_requires_gradient_sync(True)
                model.set_reshard_after_backward(True)
                handle = model.finalize_backward(async_op=async_op)
                if async_op:
                    with self.assertRaisesRegex(RuntimeError, "wait on the previous"):
                        model.finalize_backward()
                    handle.wait()
                else:
                    self.assertIsNone(handle)
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.clone()
                    dist.all_reduce(expected, op=dist.ReduceOp.AVG)
                    self.assertEqual(
                        param.grad,
                        expected_shard(
                            expected, rank=self.rank, world_size=self.world_size
                        ),
                    )

    @skip_if_lt_x_gpu(2)
    def test_finalize_backward_reduces_in_reverse_module_order(self):
        # As FSDP2's finalize_backward, which a pipeline stage calls after its
        # last backward, the reduce-scatters go in reverse module order, as
        # backward runs, whatever order the buckets were given in. Then a
        # block's dense bucket reuses the memory its routed experts, reduced
        # first, freed. As in torchtitan, the norm and output projection's
        # bucket comes first.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 6),
        ).to(device_type)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    modules,
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
                for modules in (["3", "4"], ["0"], ["2"])
            ],
        )
        model.set_manual_backward_finalization(True)
        model.set_requires_gradient_sync(False)
        model.set_reshard_after_backward(False)
        model(torch.randn(4, 8, device=device_type)).sum().backward()
        model.set_requires_gradient_sync(True)
        reduced = []
        begin_reduce_grad = bucket_runtime.begin_reduce_grad

        def recording_begin(tensors, infos, *args, **kwargs):
            reduced.append(infos[0].fqn.split(".")[0])
            return begin_reduce_grad(tensors, infos, *args, **kwargs)

        with mock.patch.object(bucket_runtime, "begin_reduce_grad", recording_begin):
            model.finalize_backward()
        self.assertEqual(reduced, ["3", "2", "0"])

    @skip_if_lt_x_gpu(2)
    def test_defer_post_backward(self):
        # A weight grad computed after its layer's backward, as with
        # TransformerEngine's delay_wgrad_compute: the layer's bucket defers its
        # post-backward, which would otherwise reduce-scatter before that grad
        # exists, until finish_deferred_backward. Once with sync and
        # reshard-after-forward, once with a backward without sync first.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        for reshard_after_forward, no_sync in ((True, False), (False, True)):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                torch.manual_seed(0)
                model = _DelayedWgradModel(device=device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [f"{name}.*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=reshard_after_forward,
                            defer_post_backward=name == "layer",
                        )
                        for name in ("inp", "layer", "out")
                    ],
                )
                model.on_wgrad = lambda model=model: model.finish_deferred_backward(
                    model.layer.weight
                )
                with self.assertRaisesRegex(ValueError, "does not defer"):
                    model.finish_deferred_backward(model.out.weight)
                torch.manual_seed(1 + self.rank)
                for step in range(2 if no_sync else 1):
                    model.set_requires_gradient_sync(step == int(no_sync))
                    x = torch.randn(4, 8, device=device_type)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.clone()
                    dist.all_reduce(expected, op=dist.ReduceOp.AVG)
                    self.assertEqual(
                        param.grad,
                        expected_shard(
                            expected, rank=self.rank, world_size=self.world_size
                        ),
                    )
        # A syncing backward that ends with the bucket unfinished raises instead
        # of reduce-scattering without the late weight grad.
        model.on_wgrad = lambda: None
        with self.assertRaisesRegex(RuntimeError, "finish_deferred_backward"):
            model(torch.randn(4, 8, device=device_type)).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_unshard(self):
        # A schedule that runs a layer's computation directly, bypassing the
        # forward hooks that gather its bucket, unshards every bucket first, as
        # Megatron-LM's EP overlap schedule does. It also splits each backward
        # into two calls, so backward finalization is manual: the buckets keep
        # their grads until finalize_backward after the last microbatch, and,
        # without reshard_after_backward, their params, as with FSDP2. Two
        # steps, so the second unshard gathers the updated shards, which its
        # outputs check. With async_op, the schedule waits on the unshard's
        # handle before it uses the params.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        for async_op in (False, True):
            with self.subTest(async_op=async_op):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
                ).to(device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["0.*"],
                            placement_fn=make_bucketed_block_placement_fn(
                                dims=(0,), blocks_per_rank=(1,) * self.world_size
                            ),
                            mesh=mesh,
                            reshard_after_forward=False,
                        ),
                        BucketSpec(
                            ["2.*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        ),
                    ],
                )
                model.set_manual_backward_finalization(True)
                optimizer = make_test_sgd(model.parameters(), lr=0.1)
                reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
                torch.manual_seed(1 + self.rank)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    model.set_reshard_after_backward(False)
                    handle = model.unshard(async_op=async_op)
                    if async_op:
                        handle.wait()
                    for microbatch in range(2):
                        model.set_requires_gradient_sync(microbatch == 1)
                        x = torch.randn(4, 8, device=device_type)
                        # The first layer's forward hooks never run.
                        hidden = torch.relu(
                            torch.nn.functional.linear(
                                x, model[0].weight, model[0].bias
                            )
                        )
                        detached = hidden.detach().requires_grad_()
                        output = model[2](detached)
                        reference_output = reference(x)
                        self.assertEqual(output, reference_output)
                        output.sum().backward()
                        hidden.backward(detached.grad)
                        reference_output.sum().backward()
                    model.set_reshard_after_backward(True)
                    model.finalize_backward()
                    _average_reference_grads(reference)
                    optimizer.step()
                    reference_optimizer.step()

    @skip_if_lt_x_gpu(2)
    def test_unshard_async_op_finished_by_forward(self):
        # A pipeline schedule unshards a stage ahead of its forward with
        # async_op=True. The forward finishes each bucket's pending all-gather
        # in its pre-forward hook, so the handle's wait() is left with nothing.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
        ).to(device_type)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [pattern],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
                for pattern in ("0.*", "2.*")
            ],
        )
        (context,) = getattr(model, bucket_runtime._EAGER_COMM_CONTEXTS_ATTR).values()
        handle = model.unshard(async_op=True)
        self.assertEqual(len(context.pending_unshards), 2)
        self.assertFalse(any(bucket.is_unsharded for bucket in context.buckets))
        x = torch.randn(4, 8, device=device_type)
        self.assertEqual(model(x), reference(x))
        self.assertEqual(context.pending_unshards, [])
        handle.wait()
        self.assertTrue(all(bucket.is_unsharded for bucket in context.buckets))

    def _two_bucket_mlp(self, mesh) -> torch.nn.Module:
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
        ).to(device_type)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [pattern],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
                for pattern in ("0.*", "2.*")
            ],
        )
        return model

    @skip_if_lt_x_gpu(2)
    def test_unshard_all_gathers_on_current_stream(self):
        # As FSDP2's FSDPModule.unshard: the all-gathers run on the caller's
        # stream, asynchronously with async_op=True, and the handle's wait()
        # waits on their work.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        original = dist.all_gather_single
        for async_op in (False, True):
            with self.subTest(async_op=async_op):
                model = self._two_bucket_mlp(mesh)
                calls = []
                waits = []

                class RecordingWork:
                    def __init__(self, work) -> None:
                        self.work = work

                    def wait(self, *wait_args, **wait_kwargs):
                        waits.append(True)
                        return self.work.wait(*wait_args, **wait_kwargs)

                def record(*args, **kwargs):
                    calls.append(
                        (torch.cuda.current_stream(), kwargs.get("async_op", False))
                    )
                    work = original(*args, **kwargs)
                    return None if work is None else RecordingWork(work)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with (
                    mock.patch.object(dist, "all_gather_single", record),
                    torch.cuda.stream(stream),
                ):
                    handle = model.unshard(async_op=async_op)
                    self.assertEqual(len(waits), 0)
                    if async_op:
                        handle.wait()
                self.assertEqual(calls, [(stream, async_op)] * 2)
                self.assertEqual(len(waits), 2 if async_op else 0)
                torch.cuda.current_stream().wait_stream(stream)

    @skip_if_lt_x_gpu(2)
    def test_single_rank_bucket_skips_all_gather_and_reduce_scatter(self):
        # As FSDP2 on a 1-rank group, e.g. torchtitan's routed experts when the
        # expert-parallel degree equals the data-parallel one: the unshard
        # casts the local shards straight into the unsharded params, with no
        # all-gather or gather buffer, through a forward's hook and through
        # unshard(async_op=True) alike, and the post-backward reduce keeps the
        # packed grads as the sharded grads, with no reduce-scatter, on the
        # current stream, so the cast sharded grads come from its allocator pool.
        # Without collectives, the 1-rank group can use a fake backend.
        mesh = init_device_mesh(
            device_type.type, (self.world_size, 1), mesh_dim_names=("dp", "edp")
        )["edp"]
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
        ).to(device_type)
        reference = copy.deepcopy(model).to(torch.bfloat16)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [pattern],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                    reshard_after_forward=False,
                )
                for pattern in ("0.*", "2.*")
            ],
        )
        x = torch.randn(4, 8, device=device_type, dtype=torch.bfloat16)
        device_module = torch.get_device_module(device_type)
        backward_streams = set()

        class RecordStreams(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                backward_streams.add(device_module.current_stream())
                return func(*args, **(kwargs or {}))

        def check_unsharded_params() -> None:
            for module, reference_module in (
                (model[0], reference[0]),
                (model[2], reference[2]),
            ):
                self.assertEqual(module.weight.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(module.weight, reference_module.weight))
                self.assertTrue(torch.equal(module.bias, reference_module.bias))

        with (
            mock.patch.object(
                dist,
                "all_gather_single",
                side_effect=AssertionError("all-gather on a 1-rank mesh"),
            ),
            mock.patch.object(
                dist,
                "reduce_scatter_single",
                side_effect=AssertionError("reduce-scatter on a 1-rank mesh"),
            ),
        ):
            output = model(x)
            check_unsharded_params()
            reference_output = reference(x)
            self.assertTrue(torch.equal(output, reference_output))
            with RecordStreams():
                output.sum().backward()
            for context in model._flex_shard_eager_comm_contexts.values():
                self.assertNotIn(context.reduce_grad_stream, backward_streams)
            reference_output.sum().backward()
            for param, reference_param in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(param.grad, reference_param.grad.float()))
            model.reshard()
            model.unshard(async_op=True).wait()
            check_unsharded_params()
            model.reshard()

    @skip_if_lt_x_gpu(2)
    def test_shard_all_gathers_in_place(self):
        # With a narrower unsharded dtype, Shard casts the local shard straight
        # into its chunk of the gather buffer and all-gathers in place, as
        # FSDP2's copy-in does. A 1-rank mesh skips the all-gather (see
        # test_single_rank_bucket_skips_all_gather_and_reduce_scatter).
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8)
        ).to(device_type)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [pattern],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                    reshard_after_forward=True,
                )
                for pattern in ("0.*", "2.*")
            ],
        )
        bucket = [
            bucket
            for context in model._flex_shard_eager_comm_contexts.values()
            for bucket in context.buckets
        ][-1]
        handle = bucket.begin_unshard()
        send_buf, gather_buf = handle.prepared.buffers
        self.assertEqual(
            send_buf.untyped_storage().data_ptr(),
            gather_buf.untyped_storage().data_ptr(),
        )
        bucket.finish_unshard(handle)
        bucket.reshard()

    @skip_if_lt_x_gpu(2)
    def test_torch_compile_traces_per_bucket_collectives(self):
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        args, model = flex_shard_transformer_model(mesh)
        graphs = []

        def capture_backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        compiled_model = torch.compile(model, backend=capture_backend, fullgraph=True)
        compiled_model(transformer_inputs(args, device=device_type)).sum().backward()
        self.assertEqual(1, len(graphs))
        # Collectives live in the fwd_body_*/bwd_body_* subgraphs that
        # autograd_function_apply dispatches to, not at the top level.
        subgraph_targets = set()
        for _, submodule in graphs[0].named_modules():
            if isinstance(submodule, torch.fx.GraphModule):
                subgraph_targets.update(
                    str(node.target) for node in submodule.graph.nodes
                )
        # Functional collectives keep the comm reorderable by graph passes.
        self.assertIn("_c10d_functional.all_gather_into_tensor", subgraph_targets)
        self.assertIn("_c10d_functional.reduce_scatter_tensor", subgraph_targets)
        self.assertIn("_c10d_functional.wait_tensor", subgraph_targets)

    @skip_if_lt_x_gpu(2)
    def test_unshard_leaves_no_memory_on_flex_shard_stream(self):
        # A pipeline schedule unshards each stage with async_op=True. As in
        # FSDP2, the all-gather buffers belong to the current stream, so after
        # the copy-out its pool reuses them, rather than FlexShard's unshard
        # stream keeping them reserved.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        model = self._two_bucket_mlp(mesh)
        (context,) = getattr(model, bucket_runtime._EAGER_COMM_CONTEXTS_ATTR).values()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for _ in range(2):
            model.unshard(async_op=True).wait()
            model.reshard()
        torch.cuda.synchronize()
        unshard_stream = context.unshard_stream.cuda_stream
        segments = [
            segment
            for segment in torch.cuda.memory._snapshot()["segments"]
            if segment["device"] == torch.cuda.current_device()
            and segment["stream"] == unshard_stream
        ]
        self.assertEqual(segments, [])

    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation(self):
        # Three microbatches, in three modes:
        # - "sync": every microbatch reduce-scatters.
        # - "per_bucket": sync is off for every bucket but the layer's on the
        #   first two microbatches, set through the per-bucket setters.
        # - "keep": sync off for the model on the first two microbatches,
        #   keeping every bucket but the norm's unsharded between them.
        # Only the layer reshards after forward. The last microbatch skips the
        # layer and the positional embeddings, so their kept grads are reduced
        # in the end-of-backward callback.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, base_model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(base_model)
        torch.manual_seed(42 + self.rank + 1)
        inputs = [
            transformer_inputs(args, batch_size=batch_size, device=device_type)
            for batch_size in (3, 2, 2)
        ]

        def microbatch_loss(module, idx):
            x = inputs[idx]
            if idx < len(inputs) - 1:
                return module(x).sum()
            return module.output(module.norm(module.tok_embeddings(x))).sum()

        for mode in ("sync", "per_bucket", "keep"):
            model = copy.deepcopy(base_model)
            reference = copy.deepcopy(base_model)
            buckets = transformer_bucket_specs(args.n_layers, mesh)
            buckets[2] = dataclasses.replace(buckets[2], reshard_after_forward=True)
            flex_shard(model, buckets=buckets)
            storages = model.sharded_bucket_storages
            layer, norm = storages[2], storages[3]
            if mode == "keep":
                model.set_reshard_after_backward(False)
                norm.set_reshard_after_backward(True)
            optim = make_test_sgd(model.parameters(), lr=0.1)
            ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

            runtime = bucket_runtime.BucketRuntime
            with (
                mock.patch.object(
                    runtime,
                    "begin_unshard",
                    autospec=True,
                    side_effect=runtime.begin_unshard,
                ) as unshards,
                mock.patch.object(
                    runtime,
                    "reduce_grads",
                    autospec=True,
                    side_effect=runtime.reduce_grads,
                ) as reduces,
            ):
                for idx in range(len(inputs)):
                    last = idx == len(inputs) - 1
                    if mode == "per_bucket":
                        if last:
                            model.reshard()  # keeps the accumulated grads
                        for storage in storages:
                            storage.set_requires_gradient_sync(last or storage is layer)
                    if mode == "keep":
                        model.set_requires_gradient_sync(last)
                        if last:
                            # As with FSDP2: reshard before the optimizer step
                            model.set_reshard_after_backward(True)
                    loss = microbatch_loss(model, idx)
                    ref_loss = microbatch_loss(reference, idx)
                    self.assertEqual(loss, ref_loss)
                    loss.backward()
                    ref_loss.backward()

            reduced = Counter(call.args[0].debug_fqn for call in reduces.call_args_list)
            gathered = Counter(
                call.args[0].debug_fqn for call in unshards.call_args_list
            )
            once = dict.fromkeys(
                ["tok_embeddings", "pos_embeddings", "layers.0", "norm", "output"], 1
            )
            if mode == "per_bucket":
                # The layer reduce-scatters in both microbatches that use it,
                # the other buckets once per step.
                self.assertEqual(reduced, {**once, "layers.0": 2})
            if mode == "keep":
                # One reduce-scatter per bucket per step. Kept buckets
                # all-gather once per step; the layer also re-gathers in each
                # backward and the norm in each forward.
                self.assertEqual(reduced, once)
                self.assertEqual(gathered, {**once, "layers.0": 3, "norm": 3})
            _average_reference_grads(reference)
            check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

            optim.step()
            ref_optim.step()
            check_flex_shard_parity(self, reference, model, self.rank, self.world_size)
            # The syncing backward resharded, so forward sees the stepped shards.
            self.assertEqual(model(inputs[0]).sum(), reference(inputs[0]).sum())

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_policy(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(model)
        reference = copy.deepcopy(model).to(torch.bfloat16)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["tok_embeddings.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                    ),
                    reshard_after_forward=False,
                ),
                *transformer_bucket_specs(
                    args.n_layers, mesh, reshard_after_forward=False
                )[1:],
            ],
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=2, device=device_type)
        # A microbatch without sync first: the kept bf16 param accumulates its
        # grads in the fp32 reduce dtype until the syncing backward.
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        model.tok_embeddings(x).float().sum().backward()
        model.set_requires_gradient_sync(True)
        model.set_reshard_after_backward(True)
        reference.tok_embeddings(x).float().sum().backward()
        unsharded_param = model.tok_embeddings.weight
        self.assertEqual(unsharded_param.dtype, torch.bfloat16)
        self.assertEqual(unsharded_param.grad.dtype, torch.float32)

        output = model.tok_embeddings(x)
        ref_output = reference.tok_embeddings(x)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output, ref_output)

        output.float().sum().backward()
        ref_output.float().sum().backward()
        self.assertIsNone(unsharded_param.grad)

        grad = model.tok_embeddings._parameters["weight"].grad
        self.assertIsNotNone(grad)
        self.assertEqual(grad.dtype, torch.float32)

        ref_grad = reference.tok_embeddings.weight.grad.to(torch.float32)
        dist.all_reduce(ref_grad, op=dist.ReduceOp.AVG)
        expected_grad = expected_shard(
            ref_grad,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.assertEqual(grad, expected_grad)

        # A syncing backward without a kept grad defers the upcast: the bf16
        # grad reaches the reduce-scatter, whose copy-in widens it, and
        # grad_dtype is restored afterwards.
        runtime = bucket_runtime.BucketRuntime
        reduced_dtypes = []
        orig_reduce_grads = runtime.reduce_grads

        def record_reduce_grads(self, grads, *args, **kwargs):
            # The reduce frees its list's grads once copied in
            reduced_dtypes.append([grad.dtype for grad in grads])
            return orig_reduce_grads(self, grads, *args, **kwargs)

        with mock.patch.object(runtime, "reduce_grads", record_reduce_grads):
            model.tok_embeddings(x).float().sum().backward()
        self.assertEqual(reduced_dtypes, [[torch.bfloat16]])
        self.assertEqual(unsharded_param.grad_dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_rank_dependent_unused_params(self):
        # Ranks leave different params of a bucket, or a whole bucket's
        # outputs, without grads; every rank still reduce-scatters every
        # bucket, with zeros for its unused params. Three microbatches: the
        # first without sync (keeping fp32 grads on rank 0 only), the second
        # skipping the side bucket, so the pending sync makes it reduce on
        # every rank, and routing each rank to the other expert, so kept and
        # fresh grads add up, and the third syncing with the side
        # bucket's outputs unused on rank 1.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(0)
        model = _RankRoutedExperts().to(device_type, torch.bfloat16)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [pattern],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
                    reshard_after_forward=False,
                )
                for pattern in ("experts.*", "side.*")
            ],
        )
        x = torch.randn(4, 8, device=device_type, dtype=torch.bfloat16)
        for idx, (use_side, shift) in enumerate(((True, 0), (False, 1), (True, 0))):
            model.set_requires_gradient_sync(idx > 0)
            model(x, use_side=use_side, shift=shift).sum().backward()
            reference(x, use_side=use_side, shift=shift).sum().backward()
        for param in reference.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        _average_reference_grads(reference)
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

    @skip_if_lt_x_gpu(2)
    def test_per_param_mixed_precision_policy(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(42)
        model = _MixedPrecisionWeights(device=device_type)
        reference = copy.deepcopy(model)
        reference.low_weight = torch.nn.Parameter(reference.low_weight.bfloat16())
        reference.low_weight2 = torch.nn.Parameter(reference.low_weight2.bfloat16())
        mixed_placement = MixedBucketPlacement({})

        def mixed_placement_fn(named_params, mesh):
            del mesh
            return {fqn: (mixed_placement.shard0,) for fqn, _ in named_params}

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=mixed_placement_fn,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        param_dtype_overrides={"full_weight": torch.float32},
                    ),
                    reshard_after_forward=False,
                )
            ],
        )

        torch.manual_seed(43 + self.rank)
        x = torch.randn(4, 8, device=device_type)
        with (
            mock.patch.object(
                dist,
                "all_gather_into_tensor",
                wraps=dist.all_gather_into_tensor,
            ) as all_gather,
            mock.patch.object(
                dist,
                "reduce_scatter_single",
                wraps=dist.reduce_scatter_single,
            ) as reduce_scatter,
        ):
            output = model(x)
            output.sum().backward()

        reference_output = reference(x)
        reference_output.sum().backward()
        self.assertEqual(output, reference_output)
        self.assertEqual(all_gather.call_count, 1)
        self.assertEqual(reduce_scatter.call_count, 1)

        model_params = dict(model.named_parameters())
        for fqn, reference_param in reference.named_parameters():
            grad = model_params[fqn].grad
            expected_grad = reference_param.grad.float()
            dist.all_reduce(expected_grad, op=dist.ReduceOp.AVG)
            self.assertEqual(
                grad,
                expected_shard(
                    expected_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                ),
            )

    @skip_if_lt_x_gpu(2)
    def test_grads_accumulate_in_unsharded_grad_dtype(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        bf16, fp32 = torch.bfloat16, torch.float32
        # Each case exercises one source of the unsharded accumulation dtype.
        # Grads of 256, from a backward without sync, and then 1, from a
        # syncing one, accumulate to 257 only in fp32. As in FSDP2, the
        # backward without sync starts the grad, so it keeps the upcast: the
        # grad is already in the accumulation dtype when the bucket reshards.
        cases = (
            ("storage_dtype", fp32, None, bf16, None, fp32, 257.0),
            ("explicit_grad_dtype", bf16, fp32, None, None, fp32, 257.0),
            ("reduce_dtype", bf16, fp32, None, bf16, bf16, 256.0),
        )
        for case in cases:
            (
                name,
                dtype,
                grad_dtype,
                param_dtype,
                reduce_dtype,
                accumulation_dtype,
                grad_value,
            ) = case
            compute_dtype = param_dtype or dtype
            with self.subTest(name=name):
                model = _AccumulatingWeight(device=device_type, dtype=dtype)
                if grad_dtype is not None:
                    model.weight.grad_dtype = grad_dtype
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            mp_policy=MixedPrecisionPolicy(
                                param_dtype=param_dtype,
                                reduce_dtype=reduce_dtype,
                            ),
                            reshard_after_forward=False,
                        )
                    ],
                )

                x = torch.ones(1, 8, dtype=compute_dtype, device=device_type)
                runtime = bucket_runtime.BucketRuntime
                at_reshard = []

                def record_reshard(bucket, reshard=runtime.reshard):
                    at_reshard.append(
                        [
                            p.grad.dtype
                            for p in bucket.unsharded_params
                            if p.grad is not None
                        ]
                    )
                    reshard(bucket)

                with mock.patch.object(
                    runtime, "reshard", autospec=True, side_effect=record_reshard
                ) as reshards:
                    model.set_requires_gradient_sync(False)
                    model(256 * x).backward()
                unsharded_param = reshards.call_args.args[0].unsharded_params[0]
                self.assertEqual(at_reshard, [[accumulation_dtype]])
                self.assertEqual(unsharded_param.grad.dtype, accumulation_dtype)
                model.set_requires_gradient_sync(True)
                model(x).backward()
                grad = model._parameters["weight"].grad
                self.assertIsNotNone(grad)
                self.assertEqual(grad.dtype, fp32)
                self.assertEqual(
                    grad,
                    expected_shard(
                        torch.full(
                            (8, 8),
                            grad_value,
                            dtype=fp32,
                            device=device_type,
                        ),
                        rank=self.rank,
                        world_size=self.world_size,
                    ),
                )

    @skip_if_lt_x_gpu(2)
    def test_reshard_restores_deferred_grad_upcast(self):
        # torch.distributed.pipelining runs a stage's forward and backward to
        # infer its shapes, with manual backward finalization, then calls
        # reshard(). That backward syncs, so it defers the upcast, and nothing
        # finishes it; reshard() must restore the upcast, as FSDP2's
        # post-backward does after every call.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model = _AccumulatingWeight(device=device_type, dtype=torch.float32)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                    reshard_after_forward=False,
                )
            ],
        )
        x = torch.ones(1, 8, dtype=torch.bfloat16, device=device_type)
        model.set_manual_backward_finalization(True)
        model(256 * x).backward()
        unsharded_param = model._parameters["weight"]
        self.assertIsNone(unsharded_param.grad_dtype)
        self.assertEqual(unsharded_param.grad.dtype, torch.bfloat16)

        model.reshard()
        self.assertEqual(unsharded_param.grad_dtype, torch.float32)
        self.assertEqual(unsharded_param.grad.dtype, torch.float32)
        # A backward without sync then adds 1 to the kept 256 in fp32.
        model.set_manual_backward_finalization(False)
        model.set_requires_gradient_sync(False)
        model(x).backward()
        grad = unsharded_param.grad
        self.assertEqual(grad.dtype, torch.float32)
        self.assertEqual(grad, torch.full_like(grad, 257.0))

    @skip_if_lt_x_gpu(2)
    def test_bucket_reduces_in_promoted_grad_dtype(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model = _AccumulatingWeights(device=device_type, dtype=torch.bfloat16)
        model.b.weight.grad_dtype = torch.float32
        flex_shard_cuda(model, mesh)

        x = torch.ones(1, 8, dtype=torch.bfloat16, device=device_type)
        # Grads of 256 and then 1: b accumulates them in fp32, a in bf16.
        model.set_requires_gradient_sync(False)
        model(256 * x).backward()
        model.set_requires_gradient_sync(True)
        model(x).backward()

        # b's fp32 sum of 257 survives only if the shared reduction promotes
        # a's bf16 and b's fp32 gradients to fp32.
        for module, grad_value, grad_dtype in (
            (model.a, 256.0, torch.bfloat16),
            (model.b, 257.0, torch.float32),
        ):
            grad = module._parameters["weight"].grad
            self.assertIsNotNone(grad)
            self.assertEqual(grad.dtype, grad_dtype)
            self.assertEqual(grad, torch.full_like(grad, grad_value))

    @skip_if_lt_x_gpu(2)
    def test_kept_grads_accumulate_in_param_grad_dtype(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model = _ReusedWeights(device=device_type, dtype=torch.bfloat16)
        model.b.weight.grad_dtype = torch.float32
        flex_shard_cuda(model, mesh)
        model.set_reshard_after_backward(False)
        x = torch.ones(1, 8, dtype=torch.bfloat16, device=device_type)

        model.set_requires_gradient_sync(False)
        model(x).backward()
        # Like FSDP2, a's kept grad stays in its bf16 grad dtype, not the
        # bucket's fp32 reduce dtype promoted by b's grad_dtype.
        self.assertEqual(model.a._parameters["weight"].grad.dtype, torch.bfloat16)
        model.set_requires_gradient_sync(True)
        model.set_reshard_after_backward(True)
        model(x).backward()
        # As in FSDP2, b's first backward, without sync, starts its grad and
        # keeps the upcast, so its two uses sum to 257 in fp32; the syncing
        # backward defers the upcast and sums them to 256 in bf16.
        for module, grad_value, grad_dtype in (
            (model.a, 512.0, torch.bfloat16),
            (model.b, 513.0, torch.float32),
        ):
            grad = module._parameters["weight"].grad
            self.assertEqual(grad.dtype, grad_dtype)
            self.assertEqual(grad, torch.full_like(grad, grad_value))

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_with_activation_checkpointing(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        _checkpoint_transformer_execution_units(model)
        _checkpoint_transformer_execution_units(reference)

        flex_shard(
            model,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=True,
            ),
        )

        # FlexShard leaves the user's activation-checkpoint wrapper untouched.
        self.assertIsInstance(model.layers[0], CheckpointWrapper)
        self.assertIs(
            model.layers[0].checkpoint_fn.keywords["context_fn"],
            _prefer_recompute_context_fn,
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)
        optim = make_test_sgd(model.parameters(), lr=0.1)
        ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        # A microbatch without sync first, keeping the params unsharded after
        # it: the next forward skips the all-gather and still reshards after
        # forward, and the recompute in backward re-gathers.
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        model(x).sum().backward()
        model.set_requires_gradient_sync(True)
        model.set_reshard_after_backward(True)
        reference(x).sum().backward()
        loss = model(x).sum()
        ref_loss = reference(x).sum()
        self.assertEqual(loss, ref_loss)
        loss.backward()
        ref_loss.backward()

        _average_reference_grads(reference)
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

        optim.step()
        ref_optim.step()
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_with_torch_remat_checkpointing(self):
        # torch_remat replays a block's forward directly, so the block's
        # forward hooks don't run in the recompute, and starts the replay in
        # the backward of an autograd Function viewing the block's output. The
        # block's bucket must be unsharded before that backward.
        # _checkpoint_like_torch_remat reproduces both, without depending on
        # torch_remat.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        def nested_layer_buckets(specs: list[BucketSpec]) -> list[BucketSpec]:
            # As torchtitan's MoE blocks: a bucket hooked on the block and
            # buckets hooked on modules inside it, which the replay calls.
            nested = []
            for spec in specs:
                (pattern,) = spec.patterns
                if not pattern.startswith("layers."):
                    nested.append(spec)
                    continue
                layer = pattern.removesuffix(".*")
                nested += [
                    dataclasses.replace(
                        spec,
                        patterns=[f"{layer}.attention_norm.*", f"{layer}.ffn_norm.*"],
                    ),
                    dataclasses.replace(spec, patterns=[f"{layer}.attention.*"]),
                    dataclasses.replace(spec, patterns=[f"{layer}.feed_forward.*"]),
                ]
            return nested

        for nested in (False, True):
            with self.subTest(nested=nested):
                args, model = make_transformer_model(
                    device=device_type.type,
                    n_layers=2,
                    vocab_size=15,
                )
                _init_params_deterministically(model)
                reference = copy.deepcopy(model)
                for layer in model.layers:
                    _checkpoint_like_torch_remat(layer)
                buckets = transformer_bucket_specs(
                    args.n_layers,
                    mesh,
                    reshard_after_forward=True,
                )
                flex_shard(
                    model,
                    buckets=nested_layer_buckets(buckets) if nested else buckets,
                )

                torch.manual_seed(42 + self.rank + 1)
                x = transformer_inputs(args, batch_size=3, device=device_type)
                optim = make_test_sgd(model.parameters(), lr=0.1)
                ref_optim = make_test_sgd(reference.parameters(), lr=0.1)
                for _ in range(2):
                    optim.zero_grad(set_to_none=True)
                    ref_optim.zero_grad(set_to_none=True)
                    loss = model(x).sum()
                    ref_loss = reference(x).sum()
                    self.assertEqual(loss, ref_loss)
                    loss.backward()
                    ref_loss.backward()
                    _average_reference_grads(reference)
                    check_flex_shard_parity(
                        self, reference, model, self.rank, self.world_size
                    )
                    optim.step()
                    ref_optim.step()
                check_flex_shard_parity(
                    self, reference, model, self.rank, self.world_size
                )

    @skip_if_lt_x_gpu(2)
    def test_submodule_buckets_under_selective_activation_checkpointing(self):
        # Buckets inside checkpointed blocks unshard again in the recompute,
        # which runs before their pre-backward hooks, after forward prefetched
        # them from the previous block, or after a microbatch left them
        # unsharded; selective activation checkpointing must not see it.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        blocks_per_rank = (1,) * self.world_size
        block_shard = BlockShard(blocks_per_rank=blocks_per_rank)

        def dist_muon_mixed_placements(named_params, mesh):
            # DistMuon's mixed buckets: block-sharded weights, owned biases.
            owned = {
                fqn: idx % mesh.size()
                for idx, (fqn, _) in enumerate(named_params)
                if fqn.endswith("bias")
            }
            blocks = {
                fqn: blocks_per_rank
                for fqn, _ in named_params
                if fqn.endswith("weight")
            }
            return make_mixed_placement_fn(owned, blocks)(named_params, mesh)

        placement_fns = {
            "shard": per_param_placements,
            "block_shard": lambda named_params, _mesh: {
                fqn: (block_shard,) for fqn, _ in named_params
            },
            "bucketed_block": make_bucketed_block_placement_fn(
                dims=(0,), blocks_per_rank=blocks_per_rank
            ),
            # Whole-parameter BucketedOwned buckets unshard through their
            # owners' rows, without copies.
            "owned": make_bucketed_owned_full_param_placement_fn(),
            "dist_muon_mixed": dist_muon_mixed_placements,
        }
        begin_unshard = bucket_runtime.BucketRuntime.begin_unshard
        gathers: list[bucket_runtime.BucketRuntime] = []

        def count_gathers(bucket, *args, **kwargs):
            gathers.append(bucket)
            return begin_unshard(bucket, *args, **kwargs)

        for (name, placement_fn), accumulate in (
            (case, accumulate)
            for case in placement_fns.items()
            for accumulate in (False, True)
        ):
            with self.subTest(placement=name, accumulate=accumulate):
                torch.manual_seed(0)
                reference = torch.nn.Sequential(*(_GatedBlock(8) for _ in range(3)))
                reference = reference.to(device_type)
                # The same FlexShard model without and with checkpointing.
                plain, model = copy.deepcopy(reference), copy.deepcopy(reference)
                for module in (reference, model):
                    for idx, block in enumerate(module):
                        module[idx] = checkpoint_wrapper(
                            block, context_fn=_prefer_recompute_context_fn
                        )
                buckets = [
                    BucketSpec(
                        [f"{idx}.*{part}"],
                        placement_fn=placement_fn,
                        mesh=mesh,
                        reshard_after_forward=True,
                    )
                    for idx in range(len(model))
                    for part in ("attn", "mlp")
                ]
                for module in (plain, model):
                    flex_shard(module, buckets=buckets)
                modules = (reference, plain, model)
                optims = [make_test_sgd(m.parameters(), lr=0.1) for m in modules]
                for _ in range(3):
                    if accumulate:
                        # The params stay unsharded after this microbatch, so
                        # the next forward skips the all-gather and the
                        # recompute re-gathers.
                        for module in (plain, model):
                            module.set_reshard_after_backward(False)
                            module.set_requires_gradient_sync(False)
                        x = torch.randn(4, 8, device=device_type)
                        for module in modules:
                            module(x).square().sum().backward()
                        for module in (plain, model):
                            module.set_reshard_after_backward(True)
                            module.set_requires_gradient_sync(True)
                    x = torch.randn(4, 8, device=device_type)
                    ref_loss = reference(x).square().sum()
                    plain_loss = plain(x).square().sum()
                    gathers.clear()
                    with mock.patch.object(
                        bucket_runtime.BucketRuntime, "begin_unshard", count_gathers
                    ):
                        loss = model(x).square().sum()
                        loss.backward()
                    ref_loss.backward()
                    plain_loss.backward()
                    self.assertEqual(plain_loss, ref_loss)
                    self.assertEqual(loss, ref_loss)
                    if not accumulate:
                        # One all-gather per bucket in forward and in backward:
                        # the recompute finishes the next bucket's prefetch.
                        self.assertEqual(len(gathers), 2 * len(buckets))
                    for (plain_fqn, plain_param), (fqn, param) in zip(
                        plain.named_parameters(), model.named_parameters(), strict=True
                    ):
                        self.assertEqual(
                            fqn.replace("_checkpoint_wrapped_module.", ""), plain_fqn
                        )
                        self.assertEqual(param.grad, plain_param.grad)
                    for optim in optims:
                        optim.step()
                        optim.zero_grad(set_to_none=True)
                for plain_param, param in zip(
                    plain.parameters(), model.parameters(), strict=True
                ):
                    self.assertEqual(param, plain_param)

    @skip_if_lt_x_gpu(2)
    def test_group_buckets_under_selective_activation_checkpointing(self):
        # A bucket whose patterns name several modules hooks each of them.
        # Inside a checkpointed block those group hooks unshard again in the
        # recompute, so selective activation checkpointing must not see them
        # either.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(0)
        plain = torch.nn.Sequential(*(_GatedBlock(8) for _ in range(3)))
        plain = plain.to(device_type)
        model = copy.deepcopy(plain)
        for idx, block in enumerate(model):
            model[idx] = checkpoint_wrapper(
                block, context_fn=_prefer_recompute_context_fn
            )
        for module, prefix in ((plain, ""), (model, "_checkpoint_wrapped_module.")):
            buckets = [
                BucketSpec(
                    [f"{idx}.{prefix}attn", f"{idx}.{prefix}mlp"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=True,
                )
                for idx in range(len(module))
            ]
            flex_shard(module, buckets=buckets)
        optims = [make_test_sgd(m.parameters(), lr=0.1) for m in (plain, model)]
        for _ in range(3):
            x = torch.randn(4, 8, device=device_type)
            plain_loss = plain(x).square().sum()
            plain_loss.backward()
            loss = model(x).square().sum()
            loss.backward()
            self.assertEqual(loss, plain_loss)
            for plain_param, param in zip(
                plain.parameters(), model.parameters(), strict=True
            ):
                self.assertEqual(param.grad, plain_param.grad)
            for optim in optims:
                optim.step()
                optim.zero_grad(set_to_none=True)
        for plain_param, param in zip(
            plain.parameters(), model.parameters(), strict=True
        ):
            self.assertEqual(param, plain_param)

    @skip_if_lt_x_gpu(2)
    def test_pre_backward_and_post_reduce_hooks(self):
        # Kernels that add weight grads into main_grad and give autograd none
        # (fused gradient accumulation) need the grad allocated before their
        # backward. The hooks do what a trainer would: the pre-backward hook
        # allocates each unsharded param's grad and aliases it as main_grad,
        # the fused weight grads and the bias's autograd grads accumulate there
        # over two backwards without sync, and the post-reduce hook drops the
        # alias once the third reduce-scatters them. The first bucket reshards
        # after forward, so its backward re-gathers before the hook runs.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        for dtype, reduce_dtype in (
            (torch.float32, None),
            (torch.bfloat16, torch.float32),
        ):
            grad_dtype = reduce_dtype or dtype
            calls = []

            def alias_main_grads(named_params, grad_dtype=grad_dtype):
                calls.append(("pre_backward", [fqn for fqn, _ in named_params]))
                for _, param in named_params:
                    if param.grad is None:
                        param.grad = torch.zeros(
                            param.shape, dtype=grad_dtype, device=param.device
                        )
                    param.main_grad = param.grad

            def drop_main_grads(named_params):
                calls.append(("post_reduce", [fqn for fqn, _ in named_params]))
                for _, param in named_params:
                    self.assertIsNone(param.grad)
                    del param.main_grad

            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    _FusedWgradLinear(device=device_type, dtype=dtype),
                    _FusedWgradLinear(device=device_type, dtype=dtype),
                )
                reference = copy.deepcopy(model)
                # As Megatron-LM's wrapper does, so the local shards keep
                # grads in the accumulation dtype too.
                for param in (*model.parameters(), *reference.parameters()):
                    param.grad_dtype = grad_dtype
                for layer in reference:
                    layer.weight.main_grad = torch.zeros_like(
                        layer.weight, dtype=grad_dtype
                    )
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [f"{idx}.*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            mp_policy=MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
                            reshard_after_forward=idx == 0,
                            pre_backward_hook=alias_main_grads,
                            post_reduce_hook=drop_main_grads,
                        )
                        for idx in range(2)
                    ],
                )
                torch.manual_seed(1 + self.rank)
                for idx in range(3):
                    x = torch.randn(4, 8, device=device_type, dtype=dtype)
                    model.set_requires_gradient_sync(idx == 2)
                    model(x).float().sum().backward()
                    reference(x).float().sum().backward()
                    for layer in model:
                        weight = layer.weights_seen[-1]
                        if idx < 2:
                            self.assertIs(weight.main_grad, weight.grad)
                            self.assertEqual(weight.grad.dtype, grad_dtype)
                        else:
                            self.assertIsNone(weight.grad)
                            self.assertFalse(hasattr(weight, "main_grad"))
                fqns = [["1.weight", "1.bias"], ["0.weight", "0.bias"]]
                self.assertEqual(
                    calls,
                    [("pre_backward", fqns[0]), ("pre_backward", fqns[1])] * 2
                    + [
                        ("pre_backward", fqns[0]),
                        ("post_reduce", fqns[0]),
                        ("pre_backward", fqns[1]),
                        ("post_reduce", fqns[1]),
                    ],
                )
                for layer in reference:
                    layer.weight.grad = layer.weight.main_grad
                _average_reference_grads(reference)
                check_flex_shard_parity(
                    self, reference, model, self.rank, self.world_size
                )

    @skip_if_lt_x_gpu(2)
    def test_tied_weights_in_one_bucket(self):
        # Registered tying (output.weight is tok_embeddings.weight), which FSDP2
        # supports within one FSDP group. One bucket holds every name of the
        # shared parameter, so FlexShard hooks it on the root, around the
        # per-layer buckets: both names swap together, both uses read the
        # gathered weight, and the grads of both reduce once, with
        # reshard-after-forward on or off. Each step runs a backward without
        # sync first.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                torch.manual_seed(0)
                args, model = make_transformer_model(
                    device=device_type.type, n_layers=2, weight_tying=True
                )
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                reference = copy.deepcopy(model)
                spec = dict(
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
                flex_shard(
                    model,
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
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                optim = make_test_sgd(model.parameters(), lr=0.1)
                ref_optim = make_test_sgd(reference.parameters(), lr=0.1)
                torch.manual_seed(1 + self.rank)
                for _ in range(2):
                    x = transformer_inputs(args, batch_size=2, device=device_type)
                    for sync in (False, True):
                        model.set_requires_gradient_sync(sync)
                        model(x).sum().backward()
                        reference(x).sum().backward()
                    _average_reference_grads(reference)
                    check_flex_shard_parity(
                        self, reference, model, self.rank, self.world_size
                    )
                    optim.step()
                    ref_optim.step()
                    optim.zero_grad(set_to_none=True)
                    ref_optim.zero_grad(set_to_none=True)
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                layouts = get_flex_shard_global_layouts(model)
                self.assertIs(
                    layouts["output.weight"], layouts["tok_embeddings.weight"]
                )


if __name__ == "__main__":
    run_tests()
