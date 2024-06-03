# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import unittest

import torch
from torch import nn
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Shard
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.experimental.attention import (
    _CausalBehavior,
    _is_causal_behavior,
    _scaled_dot_product_chunk_flash_attention,
    _scaled_dot_product_ring_efficient_attention,
    _scaled_dot_product_ring_flash_attention,
    attention_context_parallel,
    AttentionContextParallel,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    ModelArgs,
    Transformer,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


class RingAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @parametrize("is_causal", [True, False])
    def test_ring_attention_sdpa(self, is_causal: bool) -> None:
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 8
        context_tokens = query_tokens if is_causal else 8
        dim = 32
        nheads = 8
        query = torch.rand(
            (bs, nheads, self.world_size * query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
        )
        value = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
        )

        query_placement = [Shard(2)]
        dquery = distribute_tensor(query, device_mesh, query_placement)
        self.assertEqual(query.shape, (bs, nheads, self.world_size * query_tokens, dim))

        context_placement = [Shard(2)]
        dkey = distribute_tensor(key, device_mesh, context_placement)
        dvalue = distribute_tensor(value, device_mesh, context_placement)
        for t in [dkey, dvalue]:
            self.assertEqual(
                t.shape, (bs, nheads, context_tokens * self.world_size, dim)
            )
            self.assertEqual(t.to_local().shape, (bs, nheads, context_tokens, dim))

        # local tensors
        out, logsumexp, *others = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, is_causal=is_causal
        )

        self.assertEqual(out.shape, (bs, nheads, self.world_size * query_tokens, dim))
        out.sum().backward()
        out_grad = query.grad
        query.grad = None
        self.assertIsNotNone(out_grad)

        # compute chunked version to compare distributed to chunked implementations
        # chunked isn't numerically identical to single operator version
        (
            out_chunk,
            logsumexp_chunk,
            *others,
        ) = _scaled_dot_product_chunk_flash_attention(
            query,
            key,
            value,
            size=self.world_size,
            is_causal=is_causal,
        )

        out_chunk.sum().backward()
        self.assertEqual(
            out_chunk.shape, (bs, nheads, self.world_size * query_tokens, dim)
        )
        self.assertEqual(logsumexp_chunk, logsumexp)
        self.assertEqual(out_chunk, out)
        out_chunk_grad = query.grad
        query.grad = None
        # gradient doesn't match due to numerical issues with chunk size > 1
        # self.assertEqual(out_chunk_grad, out_grad)

        # parallel behavior
        with attention_context_parallel(), CommDebugMode() as comm_mode:
            (
                out_parallel,
                logsumexp_parallel,
                *others,
            ) = torch.ops.aten._scaled_dot_product_flash_attention(
                dquery, dkey, dvalue, is_causal=is_causal
            )
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: self.world_size - 1,
            },
        )
        self.assertEqual(out_parallel.placements, (Shard(2),))
        self.assertEqual(
            out_parallel._local_tensor.shape, (bs, nheads, query_tokens, dim)
        )
        self.assertEqual(
            out_parallel.shape, (bs, nheads, self.world_size * query_tokens, dim)
        )
        out_parallel_tensor = out_parallel.full_tensor()
        self.assertEqual(out_parallel_tensor, out)
        logsumexp_parallel_tensor = logsumexp_parallel.full_tensor()
        self.assertEqual(logsumexp_parallel_tensor, logsumexp)

        self.assertIsNone(dquery.grad)
        with attention_context_parallel(), CommDebugMode() as comm_mode:
            out_parallel.sum().backward()

        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1) * 2,
            },
        )
        out_parallel_grad = dquery.grad.full_tensor()
        dquery.grad = None
        self.assertEqual(out_parallel_grad, out_chunk_grad)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    @parametrize("is_causal", [True, False])
    def test_ring_attention_native_transformer(self, is_causal: bool) -> None:
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        dtype = torch.bfloat16
        bs = 8
        ntokens = 8
        dim = 32
        nheads = 8
        num_layers = 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nheads,
            dim_feedforward=dim,
            batch_first=True,
        ).to(dtype)
        encoder_layer = parallelize_module(
            module=encoder_layer,
            device_mesh=device_mesh,
            parallelize_plan={
                "self_attn": AttentionContextParallel(),
            },
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        model = model.to(self.device_type).to(dtype)

        mask = (
            nn.Transformer.generate_square_subsequent_mask(
                ntokens, device=self.device_type, dtype=dtype
            )
            if is_causal
            else None
        )
        seq = torch.rand((bs, ntokens, dim), device=self.device_type, dtype=dtype)

        with CommDebugMode() as comm_mode:
            out = model(seq, mask=mask, is_causal=is_causal)
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1) * num_layers,
            },
        )

        with CommDebugMode() as comm_mode:
            out.sum().backward()
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * 2
                * num_layers,
            },
        )

    def test_is_causal_behavior(self) -> None:
        # not causal
        self.assertEqual(
            _is_causal_behavior(rank=0, world_size=4, i=0, is_causal=False),
            _CausalBehavior.NOT_IS_CAUSAL,
        )

        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.SKIP],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def test_ring_attention_custom_transformer(self) -> None:
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        dtype = torch.bfloat16
        bs = 2
        args = ModelArgs()

        model = Transformer(args).to(dtype).to(self.device_type)

        model = parallelize_module(
            module=model,
            device_mesh=device_mesh,
            parallelize_plan={
                f"layers.{i}.attention": AttentionContextParallel()
                for i in range(args.n_layers)
            },
        )

        seq = torch.randint(
            args.vocab_size, (bs, args.max_seq_len), device=self.device_type
        )

        with CommDebugMode() as comm_mode:
            out = model(seq)
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * args.n_layers,
            },
        )

        with CommDebugMode() as comm_mode:
            out.sum().backward()
        self.assertDictEqual(
            comm_mode.get_comm_counts(),
            {
                c10d_functional.all_to_all_single: (self.world_size - 1)
                * 2
                * args.n_layers,
            },
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @parametrize(
        "attention_fn",
        [
            _scaled_dot_product_ring_flash_attention,
            _scaled_dot_product_ring_efficient_attention,
            # _scaled_dot_product_ring_cudnn_attention, # TODO: not built by default
        ],
    )
    def test_ring_attention_compile(self, attention_fn: object) -> None:
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 8
        context_tokens = 24
        dim = 32
        nheads = 8
        query = torch.rand(
            (bs, nheads, self.world_size * query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
        )
        value = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
        )

        query_placement = [Shard(2)]
        dquery = distribute_tensor(query, device_mesh, query_placement)
        self.assertEqual(query.shape, (bs, nheads, self.world_size * query_tokens, dim))

        context_placement = [Shard(2)]
        dkey = distribute_tensor(key, device_mesh, context_placement)
        dvalue = distribute_tensor(value, device_mesh, context_placement)

        # compiled = attention_fn
        compiled = torch.compile(attention_fn, fullgraph=True, backend="aot_eager")

        out, lse, *args = compiled(
            device_mesh.get_group(),
            dquery.to_local(),
            dkey.to_local(),
            dvalue.to_local(),
        )
        self.assertEqual(out.shape, (bs, nheads, query_tokens, dim))
        self.assertIsInstance(lse, torch.Tensor)

        (
            out_chunk,
            *others,
        ) = _scaled_dot_product_chunk_flash_attention(
            query,
            key,
            value,
            size=self.world_size,
            is_causal=False,
        )
        self.assertEqual(
            out,
            out_chunk[
                :, :, self.rank * query_tokens : (self.rank + 1) * query_tokens, :
            ],
        )

        out.sum().backward()


instantiate_parametrized_tests(RingAttentionTest)

if __name__ == "__main__":
    run_tests()
