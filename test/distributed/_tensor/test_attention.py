# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.experimental.attention import (
    _AttentionContextParallel,
    _CausalBehavior,
    _cp_options,
    _is_causal_behavior,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfRocm,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    ModelArgs,
    Transformer,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional
backends = []
if PLATFORM_SUPPORTS_FLASH_ATTENTION:
    backends.append(SDPBackend.FLASH_ATTENTION)
if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
    backends.append(SDPBackend.EFFICIENT_ATTENTION)


class RingAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @skip_if_lt_x_gpu(2)
    @skipIfRocm  # Missing _c10d_functional_autograd::all_to_all_single
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    @with_comms
    @decorateIf(
        unittest.skip, lambda params: params["load_balance"] and not params["is_causal"]
    )
    @parametrize("is_causal", [True, False])
    @parametrize("compiled", [True, False])
    @parametrize("backend", backends)
    @parametrize("load_balance", [True, False])
    @parametrize("rotater", [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER])
    def test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
    ) -> None:
        _cp_options.rotate_method = rotater
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 64
        context_tokens = 64
        dim = 32
        nheads = 8
        torch.manual_seed(10)
        dtype = (
            torch.bfloat16 if backend == SDPBackend.FLASH_ATTENTION else torch.float32
        )

        if is_causal and compiled and self.world_size > 2:
            # TODO: Fix this after we move `wait_tensor` to use `with_effect`.
            return

        _cp_options.enable_load_balance = load_balance

        q = torch.rand(
            (bs, nheads, self.world_size * query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )

        # Ensure all ranks have the same initialization data.
        with torch.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            out.sum().backward()

        cp_q = q.clone().detach()
        cp_k = k.clone().detach()
        cp_v = v.clone().detach()
        # Theoretically, context_parallel() should not be used to shard
        # parameters because when require_grad is True, resize_ is not
        # allowed. But requires_grad of cp_q, cp_k, and cp_v are False
        # now. So we can just use context_parallel() to shard q, k, v.
        # In reality, context_paralle() should be used to shard the input.
        with context_parallel(
            device_mesh, buffers=(cp_q, cp_k, cp_v), buffer_seq_dims=(2, 2, 2)
        ):
            cp_q.requires_grad = True
            cp_k.requires_grad = True
            cp_v.requires_grad = True
            with CommDebugMode() as comm_mode:
                with sdpa_kernel(backend):
                    if compiled:
                        fn = torch.compile(
                            F.scaled_dot_product_attention,
                            fullgraph=True,
                            backend="aot_eager",
                        )
                    else:
                        fn = F.scaled_dot_product_attention

                    cp_out = fn(cp_q, cp_k, cp_v, is_causal=is_causal)
                    cp_out.sum().backward()

                    if not compiled and rotater == _RotateMethod.ALL_TO_ALL:
                        # Compiler and CommDebugMode do not work well together.
                        self.assertDictEqual(
                            comm_mode.get_comm_counts(),
                            {
                                c10d_functional.all_to_all_single: self.world_size * 3
                                - 2
                            },
                        )

            # Due to numerical error, we need to choose different atol for different
            # attention kernels
            cp_out, cp_dq, cp_dk, cp_dv = context_parallel_unshard(
                device_mesh,
                [cp_out, cp_q.grad, cp_k.grad, cp_v.grad],
                [2, 2, 2, 2],
            )
            atol = (
                1e-08
                if backend == SDPBackend.EFFICIENT_ATTENTION
                else 1e-3 * self.world_size
            )
            self.assertTrue(torch.allclose(out, cp_out, atol=atol))

            atol = (
                2e-06
                if backend == SDPBackend.EFFICIENT_ATTENTION
                else 8e-3 * self.world_size
            )
            self.assertTrue(torch.allclose(q.grad, cp_dq, atol=atol))
            self.assertTrue(torch.allclose(k.grad, cp_dk, atol=atol))
            self.assertTrue(torch.allclose(v.grad, cp_dv, atol=atol))

            cp_q.grad = None
            cp_k.grad = None
            cp_v.grad = None
            cp_q.requires_grad = False
            cp_k.requires_grad = False
            cp_v.requires_grad = False

    def test_is_causal_behavior(self) -> None:
        _cp_options.enable_load_balance = False
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

        _cp_options.enable_load_balance = True
        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
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
    @parametrize("is_causal", [True, False])
    def test_ring_attention_native_transformer(self, is_causal: bool) -> None:
        _cp_options.enable_load_balance = is_causal
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
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
                "self_attn": _AttentionContextParallel(),
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
                c10d_functional.all_to_all_single: (self.world_size * 2 - 1)
                * num_layers,
            },
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def test_ring_attention_custom_transformer(self) -> None:
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
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
                f"layers.{i}.attention": _AttentionContextParallel()
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
                c10d_functional.all_to_all_single: (self.world_size * 2 - 1)
                * args.n_layers,
            },
        )


if backends:
    instantiate_parametrized_tests(RingAttentionTest)

if __name__ == "__main__":
    run_tests()
