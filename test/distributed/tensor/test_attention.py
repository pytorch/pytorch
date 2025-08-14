# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental._attention import (
    _AttentionContextParallel,
    _CausalBehavior,
    _cp_options,
    _DispatchMode,
    _is_causal_behavior,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, skipIfRocm
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
if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
    backends.append(SDPBackend.CUDNN_ATTENTION)

rotater_enum_to_str = {
    _RotateMethod.ALL_GATHER: "allgather",
    _RotateMethod.ALL_TO_ALL: "alltoall",
}  # mapping from _RotateMethod enum to string


class RingAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False

    @skip_if_lt_x_gpu(2)
    @skipIfRocm  # Missing _c10d_functional_autograd::all_to_all_single
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    @with_comms
    def test_ring_attention_sdpa(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "compiled": [True, False],
                "backend": backends,
                "load_balance": [True, False],
                "rotater": [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER],
                "test_forward_only": [True, False],
                "dispatch_mode": [
                    _DispatchMode.MONKEY_PATCH,
                    _DispatchMode.TORCH_FUNCTION,
                ],
            },
            self._test_ring_attention_sdpa,
        )

    def _test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
        test_forward_only: bool,
        dispatch_mode: _DispatchMode,
    ) -> None:
        torch.distributed.tensor.experimental._attention._dispatch_mode = dispatch_mode

        def fn_eval(fn, *args, **kwargs):
            if test_forward_only:
                with torch.no_grad():
                    return fn(*args, **kwargs)
            else:
                out = fn(*args, **kwargs)
                out.sum().backward()
                return out

        if load_balance and not is_causal:
            return

        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 64
        context_tokens = 64
        dim = 32
        nheads = 8
        torch.manual_seed(10)
        dtype = (
            torch.bfloat16
            if backend == SDPBackend.FLASH_ATTENTION
            or backend == SDPBackend.CUDNN_ATTENTION
            else torch.float32
        )

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
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q = q.detach().clone()
        cp_k = k.detach().clone()
        cp_v = v.detach().clone()
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

                    cp_out = fn_eval(fn, cp_q, cp_k, cp_v, is_causal=is_causal)

                    if not compiled and rotater == _RotateMethod.ALL_TO_ALL:
                        # Compiler and CommDebugMode do not work well together.
                        expect_all2all_count = (
                            self.world_size - 1
                            if test_forward_only
                            else self.world_size * 3 - 2
                        )
                        self.assertDictEqual(
                            comm_mode.get_comm_counts(),
                            {c10d_functional.all_to_all_single: expect_all2all_count},
                        )

            # Due to numerical error, we need to choose different atol for different
            # attention kernels
            (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])
            atol = (
                1e-08
                if backend == SDPBackend.EFFICIENT_ATTENTION
                else 1e-3 * self.world_size
            )
            self.assertTrue(torch.allclose(out, cp_out, atol=atol))

            if not test_forward_only:
                cp_dq, cp_dk, cp_dv = context_parallel_unshard(
                    device_mesh,
                    [cp_q.grad, cp_k.grad, cp_v.grad],
                    [2, 2, 2],
                )
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

        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.MONKEY_PATCH
        )

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
    def test_ring_attention_native_transformer(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "rotater": [_RotateMethod.ALL_GATHER, _RotateMethod.ALL_TO_ALL],
            },
            self._test_ring_attention_native_transformer,
        )

    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def _test_ring_attention_native_transformer(
        self, is_causal: bool, rotater: _RotateMethod
    ) -> None:
        _cp_options.enable_load_balance = is_causal
        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
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

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size - 1)
                    * num_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_gather_into_tensor: num_layers,
                },
            )

        with CommDebugMode() as comm_mode:
            out.sum().backward()

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size * 2 - 1)
                    * num_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_gather_into_tensor: num_layers,
                    c10d_functional.all_to_all_single: self.world_size * num_layers,
                },
            )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def test_ring_attention_custom_transformer(self) -> None:
        self.run_subtests(
            {"rotater": [_RotateMethod.ALL_GATHER, _RotateMethod.ALL_TO_ALL]},
            self._test_ring_attention_custom_transformer,
        )

    def _test_ring_attention_custom_transformer(self, rotater: _RotateMethod) -> None:
        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        # early init DTensor RNG tracker to avoid broadcast be captuured in comm_mode
        torch.distributed.tensor._random.manual_seed(10, device_mesh)

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

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size - 1)
                    * args.n_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {c10d_functional.all_gather_into_tensor: args.n_layers},
            )

        with CommDebugMode() as comm_mode:
            out.sum().backward()

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size * 2 - 1)
                    * args.n_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_gather_into_tensor: args.n_layers,
                    c10d_functional.all_to_all_single: self.world_size * args.n_layers,
                },
            )


# Compile the flex_attention function
compiled_flex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=True)
compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


class RingFlexAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_ring_flex_attention(self, qkv_size) -> None:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        torch.cuda.manual_seed(10)
        dtype = torch.float32
        bs = 8
        query_tokens = context_tokens = qkv_size
        dim = 32
        nheads = 8

        q = torch.rand(
            (bs, nheads, query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.rand(
            (bs, nheads, context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.rand(
            (bs, nheads, context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )

        block_mask = compiled_create_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=query_tokens,
            KV_LEN=context_tokens,
            device=self.device_type,
        )

        expect_out, expect_lse = compiled_flex_attention(
            q, k, v, block_mask=block_mask, return_lse=True
        )
        expect_out.sum().backward()

        # Prepare the required global vars for CP+Flex:
        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )
        # NOTE: cp needs to know the sharding dimension
        # TODO: see if this can be moved to the cp context
        from torch.distributed.tensor.experimental._attention import _set_cp_global_var

        _set_cp_global_var("cp_shard_dim", 2)
        self.assertEqual(
            torch.distributed.tensor.experimental._attention._cp_global_vars.cp_shard_dim,
            2,
        )

        # NOTE: we do not test load balance here
        _cp_options.enable_load_balance = False

        # set CP context dispatch mode to use TORCH_FUNCTION for flex_attention
        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.TORCH_FUNCTION
        )

        # prepare input buffer
        cp_q = q.detach().clone()
        cp_k = k.detach().clone()
        cp_v = v.detach().clone()

        # create block_mask for CP
        from torch.distributed.tensor.experimental._attention import (
            create_cp_block_mask,
        )

        # NOTE: call create_block_mask() within TorchFunctionMode would cause error in create_fw_bw_graph
        cp_block_mask = create_cp_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=query_tokens,
            KV_LEN=context_tokens,
            device_mesh=device_mesh,
        )

        # shard qkv on seq_dim
        shard_dim = 2

        with context_parallel(
            device_mesh,
            buffers=[cp_q, cp_k, cp_v],
            buffer_seq_dims=[shard_dim] * 3,
        ):
            cp_q.requires_grad = True
            cp_k.requires_grad = True
            cp_v.requires_grad = True

            cp_out, cp_lse = compiled_flex_attention(
                cp_q,
                cp_k,
                cp_v,
                block_mask=cp_block_mask,
                return_lse=True,
            )

            # check block_mask rewrite doesn't escape to the outside
            assert cp_block_mask.seq_lengths == (
                cp_q.size(dim=shard_dim),
                cp_k.size(dim=shard_dim),
            )

            # backward run
            cp_out.sum().backward()

            cp_q.requires_grad = False
            cp_k.requires_grad = False
            cp_v.requires_grad = False

        # unshard the output
        cp_out, cp_lse = context_parallel_unshard(device_mesh, [cp_out, cp_lse], [2, 2])
        torch.testing.assert_close(cp_out, expect_out, atol=1e-6, rtol=1e-2)
        torch.testing.assert_close(cp_lse, expect_lse, atol=1e-6, rtol=1e-2)

        # unshard the gradient
        cp_q_grad, cp_k_grad, cp_v_grad = context_parallel_unshard(
            device_mesh,
            [cp_q.grad, cp_k.grad, cp_v.grad],
            [2, 2, 2],
        )
        torch.testing.assert_close(cp_q_grad, q.grad, atol=1e-6, rtol=1e-2)
        torch.testing.assert_close(cp_k_grad, k.grad, atol=1e-6, rtol=1e-2)
        torch.testing.assert_close(cp_v_grad, v.grad, atol=1e-6, rtol=1e-2)

        # reset CP context dispatch mode to default
        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.MONKEY_PATCH
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_ring_flex_attention(self) -> None:
        self.run_subtests(
            {"qkv_size": [128 * self.world_size, 2048]},
            self._test_ring_flex_attention,
        )

        # NOTE: Context Parallel should not be used for small attentions (block_size < 128)
        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
            self.run_subtests(
                {"qkv_size": [64 * self.world_size]},
                self._test_ring_flex_attention,
            )


if __name__ == "__main__":
    run_tests()
