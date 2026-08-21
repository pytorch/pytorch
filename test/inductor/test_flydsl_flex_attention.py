# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    flex_flydsl_forward_template,
    is_causal_mask_graph,
    maybe_append_flydsl_flex_attention_choice,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V


class _FakeNode:
    def __init__(self, size, stride, dtype=torch.bfloat16, numel=None):
        self._size = list(size)
        self._stride = list(stride)
        self._dtype = dtype
        self._numel = int(torch.tensor(size).prod().item()) if numel is None else numel

    def get_size(self):
        return self._size

    def get_stride(self):
        return self._stride

    def get_dtype(self):
        return self._dtype

    def get_device(self):
        return torch.device("cuda", 0)

    def get_numel(self):
        return self._numel


def _contiguous_stride(size):
    stride = [1] * len(size)
    for index in range(len(size) - 2, -1, -1):
        stride[index] = stride[index + 1] * size[index + 1]
    return stride


def _fake_graph():
    return SimpleNamespace(
        sizevars=SimpleNamespace(
            guard_int=lambda value: int(value),
            shape_env=None,
        )
    )


def _has_gfx950_flydsl():
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(
            torch.cuda.get_device_properties(0),
            "gcnArchName",
            "",
        ).split(":", 1)[0]
        == "gfx950"
        and flydsl_utils.runtime_available()
    )


class TestFlyDSLFlexAttention(TestCase):
    def test_recognizes_standard_causal_mask(self):
        causal = torch.fx.symbolic_trace(lambda b, h, q, kv: q >= kv)
        reverse = torch.fx.symbolic_trace(lambda b, h, q, kv: kv <= q)
        bottom_right = torch.fx.symbolic_trace(lambda b, h, q, kv: q + 8188 >= kv)
        noncausal = torch.fx.symbolic_trace(lambda b, h, q, kv: q == kv)

        self.assertTrue(is_causal_mask_graph(causal))
        self.assertTrue(is_causal_mask_graph(reverse))
        self.assertTrue(is_causal_mask_graph(bottom_right, 8188))
        self.assertFalse(is_causal_mask_graph(causal, 8188))
        self.assertFalse(is_causal_mask_graph(noncausal))

    def test_appends_supported_bf16_gqa_choice(self):
        q_size = [1, 64, 512, 128]
        kv_size = [1, 4, 1024, 128]
        mask_counts_size = [1, 1, 4]
        mask_indices_size = [1, 1, 4, 8]

        query = _FakeNode(q_size, _contiguous_stride(q_size))
        key = _FakeNode(kv_size, _contiguous_stride(kv_size))
        value = _FakeNode(kv_size, _contiguous_stride(kv_size))
        stats_size = [1, 64, 512]
        lse = _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        )
        max_scores = _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        )
        counts = _FakeNode(
            mask_counts_size,
            _contiguous_stride(mask_counts_size),
            torch.int32,
        )
        indices = _FakeNode(
            mask_indices_size,
            _contiguous_stride(mask_indices_size),
            torch.int32,
        )
        full_counts = _FakeNode(
            mask_counts_size,
            _contiguous_stride(mask_counts_size),
            torch.int32,
        )
        full_indices = _FakeNode(
            mask_indices_size,
            _contiguous_stride(mask_indices_size),
            torch.int32,
        )

        score_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        )
        mask_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda b, h, q, kv: q + 512 >= kv)
        )
        graph = _fake_graph()
        choices = []

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                return_value=True,
            ),
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            maybe_append_flydsl_flex_attention_choice(
                choices,
                query=query,
                key=key,
                value=value,
                logsumexp=lse,
                max_scores=max_scores,
                kv_num_blocks=counts,
                kv_indices=indices,
                full_kv_num_blocks=full_counts,
                full_kv_indices=full_indices,
                layout=mock.Mock(),
                subgraph=score_graph,
                mask_graph=mask_graph,
                score_mod_other_buffers=[],
                mask_mod_other_buffers=[],
                scale=128**-0.5,
                sparse_q_block_size=128,
                sparse_kv_block_size=128,
            )

        append.assert_called_once()
        kwargs = append.call_args.kwargs
        self.assertEqual(kwargs["NUM_Q_HEADS"], 64)
        self.assertEqual(kwargs["NUM_KV_HEADS"], 4)
        self.assertEqual(kwargs["QK_HEAD_DIM"], 128)
        self.assertEqual(kwargs["V_HEAD_DIM"], 128)
        self.assertTrue(kwargs["CAUSAL_PARTIAL_BLOCKS"])
        self.assertEqual(kwargs["SPARSE_Q_BLOCK_SIZE"], 128)
        self.assertEqual(kwargs["SPARSE_KV_BLOCK_SIZE"], 128)

    def test_appends_dsv2_qk192_v128_choice(self):
        q_size = [1, 16, 256, 192]
        k_size = [1, 16, 256, 192]
        v_size = [1, 16, 256, 128]
        stats_size = [1, 16, 256]
        counts_size = [1, 1, 2]
        indices_size = [1, 1, 2, 1]
        query = _FakeNode(q_size, _contiguous_stride(q_size))
        key = _FakeNode(k_size, _contiguous_stride(k_size))
        value = _FakeNode(v_size, _contiguous_stride(v_size))
        lse = _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        )
        max_scores = _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        )
        counts = _FakeNode(
            counts_size,
            _contiguous_stride(counts_size),
            torch.int32,
        )
        indices = _FakeNode(
            indices_size,
            _contiguous_stride(indices_size),
            torch.int32,
        )
        score_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        )
        mask_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda b, h, q, kv: q >= kv)
        )
        graph = _fake_graph()

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                return_value=True,
            ),
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            maybe_append_flydsl_flex_attention_choice(
                [],
                query=query,
                key=key,
                value=value,
                logsumexp=lse,
                max_scores=max_scores,
                kv_num_blocks=counts,
                kv_indices=indices,
                full_kv_num_blocks=counts,
                full_kv_indices=indices,
                layout=mock.Mock(),
                subgraph=score_graph,
                mask_graph=mask_graph,
                score_mod_other_buffers=[],
                mask_mod_other_buffers=[],
                scale=192**-0.5,
                sparse_q_block_size=128,
                sparse_kv_block_size=128,
            )

        append.assert_called_once()
        kwargs = append.call_args.kwargs
        self.assertEqual(kwargs["NUM_Q_HEADS"], 16)
        self.assertEqual(kwargs["NUM_KV_HEADS"], 16)
        self.assertEqual(kwargs["QK_HEAD_DIM"], 192)
        self.assertEqual(kwargs["V_HEAD_DIM"], 128)

    def test_unsupported_q_block_falls_back(self):
        q_size = [1, 8, 256, 128]
        kv_size = [1, 2, 256, 128]
        counts_size = [1, 1, 1]
        indices_size = [1, 1, 1, 2]
        query = _FakeNode(q_size, _contiguous_stride(q_size))
        key = _FakeNode(kv_size, _contiguous_stride(kv_size))
        value = _FakeNode(kv_size, _contiguous_stride(kv_size))
        lse_size = [1, 8, 256]
        lse = _FakeNode(
            lse_size,
            _contiguous_stride(lse_size),
            torch.float32,
        )
        max_scores = _FakeNode(
            lse_size,
            _contiguous_stride(lse_size),
            torch.float32,
        )
        counts = _FakeNode(
            counts_size,
            _contiguous_stride(counts_size),
            torch.int32,
        )
        indices = _FakeNode(
            indices_size,
            _contiguous_stride(indices_size),
            torch.int32,
        )
        score_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        )
        mask_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda b, h, q, kv: q >= kv)
        )
        graph = _fake_graph()

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                return_value=True,
            ),
            mock.patch.object(
                flex_flydsl_forward_template,
                "maybe_append_choice",
            ) as append,
        ):
            maybe_append_flydsl_flex_attention_choice(
                [],
                query=query,
                key=key,
                value=value,
                logsumexp=lse,
                max_scores=max_scores,
                kv_num_blocks=counts,
                kv_indices=indices,
                full_kv_num_blocks=counts,
                full_kv_indices=indices,
                layout=mock.Mock(),
                subgraph=score_graph,
                mask_graph=mask_graph,
                score_mod_other_buffers=[],
                mask_mod_other_buffers=[],
                scale=128**-0.5,
                sparse_q_block_size=256,
                sparse_kv_block_size=128,
            )

        append.assert_not_called()

    def test_appends_supported_gqa_decode_choice(self):
        q_size = [32, 64, 4, 128]
        kv_size = [32, 4, 8192, 128]
        stats_size = [32, 64, 4]
        counts_size = [1, 4, 1]
        indices_size = [1, 4, 1, 16]
        query = _FakeNode(q_size, _contiguous_stride(q_size))
        key = _FakeNode(kv_size, _contiguous_stride(kv_size))
        value = _FakeNode(kv_size, _contiguous_stride(kv_size))
        lse = _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        )
        max_scores = _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        )
        counts = _FakeNode(
            counts_size,
            _contiguous_stride(counts_size),
            torch.int32,
        )
        indices = _FakeNode(
            indices_size,
            _contiguous_stride(indices_size),
            torch.int32,
        )
        score_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        )
        mask_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda b, h, q, kv: q + 8188 >= kv)
        )
        graph = _fake_graph()

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                return_value=True,
            ),
            mock.patch.object(
                flex_flydsl_forward_template,
                "maybe_append_choice",
            ) as append,
        ):
            maybe_append_flydsl_flex_attention_choice(
                [],
                query=query,
                key=key,
                value=value,
                logsumexp=lse,
                max_scores=max_scores,
                kv_num_blocks=counts,
                kv_indices=indices,
                full_kv_num_blocks=counts,
                full_kv_indices=indices,
                layout=mock.Mock(),
                subgraph=score_graph,
                mask_graph=mask_graph,
                score_mod_other_buffers=[],
                mask_mod_other_buffers=[],
                scale=128**-0.5,
                sparse_q_block_size=128,
                sparse_kv_block_size=128,
            )

        append.assert_called_once()
        kwargs = append.call_args.kwargs
        self.assertEqual(kwargs["BATCH_SIZE"], 32)
        self.assertEqual(kwargs["NUM_Q_HEADS"], 64)
        self.assertEqual(kwargs["NUM_KV_HEADS"], 4)
        self.assertEqual(kwargs["SEQ_Q"], 4)
        self.assertEqual(kwargs["SEQ_KV"], 8192)
        self.assertEqual(kwargs["BLOCK_MASK_HEADS"], 4)

    def test_four_gib_kv_buffer_falls_back(self):
        q_size = [64, 64, 4, 128]
        kv_size = [64, 4, 65536, 128]
        query = _FakeNode(q_size, _contiguous_stride(q_size))
        key = _FakeNode(kv_size, _contiguous_stride(kv_size))
        value = _FakeNode(kv_size, _contiguous_stride(kv_size))
        score_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        )
        mask_graph = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda b, h, q, kv: q + 65532 >= kv)
        )
        graph = _fake_graph()

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                return_value=True,
            ),
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            maybe_append_flydsl_flex_attention_choice(
                [],
                query=query,
                key=key,
                value=value,
                logsumexp=mock.Mock(),
                max_scores=mock.Mock(),
                kv_num_blocks=mock.Mock(),
                kv_indices=mock.Mock(),
                full_kv_num_blocks=mock.Mock(),
                full_kv_indices=mock.Mock(),
                layout=mock.Mock(),
                subgraph=score_graph,
                mask_graph=mask_graph,
                score_mod_other_buffers=[],
                mask_mod_other_buffers=[],
                scale=128**-0.5,
                sparse_q_block_size=128,
                sparse_kv_block_size=128,
            )

        append.assert_not_called()

    def test_unsupported_dtype_falls_back(self):
        q_size = [1, 8, 256, 128]
        query = _FakeNode(q_size, _contiguous_stride(q_size), dtype=torch.float16)
        choices = []
        with (
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                return_value=True,
            ),
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            maybe_append_flydsl_flex_attention_choice(
                choices,
                query=query,
                key=query,
                value=query,
                logsumexp=mock.Mock(),
                max_scores=mock.Mock(),
                kv_num_blocks=mock.Mock(),
                kv_indices=mock.Mock(),
                full_kv_num_blocks=mock.Mock(),
                full_kv_indices=mock.Mock(),
                layout=mock.Mock(),
                subgraph=mock.Mock(),
                mask_graph=mock.Mock(),
                score_mod_other_buffers=[],
                mask_mod_other_buffers=[],
                scale=1.0,
                sparse_q_block_size=256,
                sparse_kv_block_size=128,
            )

        append.assert_not_called()
        self.assertEqual(choices, [])

    def test_unsupported_mask_bytecode_operation_falls_back(self):
        mask_graph = torch.fx.symbolic_trace(lambda b, h, q, kv: torch.abs(q - kv) < 96)

        program, reason = lower_flydsl_mask_graph(mask_graph, [])

        self.assertIsNone(program)
        self.assertIn("unsupported", reason)

    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(self):
        if not (
            torch.cuda.is_available()
            and torch.version.hip is not None
            and getattr(
                torch.cuda.get_device_properties(0),
                "gcnArchName",
                "",
            ).split(":", 1)[0]
            == "gfx950"
            and flydsl_utils.runtime_available()
        ):
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_fwd_gfx950 import (
            build_flex_attn_fwd_module,
        )

        import flydsl.compiler as flyc

        batch, q_heads, kv_heads, seq, head_dim = 1, 4, 2, 512, 128
        scale = head_dim**-0.5
        torch.manual_seed(0)
        query = torch.randn(
            batch,
            q_heads,
            seq,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            batch,
            kv_heads,
            seq,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key)
        output = torch.empty_like(query)
        lse = torch.empty(
            batch,
            q_heads,
            seq,
            device="cuda",
            dtype=torch.float32,
        )
        max_scores = torch.empty_like(lse)

        kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 0]]],
            device="cuda",
            dtype=torch.int32,
        )
        kv_indices = torch.tensor(
            [[[[0], [1], [2], [0]]]],
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1, 1, 0]]],
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_indices = torch.tensor(
            [[[[0], [0], [0], [0]]]],
            device="cuda",
            dtype=torch.int32,
        )

        launcher = build_flex_attn_fwd_module(
            batch_size=batch,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            seq_q=seq,
            seq_kv=seq,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            block_mask_batch=1,
            block_mask_heads=1,
            num_q_blocks=4,
            max_partial_blocks=1,
            max_full_blocks=1,
            sparse_q_block_size=128,
            sparse_kv_block_size=128,
            causal_partial_blocks=True,
            scale=scale,
        )
        tensors = (
            query,
            key,
            value,
            lse,
            max_scores,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            output,
        )
        args = tuple(
            flyc.from_torch_tensor(tensor).mark_layout_dynamic() for tensor in tensors
        ) + (torch.cuda.default_stream(),)
        compiled = flyc.compile(launcher, *args)
        compiled(*args)
        torch.cuda.synchronize()

        allowed = torch.zeros(seq, seq, device="cuda", dtype=torch.bool)
        partial_blocks = [0, 1, 2, None]
        full_blocks = [[], [0], [0], []]
        positions = torch.arange(seq, device="cuda")
        causal = positions[:, None] >= positions[None, :]
        for q_block in range(4):
            q_begin = q_block * 128
            q_end = q_begin + 128
            for kv_block in full_blocks[q_block]:
                k_begin = kv_block * 128
                allowed[q_begin:q_end, k_begin : k_begin + 128] = True
            partial_block = partial_blocks[q_block]
            if partial_block is not None:
                k_begin = partial_block * 128
                allowed[
                    q_begin:q_end,
                    k_begin : k_begin + 128,
                ] = causal[
                    q_begin:q_end,
                    k_begin : k_begin + 128,
                ]

        repeated_key = key.repeat_interleave(q_heads // kv_heads, dim=1).float()
        repeated_value = value.repeat_interleave(q_heads // kv_heads, dim=1).float()
        scores = torch.matmul(query.float(), repeated_key.transpose(-2, -1)) * scale
        masked_scores = scores.masked_fill(~allowed, float("-inf"))
        row_has_values = allowed.any(dim=-1)
        reference_lse = torch.logsumexp(masked_scores, dim=-1)
        reference_max = masked_scores.amax(dim=-1)
        safe_lse = torch.where(
            row_has_values,
            reference_lse[:, :, :],
            torch.zeros_like(reference_lse),
        )
        probabilities = torch.where(
            allowed,
            torch.exp(masked_scores - safe_lse.unsqueeze(-1)),
            torch.zeros_like(masked_scores),
        )
        reference_output = torch.matmul(probabilities, repeated_value).to(
            torch.bfloat16
        )

        torch.testing.assert_close(output, reference_output, atol=0.08, rtol=0.02)
        torch.testing.assert_close(lse, reference_lse, atol=0.03, rtol=0.01)
        torch.testing.assert_close(max_scores, reference_max, atol=0.03, rtol=0.01)
        self.assertEqual(output[:, :, 384:].abs().max().item(), 0.0)
        self.assertTrue(torch.isneginf(lse[:, :, 384:]).all())
        self.assertTrue(torch.isneginf(max_scores[:, :, 384:]).all())

    def test_gfx950_public_api_minimax_per_kv_head_decode(self):
        if not (
            torch.cuda.is_available()
            and torch.version.hip is not None
            and getattr(
                torch.cuda.get_device_properties(0),
                "gcnArchName",
                "",
            ).split(":", 1)[0]
            == "gfx950"
            and flydsl_utils.runtime_available()
        ):
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            AuxRequest,
            BlockMask,
            flex_attention,
        )

        batch, q_heads, kv_heads = 1, 64, 4
        seq_q, seq_kv, head_dim = 4, 8192, 128
        scale = head_dim**-0.5
        torch.manual_seed(0)
        query = torch.randn(
            batch,
            q_heads,
            seq_q,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn(
            batch,
            kv_heads,
            seq_kv,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.randn_like(key)
        kv_num_blocks = torch.ones(1, kv_heads, 1, device="cuda", dtype=torch.int32)
        kv_indices = torch.zeros(1, kv_heads, 1, 16, device="cuda", dtype=torch.int32)
        kv_indices[..., 0] = 63
        full_kv_num_blocks = torch.full(
            (1, kv_heads, 1), 15, device="cuda", dtype=torch.int32
        )
        full_kv_indices = torch.zeros(
            1, kv_heads, 1, 16, device="cuda", dtype=torch.int32
        )
        for kv_head in range(kv_heads):
            full_kv_indices[0, kv_head, 0, :15] = torch.arange(
                kv_head * 8,
                kv_head * 8 + 15,
                device="cuda",
                dtype=torch.int32,
            )

        q_offset = seq_kv - seq_q

        def bottom_right_causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx + q_offset >= kv_idx

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=bottom_right_causal,
            seq_lengths=(seq_q, seq_kv),
            compute_q_blocks=False,
        )

        def run_flydsl(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=True,
                return_aux=AuxRequest(lse=True, max_scores=True),
                kernel_options={"BACKEND": "FLYDSL"},
            )

        (output, aux), code = run_and_get_code(
            torch.compile(run_flydsl, fullgraph=True),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()
        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))

        group_size = q_heads // kv_heads
        reference_block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks.repeat_interleave(group_size, dim=1),
            kv_indices.repeat_interleave(group_size, dim=1),
            full_kv_num_blocks.repeat_interleave(group_size, dim=1),
            full_kv_indices.repeat_interleave(group_size, dim=1),
            BLOCK_SIZE=128,
            mask_mod=bottom_right_causal,
            seq_lengths=(seq_q, seq_kv),
            compute_q_blocks=False,
        )
        (reference, reference_aux), reference_code = run_and_get_code(
            torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    block_mask=reference_block_mask,
                    scale=scale,
                    enable_gqa=True,
                    return_aux=AuxRequest(lse=True, max_scores=True),
                    kernel_options={"BACKEND": "TRITON"},
                ),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(reference_code))

        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)
        torch.testing.assert_close(aux.lse, reference_aux.lse, atol=0.03, rtol=0.01)
        torch.testing.assert_close(
            aux.max_scores,
            reference_aux.max_scores,
            atol=0.03,
            rtol=0.01,
        )

    def test_gfx950_public_api_transposed_document_qk192_v128(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            AuxRequest,
            create_block_mask,
            flex_attention,
        )

        batch, heads, seq = 2, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        scale = 0.07
        torch.manual_seed(2)
        query = torch.randn(
            batch,
            seq,
            heads,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        key = torch.randn_like(query)
        value = torch.randn(
            batch,
            seq,
            heads,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        document_end = torch.tensor(
            [
                [127] * 128 + [255] * 128,
                [63] * 64 + [191] * 128 + [255] * 64,
            ],
            device="cuda",
            dtype=torch.int32,
        )

        def document_causal(b, h, q_idx, kv_idx):
            del h
            return (q_idx >= kv_idx) & (q_idx <= document_end[b, kv_idx])

        block_mask = create_block_mask(
            document_causal,
            batch,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )

        def run(q, k, v, backend):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                return_aux=AuxRequest(lse=True, max_scores=True),
                kernel_options={"BACKEND": backend},
            )

        (output, aux), code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "FLYDSL"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        (reference, reference_aux), reference_code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "TRITON"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()

        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(reference_code))
        self.assertEqual(output.shape, (batch, heads, seq, v_head_dim))
        self.assertEqual(output.stride()[-1], 1)
        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)
        torch.testing.assert_close(aux.lse, reference_aux.lse, atol=0.03, rtol=0.01)
        torch.testing.assert_close(
            aux.max_scores,
            reference_aux.max_scores,
            atol=0.03,
            rtol=0.01,
        )

    def test_gfx950_public_api_sliding_window_mask(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        batch, heads, seq, head_dim = 1, 2, 256, 128
        torch.manual_seed(3)
        query, key, value = (
            torch.randn(
                batch,
                heads,
                seq,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )

        def sliding_window(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (q_idx - kv_idx < 96)

        block_mask = create_block_mask(
            sliding_window,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )

        def run(q, k, v, backend):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": backend},
            )

        output, code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "FLYDSL"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        reference, _ = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "TRITON"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()

        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        self.assertFalse(torch.isnan(output).any())
        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)

    def test_gfx950_public_api_b1_document_two_captures(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        batch, heads, seq, head_dim = 1, 2, 256, 128
        torch.manual_seed(4)
        query, key, value = (
            torch.randn(
                batch,
                heads,
                seq,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )
        document_ids = torch.arange(seq, device="cuda", dtype=torch.int32) // 128
        document_starts = torch.tensor(
            [0, 128],
            device="cuda",
            dtype=torch.int32,
        )

        def document_causal(b, h, q_idx, kv_idx):
            del b, h
            return (kv_idx >= document_starts[document_ids[q_idx]]) & (kv_idx <= q_idx)

        block_mask = create_block_mask(
            document_causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )

        def run(q, k, v, backend):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": backend},
            )

        output, code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "FLYDSL"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        reference, _ = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "TRITON"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()

        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)

    def test_gfx950_public_api_standard_width_create_block_mask(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        batch, heads, seq, head_dim = 1, 1, 4096, 128
        torch.manual_seed(5)
        query, key, value = (
            torch.randn(
                batch,
                heads,
                seq,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self.assertEqual(block_mask.kv_indices.shape[-1], seq // 128)
        self.assertEqual(block_mask.full_kv_indices.shape[-1], seq // 128)

        def run(q, k, v, backend):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": backend},
            )

        output, code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "FLYDSL"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        reference, _ = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "TRITON"),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()

        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)

    def test_gfx950_auto_keeps_flydsl_opt_in(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        batch, heads, seq, head_dim = 1, 2, 256, 128
        torch.manual_seed(6)
        query, key, value = (
            torch.randn(
                batch,
                heads,
                seq,
                head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(3)
        )

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )

        def run(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "AUTO"},
            )

        torch._dynamo.reset()
        output, code = run_and_get_code(
            torch.compile(run, fullgraph=True),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()

        self.assertFalse(torch.isnan(output).any())
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(code))

    def test_gfx950_public_api_dsv2_qk192_v128_prefill(self):
        if not (
            torch.cuda.is_available()
            and torch.version.hip is not None
            and getattr(
                torch.cuda.get_device_properties(0),
                "gcnArchName",
                "",
            ).split(":", 1)[0]
            == "gfx950"
            and flydsl_utils.runtime_available()
        ):
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        from torch.nn.attention.flex_attention import (
            AuxRequest,
            BlockMask,
            flex_attention,
        )

        batch, heads, seq = 1, 16, 256
        qk_head_dim, v_head_dim = 192, 128
        scale = qk_head_dim**-0.5
        torch.manual_seed(1)
        query = torch.randn(
            batch,
            heads,
            seq,
            qk_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.randn_like(query)
        value = torch.randn(
            batch,
            heads,
            seq,
            v_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        kv_num_blocks = torch.ones(1, 1, 2, device="cuda", dtype=torch.int32)
        kv_indices = torch.tensor(
            [[[[0], [1]]]],
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1]]],
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_indices = torch.tensor(
            [[[[0], [0]]]],
            device="cuda",
            dtype=torch.int32,
        )

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=causal,
            seq_lengths=(seq, seq),
            compute_q_blocks=False,
        )

        def run_flydsl(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                return_aux=AuxRequest(lse=True, max_scores=True),
                kernel_options={"BACKEND": "FLYDSL"},
            )

        (output, aux), code = run_and_get_code(
            torch.compile(run_flydsl, fullgraph=True),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()
        self.assertEqual(output.shape, (batch, heads, seq, v_head_dim))
        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))

        (reference, reference_aux), reference_code = run_and_get_code(
            torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    scale=scale,
                    return_aux=AuxRequest(lse=True, max_scores=True),
                    kernel_options={"BACKEND": "TRITON"},
                ),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(reference_code))

        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)
        torch.testing.assert_close(aux.lse, reference_aux.lse, atol=0.03, rtol=0.01)
        torch.testing.assert_close(
            aux.max_scores,
            reference_aux.max_scores,
            atol=0.03,
            rtol=0.01,
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
