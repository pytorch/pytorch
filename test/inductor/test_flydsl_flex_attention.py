# Owner(s): ["module: inductor"]

import operator
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    can_use_flydsl_flex_attention_forward,
    flex_flydsl_forward_template,
    is_causal_mask_graph,
    maybe_append_flydsl_flex_attention_choice,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)


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


def _supported_fake_forward_inputs(
    *,
    q_size=(1, 64, 512, 128),
    k_size=(1, 4, 1024, 128),
    v_size=None,
    mask_heads=1,
    index_width=8,
    dtype=torch.bfloat16,
    mask_fn=lambda b, h, q, kv: q + 512 >= kv,
):
    v_size = k_size if v_size is None else v_size
    stats_size = q_size[:3]
    mask_counts_size = [1, mask_heads, max(1, q_size[2] // 128)]
    mask_indices_size = [*mask_counts_size, index_width]
    return {
        "query": _FakeNode(q_size, _contiguous_stride(q_size), dtype),
        "key": _FakeNode(k_size, _contiguous_stride(k_size), dtype),
        "value": _FakeNode(v_size, _contiguous_stride(v_size), dtype),
        "logsumexp": _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        ),
        "max_scores": _FakeNode(
            stats_size,
            _contiguous_stride(stats_size),
            torch.float32,
        ),
        "kv_num_blocks": _FakeNode(
            mask_counts_size,
            _contiguous_stride(mask_counts_size),
            torch.int32,
        ),
        "kv_indices": _FakeNode(
            mask_indices_size,
            _contiguous_stride(mask_indices_size),
            torch.int32,
        ),
        "full_kv_num_blocks": _FakeNode(
            mask_counts_size,
            _contiguous_stride(mask_counts_size),
            torch.int32,
        ),
        "full_kv_indices": _FakeNode(
            mask_indices_size,
            _contiguous_stride(mask_indices_size),
            torch.int32,
        ),
        "subgraph": SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        ),
        "mask_graph": SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(mask_fn)
        ),
        "score_mod_other_buffers": [],
        "mask_mod_other_buffers": [],
        "scale": q_size[-1] ** -0.5,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
    }


def _mask_graph(fn):
    indices = tuple(torch.tensor(0, dtype=torch.int32) for _ in range(4))
    return make_fx(fn)(*indices)


def _evaluate_mask_program(program, b, h, q, kv):
    values = [b, h, q, kv]
    for instruction in program.instructions:
        op = instruction[0]
        if op in ("const_i32", "const_bool"):
            values.append(instruction[1])
            continue
        lhs = values[instruction[1]]
        rhs = values[instruction[2]]
        values.append(
            {
                "add": operator.add,
                "sub": operator.sub,
                "mul": operator.mul,
                "ge": operator.ge,
            }[op](lhs, rhs)
        )
    return values[program.output]


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


def _append_fake_choice(inputs):
    choices = []
    with (
        V.set_graph_handler(_fake_graph()),
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
            layout=mock.Mock(),
            **inputs,
        )
    return choices, append


def _make_qkv(
    *,
    batch=1,
    q_heads=2,
    kv_heads=None,
    seq_q=256,
    seq_kv=None,
    qk_dim=128,
    v_dim=128,
    seed=0,
    transposed=False,
):
    kv_heads = q_heads if kv_heads is None else kv_heads
    seq_kv = seq_q if seq_kv is None else seq_kv
    torch.manual_seed(seed)

    def make(heads, sequence, dimension):
        if not transposed:
            return torch.randn(
                batch,
                heads,
                sequence,
                dimension,
                device="cuda",
                dtype=torch.bfloat16,
            )
        return torch.randn(
            batch,
            sequence,
            heads,
            dimension,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)

    return (
        make(q_heads, seq_q, qk_dim),
        make(kv_heads, seq_kv, qk_dim),
        make(kv_heads, seq_kv, v_dim),
    )


class TestFlyDSLFlexAttention(TestCase):
    def _compare_forward(
        self,
        query,
        key,
        value,
        *,
        block_mask=None,
        reference_block_mask=None,
        scale=None,
        enable_gqa=False,
        return_aux=False,
    ):
        def run(q, k, v, backend, selected_mask):
            kwargs = {
                "block_mask": selected_mask,
                "enable_gqa": enable_gqa,
                "kernel_options": {"BACKEND": backend},
            }
            if scale is not None:
                kwargs["scale"] = scale
            if return_aux:
                kwargs["return_aux"] = AuxRequest(lse=True, max_scores=True)
            return flex_attention(q, k, v, **kwargs)

        actual, code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(q, k, v, "FLYDSL", block_mask),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        expected, reference_code = run_and_get_code(
            torch.compile(
                lambda q, k, v: run(
                    q,
                    k,
                    v,
                    "TRITON",
                    (
                        reference_block_mask
                        if reference_block_mask is not None
                        else block_mask
                    ),
                ),
                fullgraph=True,
            ),
            query,
            key,
            value,
        )
        torch.cuda.synchronize()
        self.assertIn("build_flex_attn_fwd_module", "\n".join(code))
        self.assertNotIn("build_flex_attn_fwd_module", "\n".join(reference_code))

        if not return_aux:
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.02)
            return actual

        output, aux = actual
        reference, reference_aux = expected
        torch.testing.assert_close(output, reference, atol=0.08, rtol=0.02)
        torch.testing.assert_close(aux.lse, reference_aux.lse, atol=0.03, rtol=0.01)
        torch.testing.assert_close(
            aux.max_scores,
            reference_aux.max_scores,
            atol=0.03,
            rtol=0.01,
        )
        return output, aux

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

    def test_fractional_offset_is_not_causal(self):
        graph_module = torch.fx.symbolic_trace(lambda b, h, q, kv: q + -0.5 >= kv)

        self.assertFalse(is_causal_mask_graph(graph_module))
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)

    def test_add_alpha_is_not_misclassified_as_causal(self):
        graph_module = _mask_graph(lambda b, h, q, kv: torch.add(q, 64, alpha=2) >= kv)

        self.assertFalse(is_causal_mask_graph(graph_module, 64))
        self.assertFalse(is_causal_mask_graph(graph_module, 128))

    def test_mask_lowering_preserves_add_and_sub_alpha(self):
        cases = (
            (lambda b, h, q, kv: torch.add(q, 64, alpha=2) >= kv, 0, 100, True),
            (lambda b, h, q, kv: torch.sub(q, 64, alpha=2) >= kv, 200, 100, False),
        )
        for mask_mod, q, kv, expected in cases:
            with self.subTest(mask_mod=mask_mod):
                program, reason = lower_flydsl_mask_graph(
                    _mask_graph(mask_mod),
                    (),
                )
                self.assertIsNotNone(program, reason)
                self.assertEqual(
                    _evaluate_mask_program(program, 0, 0, q, kv),
                    expected,
                )

    def test_appends_supported_bf16_gqa_choice(self):
        _, append = _append_fake_choice(_supported_fake_forward_inputs())
        append.assert_called_once()
        kwargs = append.call_args.kwargs
        self.assertEqual(
            (
                kwargs["NUM_Q_HEADS"],
                kwargs["NUM_KV_HEADS"],
                kwargs["QK_HEAD_DIM"],
                kwargs["V_HEAD_DIM"],
            ),
            (64, 4, 128, 128),
        )
        self.assertTrue(kwargs["CAUSAL_PARTIAL_BLOCKS"])
        self.assertEqual(kwargs["SPARSE_Q_BLOCK_SIZE"], 128)
        self.assertEqual(kwargs["SPARSE_KV_BLOCK_SIZE"], 128)

    def test_rejects_invalid_block_mask_metadata(self):
        cases = (
            (
                "full_kv_num_blocks",
                _FakeNode([1, 1, 3], [3, 3, 1], torch.int32),
                "count dimensions",
            ),
            (
                "kv_indices",
                _FakeNode([1, 1, 4, 8], [64, 64, 16, 1], torch.int32),
                "contiguous",
            ),
            (
                "kv_num_blocks",
                _FakeNode([1, 1, 4], [4, 4, 1], torch.float32),
                "int32",
            ),
        )
        for name, invalid_node, expected_reason in cases:
            with self.subTest(name=name):
                inputs = _supported_fake_forward_inputs()
                inputs[name] = invalid_node
                inputs.pop("logsumexp")
                inputs.pop("max_scores")
                with (
                    V.set_graph_handler(_fake_graph()),
                    mock.patch(
                        "torch._inductor.kernel.flex.flex_flydsl_attention._is_gfx950_device",
                        return_value=True,
                    ),
                ):
                    can_use, reason = can_use_flydsl_flex_attention_forward(**inputs)
                self.assertFalse(can_use)
                self.assertIn(expected_reason, reason)

    def test_appends_qk192_v128_choice(self):
        _, append = _append_fake_choice(
            _supported_fake_forward_inputs(
                q_size=(1, 16, 256, 192),
                k_size=(1, 16, 256, 192),
                v_size=(1, 16, 256, 128),
                index_width=1,
                mask_fn=lambda b, h, q, kv: q >= kv,
            )
        )
        append.assert_called_once()
        kwargs = append.call_args.kwargs
        self.assertEqual(
            (
                kwargs["NUM_Q_HEADS"],
                kwargs["NUM_KV_HEADS"],
                kwargs["QK_HEAD_DIM"],
                kwargs["V_HEAD_DIM"],
            ),
            (16, 16, 192, 128),
        )

    def test_unsupported_q_block_falls_back(self):
        inputs = _supported_fake_forward_inputs(
            q_size=(1, 8, 256, 128),
            k_size=(1, 2, 256, 128),
            index_width=2,
            mask_fn=lambda b, h, q, kv: q >= kv,
        )
        inputs["sparse_q_block_size"] = 256
        _, append = _append_fake_choice(inputs)
        append.assert_not_called()

    def test_decode_falls_back(self):
        _, append = _append_fake_choice(
            _supported_fake_forward_inputs(
                q_size=(32, 64, 4, 128),
                k_size=(32, 4, 8192, 128),
                mask_heads=4,
                index_width=16,
                mask_fn=lambda b, h, q, kv: q + 8188 >= kv,
            )
        )
        append.assert_not_called()

    def test_four_gib_kv_buffer_falls_back(self):
        inputs = _supported_fake_forward_inputs(
            q_size=(64, 64, 4, 128),
            k_size=(64, 4, 65536, 128),
            mask_heads=4,
            index_width=16,
            mask_fn=lambda b, h, q, kv: q + 65532 >= kv,
        )
        inputs["key"]._numel = 1 << 31
        _, append = _append_fake_choice(inputs)
        append.assert_not_called()

    def test_unsupported_dtype_falls_back(self):
        choices, append = _append_fake_choice(
            _supported_fake_forward_inputs(
                q_size=(1, 8, 256, 128),
                k_size=(1, 8, 256, 128),
                dtype=torch.float16,
            )
        )
        append.assert_not_called()
        self.assertEqual(choices, [])

    def test_unsupported_mask_bytecode_operation_falls_back(self):
        mask_graph = torch.fx.symbolic_trace(lambda b, h, q, kv: torch.abs(q - kv) < 96)

        program, reason = lower_flydsl_mask_graph(mask_graph, [])

        self.assertIsNone(program)
        self.assertIn("unsupported", reason)

    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        q_heads, kv_heads, seq, head_dim = 4, 2, 512, 128
        query, key, value = _make_qkv(
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            qk_dim=head_dim,
        )
        kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 0]]], device="cuda", dtype=torch.int32
        )
        kv_indices = torch.tensor(
            [[[[0], [1], [2], [0]]]], device="cuda", dtype=torch.int32
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1, 1, 0]]], device="cuda", dtype=torch.int32
        )
        full_kv_indices = torch.tensor(
            [[[[0], [0], [0], [0]]]], device="cuda", dtype=torch.int32
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
        output, aux = self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
            return_aux=True,
        )
        self.assertEqual(output[:, :, 384:].abs().max().item(), 0.0)
        self.assertTrue(torch.isneginf(aux.lse[:, :, 384:]).all())
        self.assertTrue(torch.isneginf(aux.max_scores[:, :, 384:]).all())

    def test_gfx950_public_api_transposed_document_qk192_v128(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        batch, heads, seq = 2, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        query, key, value = _make_qkv(
            batch=batch,
            q_heads=heads,
            seq_q=seq,
            qk_dim=qk_head_dim,
            v_dim=v_head_dim,
            seed=2,
            transposed=True,
        )
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
        output, _ = self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=0.07,
            return_aux=True,
        )
        self.assertEqual(output.shape, (batch, heads, seq, v_head_dim))
        self.assertEqual(output.stride()[-1], 1)

    def test_gfx950_public_api_sliding_window_mask(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        query, key, value = _make_qkv(seed=3)

        def sliding_window(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (q_idx - kv_idx < 96)

        block_mask = create_block_mask(
            sliding_window,
            1,
            1,
            256,
            256,
            device="cuda",
            BLOCK_SIZE=128,
        )
        output = self._compare_forward(query, key, value, block_mask=block_mask)
        self.assertFalse(torch.isnan(output).any())

    def test_gfx950_public_api_default_block_mask(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        query, key, value = _make_qkv(
            batch=2,
            q_heads=3,
            seed=7,
        )
        self._compare_forward(query, key, value)

    def test_gfx950_public_api_general_batched_head_dependent_mask(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        batch, heads, seq, head_dim = 2, 3, 512, 128
        query, key, value = _make_qkv(
            batch=batch,
            q_heads=heads,
            seq_q=seq,
            qk_dim=head_dim,
            seed=8,
        )

        def checkerboard_causal(b, h, q_idx, kv_idx):
            del b
            adjusted_q = torch.add(q_idx, 64, alpha=2)
            return (adjusted_q >= kv_idx) & ((adjusted_q - kv_idx) % (2 + h) == 0)

        block_mask = create_block_mask(
            checkerboard_causal,
            batch,
            heads,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_forward(query, key, value, block_mask=block_mask)

    def test_gfx950_public_api_b1_document_two_captures(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        seq = 256
        query, key, value = _make_qkv(seed=4)
        document_ids = torch.arange(seq, device="cuda", dtype=torch.int32) // 128
        document_starts = torch.tensor(
            [0, 128], device="cuda", dtype=torch.int32
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
        self._compare_forward(query, key, value, block_mask=block_mask)

    def test_gfx950_public_api_standard_width_create_block_mask(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        seq = 4096
        query, key, value = _make_qkv(
            q_heads=1,
            seq_q=seq,
            seed=5,
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
        self._compare_forward(query, key, value, block_mask=block_mask)

    def test_gfx950_auto_keeps_flydsl_opt_in(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        query, key, value = _make_qkv(seed=6)

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            256,
            256,
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

    def test_gfx950_public_api_qk192_v128_prefill(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

        batch, heads, seq = 1, 16, 256
        qk_head_dim, v_head_dim = 192, 128
        query, key, value = _make_qkv(
            q_heads=heads,
            seq_q=seq,
            qk_dim=qk_head_dim,
            v_dim=v_head_dim,
            seed=1,
        )

        kv_num_blocks = torch.ones(1, 1, 2, device="cuda", dtype=torch.int32)
        kv_indices = torch.tensor(
            [[[[0], [1]]]], device="cuda", dtype=torch.int32
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1]]], device="cuda", dtype=torch.int32
        )
        full_kv_indices = torch.tensor(
            [[[[0], [0]]]], device="cuda", dtype=torch.int32
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
        output, _ = self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=qk_head_dim**-0.5,
            return_aux=True,
        )
        self.assertEqual(output.shape, (batch, heads, seq, v_head_dim))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
