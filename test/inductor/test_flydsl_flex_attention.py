# Owner(s): ["module: inductor"]

import operator
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

    def node(size, node_dtype=dtype):
        return _FakeNode(size, _contiguous_stride(size), node_dtype)

    return {
        "query": node(q_size),
        "key": node(k_size),
        "value": node(v_size),
        "logsumexp": node(stats_size, torch.float32),
        "max_scores": node(stats_size, torch.float32),
        "kv_num_blocks": node(mask_counts_size, torch.int32),
        "kv_indices": node(mask_indices_size, torch.int32),
        "full_kv_num_blocks": node(mask_counts_size, torch.int32),
        "full_kv_indices": node(mask_indices_size, torch.int32),
        "subgraph": SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(lambda score, b, h, q, kv: score)
        ),
        "mask_graph": SimpleNamespace(graph_module=torch.fx.symbolic_trace(mask_fn)),
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


def _fake_choice_kwargs(inputs):
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
    return append.call_args.kwargs if append.called else None


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
    def _require_runtime(self):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")

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

    def _compare_created_mask(
        self,
        mask_mod,
        *,
        batch=1,
        q_heads=2,
        kv_heads=None,
        seq=256,
        qk_dim=128,
        v_dim=128,
        seed=0,
        transposed=False,
        mask_heads=1,
        scale=None,
        return_aux=False,
    ):
        query, key, value = _make_qkv(
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            qk_dim=qk_dim,
            v_dim=v_dim,
            seed=seed,
            transposed=transposed,
        )
        block_mask = create_block_mask(
            mask_mod,
            batch,
            mask_heads,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        return self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=kv_heads is not None and kv_heads != q_heads,
            return_aux=return_aux,
        )

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

    def test_mask_lowering_edge_cases(self):
        graph_module = torch.fx.symbolic_trace(lambda b, h, q, kv: q + -0.5 >= kv)
        self.assertFalse(is_causal_mask_graph(graph_module))
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)

        graph_module = _mask_graph(lambda b, h, q, kv: torch.add(q, 64, alpha=2) >= kv)
        self.assertFalse(is_causal_mask_graph(graph_module, 64))
        self.assertFalse(is_causal_mask_graph(graph_module, 128))

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

    def test_appends_supported_choices(self):
        cases = (
            (
                "bf16_gqa",
                {},
                (64, 4, 512, 1024, 128, 128, 1),
            ),
            (
                "qk192_v128",
                {
                    "q_size": (1, 16, 256, 192),
                    "k_size": (1, 16, 256, 192),
                    "v_size": (1, 16, 256, 128),
                    "index_width": 1,
                    "mask_fn": lambda b, h, q, kv: q >= kv,
                },
                (16, 16, 256, 256, 192, 128, 1),
            ),
            (
                "gqa_decode_q4",
                {
                    "q_size": (32, 64, 4, 128),
                    "k_size": (32, 4, 8192, 128),
                    "mask_heads": 4,
                    "index_width": 16,
                    "mask_fn": lambda b, h, q, kv: q + 8188 >= kv,
                },
                (64, 4, 4, 8192, 128, 128, 4),
            ),
        )
        keys = (
            "NUM_Q_HEADS",
            "NUM_KV_HEADS",
            "SEQ_Q",
            "SEQ_KV",
            "QK_HEAD_DIM",
            "V_HEAD_DIM",
            "BLOCK_MASK_HEADS",
        )
        for name, input_kwargs, expected in cases:
            with self.subTest(name=name):
                kwargs = _fake_choice_kwargs(
                    _supported_fake_forward_inputs(**input_kwargs)
                )
                self.assertIsNotNone(kwargs)
                self.assertEqual(tuple(kwargs[key] for key in keys), expected)
                self.assertTrue(kwargs["CAUSAL_PARTIAL_BLOCKS"])
                self.assertEqual(kwargs["SPARSE_Q_BLOCK_SIZE"], 128)
                self.assertEqual(kwargs["SPARSE_KV_BLOCK_SIZE"], 128)

    def test_unsupported_choices_fall_back(self):
        huge_seq = 1 << 24
        cases = (
            (
                "count_dtype",
                {},
                {"kv_num_blocks": _FakeNode([1, 1, 4], [4, 4, 1], torch.float32)},
            ),
            (
                "q_block_size",
                {
                    "q_size": (1, 8, 256, 128),
                    "k_size": (1, 2, 256, 128),
                    "index_width": 2,
                    "mask_fn": lambda b, h, q, kv: q >= kv,
                },
                {"sparse_q_block_size": 256},
            ),
            (
                "decode_per_q_head_mask",
                {
                    "q_size": (32, 64, 4, 128),
                    "k_size": (32, 4, 8192, 128),
                    "mask_heads": 64,
                    "index_width": 16,
                    "mask_fn": lambda b, h, q, kv: q + 8188 >= kv,
                },
                {},
            ),
            (
                "four_gib_head_slice",
                {
                    "q_size": (1, 16, 1, 128),
                    "k_size": (1, 1, huge_seq, 128),
                    "mask_heads": 1,
                    "index_width": 16,
                    "mask_fn": lambda b, h, q, kv: q + huge_seq - 1 >= kv,
                },
                {},
            ),
            (
                "dtype",
                {
                    "q_size": (1, 8, 256, 128),
                    "k_size": (1, 8, 256, 128),
                    "dtype": torch.float16,
                },
                {},
            ),
        )
        for name, input_kwargs, overrides in cases:
            with self.subTest(name=name):
                inputs = _supported_fake_forward_inputs(**input_kwargs)
                inputs.update(overrides)
                self.assertIsNone(_fake_choice_kwargs(inputs))

    def test_four_gib_kv_buffer_uses_rebased_head_slice(self):
        inputs = _supported_fake_forward_inputs(
            q_size=(64, 64, 4, 128),
            k_size=(64, 4, 65536, 128),
            mask_heads=4,
            index_width=16,
            mask_fn=lambda b, h, q, kv: q + 65532 >= kv,
        )
        inputs["key"]._numel = 1 << 31
        self.assertIsNotNone(_fake_choice_kwargs(inputs))

    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(self):
        self._require_runtime()

        q_heads, kv_heads, seq, head_dim = 4, 2, 512, 128
        query, key, value = _make_qkv(
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            qk_dim=head_dim,
        )
        kv_num_blocks = torch.tensor([[[1, 1, 1, 0]]], device="cuda", dtype=torch.int32)
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

    def _check_gfx950_public_api_per_kv_head_decode(self, seq_q):
        batch, q_heads, kv_heads = 1, 64, 4
        seq_kv, head_dim = 8192, 128
        query, key, value = _make_qkv(
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_dim=head_dim,
            seed=9,
        )
        kv_num_blocks = torch.ones(
            1,
            kv_heads,
            1,
            device="cuda",
            dtype=torch.int32,
        )
        kv_indices = torch.zeros(
            1,
            kv_heads,
            1,
            16,
            device="cuda",
            dtype=torch.int32,
        )
        kv_indices[..., 0] = 63
        full_kv_num_blocks = torch.full(
            (1, kv_heads, 1),
            15,
            device="cuda",
            dtype=torch.int32,
        )
        full_kv_indices = torch.arange(
            15,
            device="cuda",
            dtype=torch.int32,
        ).view(1, 1, 1, 15) + (
            torch.arange(kv_heads, device="cuda", dtype=torch.int32).view(
                1, kv_heads, 1, 1
            )
            * 8
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
        self._compare_forward(
            query,
            key,
            value,
            block_mask=block_mask,
            reference_block_mask=reference_block_mask,
            scale=head_dim**-0.5,
            enable_gqa=True,
            return_aux=True,
        )

    def test_gfx950_public_api_per_kv_head_decode(self):
        self._require_runtime()
        for seq_q in (1, 4, 8):
            with self.subTest(seq_q=seq_q):
                self._check_gfx950_public_api_per_kv_head_decode(seq_q)

    def test_gfx950_public_api_transposed_document_qk192_v128(self):
        self._require_runtime()

        batch, heads, seq = 2, 2, 256
        qk_head_dim, v_head_dim = 192, 128
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

        output, _ = self._compare_created_mask(
            document_causal,
            batch=batch,
            q_heads=heads,
            seq=seq,
            qk_dim=qk_head_dim,
            v_dim=v_head_dim,
            seed=2,
            transposed=True,
            scale=0.07,
            return_aux=True,
        )
        self.assertEqual(output.shape, (batch, heads, seq, v_head_dim))
        self.assertEqual(output.stride()[-1], 1)

    def test_gfx950_public_api_supported_masks(self):
        self._require_runtime()

        def sliding_window(b, h, q_idx, kv_idx):
            del b, h
            return (q_idx >= kv_idx) & (q_idx - kv_idx < 96)

        def checkerboard_causal(b, h, q_idx, kv_idx):
            del b
            adjusted_q = torch.add(q_idx, 64, alpha=2)
            return (adjusted_q >= kv_idx) & ((adjusted_q - kv_idx) % (2 + h) == 0)

        seq = 256
        document_ids = torch.arange(seq, device="cuda", dtype=torch.int32) // 128
        document_starts = torch.tensor([0, 128], device="cuda", dtype=torch.int32)

        def document_causal(b, h, q_idx, kv_idx):
            del b, h
            return (kv_idx >= document_starts[document_ids[q_idx]]) & (kv_idx <= q_idx)

        cases = (
            ("sliding_window", sliding_window, {"seed": 3}),
            (
                "head_dependent",
                checkerboard_causal,
                {
                    "batch": 2,
                    "q_heads": 3,
                    "seq": 512,
                    "seed": 8,
                    "mask_heads": 3,
                },
            ),
            ("two_captures", document_causal, {"seed": 4}),
        )
        for name, mask_mod, kwargs in cases:
            with self.subTest(name=name):
                self._compare_created_mask(mask_mod, **kwargs)

        with self.subTest(name="default"):
            self._compare_forward(*_make_qkv(batch=2, q_heads=3, seed=7))

    def test_gfx950_auto_keeps_flydsl_opt_in(self):
        self._require_runtime()

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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
