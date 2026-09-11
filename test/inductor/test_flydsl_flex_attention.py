# Owner(s): ["module: inductor"]

import operator
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _get_supported_bhsd_stride,
    _is_contiguous_shape_stride,
    flex_flydsl_forward_template,
    maybe_append_flydsl_flex_attention_choice,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import (
    and_masks,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
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
                "lt": operator.lt,
                "and": operator.and_,
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


def _fake_choice_result(inputs):
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
        _, reason = maybe_append_flydsl_flex_attention_choice(
            choices,
            layout=mock.Mock(),
            **inputs,
        )
    kwargs = append.call_args.kwargs if append.called else None
    return kwargs, reason


def _make_qkv(
    *,
    device="cuda",
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
                device=device,
                dtype=torch.bfloat16,
            )
        return torch.randn(
            batch,
            sequence,
            heads,
            dimension,
            device=device,
            dtype=torch.bfloat16,
        ).transpose(1, 2)

    return (
        make(q_heads, seq_q, qk_dim),
        make(kv_heads, seq_kv, qk_dim),
        make(kv_heads, seq_kv, v_dim),
    )


class TestFlyDSLFlexAttentionBackend(TestCase):
    @parametrize("qk_dim", [128, 192])
    @parametrize("partial", [False, True])
    def test_split_kv_reduction(self, device, qk_dim, partial):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        query, key, value = _make_qkv(
            device=device,
            q_heads=4,
            seq_q=1,
            seq_kv=2048,
            qk_dim=qk_dim,
            seed=17,
        )
        counts = torch.arange(4, device=device, dtype=torch.int32).view(1, 4, 1)
        zero_counts = torch.zeros_like(counts)
        block_ids = torch.tensor([0, 5, 15], device=device, dtype=torch.int32)
        indices = block_ids.view(1, 1, 1, 3).expand(1, 4, 1, 3).contiguous()

        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx % 2 == 0 if partial else kv_idx >= 0

        block_mask = BlockMask.from_kv_blocks(
            counts if partial else zero_counts,
            indices,
            zero_counts if partial else counts,
            indices,
            BLOCK_SIZE=128,
            mask_mod=mask_mod,
            seq_lengths=(1, 2048),
            compute_q_blocks=False,
        )

        def attention(q, k, v):
            return flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLYDSL"},
                return_aux=AuxRequest(lse=True, max_scores=True),
            )

        output, aux = torch.compile(attention, fullgraph=True)(query, key, value)
        positions = torch.arange(2048, device=device)
        active_blocks = (
            torch.arange(3, device=device)[None, :, None]
            < torch.arange(4, device=device)[:, None, None]
        )
        keep = (
            active_blocks
            & ((positions // 128)[None, None, :] == block_ids[None, :, None])
        ).any(dim=1).view(1, 4, 1, 2048)
        if partial:
            keep = keep & (positions % 2 == 0)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), attn_mask=keep
        ).to(query.dtype)
        scores = (query.float() @ key.float().transpose(-1, -2)) * qk_dim**-0.5
        scores = scores.masked_fill(~keep, float("-inf"))
        # FlyDSL rounds scaled Q to bf16 before QK.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)
        self.assertEqual(aux.lse, scores.logsumexp(-1), atol=0.025, rtol=0.025)
        self.assertEqual(aux.max_scores, scores.amax(-1), atol=0.025, rtol=0.025)
        self.assertEqual(output[:, 0], torch.zeros_like(output[:, 0]), atol=0, rtol=0)
        self.assertTrue(torch.isneginf(aux.lse[:, 0]).all())
        self.assertTrue(torch.isneginf(aux.max_scores[:, 0]).all())

    @parametrize("seq_q", [1, 256])
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_dense_default_uses_isolated_backend(self, device, seq_q):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        query, key, value = _make_qkv(device=device, seq_q=seq_q, seq_kv=256)
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, kernel_options={"BACKEND": "FLYDSL"}
            ),
            fullgraph=True,
        )
        with (
            mock.patch(
                "torch._inductor.kernel.flex.flex_attention._use_flex_decoding",
                side_effect=AssertionError("FlyDSL reached Triton decode selection"),
            ),
            mock.patch(
                "torch._inductor.kernel.flex.flex_attention."
                "flex_attention_template.maybe_append_choice",
                side_effect=AssertionError("FlyDSL registered a Triton candidate"),
            ),
        ):
            output = compiled(query, key, value)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float()
        ).to(query.dtype)
        # FlyDSL rounds the scaled Q to bf16 before QK.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_computed_mask_capture(self, device):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        query, key, value = _make_qkv(device=device)
        ends = torch.arange(256, device=device, dtype=torch.int32)
        counts = torch.full((1, 1, 2), 2, device=device, dtype=torch.int32)
        indices = torch.tensor([[[[0, 1], [0, 1]]]], device=device, dtype=torch.int32)
        full_counts = torch.zeros_like(counts)
        full_indices = torch.zeros_like(indices)

        def attention(q, k, v, document_ends):
            computed_ends = document_ends + 1

            def mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < computed_ends[q_idx]

            block_mask = BlockMask.from_kv_blocks(
                counts,
                indices,
                full_counts,
                full_indices,
                BLOCK_SIZE=128,
                mask_mod=mask_mod,
                seq_lengths=(256, 256),
                compute_q_blocks=False,
            )
            return flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLYDSL"}
            )

        output = torch.compile(attention, fullgraph=True)(query, key, value, ends)
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), is_causal=True
        ).to(query.dtype)
        # Match the bf16 scaled-Q precision used by the forward kernel.
        self.assertEqual(output, reference, atol=0.025, rtol=0.025)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_backward_uses_triton_with_flydsl_forward(self, device):
        if not _has_gfx950_flydsl():
            self.skipTest("requires gfx950 and a built FlyDSL runtime")
        base_inputs = _make_qkv(
            device=device, seq_q=128, seq_kv=128, seed=23
        )
        torch.manual_seed(24)
        grad_output = torch.randn_like(base_inputs[2])

        def run(backend):
            inputs = tuple(t.detach().clone().requires_grad_() for t in base_inputs)
            compiled = torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )
            output, code = run_and_get_code(compiled, *inputs)
            output.backward(grad_output)
            return (
                output.detach(),
                tuple(t.grad.detach().clone() for t in inputs),
                "\n".join(code),
            )

        output, grads, code = run("FLYDSL")
        reference, reference_grads, reference_code = run("TRITON")
        self.assertIn("build_flex_attn_fwd_module", code)
        self.assertNotIn("build_flex_attn_fwd_module", reference_code)
        self.assertEqual(output, reference, atol=0.03, rtol=0.02)
        for grad, reference_grad in zip(grads, reference_grads):
            self.assertEqual(grad, reference_grad, atol=0.03, rtol=0.02)


instantiate_device_type_tests(
    TestFlyDSLFlexAttentionBackend, globals(), only_for=("cuda",)
)


@instantiate_parametrized_tests
class TestFlyDSLFlexAttentionConfig(TestCase):
    @parametrize(
        "shape,stride,strict,metadata",
        [
            subtest(((2, 3, 4, 5), (60, 20, 5, 1), True, True), name="contiguous"),
            subtest(
                ((2, 1, 4, 5), (20, 99, 5, 1), False, True),
                name="size_one_dimension",
            ),
            subtest(
                ((2, 3, 4, 5), (120, 40, 5, 1), False, False),
                name="strided",
            ),
            subtest(
                ((2, 0, 4, 5), (0, 20, 5, 1), True, False),
                name="zero_size_strict",
            ),
            subtest(
                ((2, 0, 4, 5), (20, 20, 5, 1), False, True),
                name="zero_size_metadata",
            ),
            subtest(
                ((2, 3, 4, 5), (20, 5, 1), False, False),
                name="rank_mismatch",
            ),
            subtest(((), (), False, True), name="scalar_metadata"),
        ],
    )
    def test_contiguous_stride_checks(self, shape, stride, strict, metadata):
        with V.set_graph_handler(_fake_graph()):
            self.assertEqual(
                _get_supported_bhsd_stride(
                    _FakeNode(shape, stride), allow_strided=False
                ),
                stride if strict else None,
            )
            self.assertEqual(_is_contiguous_shape_stride(shape, stride), metadata)

    def test_unrealized_metadata_is_rejected(self):
        inputs = _supported_fake_forward_inputs()
        with (
            V.set_graph_handler(_fake_graph()),
            mock.patch.object(
                inputs["kv_indices"], "get_stride", side_effect=NotImplementedError
            ),
            mock.patch.object(flex_flydsl_forward_template, "maybe_append_choice") as append,
        ):
            appended, reason = maybe_append_flydsl_flex_attention_choice(
                [], layout=mock.Mock(), **inputs
            )
        self.assertFalse(appended)
        self.assertIn("requires statically known BlockMask metadata strides", reason)
        append.assert_not_called()


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
            # FlyDSL rounds scaled Q to bf16 before QK.
            self.assertEqual(actual, expected, atol=0.03, rtol=0.02)
            return actual

        output, aux = actual
        reference, reference_aux = expected
        # FlyDSL rounds scaled Q to bf16 before QK.
        self.assertEqual(output, reference, atol=0.03, rtol=0.02)
        self.assertEqual(aux.lse, reference_aux.lse, atol=0.03, rtol=0.01)
        self.assertEqual(
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
        device,
        batch=1,
        q_heads=2,
        kv_heads=None,
        seq=256,
        seq_kv=None,
        qk_dim=128,
        v_dim=128,
        seed=0,
        transposed=False,
        mask_heads=1,
        scale=None,
        return_aux=False,
    ):
        query, key, value = _make_qkv(
            device=device,
            batch=batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            seq_kv=seq_kv,
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
            seq if seq_kv is None else seq_kv,
            device=device,
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

    def test_mask_lowering_rejects_float_constant(self):
        graph_module = torch.fx.symbolic_trace(lambda b, h, q, kv: q + -0.5 >= kv)
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)

    @parametrize(
        "mask_mod,q,kv,expected",
        [
            subtest(
                (lambda b, h, q, kv: torch.add(q, 64, alpha=2) >= kv, 0, 100, True),
                name="add_alpha",
            ),
            subtest(
                (
                    lambda b, h, q, kv: torch.sub(q, 64, alpha=2) >= kv,
                    200,
                    100,
                    False,
                ),
                name="sub_alpha",
            ),
            subtest(
                (lambda b, h, q, kv: q + 8188 >= kv, 3, 8191, True),
                name="decode_boundary_inside",
            ),
            subtest(
                (lambda b, h, q, kv: q + 8188 >= kv, 2, 8191, False),
                name="decode_boundary_outside",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 96),
                    95,
                    0,
                    True,
                ),
                name="window_boundary_inside",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 96),
                    96,
                    0,
                    False,
                ),
                name="window_boundary_outside",
            ),
        ],
    )
    def test_mask_lowering_edge_cases(self, mask_mod, q, kv, expected):
        program, reason = lower_flydsl_mask_graph(
            _mask_graph(mask_mod),
            (),
        )
        self.assertIsNotNone(program, reason)
        self.assertEqual(
            _evaluate_mask_program(program, 0, 0, q, kv),
            expected,
        )

    @parametrize(
        "input_kwargs,expected",
        [
            subtest(
                (
                    {},
                    (64, 4, 512, 1024, 128, 128, 1),
                ),
                name="bf16_gqa",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 16, 256, 192),
                        "k_size": (1, 16, 256, 192),
                        "v_size": (1, 16, 256, 128),
                        "index_width": 1,
                        "mask_fn": lambda b, h, q, kv: q >= kv,
                    },
                    (16, 16, 256, 256, 192, 128, 1),
                ),
                name="qk192_v128",
            ),
            subtest(
                (
                    {
                        "q_size": (32, 64, 4, 128),
                        "k_size": (32, 4, 8192, 128),
                        "mask_heads": 4,
                        "index_width": 16,
                        "mask_fn": lambda b, h, q, kv: q + 8188 >= kv,
                    },
                    (64, 4, 4, 8192, 128, 128, 4),
                ),
                name="gqa_decode_q4",
            ),
        ],
    )
    def test_appends_supported_choices(self, input_kwargs, expected):
        keys = (
            "NUM_Q_HEADS",
            "NUM_KV_HEADS",
            "SEQ_Q",
            "SEQ_KV",
            "QK_HEAD_DIM",
            "V_HEAD_DIM",
            "BLOCK_MASK_HEADS",
        )
        kwargs, reason = _fake_choice_result(
            _supported_fake_forward_inputs(**input_kwargs)
        )
        self.assertIsNotNone(kwargs, reason)
        self.assertEqual(tuple(kwargs[key] for key in keys), expected)
        self.assertNotIn("CAUSAL_PARTIAL_BLOCKS", kwargs)
        self.assertTrue(kwargs["MASK_PROGRAM"])
        self.assertEqual(kwargs["SPARSE_Q_BLOCK_SIZE"], 128)
        self.assertEqual(kwargs["SPARSE_KV_BLOCK_SIZE"], 128)

    @parametrize(
        "input_kwargs,overrides,expected_reason",
        [
            subtest(
                (
                    {},
                    {
                        "kv_num_blocks": _FakeNode(
                            [1, 1, 4], [4, 4, 1], torch.float32
                        )
                    },
                    "requires int32 BlockMask metadata",
                ),
                name="count_dtype",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 8, 256, 128),
                        "k_size": (1, 2, 256, 128),
                        "index_width": 2,
                        "mask_fn": lambda b, h, q, kv: q >= kv,
                    },
                    {"sparse_q_block_size": 256},
                    "requires sparse Q/KV block sizes of 128",
                ),
                name="q_block_size",
            ),
            subtest(
                (
                    {
                        "q_size": (32, 64, 4, 128),
                        "k_size": (32, 4, 8192, 128),
                        "mask_heads": 64,
                        "index_width": 16,
                        "mask_fn": lambda b, h, q, kv: q + 8188 >= kv,
                    },
                    {},
                    "requires prefill Sq divisible by 128",
                ),
                name="decode_per_q_head_mask",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 16, 1, 128),
                        "k_size": (1, 1, 1 << 24, 128),
                        "mask_heads": 1,
                        "index_width": 16,
                        "mask_fn": lambda b, h, q, kv: q + (1 << 24) - 1 >= kv,
                    },
                    {},
                    "requires every per-head tensor slice to be smaller than 4 GiB",
                ),
                name="four_gib_head_slice",
            ),
            subtest(
                (
                    {
                        "q_size": (1, 8, 256, 128),
                        "k_size": (1, 8, 256, 128),
                        "dtype": torch.float16,
                    },
                    {},
                    "supports BF16 only",
                ),
                name="dtype",
            ),
        ],
    )
    def test_unsupported_choices_fall_back(
        self, input_kwargs, overrides, expected_reason
    ):
        inputs = _supported_fake_forward_inputs(**input_kwargs)
        inputs.update(overrides)
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNone(kwargs)
        self.assertIn(expected_reason, reason)

    def test_four_gib_kv_buffer_uses_rebased_head_slice(self):
        inputs = _supported_fake_forward_inputs(
            q_size=(64, 64, 4, 128),
            k_size=(64, 4, 65536, 128),
            mask_heads=4,
            index_width=16,
            mask_fn=lambda b, h, q, kv: q + 65532 >= kv,
        )
        inputs["key"]._numel = 1 << 31
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNotNone(kwargs, reason)

    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(self, device):
        self._require_runtime()

        q_heads, kv_heads, seq, head_dim = 4, 2, 512, 128
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            qk_dim=head_dim,
        )
        kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 0]]], device=device, dtype=torch.int32
        )
        kv_indices = torch.tensor(
            [[[[0], [1], [2], [0]]]], device=device, dtype=torch.int32
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1, 1, 0]]], device=device, dtype=torch.int32
        )
        full_kv_indices = torch.tensor(
            [[[[0], [0], [0], [0]]]], device=device, dtype=torch.int32
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

    def _check_gfx950_public_api_per_kv_head_decode(self, device, seq_q):
        batch, q_heads, kv_heads = 1, 64, 4
        seq_kv, head_dim = 8192, 128
        query, key, value = _make_qkv(
            device=device,
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
            device=device,
            dtype=torch.int32,
        )
        kv_indices = torch.zeros(
            1,
            kv_heads,
            1,
            16,
            device=device,
            dtype=torch.int32,
        )
        kv_indices[..., 0] = 63
        full_kv_num_blocks = torch.full(
            (1, kv_heads, 1),
            15,
            device=device,
            dtype=torch.int32,
        )
        full_kv_indices = torch.arange(
            15,
            device=device,
            dtype=torch.int32,
        ).view(1, 1, 1, 15) + (
            torch.arange(kv_heads, device=device, dtype=torch.int32).view(
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

    @parametrize("seq_q", [1, 4, 8])
    def test_gfx950_public_api_per_kv_head_decode(self, device, seq_q):
        self._require_runtime()
        self._check_gfx950_public_api_per_kv_head_decode(device, seq_q)

    def test_gfx950_public_api_transposed_document_qk192_v128(self, device):
        self._require_runtime()

        batch, heads, seq = 2, 2, 256
        qk_head_dim, v_head_dim = 192, 128
        document_end = torch.tensor(
            [
                [127] * 128 + [255] * 128,
                [63] * 64 + [191] * 128 + [255] * 64,
            ],
            device=device,
            dtype=torch.int32,
        )

        def document_causal(b, h, q_idx, kv_idx):
            del h
            return (q_idx >= kv_idx) & (q_idx <= document_end[b, kv_idx])

        output, _ = self._compare_created_mask(
            document_causal,
            device=device,
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

    @parametrize(
        "mask_mod,kwargs",
        [
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 96),
                    {
                        "q_heads": 4,
                        "kv_heads": 2,
                        "return_aux": True,
                        "seed": 3,
                    },
                ),
                name="sliding_window",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q >= kv) & (q - kv < 256),
                    {
                        "seq": 512,
                        "q_heads": 4,
                        "kv_heads": 2,
                        "return_aux": True,
                    },
                ),
                name="wide_window",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (q + 2044 >= kv)
                    & (q + 2044 - kv < 96),
                    {
                        "seq": 4,
                        "seq_kv": 2048,
                        "q_heads": 4,
                        "kv_heads": 2,
                        "return_aux": True,
                    },
                ),
                name="window_decode",
            ),
            subtest(
                (
                    lambda b, h, q, kv: (torch.add(q, 64, alpha=2) >= kv)
                    & ((torch.add(q, 64, alpha=2) - kv) % (2 + h) == 0),
                    {
                        "batch": 2,
                        "q_heads": 3,
                        "seq": 512,
                        "seed": 8,
                        "mask_heads": 3,
                    },
                ),
                name="head_dependent",
            ),
        ],
    )
    def test_gfx950_public_api_supported_masks(self, device, mask_mod, kwargs):
        self._require_runtime()
        self._compare_created_mask(mask_mod, device=device, **kwargs)

    def test_gfx950_public_api_document_mask_lowering(self, device):
        self._require_runtime()

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            is_causal_document_mask_program,
        )

        seq = 512
        document_ids = torch.arange(seq, device=device, dtype=torch.int32) // 256
        document_starts = torch.tensor([0, 256], device=device, dtype=torch.int32)
        document_causal = and_masks(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            lambda b, h, q_idx, kv_idx: kv_idx >= document_starts[document_ids[q_idx]],
        )

        def check_document_lowering(*args, **kwargs):
            program, reason = lower_flydsl_mask_graph(*args, **kwargs)
            self.assertIsNotNone(program, reason)
            self.assertTrue(
                is_causal_document_mask_program(
                    program.instructions, program.output, program.buffer_strides
                )
            )
            return program, reason

        torch._dynamo.reset()
        with (
            torch._inductor.config.patch({"fx_graph_cache": False}),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention.lower_flydsl_mask_graph",
                side_effect=check_document_lowering,
            ) as lower,
        ):
            self._compare_created_mask(
                document_causal,
                device=device,
                seq=seq,
                seed=4,
                return_aux=True,
            )
            lower.assert_called_once()

    def test_gfx950_auto_keeps_flydsl_opt_in(self, device):
        self._require_runtime()

        query, key, value = _make_qkv(device=device, seed=6)

        def causal(b, h, q_idx, kv_idx):
            del b, h
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal,
            1,
            1,
            256,
            256,
            device=device,
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


instantiate_device_type_tests(
    TestFlyDSLFlexAttention, globals(), only_for=("cuda",)
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
