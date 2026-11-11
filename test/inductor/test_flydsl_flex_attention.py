# Owner(s): ["module: inductor"]

import math
import operator
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel.flex import flex_flydsl_attention, flex_flydsl_backward
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _get_supported_bhsd_stride,
    _is_contiguous_shape_stride,
    flex_flydsl_forward_template,
    maybe_append_flydsl_flex_attention_choice,
)
from torch._inductor.kernel.flex.flex_flydsl_backward import (
    _get_flydsl_flex_attention_backward_config,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.kernel.flex.flex_flydsl_score import (
    classify_capture_gradient,
    lower_flydsl_score_graph,
)
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
    DeterministicGuard,
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


def _supported_fake_backward_inputs():
    inputs = _supported_fake_forward_inputs(
        q_size=(1, 2, 256, 128),
        k_size=(1, 2, 256, 128),
        v_size=(1, 2, 256, 128),
        index_width=2,
        mask_fn=lambda b, h, q, kv: q >= kv,
    )
    output_shape = (1, 2, 256, 128)
    inputs["fw_subgraph"] = inputs.pop("subgraph")
    inputs.pop("logsumexp")
    inputs.pop("max_scores")
    inputs.update(
        out=_FakeNode(output_shape, _contiguous_stride(output_shape)),
        grad_out=_FakeNode(output_shape, _contiguous_stride(output_shape)),
        grad_logsumexp=None,
    )
    return inputs


def _backward_config_result(inputs):
    with (
        V.set_graph_handler(_fake_graph()),
        mock.patch.object(
            flex_flydsl_attention, "_is_gfx950_device", return_value=True
        ),
    ):
        return _get_flydsl_flex_attention_backward_config(**inputs)


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
                "remainder": operator.mod,
                "ge": operator.ge,
                "gt": operator.gt,
                "lt": operator.lt,
                "eq": operator.eq,
                "and": operator.and_,
            }[op](lhs, rhs)
        )
    return values[program.output]


def _gather(buffer, *indices):
    # flex_attention traces captured-buffer indexing as aten.index.Tensor;
    # plain make_fx would instead read 0-d indices as Python integers.
    return torch.ops.aten.index.Tensor(buffer, list(indices))


def _score_graph(fn, *captures):
    score = torch.tensor(0.0)
    indices = tuple(torch.tensor(0, dtype=torch.int32) for _ in range(4))
    return make_fx(fn)(score, *indices, *captures)


_SCORE_PROGRAM_OPS = {
    "i2f": float,
    "b2f": float,
    "f2i": int,
    "b2i": int,
    "f2b": bool,
    "i2b": bool,
    "not": operator.not_,
    "fneg": operator.neg,
    "fabs": abs,
    "exp": math.exp,
    "exp2": lambda x: 2.0**x,
    "log": math.log,
    "log2": math.log2,
    "tanh": math.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "sqrt": math.sqrt,
    "rsqrt": lambda x: 1.0 / math.sqrt(x),
    "recip": lambda x: 1.0 / x,
    "sin": math.sin,
    "cos": math.cos,
    "fadd": operator.add,
    "fsub": operator.sub,
    "fmul": operator.mul,
    "fdiv": operator.truediv,
    "fmax": max,
    "fmin": min,
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "max": max,
    "min": min,
    "floordiv": operator.floordiv,
    "remainder": operator.mod,
    "and": operator.and_,
    "or": operator.or_,
    **{
        prefix + name: function
        for prefix in ("", "f")
        for name, function in (
            ("ge", operator.ge),
            ("gt", operator.gt),
            ("le", operator.le),
            ("lt", operator.lt),
            ("eq", operator.eq),
            ("ne", operator.ne),
        )
    },
}


def _evaluate_score_program(program, score, b, h, q, kv, buffers=()):
    values = [float(score), b, h, q, kv]
    for op, *operands in program.instructions:
        if op == "const_f32":
            values.append(float.fromhex(operands[0]))
        elif op in ("const_i32", "const_bool"):
            values.append(operands[0])
        elif op in ("load_f32", "load_i32"):
            index = tuple(values[value] for value in operands[1])
            loaded = buffers[operands[0]][index].item()
            values.append(float(loaded) if op == "load_f32" else int(loaded))
        elif op == "select":
            condition, lhs, rhs = (values[value] for value in operands)
            values.append(lhs if condition else rhs)
        else:
            values.append(
                _SCORE_PROGRAM_OPS[op](*(values[value] for value in operands))
            )
    return values[program.output]


def _score_mod_case(kind, *, device, heads):
    """Return ``(make_score_mod, captures)``; captures feed ``make_score_mod``."""
    if kind == "relative":
        return (lambda: lambda s, b, h, q, kv: s + (q - kv) * 0.01), ()
    if kind == "alibi":
        return (
            lambda: (
                lambda s, b, h, q, kv: (
                    s + (kv - q) * torch.exp2(-((h + 1) * 8.0 / heads))
                )
            )
        ), ()
    if kind == "softcap":
        return (lambda: lambda s, b, h, q, kv: torch.tanh(s / 2.0) * 2.0), ()
    if kind == "where_inf":
        return (
            lambda: lambda s, b, h, q, kv: torch.where(q >= kv, s * 1.5, float("-inf"))
        ), ()
    torch.manual_seed(5)
    if kind == "head_bias":
        bias = torch.randn(heads, 256, device=device).bfloat16()
        return (lambda bias: lambda s, b, h, q, kv: s + bias[h, kv]), (bias,)
    # Trainable captures, one per gradient accumulation class (seq 256).
    trainable = {
        "trainable_slope": (
            torch.rand(heads, device=device),
            lambda slope: lambda s, b, h, q, kv: s + slope[h] * (kv - q) * 0.01,
        ),
        "trainable_key_bias": (
            torch.randn(heads, 256, device=device),
            lambda bias: lambda s, b, h, q, kv: s + bias[h, kv],
        ),
        "trainable_query_scale": (
            torch.rand(heads, 256, device=device) + 0.5,
            lambda scale: lambda s, b, h, q, kv: s * scale[h, q],
        ),
        "trainable_relative_table": (
            torch.randn(heads, 511, device=device),
            lambda table: lambda s, b, h, q, kv: s + table[h, q - kv + 255],
        ),
        "trainable_full_bias": (
            torch.randn(1, heads, 256, 256, device=device),
            lambda bias: lambda s, b, h, q, kv: s + bias[b, h, q, kv],
        ),
        "trainable_clamped_table": (
            torch.randn(8, device=device),
            lambda table: (
                lambda s, b, h, q, kv: s + table[torch.clamp(torch.abs(q - kv), max=7)]
            ),
        ),
    }
    capture, make_score_mod = trainable[kind]
    return make_score_mod, (capture.requires_grad_(),)


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
            (
                active_blocks
                & ((positions // 128)[None, None, :] == block_ids[None, :, None])
            )
            .any(dim=1)
            .view(1, 4, 1, 2048)
        )
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
            mock.patch.object(
                flex_flydsl_forward_template, "maybe_append_choice"
            ) as append,
        ):
            appended, reason = maybe_append_flydsl_flex_attention_choice(
                [], layout=mock.Mock(), **inputs
            )
        self.assertFalse(appended)
        self.assertIn("requires statically known BlockMask metadata strides", reason)
        append.assert_not_called()

    @parametrize(
        "unsupported,expected",
        (
            ("gqa", "matching Q/K shapes"),
            ("decode", "sequence length divisible by 128"),
            ("head_dim", "supports only"),
            ("grad_lse", "gradients through LSE"),
            ("workspace", "workspace larger than 16 GiB"),
        ),
    )
    def test_backward_config_gates(self, unsupported, expected):
        inputs = _supported_fake_backward_inputs()
        if unsupported == "gqa":
            shape = (1, 1, 256, 128)
            inputs["key"] = _FakeNode(shape, _contiguous_stride(shape))
            inputs["value"] = _FakeNode(shape, _contiguous_stride(shape))
        elif unsupported == "decode":
            shape = (1, 2, 4, 128)
            for name in ("query", "key", "value", "out", "grad_out"):
                inputs[name] = _FakeNode(shape, _contiguous_stride(shape))
        elif unsupported == "head_dim":
            shape = (1, 2, 256, 64)
            inputs["query"] = _FakeNode(shape, _contiguous_stride(shape))
            inputs["key"] = _FakeNode(shape, _contiguous_stride(shape))
        elif unsupported == "grad_lse":
            inputs["grad_logsumexp"] = inputs["out"]
        else:
            qk_shape = (32, 64, 16384, 192)
            value_shape = (*qk_shape[:-1], 128)
            inputs["query"] = _FakeNode(qk_shape, _contiguous_stride(qk_shape))
            inputs["key"] = _FakeNode(qk_shape, _contiguous_stride(qk_shape))
            inputs["value"] = _FakeNode(value_shape, _contiguous_stride(value_shape))
            inputs["out"] = _FakeNode(value_shape, _contiguous_stride(value_shape))
            inputs["grad_out"] = _FakeNode(value_shape, _contiguous_stride(value_shape))
        config, reason = _backward_config_result(inputs)
        self.assertIsNone(config)
        self.assertIn(expected, reason)

    def test_backward_config(self):
        config, reason = _backward_config_result(_supported_fake_backward_inputs())
        self.assertIsNotNone(config, reason)
        self.assertEqual(config["MASK_PROGRAM"], (("ge", 2, 3),))
        self.assertEqual(config["Q_STRIDE"], (65536, 32768, 128, 1))
        self.assertEqual(config["SCORE_PROGRAM"], ())
        self.assertEqual(config["SCORE_PROGRAM_OUTPUT"], 0)

    def test_backward_score_mod_config(self):
        inputs = _supported_fake_backward_inputs()
        inputs["fw_subgraph"] = SimpleNamespace(
            graph_module=torch.fx.symbolic_trace(
                lambda score, b, h, q, kv: score + (q - kv)
            )
        )
        config, reason = _backward_config_result(inputs)
        self.assertIsNotNone(config, reason)
        # float(q - kv) lowers to float(q) - float(kv); the int sub is dead.
        self.assertEqual(
            config["SCORE_PROGRAM"],
            (("sub", 3, 4), ("i2f", 3), ("i2f", 4), ("fsub", 6, 7), ("fadd", 0, 8)),
        )
        self.assertEqual(config["SCORE_PROGRAM_OUTPUT"], 9)

    @parametrize("dq_accum_fp32", (True, False))
    def test_backward_workspace_bound(self, dq_accum_fp32):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            choose_dq_partitions,
            MAX_DQ_WORKSPACE_BYTES,
        )

        element_bytes = 4 if dq_accum_fp32 else 2
        batch_heads, sequence_length, head_dim = 1024, 8192, 192
        partitions = choose_dq_partitions(
            batch_heads,
            sequence_length,
            192,
            element_bytes,
            qk_head_dim=head_dim,
        )
        self.assertLessEqual(
            batch_heads * partitions * sequence_length * head_dim * element_bytes,
            MAX_DQ_WORKSPACE_BYTES,
        )

    def test_block_list_dq_partitions_fill_device(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            choose_dq_partitions,
        )

        # (batch * heads, sequence) -> slots; about one CTA per compute unit.
        for batch_heads, sequence_length, expected in (
            (64, 4096, 4),
            (16, 4096, 16),
            (256, 2048, 2),
            (4, 256, 2),
        ):
            with self.subTest(batch_heads=batch_heads):
                self.assertEqual(
                    choose_dq_partitions(
                        batch_heads, sequence_length, 128, qk_head_dim=128
                    ),
                    expected,
                )

    def test_backward_mask_traversal_classification(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            DIRECT_RANGE_CAUSAL,
            MASK_TRAVERSAL_BLOCK_LIST,
            MASK_TRAVERSAL_DIRECT_RANGE,
            MASK_TRAVERSAL_UNMASKED,
        )

        cases = (
            ((), 0, True, MASK_TRAVERSAL_UNMASKED, None),
            (
                (("ge", 2, 3),),
                4,
                True,
                MASK_TRAVERSAL_DIRECT_RANGE,
                DIRECT_RANGE_CAUSAL,
            ),
            ((), 0, False, MASK_TRAVERSAL_BLOCK_LIST, None),
            (
                (("ge", 2, 3),),
                4,
                False,
                MASK_TRAVERSAL_BLOCK_LIST,
                None,
            ),
            (
                (("const_i32", 96), ("sub", 2, 3), ("lt", 5, 4)),
                6,
                True,
                MASK_TRAVERSAL_BLOCK_LIST,
                None,
            ),
        )
        for instructions, output, block_mask_is_dense, traversal, kind in cases:
            with self.subTest(instructions=instructions):
                self.assertEqual(
                    classify_mask_traversal(
                        instructions,
                        output,
                        sequence_length=256,
                        block_mask_is_dense=block_mask_is_dense,
                    ),
                    (traversal, kind),
                )

    def test_unavailable_backward_falls_back_before_allocation(self):
        with (
            mock.patch.object(
                flex_flydsl_backward, "runtime_available", return_value=False
            ),
            mock.patch.object(flex_flydsl_backward, "empty_strided") as allocate,
        ):
            result = flex_flydsl_backward.create_flydsl_flex_attention_backward_kernel(
                None, None, None, None, None, None, None, 1.0, 128, 128
            )
        self.assertIsNone(result)
        allocate.assert_not_called()

    def test_invalid_dq_accumulation_option(self):
        with self.assertRaisesRegex(ValueError, "DQ_ACCUM_FP32 must be a bool"):
            flex_flydsl_backward.create_flydsl_flex_attention_backward_kernel(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                1.0,
                128,
                128,
                dq_accum_fp32=1,
            )


class _FlyDSLFlexAttentionRuntimeMixin:
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
        score_mod=None,
    ):
        def run(q, k, v, backend, selected_mask):
            kwargs = {
                "block_mask": selected_mask,
                "enable_gqa": enable_gqa,
                "score_mod": score_mod,
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
            # Without score_mod, FlyDSL rounds scaled Q to bf16 before QK.
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

    def _compare_backward(
        self,
        query,
        key,
        value,
        *,
        block_mask,
        dq_accum_fp32,
        score_mod=None,
        score_captures=(),
        expect_flydsl_backward=True,
    ):
        torch.manual_seed(24)
        grad_output = torch.randn_like(value)

        def run(backend):
            inputs = tuple(
                tensor.detach().clone().requires_grad_()
                for tensor in (query, key, value)
            )
            # Captures keep their own requires_grad; trainable ones get gradients.
            captures = tuple(
                capture.detach().clone().requires_grad_(capture.requires_grad)
                for capture in score_captures
            )
            trainable = inputs + tuple(c for c in captures if c.requires_grad)
            kernel_options = {"BACKEND": backend}
            if backend == "FLYDSL":
                kernel_options["bwd_FLYDSL_DQ_ACCUM_FP32"] = dq_accum_fp32
            compiled = torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    score_mod=(None if score_mod is None else score_mod(*captures)),
                    block_mask=block_mask,
                    kernel_options=kernel_options,
                ),
                fullgraph=True,
            )
            with DeterministicGuard(backend == "FLYDSL"):
                output, code = run_and_get_code(compiled, *inputs)
                _, backward_code = run_and_get_code(
                    lambda: output.backward(grad_output)
                )
            return (
                output.detach(),
                tuple(t.grad.detach().clone() for t in trainable),
                "\n".join(code),
                "\n".join(backward_code),
            )

        output, grads, code, backward_code = run("FLYDSL")
        reference, reference_grads, reference_code, reference_backward_code = run(
            "TRITON"
        )
        self.assertIn("flydsl_fused_", code)
        self.assertNotIn("flydsl_fused_", reference_code)
        if expect_flydsl_backward:
            self.assertIn("flydsl_fused_flex_attention_backward_", backward_code)
        else:
            self.assertNotIn("flydsl_fused_", backward_code)
        self.assertNotIn("flydsl_fused_", reference_backward_code)
        self.assertEqual(output, reference, atol=0.03, rtol=0.02)
        for grad, reference_grad in zip(grads, reference_grads):
            self.assertEqual(grad, reference_grad, atol=0.04, rtol=0.03)

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


@instantiate_parametrized_tests
class TestFlyDSLFlexAttentionLowering(TestCase):
    def test_mask_lowering_rejects_float_constant(self):
        graph_module = torch.fx.symbolic_trace(lambda b, h, q, kv: q + -0.5 >= kv)
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)

    def test_mask_lowering_full_constant_respects_dtype(self):
        program, reason = lower_flydsl_mask_graph(
            _mask_graph(
                lambda b, h, q, kv: (
                    q > torch.full((), 3, dtype=torch.bool, device=q.device)
                )
            ),
            (),
        )
        self.assertIsNotNone(program, reason)
        self.assertTrue(_evaluate_mask_program(program, 0, 0, 2, 0))

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
            subtest(
                (
                    lambda b, h, q, kv: (q - kv) % 8 == 1,
                    1,
                    8,
                    True,
                ),
                name="negative_lhs_remainder",
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
        "score_mod,captures",
        [
            subtest((lambda s, b, h, q, kv: s + (q - kv), ()), name="relative"),
            subtest(
                (
                    lambda s, b, h, q, kv: (
                        s + (kv - q) * torch.exp2(-((h + 1) * 8.0 / 8))
                    ),
                    (),
                ),
                name="alibi",
            ),
            subtest(
                (lambda s, b, h, q, kv: torch.tanh(s / 30.0) * 30.0, ()),
                name="softcap",
            ),
            subtest(
                (lambda s, b, h, q, kv: torch.where(q >= kv, s, float("-inf")), ()),
                name="where_inf",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv: (
                        torch.clamp(s, min=-1.0, max=2.0) ** 2
                        + torch.maximum(s, s * 0.5)
                        - torch.minimum(s, torch.abs(-s))
                    ),
                    (),
                ),
                name="clamp_pow_max",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv: (
                        torch.sigmoid(s)
                        + torch.exp(s * 0.1)
                        + torch.log(torch.abs(s) + 1)
                        + torch.sqrt(torch.abs(s))
                        + torch.rsqrt(torch.abs(s) + 1)
                        + torch.reciprocal(s + 4)
                        + torch.sin(s) * torch.cos(s)
                    ),
                    (),
                ),
                name="transcendental",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv: torch.where(
                        ((q // 3) % 4 == 1) | ((kv - q) % 5 > b),
                        s * ((q - kv) % 7).float(),
                        (s * 2).to(torch.int32) - h,
                    ),
                    (),
                ),
                name="integer",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, bias, table: (
                        s * _gather(bias, h) + _gather(table, q - kv + 4)
                    ),
                    (
                        torch.linspace(0.5, 2.0, 4),
                        torch.arange(-4, 5, dtype=torch.int32),
                    ),
                ),
                name="captured",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, scale: s * scale,
                    (torch.tensor(0.25),),
                ),
                name="rank0_capture",
            ),
        ],
    )
    def test_score_lowering_matches_eager(self, score_mod, captures):
        buffers = [
            _FakeNode(tuple(capture.shape), capture.stride(), capture.dtype)
            for capture in captures
        ]
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_score_graph(
                _score_graph(score_mod, *captures), buffers
            )
        self.assertIsNotNone(program, reason)
        self.assertEqual(program.buffer_count, len(captures))
        for score, b, h, q, kv in (
            (1.5, 0, 0, 3, 1),
            (-0.75, 1, 3, 2, 5),
            (2.25, 1, 2, 4, 4),
            (-3.0, 0, 1, 0, 3),
        ):
            expected = score_mod(
                torch.tensor(score),
                *(torch.tensor(index, dtype=torch.int32) for index in (b, h, q, kv)),
                *captures,
            ).item()
            actual = _evaluate_score_program(program, score, b, h, q, kv, captures)
            if math.isinf(expected):
                self.assertEqual(actual, expected)
            else:
                self.assertAlmostEqual(actual, expected, places=4)

    @parametrize(
        "score_mod,captures,dtypes,expected",
        [
            subtest(
                (lambda s, b, h, q, kv: torch.erf(s), (), (), "erf"),
                name="unsupported_op",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv: torch.floor_divide(s, 2.0),
                    (),
                    (),
                    "floor division supports integers only",
                ),
                name="float_floor_divide",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, table: s + _gather(table, q),
                    (torch.zeros(8),),
                    (torch.int64,),
                    "must be float32, bfloat16, float16, or int32",
                ),
                name="int64_capture",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, table: s * table,
                    (torch.zeros(8),),
                    (torch.float32,),
                    "must be indexed to scalars",
                ),
                name="unindexed_capture",
            ),
        ],
    )
    def test_score_lowering_rejects(self, score_mod, captures, dtypes, expected):
        buffers = [
            _FakeNode(tuple(capture.shape), capture.stride(), dtype)
            for capture, dtype in zip(captures, dtypes)
        ]
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_score_graph(
                _score_graph(score_mod, *captures), buffers
            )
        self.assertIsNone(program)
        self.assertIn(expected, reason)

    def test_appends_score_mod_choice(self):
        bias = torch.zeros(64)
        inputs = _supported_fake_forward_inputs()
        inputs["subgraph"] = SimpleNamespace(
            graph_module=_score_graph(
                lambda s, b, h, q, kv, bias: (
                    torch.tanh(s / 30.0) * 30.0 + _gather(bias, h)
                ),
                bias,
            )
        )
        inputs["score_mod_other_buffers"] = [_FakeNode((64,), (1,), torch.float32)]
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNotNone(kwargs, reason)
        self.assertTrue(kwargs["SCORE_PROGRAM"])
        self.assertEqual(kwargs["SCORE_BUFFER_COUNT"], 1)
        self.assertEqual(kwargs["SCORE_BUFFER_SHAPES"], ((64,),))

    def test_captured_buffer_slots_are_shared(self):
        document = torch.zeros(512, dtype=torch.int32)
        tables = [torch.zeros(64) for _ in range(4)]
        inputs = _supported_fake_forward_inputs()
        inputs["mask_graph"] = SimpleNamespace(
            graph_module=make_fx(lambda b, h, q, kv, doc: _gather(doc, q) >= kv)(
                *(torch.tensor(0, dtype=torch.int32) for _ in range(4)), document
            )
        )
        inputs["mask_mod_other_buffers"] = [_FakeNode((512,), (1,), torch.int32)]
        inputs["subgraph"] = SimpleNamespace(
            graph_module=_score_graph(
                lambda s, b, h, q, kv, t0, t1, t2, t3: (
                    s
                    + _gather(t0, h)
                    + _gather(t1, h)
                    + _gather(t2, h)
                    + _gather(t3, h)
                ),
                *tables,
            )
        )
        inputs["score_mod_other_buffers"] = [
            _FakeNode((64,), (1,), torch.float32) for _ in tables
        ]
        kwargs, reason = _fake_choice_result(inputs)
        self.assertIsNone(kwargs)
        self.assertIn("at most 4 captured mask_mod/score_mod buffers", reason)

    def test_backward_score_mod_capture_grad_mask(self):
        def mask(fn, count):
            return flex_flydsl_backward._score_mod_capture_grad_mask(
                SimpleNamespace(graph_module=torch.fx.symbolic_trace(fn)), count
            )

        self.assertEqual(mask(lambda s, g: (s * g, None, None, None, None), 1), [False])
        self.assertEqual(
            mask(lambda s, g: (s * g, None, None, None, None, None, g), 2),
            [False, True],
        )
        self.assertIsNone(mask(lambda s, g: (s * g, None, None, None, None, g), 0))

    @parametrize(
        "score_mod,shape,expected",
        [
            subtest(
                (
                    lambda s, b, h, q, kv, slope: s + _gather(slope, h) * (kv - q),
                    (4,),
                    ("invariant", (("h",),)),
                ),
                name="invariant",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, bias: s + _gather(bias, kv, h),
                    (256, 4),
                    ("key", (("kv",), ("h",))),
                ),
                name="key",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, scale: s * _gather(scale, q),
                    (256,),
                    ("query", (("q",),)),
                ),
                name="query",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, table: s + _gather(table, h, q - kv + 255),
                    (4, 511),
                    ("relative", (("h",), ("rel", 1, 255))),
                ),
                name="relative",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, table: s + _gather(table, 300 - (q - kv)),
                    (600,),
                    ("relative", (("rel", -1, 300),)),
                ),
                name="relative_flipped",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, bias: s + _gather(bias, h, q, kv),
                    (4, 256, 256),
                    ("full", (("h",), ("q",), ("kv",))),
                ),
                name="full",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, bias: s + _gather(bias, q, kv),
                    (256, 256),
                    "indexed by every batch and head dimension",
                ),
                name="full_shared_heads",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, table: s + _gather(table, q // 2),
                    (128,),
                    "indices must be b, h, q_idx, kv_idx",
                ),
                name="computed_index",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, table: s + _gather(table, q - kv + 255, q),
                    (511, 256),
                    "mixes relative and absolute positions",
                ),
                name="mixed",
            ),
            subtest(
                (
                    lambda s, b, h, q, kv, bias: s + _gather(bias, kv),
                    (128,),
                    "must have size 256",
                ),
                name="size_mismatch",
            ),
        ],
    )
    def test_classify_capture_gradient(self, score_mod, shape, expected):
        capture = torch.zeros(shape)
        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_score_graph(
                _score_graph(score_mod, capture),
                [_FakeNode(shape, _contiguous_stride(shape), torch.float32)],
            )
            self.assertIsNotNone(program, reason)
            capture_grad, reason = classify_capture_gradient(
                program, 0, batch=1, heads=4, sequence_length=256
            )
        if isinstance(expected, str):
            self.assertIsNone(capture_grad)
            self.assertIn(expected, reason)
        else:
            self.assertIsNotNone(capture_grad, reason)
            self.assertEqual((capture_grad.kind, capture_grad.dims), expected)

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
                    {"kv_num_blocks": _FakeNode([1, 1, 4], [4, 4, 1], torch.float32)},
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


class TestFlyDSLFlexAttention(_FlyDSLFlexAttentionRuntimeMixin, TestCase):
    @parametrize(
        "qk_dim,mask_kind,dq_accum_fp32",
        [
            subtest((128, "dense", True), name="d128_dense_fp32"),
            subtest((192, "causal", False), name="d192_causal_bf16"),
            subtest((192, "document", True), name="d192_document_fp32"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward(self, device, qk_dim, mask_kind, dq_accum_fp32):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, qk_dim=qk_dim, seed=23)
        block_mask = None
        if mask_kind != "dense":
            if mask_kind == "causal":

                def mask_mod(b, h, q, kv):
                    return q >= kv

            else:
                document_ids = (
                    torch.arange(256, device=device, dtype=torch.int32) // 128
                )
                document_starts = torch.tensor(
                    [0, 128], device=device, dtype=torch.int32
                )

                def mask_mod(b, h, q, kv):
                    return (q >= kv) & (kv >= document_starts[document_ids[q]])

            block_mask = create_block_mask(
                mask_mod, 1, 1, 256, 256, device=device, BLOCK_SIZE=128
            )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=dq_accum_fp32,
        )

    @parametrize(
        "score_kind,mask_kind,qk_dim",
        [
            subtest(("relative", "dense", 128), name="relative_dense"),
            subtest(("alibi", "causal", 128), name="alibi_causal"),
            subtest(("softcap", "document", 192), name="softcap_document_d192"),
            subtest(("head_bias", "causal", 192), name="head_bias_causal_d192"),
            subtest(("where_inf", "dense", 128), name="where_inf_dense"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_score_mod_backward(self, device, score_kind, mask_kind, qk_dim):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, q_heads=4, qk_dim=qk_dim, seed=31)
        block_mask = None
        if mask_kind == "causal":
            block_mask = create_block_mask(
                lambda b, h, q, kv: q >= kv, 1, 1, 256, 256, device=device
            )
        elif mask_kind == "document":
            document_ids = torch.arange(256, device=device, dtype=torch.int32) // 96
            block_mask = create_block_mask(
                lambda b, h, q, kv: (q >= kv) & (document_ids[q] == document_ids[kv]),
                1,
                1,
                256,
                256,
                device=device,
                BLOCK_SIZE=128,
            )
        score_mod, captures = _score_mod_case(score_kind, device=device, heads=4)
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            score_mod=score_mod,
            score_captures=captures,
        )

    @parametrize(
        "score_kind,mask_kind",
        [
            subtest(("trainable_slope", "dense"), name="slope_dense"),
            subtest(("trainable_key_bias", "causal"), name="key_bias_causal"),
            subtest(("trainable_query_scale", "dense"), name="query_scale_dense"),
            subtest(("trainable_relative_table", "causal"), name="relative_causal"),
            subtest(("trainable_full_bias", "dense"), name="full_bias_dense"),
        ],
    )
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_trainable_score_mod_capture_backward(
        self, device, score_kind, mask_kind
    ):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, q_heads=4, seed=32)
        block_mask = None
        if mask_kind == "causal":
            block_mask = create_block_mask(
                lambda b, h, q, kv: q >= kv, 1, 1, 256, 256, device=device
            )
        score_mod, captures = _score_mod_case(score_kind, device=device, heads=4)
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
            score_mod=score_mod,
            score_captures=captures,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_computed_index_trainable_capture_uses_triton_backward(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, q_heads=4, seed=34)
        score_mod, captures = _score_mod_case(
            "trainable_clamped_table", device=device, heads=4
        )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=None,
            dq_accum_fp32=True,
            score_mod=score_mod,
            score_captures=captures,
            expect_flydsl_backward=False,
        )

    @parametrize(
        "score_kind,seq",
        [
            subtest(("alibi", 4), name="alibi_decode"),
            subtest(("softcap", 1), name="softcap_single_query"),
            subtest(("head_bias", 256), name="head_bias_gqa_prefill"),
        ],
    )
    def test_gfx950_score_mod_forward_gqa(self, device, score_kind, seq):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, q_heads=8, kv_heads=2, seq_q=seq, seq_kv=256, seed=33
        )
        make_score_mod, captures = _score_mod_case(score_kind, device=device, heads=8)
        self._compare_forward(
            query,
            key,
            value,
            enable_gqa=True,
            score_mod=make_score_mod(*captures),
        )

    @parametrize("mask_kind", ("causal", "document"))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_shared_dq_slots(self, device, mask_kind):
        # 64 batch-heads use 4 dQ slots for 8 key owners, so owners share slots
        # and skip query blocks that an earlier owner did not write.
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device, batch=4, q_heads=16, seq_q=1024, seed=35
        )
        document_ids = torch.arange(1024, device=device, dtype=torch.int32) // 300

        def mask_mod(b, h, q, kv):
            if mask_kind == "causal":
                return q >= kv
            return (q >= kv) & (document_ids[q] == document_ids[kv])

        block_mask = create_block_mask(mask_mod, 1, 1, 1024, 1024, device=device)
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @parametrize("qk_dim", (128, 192))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_long_causal_dq_matches_fp64(self, device, qk_dim):
        self._require_runtime()
        sequence_length = 16384
        # Multiple KV owners contribute to each private dQ partition.
        query, key, value = _make_qkv(
            device=device, q_heads=16, seq_q=sequence_length, qk_dim=qk_dim, seed=41
        )
        query.requires_grad_()
        torch.manual_seed(42)
        grad_output = torch.randn_like(value)
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: query_index >= key_index,
            1,
            1,
            sequence_length,
            sequence_length,
            device=device,
            BLOCK_SIZE=128,
        )
        compiled = torch.compile(
            lambda q, k, v: flex_attention(
                q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLYDSL"}
            ),
            fullgraph=True,
        )
        output, forward_code = run_and_get_code(compiled, query, key, value)
        gradients, backward_code = run_and_get_code(
            lambda: torch.autograd.grad(output, query, grad_output)
        )
        self.assertIn("build_flex_attn_fwd_module", "\n".join(forward_code))
        self.assertIn("build_flex_attn_bwd_module", "\n".join(backward_code))
        grad_query = gradients[0]
        reference = torch.empty_like(query, dtype=torch.float64)
        scale = qk_dim**-0.5
        # Chunk rows and heads to avoid materializing the full S x S matrix.
        with torch.no_grad():
            key_positions = torch.arange(sequence_length, device=device)
            for head in range(query.size(1)):
                key64 = key[:, head].double()
                value64 = value[:, head].double()
                for start in range(0, sequence_length, 256):
                    end = start + 256
                    scores = (
                        query[:, head, start:end].double() @ key64.transpose(-2, -1)
                    ) * scale
                    rows = torch.arange(start, end, device=device)
                    scores.masked_fill_(
                        rows[:, None] < key_positions[None, :], -float("inf")
                    )
                    probability = scores.softmax(-1)
                    grad_probability = grad_output[
                        :, head, start:end
                    ].double() @ value64.transpose(-2, -1)
                    grad_score = probability * (
                        grad_probability
                        - (probability * grad_probability).sum(-1, keepdim=True)
                    )
                    reference[:, head, start:end] = (grad_score @ key64) * scale
        error = grad_query.double() - reference
        self.assertEqual(grad_query.double(), reference, atol=0.01, rtol=0.02)
        self.assertLess((error.norm() / reference.norm()).item(), 0.003)
        late_rows = sequence_length // 2
        self.assertLess(
            (
                error[:, :, late_rows:].norm() / reference[:, :, late_rows:].norm()
            ).item(),
            0.003,
        )

    @parametrize("query_length", (4, 256))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gqa_backward_falls_back_to_triton(self, device, query_length):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            q_heads=4,
            kv_heads=2,
            seq_q=query_length,
            seq_kv=256,
            seed=43,
        )
        grad_output = torch.randn_like(query)
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: (
                query_index + 256 - query_length >= key_index
            ),
            1,
            1,
            query_length,
            256,
            device=device,
            BLOCK_SIZE=128,
        )

        def run(backend):
            inputs = tuple(
                tensor.detach().clone().requires_grad_()
                for tensor in (query, key, value)
            )
            compiled = torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    enable_gqa=True,
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )
            output, forward_code = run_and_get_code(compiled, *inputs)
            gradients, backward_code = run_and_get_code(
                lambda: torch.autograd.grad(output, inputs, grad_output)
            )
            return output, gradients, "\n".join(forward_code), "\n".join(backward_code)

        output, gradients, forward_code, backward_code = run("FLYDSL")
        reference, reference_gradients, _, reference_backward_code = run("TRITON")
        self.assertIn("build_flex_attn_fwd_module", forward_code)
        self.assertNotIn("build_flex_attn_bwd_module", backward_code)
        self.assertNotIn("build_flex_attn_bwd_module", reference_backward_code)
        self.assertIn("triton", backward_code)
        self.assertEqual(output, reference, atol=0.03, rtol=0.02)
        for gradient, expected in zip(gradients, reference_gradients):
            self.assertEqual(gradient, expected, atol=0.03, rtol=0.02)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_lse_aux_backward_falls_back_to_triton(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seed=44)
        grad_output = torch.randn_like(value)
        grad_lse = torch.randn(query.shape[:3], device=device, dtype=torch.float32)
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: query_index >= key_index,
            1,
            1,
            256,
            256,
            device=device,
            BLOCK_SIZE=128,
        )

        def run(backend):
            inputs = tuple(
                tensor.detach().clone().requires_grad_()
                for tensor in (query, key, value)
            )
            compiled = torch.compile(
                lambda q, k, v: flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    return_aux=AuxRequest(lse=True),
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )
            (output, aux), forward_code = run_and_get_code(compiled, *inputs)
            gradients, backward_code = run_and_get_code(
                lambda: torch.autograd.grad(
                    (output, aux.lse), inputs, (grad_output, grad_lse)
                )
            )
            return gradients, "\n".join(forward_code), "\n".join(backward_code)

        gradients, forward_code, backward_code = run("FLYDSL")
        reference, _, _ = run("TRITON")
        self.assertIn("build_flex_attn_fwd_module", forward_code)
        self.assertNotIn("build_flex_attn_bwd_module", backward_code)
        self.assertIn("triton", backward_code)
        for gradient, expected in zip(gradients, reference):
            self.assertEqual(gradient, expected, atol=0.03, rtol=0.02)

    @parametrize("qk_dim", (128, 192))
    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_strided_sliding_window_backward(self, device, qk_dim):
        self._require_runtime()
        query, key, value = _make_qkv(
            device=device,
            batch=2,
            q_heads=4,
            seq_q=512,
            qk_dim=qk_dim,
            seed=45,
            transposed=True,
        )
        for tensor in (query, key, value):
            self.assertFalse(tensor.is_contiguous())
        block_mask = create_block_mask(
            lambda batch, head, query_index, key_index: (
                (query_index >= key_index) & (query_index - key_index < 256)
            ),
            2,
            1,
            512,
            512,
            device=device,
            BLOCK_SIZE=128,
        )
        self._compare_backward(
            query, key, value, block_mask=block_mask, dq_accum_fp32=True
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    def test_gfx950_backward_respects_block_mask_metadata(self, device):
        self._require_runtime()
        query, key, value = _make_qkv(device=device, seed=25)
        kv_num_blocks = torch.tensor([[[0, 1]]], device=device, dtype=torch.int32)
        kv_indices = torch.tensor([[[[0], [1]]]], device=device, dtype=torch.int32)
        full_kv_num_blocks = torch.zeros_like(kv_num_blocks)
        full_kv_indices = torch.zeros_like(kv_indices)

        def causal(b, h, q, kv):
            return q >= kv

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=causal,
            seq_lengths=(256, 256),
            compute_q_blocks=True,
        )
        self._compare_backward(
            query,
            key,
            value,
            block_mask=block_mask,
            dq_accum_fp32=True,
        )

    @parametrize(
        "q_heads,kv_heads,seq,captured",
        [
            (4, 2, 512, False),
            (64, 8, 2048, False),
            (8, 2, 4096, True),
            (16, 4, 8192, True),
        ],
    )
    def test_gfx950_forward_full_partial_gqa_and_empty_q_block(
        self, device, q_heads, kv_heads, seq, captured
    ):
        self._require_runtime()

        head_dim = 192 if captured else 128
        query, key, value = _make_qkv(
            device=device,
            q_heads=q_heads,
            kv_heads=kv_heads,
            seq_q=seq,
            qk_dim=head_dim,
        )
        kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 0] * (seq // 512)]], device=device, dtype=torch.int32
        )
        kv_indices = torch.tensor(
            [[[[0], [1], [2], [0]] * (seq // 512)]], device=device, dtype=torch.int32
        )
        full_kv_num_blocks = torch.tensor(
            [[[0, 1, 1, 0] * (seq // 512)]], device=device, dtype=torch.int32
        )
        full_kv_indices = torch.tensor(
            [[[[0]] * (seq // 128)]], device=device, dtype=torch.int32
        )

        if captured:
            offsets = torch.zeros(seq, device=device, dtype=torch.int32)

            def causal(b, h, q_idx, kv_idx):
                del b, h
                return q_idx >= kv_idx + offsets[q_idx]

        else:

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
        for start in range(384, seq, 512):
            self.assertEqual(output[:, :, start : start + 128].abs().max().item(), 0.0)
            self.assertTrue(torch.isneginf(aux.lse[:, :, start : start + 128]).all())
            self.assertTrue(
                torch.isneginf(aux.max_scores[:, :, start : start + 128]).all()
            )

    @parametrize("seq,document_starts", [(128, [0, 29, 83]), (8192, [0, 1701, 5387])])
    def test_gfx950_forward_unequal_packed_documents(
        self, device, seq, document_starts
    ):
        self._require_runtime()
        starts = torch.tensor(document_starts, device=device, dtype=torch.int32)
        document_ids = torch.bucketize(
            torch.arange(seq, device=device), starts[1:], right=True
        ).to(torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (kv_idx >= starts[document_ids[q_idx]])

        self._compare_created_mask(
            mask_mod,
            device=device,
            q_heads=16,
            kv_heads=4,
            seq=seq,
            qk_dim=192,
            transposed=True,
            return_aux=True,
        )

    @parametrize("seq_q", [1, 3, 4, 8, 16])
    def test_gfx950_public_api_per_kv_head_decode(self, device, seq_q):
        self._require_runtime()
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
                    lambda b, h, q, kv: (q + 2044 >= kv) & (q + 2044 - kv < 96),
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
                    lambda b, h, q, kv: (
                        (torch.add(q, 64, alpha=2) >= kv)
                        & ((torch.add(q, 64, alpha=2) - kv) % (2 + h) == 0)
                    ),
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

    @parametrize("seq", [128, 512])
    def test_gfx950_public_api_composed_document_mask(self, device, seq):
        self._require_runtime()
        document_ids = torch.arange(seq, device=device, dtype=torch.int32) // (seq // 2)
        document_starts = torch.tensor([0, seq // 2], device=device, dtype=torch.int32)
        document_causal = and_masks(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            lambda b, h, q_idx, kv_idx: kv_idx >= document_starts[document_ids[q_idx]],
        )
        self._compare_created_mask(
            document_causal, device=device, seq=seq, seed=4, return_aux=True
        )

    def test_gfx950_public_api_sliding_window_mask_lowering(self, device):
        self._require_runtime()

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_utils import (
            is_sliding_window_mask_program,
        )

        def check_match(mask_mod, expected):
            matched = []

            def check_window_lowering(*args, **kwargs):
                program, reason = lower_flydsl_mask_graph(*args, **kwargs)
                self.assertIsNotNone(program, reason)
                matched.append(
                    is_sliding_window_mask_program(
                        program.instructions, program.output, program.buffer_strides
                    )
                )
                return program, reason

            torch._dynamo.reset()
            with (
                torch._inductor.config.patch({"fx_graph_cache": False}),
                mock.patch(
                    "torch._inductor.kernel.flex.flex_flydsl_attention.lower_flydsl_mask_graph",
                    side_effect=check_window_lowering,
                ) as lower,
            ):
                self._compare_created_mask(
                    mask_mod, device=device, seed=3, return_aux=True
                )
                lower.assert_called_once()
            self.assertEqual(matched, [expected])

        # The two spellings lower to different programs, so both need covering:
        # and_masks() seeds a const_bool True that the bare lambda never emits.
        check_match(lambda b, h, q, kv: (q >= kv) & (q - kv < 96), True)
        check_match(
            and_masks(
                lambda b, h, q, kv: q >= kv,
                lambda b, h, q, kv: q - kv < 96,
            ),
            True,
        )
        # Unbounded causal must not match: it is the ragged-tail case the reversed
        # query-block dispatch exists for, and matching here would silently disable it.
        check_match(lambda b, h, q, kv: q >= kv, False)

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


instantiate_device_type_tests(TestFlyDSLFlexAttention, globals(), only_for=("cuda",))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
