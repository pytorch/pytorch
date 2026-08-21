# Owner(s): ["module: inductor"]
import operator
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._inductor.kernel.flex import flex_flydsl_attention
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _can_use_flydsl_flex_attention_backward,
    flex_flydsl_backward_template,
    is_causal_mask_graph,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _score_graph(fn) -> SimpleNamespace:
    return SimpleNamespace(graph_module=torch.fx.symbolic_trace(fn))


def _identity_score_graph() -> SimpleNamespace:
    return _score_graph(lambda score, b, h, m, n: score)


def _fake_query(dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(get_dtype=lambda: dtype)


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


def _gfx950_available() -> bool:
    return (
        runtime_available()
        and torch.cuda.is_available()
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
            ":", 1
        )[0]
        == "gfx950"
    )


_GATE_CASES = (
    ("runtime", False, "6.0.0", torch.bfloat16, False, "unavailable"),
    ("rocm", True, None, torch.bfloat16, False, "ROCm"),
    ("dtype", True, "6.0.0", torch.float16, False, "bf16"),
    ("score_mod", True, "6.0.0", torch.bfloat16, True, "identity score_mod"),
)

_MASK_CASES = (
    ("d128_dense", 128, "dense", None),
    ("d128_causal", 128, "causal", 0.07),
    ("d128_alpha", 128, "alpha", None),
    ("d192_dense", 192, "dense", 0.07),
    ("d192_causal", 192, "causal", 0.07),
    ("d192_window", 192, "window", None),
    ("d192_per_head", 192, "per_head", None),
    ("d192_two_buffers", 192, "two_buffer_document", None),
)

_DOCUMENT_CASES = (
    (
        "transposed",
        2,
        2,
        256,
        True,
        ((128, 128), (64, 128, 64)),
        0.07,
    ),
    ("b1_batched", 1, 2, 4096, False, ((2048, 2048),), None),
    ("standard_width", 1, 1, 4096, False, None, 0.07),
)


@instantiate_parametrized_tests
class TestFlexFlyDSLGates(TestCase):
    def test_template_registered(self):
        self.assertIs(
            FlyDSLTemplate.all_templates["flex_flydsl_backward"],
            flex_flydsl_backward_template,
        )

    @parametrize("case", _GATE_CASES, name_fn=lambda case: case[0])
    def test_gate_declines_unsupported(self, case):
        _, runtime, hip, dtype, nontrivial_score, expected = case
        score_graph = (
            _score_graph(lambda score, b, h, m, n: score * 2.0)
            if nontrivial_score
            else _identity_score_graph()
        )
        with (
            mock.patch.object(
                flex_flydsl_attention, "runtime_available", return_value=runtime
            ),
            mock.patch.object(torch.version, "hip", hip),
        ):
            can_use, reason = _can_use_flydsl_flex_attention_backward(
                score_graph,
                _identity_score_graph(),
                _fake_query(dtype),
                key=_fake_query(dtype),
                value=_fake_query(dtype),
            )
        self.assertFalse(can_use)
        self.assertIn(expected, reason)


class TestFlexFlyDSLMaskLowering(TestCase):
    def test_add_and_sub_alpha(self):
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

    def test_fractional_offset_is_not_causal(self):
        graph_module = torch.fx.symbolic_trace(lambda b, h, q, kv: q + -0.5 >= kv)
        self.assertFalse(is_causal_mask_graph(graph_module))
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)


@unittest.skipUnless(_gfx950_available(), "requires FlyDSL on ROCm gfx950")
@instantiate_parametrized_tests
class TestFlexFlyDSLRuntime(TestCase):
    def _make_inputs(
        self,
        *,
        batch=1,
        heads=2,
        seq=256,
        qk_dim=192,
        v_dim=128,
        transposed=False,
        seed=0,
    ):
        torch.manual_seed(seed)

        def make(dim):
            shape = (batch, seq, heads, dim) if transposed else (batch, heads, seq, dim)
            tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            return tensor.transpose(1, 2) if transposed else tensor

        return make(qk_dim), make(qk_dim), make(v_dim), make(v_dim)

    def _make_mask(self, kind, *, heads=2, seq=256):
        mask_heads = 1
        if kind == "dense":

            def mask_mod(b, h, q, kv):
                del b, h, kv
                return q >= 0

        elif kind == "causal":

            def mask_mod(b, h, q, kv):
                del b, h
                return q >= kv

        elif kind == "alpha":

            def mask_mod(b, h, q, kv):
                del b, h
                return torch.add(q, 64, alpha=2) >= kv

        elif kind == "window":

            def mask_mod(b, h, q, kv):
                del b, h
                return (q >= kv) & (q - kv < 96)

        elif kind == "per_head":
            mask_heads = heads

            def mask_mod(b, h, q, kv):
                del b
                return (q >= kv) & (q - kv < 64 + h * 64)

        elif kind == "two_buffer_document":
            document_ids = torch.arange(seq, device="cuda", dtype=torch.int32) // 128
            document_starts = torch.arange(
                0, seq, 128, device="cuda", dtype=torch.int32
            )

            def mask_mod(b, h, q, kv):
                del b, h
                return (q >= kv) & (kv >= document_starts[document_ids[q]])

        else:
            raise AssertionError(f"unknown mask kind {kind}")

        return create_block_mask(
            mask_mod,
            1,
            mask_heads,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )

    def _compare_backward(
        self,
        q,
        k,
        v,
        grad_out,
        *,
        block_mask,
        scale=None,
        atol=0.1,
        rtol=0.05,
    ):
        def clone(tensor):
            result = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return result.copy_(tensor).requires_grad_(True)

        def run(backend):
            inputs = tuple(clone(tensor) for tensor in (q, k, v))
            torch._dynamo.reset()
            compiled = torch.compile(
                lambda q_, k_, v_: flex_attention(
                    q_,
                    k_,
                    v_,
                    block_mask=block_mask,
                    scale=scale,
                    kernel_options={"BACKEND": backend},
                ),
                fullgraph=True,
            )
            output = compiled(*inputs)
            grads = torch.autograd.grad(output, inputs, grad_out)
            torch.cuda.synchronize()
            return grads

        flydsl = run("FLYDSL")
        triton = run("TRITON")
        for name, actual, expected in zip(
            ("dQ", "dK", "dV"), flydsl, triton, strict=True
        ):
            torch.testing.assert_close(
                actual, expected, atol=atol, rtol=rtol, msg=f"{name} mismatch"
            )
        return flydsl

    @parametrize("case", _MASK_CASES, name_fn=lambda case: case[0])
    def test_masks_match_triton(self, case):
        _, qk_dim, mask_kind, scale = case
        q, k, v, grad_out = self._make_inputs(qk_dim=qk_dim, seed=2)
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=self._make_mask(mask_kind),
            scale=scale,
        )

    def test_padded_positive_strides_match_triton(self):
        batch, heads, seq = 1, 2, 256
        torch.manual_seed(32)

        def make(dim, row_stride, head_padding, batch_padding):
            head_stride = seq * row_stride + head_padding
            tensor = torch.empty_strided(
                (batch, heads, seq, dim),
                (
                    heads * head_stride + batch_padding,
                    head_stride,
                    row_stride,
                    1,
                ),
                device="cuda",
                dtype=torch.bfloat16,
            )
            return tensor.normal_()

        q = make(192, 208, 64, 128)
        k = make(192, 216, 32, 64)
        v = make(128, 136, 64, 128)
        grad_out = make(128, 144, 32, 64)
        self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=self._make_mask("causal"),
        )

    def test_default_block_mask_matches_triton(self):
        q, k, v, grad_out = self._make_inputs(qk_dim=128, seed=3)
        self._compare_backward(q, k, v, grad_out, block_mask=None)

    def test_non_dsv3_shape_and_mask_match_triton(self):
        batch, heads, seq = 2, 3, 384
        q, k, v, grad_out = self._make_inputs(
            batch=batch,
            heads=heads,
            seq=seq,
            qk_dim=128,
            seed=6,
        )

        def mask_mod(b, h, q_idx, kv_idx):
            del b
            return (q_idx >= kv_idx) & ((q_idx - kv_idx) % (2 + h) == 0)

        block_mask = create_block_mask(
            mask_mod,
            batch,
            heads,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        self._compare_backward(q, k, v, grad_out, block_mask=block_mask)

    def test_fractional_causal_mask_is_rejected(self):
        q, k, v, grad_out = self._make_inputs(qk_dim=128, seed=5)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        def mask_mod(b, h, q_idx, kv_idx):
            del b, h
            return q_idx + -0.5 >= kv_idx

        block_mask = create_block_mask(
            mask_mod,
            1,
            1,
            q.size(-2),
            k.size(-2),
            device="cuda",
            BLOCK_SIZE=128,
        )
        compiled = torch.compile(
            lambda q_, k_, v_: flex_attention(
                q_,
                k_,
                v_,
                block_mask=block_mask,
                kernel_options={"BACKEND": "FLYDSL"},
            ),
            fullgraph=True,
        )
        with self.assertRaisesRegex(
            RuntimeError, "scalar constant -0.5 is unsupported"
        ):
            compiled(q, k, v).backward(grad_out)

    @parametrize("case", _DOCUMENT_CASES, name_fn=lambda case: case[0])
    def test_document_masks_match_triton(self, case):
        name, batch, heads, seq, transposed, partitions, scale = case
        q, k, v, grad_out = self._make_inputs(
            batch=batch,
            heads=heads,
            seq=seq,
            transposed=transposed,
            seed=4,
        )
        if partitions is None:
            document_size = 512
            document_end = (
                torch.arange(seq, device="cuda", dtype=torch.int32) // document_size + 1
            ) * document_size - 1

            def mask_mod(b, h, q_idx, kv_idx):
                del b, h
                return (q_idx >= kv_idx) & (q_idx <= document_end[kv_idx])

            mask_batch = 1
        else:
            rows = []
            for lengths in partitions:
                end = 0
                row = []
                for length in lengths:
                    end += length
                    row.extend([end - 1] * length)
                self.assertEqual(end, seq)
                rows.append(row)
            document_end = torch.tensor(rows, device="cuda", dtype=torch.int32)

            def mask_mod(b, h, q_idx, kv_idx):
                del h
                return (q_idx >= kv_idx) & (q_idx <= document_end[b, kv_idx])

            mask_batch = batch

        block_mask = create_block_mask(
            mask_mod,
            mask_batch,
            1,
            seq,
            seq,
            device="cuda",
            BLOCK_SIZE=128,
        )
        if name == "standard_width":
            self.assertEqual(block_mask.kv_indices.shape[-1], seq // 128)
            self.assertEqual(block_mask.full_kv_indices.shape[-1], seq // 128)
        result = self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=block_mask,
            scale=scale,
            atol=0.12 if name == "standard_width" else 0.1,
            rtol=0.06 if name == "standard_width" else 0.05,
        )
        if transposed:
            for grad, source in zip(result, (q, k, v), strict=True):
                self.assertEqual(grad.stride(), source.stride())

    def test_gqa_is_rejected(self):
        batch, seq = 1, 256
        q = torch.randn(
            batch,
            4,
            seq,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            batch,
            2,
            seq,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        v = torch.randn_like(k, requires_grad=True)
        grad_out = torch.randn_like(q)
        block_mask = self._make_mask("dense", heads=4, seq=seq)
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda q_, k_, v_: flex_attention(
                q_,
                k_,
                v_,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options={"BACKEND": "FLYDSL"},
            ),
            fullgraph=True,
        )
        with self.assertRaisesRegex(RuntimeError, "MHA with matching Q/K"):
            compiled(q, k, v).backward(grad_out)


if __name__ == "__main__":
    run_tests()
