# Owner(s): ["module: inductor"]
import operator
from importlib.util import find_spec
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl.flydsl_template import FlyDSLTemplate
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._inductor.kernel.flex import flex_flydsl_attention
from torch._inductor.kernel.flex.flex_flydsl_attention import (
    _fits_u32_head_slice,
    _get_flydsl_flex_attention_backward_config,
    _get_supported_bhsd_stride,
    _is_contiguous_shape_stride,
    flex_flydsl_backward_template,
)
from torch._inductor.kernel.flex.flex_flydsl_mask import lower_flydsl_mask_graph
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


def _score_graph(fn) -> SimpleNamespace:
    return SimpleNamespace(graph_module=torch.fx.symbolic_trace(fn))


def _identity_score_graph() -> SimpleNamespace:
    return _score_graph(lambda score, b, h, m, n: score)


def _fake_query(dtype: torch.dtype) -> SimpleNamespace:
    return SimpleNamespace(get_dtype=lambda: dtype)


class _FakeNode:
    def __init__(
        self,
        size,
        stride,
        dtype=torch.bfloat16,
        device=torch.device("cuda", 0),
    ):
        self._size = list(size)
        self._stride = list(stride)
        self._dtype = dtype
        self._device = device

    def get_size(self):
        return self._size

    def get_stride(self):
        return self._stride

    def get_dtype(self):
        return self._dtype

    def get_device(self):
        return self._device


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


def _supported_fake_backward_inputs():
    qk_shape = (1, 2, 256, 128)
    value_shape = (1, 2, 256, 128)
    count_shape = (1, 1, 2)
    index_shape = (1, 1, 2, 2)

    def node(shape, dtype=torch.bfloat16, *, device=torch.device("cuda", 0)):
        return _FakeNode(shape, _contiguous_stride(shape), dtype, device)

    return {
        "fw_subgraph": _identity_score_graph(),
        "mask_graph": _score_graph(lambda b, h, q, kv: q >= kv),
        "query": node(qk_shape),
        "key": node(qk_shape),
        "value": node(value_shape),
        "out": node(value_shape),
        "grad_out": node(value_shape),
        "grad_logsumexp": None,
        "score_mod_other_buffers": [],
        "mask_mod_other_buffers": [],
        "kv_num_blocks": node(count_shape, torch.int32),
        "kv_indices": node(index_shape, torch.int32),
        "full_kv_num_blocks": node(count_shape, torch.int32),
        "full_kv_indices": node(index_shape, torch.int32),
        "scale": 128**-0.5,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
    }


def _backward_config_result(inputs):
    with (
        V.set_graph_handler(_fake_graph()),
        mock.patch.object(
            flex_flydsl_attention, "runtime_available", return_value=True
        ),
        mock.patch.object(torch.version, "hip", "6.0.0"),
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
    ("d128_empty_rows", 128, "empty_rows", None),
    ("d192_dense", 192, "dense", 0.07),
    ("d192_causal", 192, "causal", 0.07),
    ("d192_window", 192, "window", None),
    ("d192_per_head", 192, "per_head", None),
    ("d192_two_buffers", 192, "two_buffer_document", None),
    ("d192_empty_rows", 192, "empty_rows", None),
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
    ("reused_slots", 1, 1, 4352, False, ((2112, 2240),), 0.07),
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
            config, reason = _get_flydsl_flex_attention_backward_config(
                score_graph,
                _identity_score_graph(),
                _fake_query(dtype),
                key=_fake_query(dtype),
                value=_fake_query(dtype),
            )
        self.assertIsNone(config)
        self.assertIn(expected, reason)


@instantiate_parametrized_tests
class TestFlexFlyDSLConfig(TestCase):
    @parametrize("value", (None, 0, 1, "bf16", torch.float32))
    def test_invalid_dq_accumulation_option(self, value):
        with self.assertRaisesRegex(ValueError, "DQ_ACCUM_FP32 must be a bool"):
            flex_flydsl_attention.create_flydsl_flex_attention_backward_kernel(
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
                dq_accum_fp32=value,
            )

    @parametrize("mode", ("off", "strict", "warn_only"))
    def test_deterministic_backward(self, mode):
        with DeterministicGuard(mode != "off", warn_only=mode == "warn_only"):
            config, reason = _backward_config_result(_supported_fake_backward_inputs())
        self.assertIsNotNone(config, reason)

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
                ((2, 3, 4, 5), (20, 5, 1), False, False),
                name="rank_mismatch",
            ),
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

    @parametrize(
        "shape,stride,supported",
        [
            subtest(
                ((1, 128, 4096, 192), (0, 192, 24576, 1), True),
                name="size_one_zero_stride",
            ),
            subtest(
                ((2, 128, 4096, 192), (0, 192, 24576, 1), False),
                name="broadcast_batch",
            ),
            subtest(
                ((1, 128, 4096, 192), (-1, 192, 24576, 1), False),
                name="negative_size_one_stride",
            ),
        ],
    )
    def test_strided_bhsd_zero_stride(self, shape, stride, supported):
        with V.set_graph_handler(_fake_graph()):
            self.assertEqual(
                _get_supported_bhsd_stride(
                    _FakeNode(shape, stride), allow_strided=True
                ),
                stride if supported else None,
            )

    @parametrize(
        "metadata_name,replacement,expected_reason",
        [
            subtest(
                (
                    "kv_num_blocks",
                    _FakeNode((1, 1, 2), (2, 2, 1), torch.float32),
                    "requires int32 BlockMask metadata",
                ),
                name="dtype",
            ),
            subtest(
                (
                    "kv_indices",
                    _FakeNode(
                        (1, 1, 2, 2),
                        (4, 4, 2, 1),
                        torch.int32,
                        torch.device("cpu"),
                    ),
                    "requires BlockMask metadata on the query device",
                ),
                name="device",
            ),
            subtest(
                (
                    "full_kv_indices",
                    _FakeNode((1, 1, 2, 2), (8, 8, 4, 2), torch.int32),
                    "requires contiguous BlockMask metadata",
                ),
                name="stride",
            ),
        ],
    )
    def test_block_mask_metadata_validation(
        self, metadata_name, replacement, expected_reason
    ):
        inputs = _supported_fake_backward_inputs()
        inputs[metadata_name] = replacement
        config, reason = _backward_config_result(inputs)
        self.assertIsNone(config)
        self.assertIn(expected_reason, reason)

    def test_unrealized_metadata_is_rejected(self):
        inputs = _supported_fake_backward_inputs()
        with mock.patch.object(
            inputs["kv_indices"], "get_stride", side_effect=NotImplementedError
        ):
            config, reason = _backward_config_result(inputs)
        self.assertIsNone(config)
        self.assertIn("requires statically known BlockMask metadata strides", reason)

    def test_four_gib_check_is_per_head(self):
        with V.set_graph_handler(_fake_graph()):
            self.assertTrue(
                _fits_u32_head_slice(
                    _FakeNode(
                        (64, 4, 65536, 128),
                        _contiguous_stride((64, 4, 65536, 128)),
                    )
                )
            )
            self.assertFalse(
                _fits_u32_head_slice(
                    _FakeNode(
                        (1, 1, 1 << 24, 128),
                        _contiguous_stride((1, 1, 1 << 24, 128)),
                    )
                )
            )


@instantiate_parametrized_tests
class TestFlexFlyDSLMaskLowering(TestCase):
    def _require_flydsl(self):
        if find_spec("flydsl") is None:
            self.skipTest("requires the optional FlyDSL package")

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
        ],
    )
    def test_add_and_sub_alpha(self, mask_mod, q, kv, expected):
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
        program, reason = lower_flydsl_mask_graph(graph_module, ())
        self.assertIsNone(program)
        self.assertIn("scalar constant -0.5 is unsupported", reason)

    @parametrize("kind", ("document_start", "batched_document_end"))
    def test_document_mask_uses_block_list_traversal(self, kind):
        self._require_flydsl()
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            MASK_TRAVERSAL_BLOCK_LIST,
        )

        batch, sequence_length = 2, 256
        graph = torch.fx.Graph()
        batch_index = graph.placeholder("batch")
        graph.placeholder("head")
        query_index = graph.placeholder("query")
        kv_index = graph.placeholder("key_value")
        mask_buffer_0 = graph.placeholder("mask_buffer_0")
        mask_buffer_1 = (
            graph.placeholder("mask_buffer_1") if kind == "document_start" else None
        )
        mask = graph.call_function(torch.ops.aten.full.default, ([], True))
        causal = graph.call_function(torch.ops.aten.ge.Tensor, (query_index, kv_index))
        mask = graph.call_function(torch.ops.aten.bitwise_and.Tensor, (mask, causal))
        if kind == "document_start":
            document_id = graph.call_function(
                torch.ops.aten.index.Tensor, (mask_buffer_0, [query_index])
            )
            document_start = graph.call_function(
                torch.ops.aten.index.Tensor, (mask_buffer_1, [document_id])
            )
            in_document = graph.call_function(
                torch.ops.aten.ge.Tensor, (kv_index, document_start)
            )
            captures = (
                _FakeNode((sequence_length,), (1,), torch.int32),
                _FakeNode((2,), (1,), torch.int32),
            )
        else:
            document_end = graph.call_function(
                torch.ops.aten.index.Tensor,
                (mask_buffer_0, [batch_index, kv_index]),
            )
            in_document = graph.call_function(
                torch.ops.aten.le.Tensor, (query_index, document_end)
            )
            captures = (
                _FakeNode(
                    (batch, sequence_length),
                    (sequence_length, 1),
                    torch.int32,
                ),
            )
        mask = graph.call_function(
            torch.ops.aten.bitwise_and.Tensor, (mask, in_document)
        )
        graph.output(mask)

        with V.set_graph_handler(_fake_graph()):
            program, reason = lower_flydsl_mask_graph(
                torch.fx.GraphModule({}, graph),
                captures,
            )
        self.assertIsNotNone(program, reason)
        traversal, direct_range_kind = classify_mask_traversal(
            program.instructions,
            program.output,
            program.buffer_shapes,
        )
        self.assertEqual(traversal, MASK_TRAVERSAL_BLOCK_LIST)
        self.assertIsNone(direct_range_kind)

    def test_causal_mask_uses_direct_range_traversal(self):
        self._require_flydsl()
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            DIRECT_RANGE_CAUSAL,
            MASK_TRAVERSAL_DIRECT_RANGE,
        )

        program, reason = lower_flydsl_mask_graph(
            _mask_graph(lambda b, h, q, kv: q >= kv),
            (),
        )
        self.assertIsNotNone(program, reason)
        traversal, direct_range_kind = classify_mask_traversal(
            program.instructions,
            program.output,
            program.buffer_shapes,
        )
        self.assertEqual(traversal, MASK_TRAVERSAL_DIRECT_RANGE)
        self.assertEqual(direct_range_kind, DIRECT_RANGE_CAUSAL)

    def test_equivalent_causal_programs_use_direct_range_traversal(self):
        self._require_flydsl()
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            DIRECT_RANGE_CAUSAL,
            MASK_TRAVERSAL_DIRECT_RANGE,
        )

        cases = (
            ((("ge", 2, 3),), 4),
            ((("le", 3, 2),), 4),
            ((("const_bool", True), ("ge", 2, 3), ("and", 4, 5)), 6),
            ((("const_i32", 0), ("add", 2, 4), ("ge", 5, 3)), 6),
            ((("const_i32", 0), ("sub", 2, 4), ("ge", 5, 3)), 6),
        )
        for instructions, output in cases:
            with self.subTest(instructions=instructions):
                traversal, direct_range_kind = classify_mask_traversal(
                    instructions,
                    output,
                )
                self.assertEqual(traversal, MASK_TRAVERSAL_DIRECT_RANGE)
                self.assertEqual(direct_range_kind, DIRECT_RANGE_CAUSAL)

    @parametrize(
        "sequence_length,window",
        [(128, 127), (128, 128), (128, 129), (512, 512), (1024, 512)],
    )
    def test_window_traversal_respects_sequence_bounds(self, sequence_length, window):
        self._require_flydsl()
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            DIRECT_RANGE_CAUSAL,
            MASK_TRAVERSAL_BLOCK_LIST,
            MASK_TRAVERSAL_DIRECT_RANGE,
        )

        program, reason = lower_flydsl_mask_graph(
            _mask_graph(lambda b, h, q, kv: (q >= kv) & (q - kv < window)),
            (),
        )
        self.assertIsNotNone(program, reason)
        traversal, kind = classify_mask_traversal(
            program.instructions,
            program.output,
            program.buffer_shapes,
            sequence_length=sequence_length,
        )
        expected = (
            MASK_TRAVERSAL_DIRECT_RANGE
            if window >= sequence_length
            else MASK_TRAVERSAL_BLOCK_LIST
        )
        self.assertEqual(traversal, expected)
        self.assertEqual(
            kind, DIRECT_RANGE_CAUSAL if window >= sequence_length else None
        )

    @parametrize("operation,bound", [("ge", 0), ("gt", -1), ("lt", 512), ("le", 511)])
    def test_constant_coordinate_bounds_are_unmasked(self, operation, bound):
        self._require_flydsl()
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            MASK_TRAVERSAL_UNMASKED,
        )

        instructions = (("const_i32", bound), (operation, 2, 4))
        self.assertEqual(
            classify_mask_traversal(instructions, 5, sequence_length=512),
            (MASK_TRAVERSAL_UNMASKED, None),
        )

    @parametrize(
        "instructions,output",
        [
            (
                (("const_i32", 1 << 30), ("mul", 2, 4), ("const_i32", 0), ("ge", 5, 6)),
                7,
            ),
            ((("load_i32", 0, (2,)), ("ge", 2, 4)), 5),
            ((("const_i32", 2), ("floordiv", 2, 4), ("ge", 5, 3)), 6),
        ],
    )
    def test_uncertain_coordinate_bounds_keep_block_lists(self, instructions, output):
        self._require_flydsl()
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            MASK_TRAVERSAL_BLOCK_LIST,
        )

        self.assertEqual(
            classify_mask_traversal(instructions, output, sequence_length=512),
            (MASK_TRAVERSAL_BLOCK_LIST, None),
        )


class TestFlexFlyDSLRuntime(TestCase):
    def _require_runtime(self):
        if not _gfx950_available():
            self.skipTest("requires FlyDSL on ROCm gfx950")

    def _make_inputs(
        self,
        *,
        device,
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
            tensor = torch.randn(shape, device=device, dtype=torch.bfloat16)
            return tensor.transpose(1, 2) if transposed else tensor

        return make(qk_dim), make(qk_dim), make(v_dim), make(v_dim)

    def _make_mask(self, kind, *, device, heads=2, seq=256):
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

        elif kind == "empty_rows":
            mask_heads = heads

            def mask_mod(b, h, q, kv):
                del b
                return (q >= kv) & (kv >= 128) & (h != 0)

        elif kind == "two_buffer_document":
            document_ids = torch.arange(seq, device=device, dtype=torch.int32) // 128
            document_starts = torch.arange(
                0, seq, 128, device=device, dtype=torch.int32
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
            device=device,
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
        flydsl_options=None,
        dq_atol=None,
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
                    kernel_options={
                        "BACKEND": backend,
                        **((flydsl_options or {}) if backend == "FLYDSL" else {}),
                    },
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
            # FlyDSL rounds intermediate probability and dS fragments to bf16.
            self.assertEqual(
                actual,
                expected,
                atol=dq_atol if name == "dQ" and dq_atol is not None else atol,
                rtol=rtol,
                msg=f"{name} mismatch",
            )
        return flydsl

    @parametrize("dq_accum_fp32", (True, False))
    @parametrize("qk_dim", (128, 192))
    @parametrize("mask_kind", ("dense", "causal", "two_buffer_document"))
    def test_dq_accumulation_modes(self, device, dq_accum_fp32, qk_dim, mask_kind):
        self._require_runtime()
        # The sequence and head count reuse private slots in both traversals.
        q, k, v, grad_out = self._make_inputs(
            device=device, heads=16, seq=4352, qk_dim=qk_dim, seed=41
        )
        block_mask = self._make_mask(mask_kind, device=device, heads=16, seq=4352)
        with DeterministicGuard(True):
            first = self._compare_backward(
                q,
                k,
                v,
                grad_out,
                block_mask=block_mask,
                rtol=0.02,
                flydsl_options={"bwd_FLYDSL_DQ_ACCUM_FP32": dq_accum_fp32},
                dq_atol=0.02,
            )
            inputs = tuple(tensor.detach().requires_grad_(True) for tensor in (q, k, v))
            compiled = torch.compile(
                lambda q_, k_, v_: flex_attention(
                    q_,
                    k_,
                    v_,
                    block_mask=block_mask,
                    kernel_options={
                        "BACKEND": "FLYDSL",
                        "bwd_FLYDSL_DQ_ACCUM_FP32": dq_accum_fp32,
                    },
                ),
                fullgraph=True,
            )
            from torch._inductor.utils import run_and_get_code

            def run():
                output = compiled(*inputs)
                grads = torch.autograd.grad(output, inputs, grad_out, retain_graph=True)
                return output, grads

            (output, second), code = run_and_get_code(run)
            self.assertIn(
                f"DQ_ACCUM_FP32: fx.Constexpr = {dq_accum_fp32}", "\n".join(code)
            )
            third = torch.autograd.grad(output, inputs, grad_out)
            for expected, actual, repeated in zip(first, second, third, strict=True):
                self.assertEqual(actual, expected, atol=0, rtol=0)
                self.assertEqual(repeated, actual, atol=0, rtol=0)

            other_mode = torch.compile(
                lambda q_, k_, v_: flex_attention(
                    q_,
                    k_,
                    v_,
                    block_mask=block_mask,
                    kernel_options={
                        "BACKEND": "FLYDSL",
                        "bwd_FLYDSL_DQ_ACCUM_FP32": not dq_accum_fp32,
                    },
                ),
                fullgraph=True,
            )
            other_grads = torch.autograd.grad(other_mode(*inputs), inputs, grad_out)
            # Only dQ storage changes; retain exact dK/dV agreement across modes.
            for actual, expected in zip(second[1:], other_grads[1:], strict=True):
                self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("case", _MASK_CASES, name_fn=lambda case: case[0])
    def test_masks_match_triton(self, device, case):
        self._require_runtime()
        _, qk_dim, mask_kind, scale = case
        q, k, v, grad_out = self._make_inputs(device=device, qk_dim=qk_dim, seed=2)
        grads = self._compare_backward(
            q,
            k,
            v,
            grad_out,
            block_mask=self._make_mask(mask_kind, device=device),
            scale=scale,
        )
        if mask_kind == "empty_rows":
            for grad in grads:
                self.assertEqual(grad[:, 0], torch.zeros_like(grad[:, 0]))
                self.assertEqual(grad[:, :, :128], torch.zeros_like(grad[:, :, :128]))

    @parametrize("mask_kind", ("causal", "window"))
    def test_padded_positive_strides_match_triton(self, device, mask_kind):
        self._require_runtime()
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
                device=device,
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
            block_mask=self._make_mask(mask_kind, device=device),
        )

    def test_default_block_mask_matches_triton(self, device):
        self._require_runtime()
        q, k, v, grad_out = self._make_inputs(device=device, qk_dim=128, seed=3)
        self._compare_backward(q, k, v, grad_out, block_mask=None)

    @parametrize("mode", ("strict", "warn_only"))
    @parametrize("mask_kind", ("causal", "window", "two_buffer_document", "empty_rows"))
    @parametrize("transposed", (False, True))
    def test_deterministic_backward(self, device, mode, mask_kind, transposed):
        self._require_runtime()
        q, k, v, grad_out = self._make_inputs(
            device=device, batch=2, heads=3, seq=384, transposed=transposed, seed=9
        )
        inputs = tuple(t.requires_grad_(True) for t in (q, k, v))
        block_mask = self._make_mask(mask_kind, device=device, heads=3, seq=384)
        torch._dynamo.reset()
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
        with DeterministicGuard(True, warn_only=mode == "warn_only"):
            output = compiled(*inputs)
            expected = torch.autograd.grad(output, inputs, grad_out, retain_graph=True)
            for _ in range(3):
                actual = torch.autograd.grad(
                    output, inputs, grad_out, retain_graph=True
                )
                for grad, reference in zip(actual, expected, strict=True):
                    self.assertEqual(
                        grad.contiguous().view(torch.uint8),
                        reference.contiguous().view(torch.uint8),
                    )

    def test_non_dsv3_shape_and_mask_match_triton(self, device):
        self._require_runtime()
        batch, heads, seq = 2, 3, 384
        q, k, v, grad_out = self._make_inputs(
            device=device,
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
            device=device,
            BLOCK_SIZE=128,
        )
        self._compare_backward(q, k, v, grad_out, block_mask=block_mask)

    def test_fractional_causal_mask_is_rejected(self, device):
        self._require_runtime()
        q, k, v, grad_out = self._make_inputs(device=device, qk_dim=128, seed=5)
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
            device=device,
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
    def test_document_masks_match_triton(self, device, case):
        self._require_runtime()
        name, batch, heads, seq, transposed, partitions, scale = case
        q, k, v, grad_out = self._make_inputs(
            device=device,
            batch=batch,
            heads=heads,
            seq=seq,
            transposed=transposed,
            seed=4,
        )
        if partitions is None:
            document_size = 512
            document_end = (
                torch.arange(seq, device=device, dtype=torch.int32) // document_size + 1
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
            document_end = torch.tensor(rows, device=device, dtype=torch.int32)

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
            device=device,
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

    @parametrize("kind", ("document_start", "batched_document_end"))
    def test_document_mask_lowering(self, device, kind):
        self._require_runtime()

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            classify_mask_traversal,
            MASK_TRAVERSAL_BLOCK_LIST,
        )

        batch, heads, seq = 2, 2, 256
        q, k, v, grad_out = self._make_inputs(
            device=device,
            batch=batch,
            heads=heads,
            seq=seq,
            seed=7,
        )
        if kind == "document_start":
            document_ids = torch.arange(seq, device=device, dtype=torch.int32) // 128
            document_starts = torch.tensor(
                [0, 128],
                device=device,
                dtype=torch.int32,
            )

            mask_mod = and_masks(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                lambda b, h, q_idx, kv_idx: (
                    kv_idx >= document_starts[document_ids[q_idx]]
                ),
            )

            mask_batch = 1
        else:
            document_end = torch.tensor(
                [[127] * 128 + [255] * 128] * batch,
                device=device,
                dtype=torch.int32,
            )

            mask_mod = and_masks(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                lambda b, h, q_idx, kv_idx: q_idx <= document_end[b, kv_idx],
            )

            mask_batch = batch

        block_mask = create_block_mask(
            mask_mod,
            mask_batch,
            1,
            seq,
            seq,
            device=device,
            BLOCK_SIZE=128,
        )

        lowered_programs = []

        def check_document_lowering(*args, **kwargs):
            program, reason = lower_flydsl_mask_graph(*args, **kwargs)
            lowered_programs.append((program, reason))
            return program, reason

        torch._dynamo.reset()
        with (
            torch._inductor.config.patch({"fx_graph_cache": False}),
            mock.patch(
                "torch._inductor.kernel.flex.flex_flydsl_attention.lower_flydsl_mask_graph",
                side_effect=check_document_lowering,
            ) as lower,
        ):
            self._compare_backward(
                q,
                k,
                v,
                grad_out,
                block_mask=block_mask,
            )
            lower.assert_called_once()
        program, reason = lowered_programs[0]
        self.assertIsNotNone(program, reason)
        traversal, direct_range_kind = classify_mask_traversal(
            program.instructions,
            program.output,
            program.buffer_shapes,
        )
        self.assertEqual(traversal, MASK_TRAVERSAL_BLOCK_LIST)
        self.assertIsNone(direct_range_kind)

    def test_gqa_is_rejected(self, device):
        self._require_runtime()
        batch, seq = 1, 256
        q = torch.randn(
            batch,
            4,
            seq,
            128,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn(
            batch,
            2,
            seq,
            128,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        v = torch.randn_like(k, requires_grad=True)
        grad_out = torch.randn_like(q)
        block_mask = self._make_mask("dense", device=device, heads=4, seq=seq)
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

    @largeTensorTest("36GB")
    def test_standalone_four_gib_buffers(self, device):
        self._require_runtime()

        import flydsl.compiler as flyc

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_gfx950 import (
            build_flex_attn_bwd_module,
        )

        batch, heads, seq, dim = 16, 64, 16384, 128
        # Each tensor spans 4 GiB, while each head fits a buffer descriptor.
        source = torch.zeros(
            (batch, heads, seq, dim), device=device, dtype=torch.bfloat16
        )
        grads = [torch.empty_like(source) for _ in range(3)]
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            choose_dq_partitions,
        )

        partitions = choose_dq_partitions(batch * heads, seq, 192)
        workspace = torch.empty(
            (batch * heads // 32, partitions, seq, dim, 32),
            device=device,
            dtype=torch.float32,
        )
        lse = torch.full((batch, heads, seq), 14.0, device=device, dtype=torch.float32)
        delta = torch.empty_like(lse)
        unused = torch.empty(1, device=device, dtype=torch.int32)
        launcher = build_flex_attn_bwd_module(
            batch, heads, seq, dim, dim, "bf16", 128, 128, lse_in_log2=True
        )
        args = (
            source,
            source,
            source,
            source,
            lse,
            source,
            *grads,
            unused,
            unused,
            unused,
            unused,
            delta,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            unused,
            workspace,
        )
        stream = torch.cuda.current_stream(device).cuda_stream
        flyc.compile(launcher, *(flyc.from_torch_tensor(arg) for arg in args), stream)
        for grad in grads:
            self.assertEqual(torch.count_nonzero(grad).item(), 0)

    @parametrize("qk_dim", (128, 192))
    def test_long_causal_dq_matches_fp64(self, device, qk_dim):
        self._require_runtime()
        # Enough heads keep multiple KV owners in each private dQ slot.
        q, k, v, grad_out = self._make_inputs(
            device=device, heads=16, seq=16384, qk_dim=qk_dim, seed=41
        )
        q, k, v = (tensor.requires_grad_(True) for tensor in (q, k, v))
        block_mask = self._make_mask("causal", device=device, heads=16, seq=16384)
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
        grad_query = torch.autograd.grad(compiled(q, k, v), q, grad_out)[0]
        scale = qk_dim**-0.5
        reference = torch.empty_like(q, dtype=torch.float64)
        # Chunk rows and heads so the reference never materializes an S x S tensor.
        with torch.no_grad():
            key_positions = torch.arange(16384, device=device)
            for head in range(q.size(1)):
                key64, value64 = k[:, head].double(), v[:, head].double()
                for start in range(0, 16384, 256):
                    end = start + 256
                    scores = (
                        q[:, head, start:end].double() @ key64.transpose(-2, -1)
                    ) * scale
                    rows = torch.arange(start, end, device=device)
                    scores.masked_fill_(
                        rows[:, None] < key_positions[None, :], -float("inf")
                    )
                    probability = scores.softmax(-1)
                    dp = grad_out[:, head, start:end].double() @ value64.transpose(
                        -2, -1
                    )
                    ds = probability * (dp - (probability * dp).sum(-1, keepdim=True))
                    reference[:, head, start:end] = (ds @ key64) * scale
        self.assertEqual(grad_query.double(), reference, atol=0.01, rtol=0.02)
        relative_rms = (grad_query.double() - reference).norm() / reference.norm()
        self.assertLess(relative_rms.item(), 0.003)
        # Late rows visit many KV owners and expose repeated BF16 rounding.
        late_reference = reference[:, :, 8192:]
        late_error = grad_query[:, :, 8192:].double() - late_reference
        self.assertLess((late_error.norm() / late_reference.norm()).item(), 0.003)

    @parametrize("causal", [False, True])
    @parametrize("sequence_length", (256, 512, 1024))
    def test_standalone_default_lse(self, device, causal, sequence_length):
        self._require_runtime()

        import flydsl.compiler as flyc

        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_gfx950 import (
            build_flex_attn_bwd_module,
        )

        batch, heads, head_dim = 1, 8, 128
        query, key, value, grad_output = self._make_inputs(
            device=device,
            batch=batch,
            heads=heads,
            seq=sequence_length,
            qk_dim=head_dim,
            v_dim=head_dim,
            seed=8,
        )
        scale = head_dim**-0.5
        scores = query.float() @ key.float().transpose(-2, -1) * scale
        if causal:
            positions = torch.arange(sequence_length, device=device)
            scores.masked_fill_(positions[:, None] < positions[None, :], -torch.inf)
        probabilities = scores.softmax(dim=-1)
        attention_output = (probabilities @ value.float()).to(torch.bfloat16)
        logsumexp = scores.logsumexp(dim=-1)
        grad_probabilities = grad_output.float() @ value.float().transpose(-2, -1)
        delta_reference = (attention_output.float() * grad_output.float()).sum(dim=-1)
        grad_scores = probabilities * (
            grad_probabilities - delta_reference.unsqueeze(-1)
        )
        expected_grads = (
            (grad_scores @ key.float() * scale).to(torch.bfloat16),
            (grad_scores.transpose(-2, -1) @ query.float() * scale).to(torch.bfloat16),
            (probabilities.transpose(-2, -1) @ grad_output.float()).to(torch.bfloat16),
        )
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        delta = torch.empty(
            batch * heads * sequence_length,
            device=device,
            dtype=torch.float32,
        )
        from torch._inductor.kernel.vendored_templates.flydsl.kernels.flex_attn_bwd_utils import (
            choose_dq_partitions,
        )

        partitions = choose_dq_partitions(batch * heads, sequence_length, 192)
        grad_query_workspace = torch.empty(
            batch * heads * partitions * sequence_length * head_dim,
            device=device,
            dtype=torch.float32,
        )
        # Dense ranges do not read the reverse-list metadata.
        partial_kv_counts = torch.empty(1, device=device, dtype=torch.int32)
        partial_kv_indices = torch.empty_like(partial_kv_counts)
        full_kv_counts = torch.empty_like(partial_kv_counts)
        full_kv_indices_transposed = torch.empty_like(partial_kv_counts)
        sparse_blocks = sequence_length // 128
        kv_num_blocks = torch.zeros(
            (1, 1, sparse_blocks), device=device, dtype=torch.int32
        )
        kv_indices = torch.zeros(
            (1, 1, sparse_blocks, sparse_blocks),
            device=device,
            dtype=torch.int32,
        )
        full_kv_num_blocks = torch.full_like(kv_num_blocks, sparse_blocks)
        full_kv_indices = (
            torch.arange(sparse_blocks, device=device, dtype=torch.int32)
            .view(1, 1, 1, -1)
            .expand(1, 1, sparse_blocks, sparse_blocks)
            .contiguous()
        )
        launcher = build_flex_attn_bwd_module(
            batch,
            heads,
            sequence_length,
            head_dim,
            head_dim,
            "bf16",
            128,
            128,
            scale=scale,
            max_partial_blocks=sparse_blocks,
            max_full_blocks=sparse_blocks,
            mask_program=(("ge", 2, 3),) if causal else (),
            mask_program_output=4 if causal else 0,
        )
        tensor_args = (
            query,
            key,
            value,
            attention_output,
            logsumexp,
            grad_output,
            grad_query,
            grad_key,
            grad_value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            delta,
            partial_kv_counts,
            partial_kv_indices,
            full_kv_counts,
            full_kv_indices_transposed,
            kv_num_blocks,
            kv_num_blocks,
            kv_num_blocks,
            kv_num_blocks,
            grad_query_workspace,
        )
        stream = torch.cuda.current_stream(device).cuda_stream
        compile_args = tuple(flyc.from_torch_tensor(arg) for arg in tensor_args) + (
            stream,
        )
        compiled = flyc.compile(launcher, *compile_args)
        # Repeated launches exercise dV reads before the double buffer is reused.
        for _ in range(5):
            grad_query_workspace.fill_(float("nan"))
            compiled(*tensor_args, stream)
            for name, actual, expected in zip(
                ("dQ", "dK", "dV"),
                (grad_query, grad_key, grad_value),
                expected_grads,
                strict=True,
            ):
                self.assertEqual(actual, expected, atol=0.02, rtol=0.02, msg=name)


instantiate_device_type_tests(TestFlexFlyDSLRuntime, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
