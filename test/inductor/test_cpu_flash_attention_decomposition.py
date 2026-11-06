# Owner(s): ["oncall: cpu inductor"]
import operator

import torch
from torch._dynamo.utils import counters
from torch._inductor import config, cpu_vec_isa, metrics
from torch._inductor.fx_passes.post_grad import pass_patterns
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
)


FLASH = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default
NATIVE_CALLS = (
    "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(",
    "aoti_torch_cpu__scaled_dot_product_flash_attention_for_cpu(",
)


def sdpa(*, output_only=True, **options):
    def attention(q, k, v):
        outputs = FLASH(q, k, v, **options)
        return outputs[0] if output_only else outputs

    return attention


ATTENTION = sdpa(scale=0.0125)


def qkv(device, batch, heads, queries, keys, features):
    q = torch.randn(batch, heads, queries, features, device=device)
    k = torch.randn(batch, heads, keys, features, device=device)
    return q, k, torch.randn_like(k)


def last_dim_strided(x):
    return torch.randn(*x.shape[:-1], 2 * x.size(-1), device=x.device)[..., ::2]


def math_flags(unsafe):
    return {
        "cpp.enable_unsafe_math_opt_flag": unsafe,
        "cpp.enable_floating_point_contract_flag": "fast" if unsafe else "off",
    }


def calls_native(code):
    generated = "\n".join(code)
    return any(call in generated for call in NATIVE_CALLS)


class CpuFlashAttentionDecompositionTests(TestCase):
    hw_classification = HardwareClassification.CPU

    def setUp(self):
        super().setUp()
        self.enterContext(
            config.patch(
                {
                    "force_disable_caches": True,
                    "cpp.threads": 1,
                    "cpp.enable_cpu_flash_attention_decomposition": True,
                }
            )
        )

    def assert_close(self, actual, expected):
        self.assertEqual(actual, expected, atol=1e-5, rtol=1e-4)

    def assert_rewritten(self, code, before, rewritten=True):
        self.assertTrue(code)
        rewrites = counters["inductor"]["cpu_flash_attention_decomposition"]
        self.assertEqual(rewrites > before, rewritten)
        self.assertEqual(calls_native(code), not rewritten)

    def check(
        self, fn, *args, rewritten=True, expected=None, patches=None, **compile_kwargs
    ):
        if expected is None:
            expected = fn(*args)
        before = counters["inductor"]["cpu_flash_attention_decomposition"]
        with config.patch(patches or {}):
            actual, code = run_and_get_code(
                torch.compile(fn, fullgraph=True, **compile_kwargs), *args
            )
        self.assert_close(actual, expected)
        self.assert_rewritten(code, before, rewritten)
        return actual, code

    @parametrize(
        "shape",
        [
            (1, 3, 1, 24, 64),
            (1, 3, 2, 24, 64),
            (8, 4, 1, 32, 64),
            (1, 1, 1, 24, 160),
            (1, 1, 1, 513, 64),
        ],
    )
    def test_decomposition_without_size_cutoff(self, device, shape):
        self.check(ATTENTION, *qkv(device, *shape))

    @parametrize("shape", [(1, 1, 16, 15), (2, 2, 16, 32), (1, 4, 32, 65)])
    def test_nearby_shapes_and_strided_keys(self, device, shape):
        batch, heads, keys, features = shape
        q = torch.randn(batch, heads, 1, features, device=device)
        k = torch.randn(batch, heads, keys * 2, features, device=device)[:, :, ::2]
        self.check(sdpa(), q, k, torch.randn_like(k))

    def test_dequantized_query(self, device):
        def attention(q, scale, bias, k, v):
            return FLASH(q.float() * scale + bias, k, v)[0]

        q = torch.randint(0, 256, (1, 3, 1, 64), dtype=torch.uint8, device=device)
        scale = torch.randn(1, 3, 1, 1, device=device) * 0.01
        bias = torch.randn(1, 3, 1, 1, device=device)
        _, k, v = qkv(device, 1, 3, 1, 32, 64)
        self.check(attention, q, scale, bias, k, v)

    def test_multiple_output_consumers(self, device):
        def attention(q, k, v):
            y = ATTENTION(q, k, v)
            return y, y * 2

        self.check(attention, *qkv(device, 1, 3, 1, 32, 64))

    @parametrize("output_only", [True, False])
    def test_cpp_wrapper(self, device, output_only):
        self.check(
            sdpa(output_only=output_only, scale=0.0125),
            *qkv(device, 1, 3, 1, 24, 64),
            rewritten=output_only,
            patches={"cpp_wrapper": True},
        )

    @parametrize(
        "case",
        [
            "query_nan",
            "key_inf",
            "value_inf",
            "all_negative_inf",
            "masked_value_inf",
            "overflow",
            "value_overflow",
        ],
    )
    @parametrize("unsafe_math", [False, True])
    @parametrize("threads", [1, 4])
    def test_nonfinite_and_fully_masked_rows(self, device, case, unsafe_math, threads):
        q = torch.ones(1, 3, 1, 64, device=device)
        k = torch.randn(1, 3, 32, 64, device=device)
        v = torch.randn_like(k)
        if case == "query_nan":
            q[0, 0, 0, 0] = float("nan")
        elif case == "key_inf":
            k[0, 0, 0, 0] = float("inf")
        elif case == "value_inf":
            v[0, 0, 0, 0] = float("inf")
        elif case in ("all_negative_inf", "masked_value_inf"):
            k.fill_(-float("inf"))
            if case == "masked_value_inf":
                v[0, 0, 0, 0] = float("inf")
        elif case == "value_overflow":
            q.zero_()
            v.fill_(1e38)
        else:
            q.fill_(1e20)
            k.fill_(1e20)
        patches = {"cpp.threads": threads, **math_flags(unsafe_math)}
        actual, _ = self.check(ATTENTION, q, k, v, patches=patches)
        if case == "all_negative_inf":
            self.assertEqual(actual, torch.zeros_like(actual))

    @parametrize(
        "case",
        [
            "fully_masked",
            "masked_value_inf",
            "masked_then_finite",
            "finite_then_masked",
            "late_nan",
            "late_inf",
        ],
    )
    @parametrize("threads", [1, 4])
    def test_multiblock_rows(self, device, case, threads):
        q = torch.ones(1, 2, 3, 8, device=device)
        k = torch.zeros(1, 2, 1024, 8, device=device)
        v = torch.randn_like(k)
        if case in ("fully_masked", "masked_value_inf"):
            k.fill_(-float("inf"))
            if case == "masked_value_inf":
                v[:, :, -1, 0] = float("inf")
        elif case == "masked_then_finite":
            k[:, :, :512].fill_(-float("inf"))
        elif case == "finite_then_masked":
            k[:, :, 512:].fill_(-float("inf"))
        else:
            k[:, :, -1, 0] = float("nan") if case == "late_nan" else float("inf")
        _, code = self.check(ATTENTION, q, k, v, patches={"cpp.threads": threads})
        if threads > 1:
            self.assertIn("#pragma omp", "\n".join(code))

    @parametrize("keys", [513, 1024])
    def test_late_maximum_preserves_value_overflow(self, device, keys):
        q = torch.ones(1, 1, 1, 1, device=device)
        k = torch.zeros(1, 1, keys, 1, device=device)
        k[:, :, -1] = 80.0
        v = torch.full_like(k, 1e38)
        actual, _ = self.check(sdpa(scale=1.0), q, k, v, patches=math_flags(False))
        self.assertTrue(torch.isinf(actual).all())

    def test_cancelling_values_can_overflow_per_block(self, device):
        q = torch.zeros(1, 1, 3, 64, device=device)
        k = torch.zeros(1, 1, 1024, 64, device=device)
        v = torch.cat(
            [
                torch.full((1, 1, 512, 64), -5e35, device=device),
                torch.full((1, 1, 512, 64), 7e35, device=device),
            ],
            2,
        )
        # Native's GEMM accumulates across blocks and can return ~1e35 here.
        self.check(sdpa(scale=1.0), q, k, v, expected=torch.full_like(q, float("inf")))

    @parametrize("keys", [1, 7, 15, 17, 23, 31])
    def test_scalar_key_tail_nan(self, device, keys):
        q = torch.ones(1, 1, 1, 2, device=device)
        k = torch.full((1, 1, keys, 2), -float("inf"), device=device)
        k[0, 0, -1, 0] = float("nan")
        v = torch.ones_like(k)
        # Native returns 0 here; see Note [CPU flash attention accumulation order].
        self.check(ATTENTION, q, k, v, expected=torch.full_like(q, float("nan")))

    @parametrize("case", ["late_nan", "fully_masked", "masked_value_inf"])
    @parametrize("vec_isa_ok", [False, True])
    def test_parallel_reduction_handles_nonfinite(self, device, case, vec_isa_ok):
        if vec_isa_ok and not cpu_vec_isa.pick_vec_isa():
            self.skipTest("requires a vector ISA")
        q = torch.ones(1, 1, 1, 1, device=device)
        k = torch.zeros(1, 1, 1024, 1, device=device)
        v = torch.ones_like(k)
        if case == "late_nan":
            k[:, :, -1] = float("nan")
        else:
            k.fill_(-float("inf"))
            if case == "masked_value_inf":
                v[:, :, -1] = float("inf")
        reductions = metrics.parallel_reduction_count
        patches = {
            "cpp.threads": 4,
            "cpp.min_chunk_size": 1,
            "cpp.vec_isa_ok": vec_isa_ok,
        }
        _, code = self.check(ATTENTION, q, k, v, patches=patches)
        self.assertGreater(metrics.parallel_reduction_count, reductions)
        if vec_isa_ok:
            self.assertRegex(
                "\n".join(code), r"at::vec::maximum\([^\n]*_vec_arr\[tid\]"
            )

    @parametrize("scale", [0.0, -0.1])
    @parametrize("keys", [24, 513])
    def test_finite_nonpositive_scale(self, device, scale, keys):
        self.check(sdpa(scale=scale), *qkv(device, 1, 2, 3, keys, 16))

    @parametrize("scale", [0.0, -0.1])
    @parametrize("unsafe_math", [False, True])
    def test_nonpositive_scale_preserves_nonfinite_scores(
        self, device, scale, unsafe_math
    ):
        q = torch.ones(1, 1, 1, 2, device=device)
        k = torch.full((1, 1, 32, 2), float("inf"), device=device)
        v = torch.randn_like(k)
        self.check(sdpa(scale=scale), q, k, v, patches=math_flags(unsafe_math))

    @parametrize(
        "case",
        [
            "disabled",
            "mask",
            "causal",
            "nan_scale",
            "infinite_scale",
            "float64",
            "bfloat16",
            "noncontiguous_query",
            "noncontiguous_key",
            "noncontiguous_value",
            "gqa",
            "logsumexp",
            "dynamic_shapes",
            "independent_dynamic_batches",
        ],
    )
    def test_native_fallback(self, device, case):
        q, k, v = qkv(device, 2, 4, 1, 32, 64)
        options = {"scale": 0.0125}
        patches = {}
        compile_kwargs = {}
        if case == "disabled":
            patches["cpp.enable_cpu_flash_attention_decomposition"] = False
        elif case == "mask":
            options["attn_mask"] = torch.zeros(1, 32, device=device)
        elif case == "causal":
            options["is_causal"] = True
        elif case == "nan_scale":
            options["scale"] = float("nan")
        elif case == "infinite_scale":
            options["scale"] = float("inf")
        elif case in ("float64", "bfloat16"):
            dtype = getattr(torch, case)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        elif case == "noncontiguous_query":
            q = last_dim_strided(q)
        elif case == "noncontiguous_key":
            k = last_dim_strided(k)
        elif case == "noncontiguous_value":
            v = last_dim_strided(v)
        elif case == "gqa":
            k, v = k[:, :2].contiguous(), v[:, :2].contiguous()
        elif case == "dynamic_shapes":
            compile_kwargs["dynamic"] = True
        elif case == "independent_dynamic_batches":
            for x in (q, k, v):
                torch._dynamo.mark_dynamic(x, 0)
            compile_kwargs["dynamic"] = False
        fn = sdpa(output_only=case != "logsumexp", **options)
        self.check(fn, q, k, v, rewritten=False, patches=patches, **compile_kwargs)

    def test_gradients_keep_native(self, device):
        inputs = [x.requires_grad_() for x in qkv(device, 1, 3, 1, 32, 64)]
        reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        expected = ATTENTION(*reference_inputs)
        expected.sum().backward()
        before = counters["inductor"]["cpu_flash_attention_decomposition"]
        actual, code = run_and_get_code(
            torch.compile(ATTENTION, fullgraph=True), *inputs
        )
        actual.sum().backward()
        self.assert_close(actual, expected)
        for x, reference in zip(inputs, reference_inputs):
            self.assert_close(x.grad, reference.grad)
        self.assert_rewritten(code, before, rewritten=False)

    def test_dynamic_batch(self, device):
        def attention(stacked):
            q = stacked[:, :, :1].contiguous()
            k = stacked[:, :, 1:25].contiguous()
            v = stacked[:, :, 25:49].contiguous()
            return ATTENTION(q, k, v)

        first = torch.randn(2, 3, 49, 64, device=device)
        torch._dynamo.mark_dynamic(first, 0)
        compiled = torch.compile(attention, fullgraph=True, dynamic=False)
        before = counters["inductor"]["cpu_flash_attention_decomposition"]
        actual, code = run_and_get_code(compiled, first)
        self.assert_close(actual, attention(first))
        self.assert_rewritten(code, before)
        graphs = counters["stats"]["unique_graphs"]
        for batch in (3, 8, 32, 256):
            stacked = torch.randn(batch, 3, 49, 64, device=device)
            self.assert_close(compiled(stacked), attention(stacked))
        self.assertEqual(counters["stats"]["unique_graphs"], graphs)
        for batch in (0, 1):
            stacked = torch.randn(batch, 3, 49, 64, device=device)
            actual, code = run_and_get_code(compiled, stacked)
            self.assert_close(actual, attention(stacked))
            self.assertFalse(calls_native(code))

    @parametrize("unsafe_math", [False, True])
    def test_generated_query_dynamic_batch(self, device, unsafe_math):
        def attention(packed, key, value):
            batch = packed.size(0)
            q = (packed.float() * 0.01 - 1.0).contiguous()
            k = key.expand(batch, -1, -1, -1).contiguous()
            v = value.expand(batch, -1, -1, -1).contiguous()
            return ATTENTION(q, k, v)

        packed = torch.randint(0, 256, (2, 3, 1, 64), device=device, dtype=torch.uint8)
        torch._dynamo.mark_dynamic(packed, 0)
        key = torch.randn(1, 3, 24, 64, device=device)
        value = torch.randn_like(key)
        with config.patch(math_flags(unsafe_math)):
            compiled = torch.compile(attention, fullgraph=True, dynamic=False)
            before = counters["inductor"]["cpu_flash_attention_decomposition"]
            actual, code = run_and_get_code(compiled, packed, key, value)
            self.assert_close(actual, attention(packed, key, value))
            self.assert_rewritten(code, before)
            graphs = counters["stats"]["unique_graphs"]
            for batch in (8, 32, 256):
                p = torch.randint(
                    0, 256, (batch, 3, 1, 64), device=device, dtype=torch.uint8
                )
                self.assert_close(compiled(p, key, value), attention(p, key, value))
            self.assertEqual(counters["stats"]["unique_graphs"], graphs)
            key[0, 1, -1, 0] = float("nan")
            self.assert_close(
                compiled(packed, key, value), attention(packed, key, value)
            )
            packed.fill_(100)
            key.zero_()
            value.fill_(1e38)
            self.assert_close(
                compiled(packed, key, value), attention(packed, key, value)
            )

    @staticmethod
    def make_attention_graph(device, *, mutation=False, dropout=0.0):
        graph = torch.fx.Graph()
        inputs = [graph.placeholder(name) for name in ("q", "k", "v")]
        flash = graph.call_function(
            FLASH, tuple(inputs), {"scale": 0.0125, "dropout_p": dropout}
        )
        mutation_node = (
            graph.call_function(torch.ops.aten.add_.Tensor, (inputs[2], 1))
            if mutation
            else None
        )
        output = graph.call_function(operator.getitem, (flash, 0))
        graph.output(output)
        with FakeTensorMode() as fake_mode:
            q, k, v = qkv(device, 1, 3, 1, 32, 64)
            lse = torch.empty(1, 3, 1, device=device)
        for node, value in zip(inputs, (q, k, v)):
            node.meta["val"] = value
        flash.meta["val"] = (q, lse)
        if mutation_node is not None:
            mutation_node.meta["val"] = v
        output.meta["val"] = q
        return torch.fx.GraphModule({}, graph), fake_mode

    @parametrize("mutation", [False, True])
    def test_mutation_between_operator_and_getitem(self, device, mutation):
        gm, fake_mode = self.make_attention_graph(device, mutation=mutation)
        with fake_mode:
            self.assertEqual(pass_patterns[0].apply(gm.graph), 0 if mutation else 1)
        gm.recompile()
        q = torch.ones(1, 3, 1, 64, device=device)
        k = torch.ones(1, 3, 32, 64, device=device)
        actual = gm(q, k, torch.ones_like(k))
        self.assertEqual(actual, torch.ones_like(actual))

    def test_nonzero_dropout_keeps_native_error(self, device):
        gm, fake_mode = self.make_attention_graph(device, dropout=0.1)
        with fake_mode:
            self.assertEqual(pass_patterns[0].apply(gm.graph), 0)
        q, k, v = qkv(device, 1, 3, 1, 32, 64)
        with self.assertRaisesRegex(RuntimeError, "do not support dropout"):
            gm(q, k, v)


instantiate_device_type_tests(
    CpuFlashAttentionDecompositionTests, globals(), only_for="cpu"
)


class CpuFlashAttentionSchemaTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_schema_matches_eligibility_check(self):
        # The pattern reads dropout_p/is_causal by position, attn_mask/scale by name.
        arguments = FLASH._schema.arguments
        self.assertEqual(
            [arg.name for arg in arguments],
            ["query", "key", "value", "dropout_p", "is_causal", "attn_mask", "scale"],
        )
        self.assertEqual(
            {arg.name for arg in arguments if arg.kwarg_only}, {"attn_mask", "scale"}
        )


if __name__ == "__main__":
    run_tests()
