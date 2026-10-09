# Owner(s): ["module: inductor"]

import os
from unittest import mock

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.utils import run_and_get_kernels
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE


def combine_inputs_asm(outputs):
    """Make every input word contribute to one of the assembly outputs."""
    if torch.version.hip:
        instructions = []
        for i in range(outputs):
            instructions.append(f"v_xor_b32 ${i}, ${outputs + i}, ${outputs + 16}")
            for j in range(i + outputs, 16, outputs):
                instructions.append(f"v_xor_b32 ${i}, ${i}, ${outputs + j}")
        return "\n".join(instructions), ",".join(["=&v"] * outputs + ["v"] * 17)
    asm = f"{{ .reg .b32 a<16>; .reg .b32 m; mov.b32 m, ${outputs + 16}; "
    asm += " ".join(f"mov.b32 a{i}, ${outputs + i};" for i in range(16))
    for i in range(outputs):
        asm += f" xor.b32 ${i}, a{i}, m;"
        for j in range(i + outputs, 16, outputs):
            asm += f" xor.b32 ${i}, ${i}, a{j};"
    return asm + " }", ",".join(["=r"] * outputs + ["r"] * 17)


def xor_words(pairs, maximum, outputs):
    words = []
    for start in range(outputs):
        word = pairs[..., start] ^ maximum
        for lane in range(start + outputs, 16, outputs):
            word = word ^ pairs[..., lane]
        words.append(word)
    return words


class TestInlineAsmConcat(TestCase):
    @parametrize("multiple_outputs", [False, True])
    @config.patch(
        {
            "force_pointwise_cat": False,
            "max_complex_pointwise_cat_inputs": 8,
            "triton.multi_kernel": 0,
            "force_disable_caches": True,
        }
    )
    def test_unrelated_asm_outputs(self, device, multiple_outputs):
        outputs = 2 if multiple_outputs else 1
        if torch.version.hip:
            asm = "\n".join(
                f"v_add_u32 ${i}, ${outputs}, {i + 1}" for i in range(outputs)
            )
            constraints = ",".join(["=&v"] * outputs + ["v"])
        else:
            asm = f"{{ .reg .b32 a; mov.b32 a, ${outputs}; "
            asm += " ".join(f"add.u32 ${i}, a, {i + 1};" for i in range(outputs))
            asm += " }"
            constraints = ",".join(["=r"] * outputs + ["r"])
        dtype = (torch.int32, torch.int32) if multiple_outputs else torch.int32

        def fn(x):
            a = inline_asm_elementwise(
                x, asm_str=asm, constraints=constraints, dtype=dtype
            )
            b = inline_asm_elementwise(
                x + 5, asm_str=asm, constraints=constraints, dtype=dtype
            )
            if multiple_outputs:
                a, b = a[0], b[1]
            return (torch.stack((a, b), -1) + 3).view(torch.uint8)

        x = torch.arange(1024, device=device, dtype=torch.int32).view(64, 16)
        expected = (torch.stack((x + 1, x + 5 + outputs), -1) + 3).view(torch.uint8)
        torch._dynamo.reset()
        metrics.reset()
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), expected)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @parametrize("outputs,dim", [(2, 0), (6, 1)])
    @config.patch(
        {
            "force_pointwise_cat": False,
            "max_complex_pointwise_cat_inputs": 8,
            "triton.multi_kernel": 0,
            "fx_graph_cache": False,
        }
    )
    def test_stack_outputs(self, device, outputs, dim):
        asm, constraints = combine_inputs_asm(outputs)

        def fn(x):
            maximum = x.amax(-1)
            values = inline_asm_elementwise(
                *x.unbind(-1),
                maximum,
                asm_str=asm,
                constraints=constraints,
                dtype=(torch.int32,) * outputs,
            )
            return torch.stack(values, dim).view(torch.uint8), maximum

        torch._dynamo.reset()
        metrics.reset()
        x = torch.arange(1024 * 16, device=device, dtype=torch.int32).view(1024, 16)
        maximum = x.amax(-1)
        expected_words = xor_words(x, maximum, outputs)
        expected = (
            torch.stack(expected_words, dim).view(torch.uint8),
            maximum,
        )
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), expected)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @parametrize("use_asm", [False, True])
    @parametrize(
        "width,multi_kernel,dynamic_batch",
        [(1024, 0, False), (1024, 1, True), (6144, 0, True), (6144, 1, False)],
    )
    def test_normal_choices_stack_reshape(
        self, device, use_asm, width, multi_kernel, dynamic_batch
    ):
        asm, constraints = combine_inputs_asm(3)

        def fn(x, native=use_asm):
            value = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)).to(
                torch.bfloat16
            )
            groups = value.reshape(x.shape[0], -1, 32)
            maximum = groups.abs().amax(-1)
            bits = groups.view(torch.int16).to(torch.int32) & 65535
            pairs = bits[..., ::2] | (bits[..., 1::2] << 16)
            if native:
                words = inline_asm_elementwise(
                    *pairs.unbind(-1),
                    maximum.to(torch.int32),
                    asm_str=asm,
                    constraints=constraints,
                    dtype=(torch.int32,) * 3,
                )
            else:
                words = xor_words(pairs, maximum.to(torch.int32), 3)
            return torch.stack(words, -1).reshape(x.shape[0], -1), maximum

        def make_input(batch):
            if not use_asm and not dynamic_batch:
                # Keep distinct row phases to detect incorrect row reads.
                return (
                    (torch.arange(batch * width, device=device) % 257 - 128).float()
                    / 64
                ).reshape(batch, width)
            # Avoid BF16 midpoints so packing comparisons can remain exact.
            row = (torch.arange(width, device=device) % 257 - 128).float() / 64
            signs = (torch.arange(batch, device=device) % 2 * 2 - 1).float()
            return signs[:, None] * row[None, :]

        x = make_input(64)
        expected = fn(x, False)
        if dynamic_batch:
            torch._dynamo.mark_dynamic(x, 0)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        metrics.reset()
        with (
            mock.patch.dict(os.environ, TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE="1"),
            config.patch(
                {
                    "triton.multi_kernel": multi_kernel,
                    "coordinate_descent_tuning": bool(multi_kernel),
                    "max_autotune": bool(multi_kernel),
                    "triton.nested_reduction": True,
                    "loop_ordering_after_fusion": True,
                    "emulate_precision_casts": True,
                    "force_disable_caches": True,
                }
            ),
        ):
            actual, kernels = run_and_get_kernels(compiled, x)
            if dynamic_batch:
                for batch in (2, 17, 129):
                    value = make_input(batch)
                    self.assertEqual(compiled(value), fn(value, False), atol=0, rtol=0)
            self.assertEqual(counter.frame_count, 1)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(
            tuple(t.stride() for t in actual), tuple(t.stride() for t in expected)
        )
        kernels = [kernel for kernel in kernels if "@triton.jit" in kernel]
        if use_asm:
            self.assertEqual(metrics.codegen_nested_reduction, 1)
            compute = [k for k in kernels if "tl.inline_asm_elementwise(" in k]
            self.assertTrue(compute)
            for kernel in compute:
                self.assertIn("tl.sum(", kernel)
                self.assertIn("triton_helpers.max2(", kernel)
                # Three FP6 words and the scale.
                self.assertEqual(kernel.count("tl.store("), 4)
            for kernel in kernels:
                if kernel not in compute:
                    self.assertNotIn("tl.load(", kernel)
        else:
            self.assertGreater(metrics.codegen_nested_reduction, 0)
            consumers = [k for k in kernels if "tl.load(" in k]
            reductions = [k for k in consumers if "tl.sum(" in k]
            pointwise = [k for k in consumers if "tl.sum(" not in k]
            self.assertTrue(reductions)
            for kernel in reductions:
                self.assertIn("triton_helpers.max2(", kernel)
            self.assertLessEqual(len(pointwise), 1)
            for kernel in pointwise:
                self.assertIn(" ^ ", kernel)  # Packing, not a payload copy.
            for kernel in consumers:
                self.assertNotIn("tl.inline_asm_elementwise(", kernel)
            fills = [k for k in kernels if "tl.load(" not in k]
            self.assertLessEqual(len(fills), 1)
            for kernel in fills:
                self.assertIn("tl.store(", kernel)


class TestPointwisePackingChoices(TestCase):
    settings = {
        "triton.nested_reduction": True,
        "loop_ordering_after_fusion": True,
        "emulate_precision_casts": True,
        "force_disable_caches": True,
    }

    @parametrize("kind", ["two_word_pack", "float_interleave"])
    def test_cheap_inputs_keep_one_kernel(self, device, kind):
        if kind == "two_word_pack":

            def fn(x):
                first = x[:, 0] | (x[:, 1] << 4)
                second = x[:, 2] | (x[:, 3] << 4)
                packed = torch.stack((first, second), -1).reshape(-1)
                return torch.nn.functional.pad(packed, (0, 5))

            x = (torch.arange(1024, device=device, dtype=torch.int32) % 16).reshape(
                256, 4
            )
        else:

            def fn(x):
                packed = torch.stack((x + 1, x + 2), -1).flatten(1)
                return torch.nn.functional.pad(packed, (0, 5))

            x = torch.arange(77, device=device, dtype=torch.float32).reshape(7, 11)

        expected = fn(x)
        torch._dynamo.reset()
        metrics.reset()
        with (
            config.patch({**self.settings, "triton.multi_kernel": 0}),
        ):
            actual = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(metrics.generated_kernel_count, 1)


# The inline asm strings are PTX or AMDGCN.
instantiate_device_type_tests(TestInlineAsmConcat, globals(), only_for="cuda")
instantiate_device_type_tests(TestPointwisePackingChoices, globals(), only_for=GPU_TYPE)


if __name__ == "__main__":
    run_tests()
