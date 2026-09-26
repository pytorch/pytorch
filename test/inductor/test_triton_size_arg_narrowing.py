# Owner(s): ["module: inductor"]
"""Tests for config.triton.narrow_proven_size_args (codegen/triton_size_arg_narrowing.py)."""

import re

import torch
import torch._dynamo
from torch._inductor import config
from torch._inductor.codegen import triton_size_arg_narrowing as narrowing
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


def nested_cat_add(q1, k1, v1a, v1b, q2, k2, v2a, v2b):
    v1 = v1a + v1b
    v2 = v2a + v2b
    return torch.cat([torch.cat([q1, k1, v1], -1), torch.cat([q2, k2, v2], -1)], -1)


def _inputs(n=64, widths=(2048, 256, 256, 256, 2048, 256, 256, 256)):
    ins = [torch.randn(n, w, device=GPU_TYPE, dtype=torch.bfloat16) for w in widths]
    for t in ins:
        torch._dynamo.mark_dynamic(t, 1)
    return ins


def _ks_types(code: str) -> dict:
    return dict(re.findall(r"'(ks\d+)': '(i\d+)'", code))


def _kernel(code: str) -> tuple:
    """(argument names, dedented body) of the first @triton.jit kernel in the generated code."""
    m = re.search(r"@triton\.jit\ndef \w+\((.*?)\):\n((?:    .*\n|\n)+)", code)
    args = [a.split(":")[0].strip() for a in m.group(1).split(",")]
    body = "\n".join(line[4:] for line in m.group(2).splitlines())
    return args, body


class TestSizeArgNarrowing(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @requires_gpu()
    @config.patch({"triton.narrow_proven_size_args": True})
    def test_narrows_proven_cat_kernel(self):
        ins = _inputs()
        out, code = run_and_get_code(torch.compile(nested_cat_add, dynamic=True), *ins)
        code = "\n".join(code)
        self.assertEqual(set(_ks_types(code).values()), {"i32"})
        self.assertEqual(out.view(torch.int16), nested_cat_add(*ins).view(torch.int16))
        # a second shape reuses the same kernel (no recompile) and stays exact
        ins2 = _inputs(n=3000, widths=(1000, 120, 136, 136, 1000, 120, 136, 136))
        self.assertEqual(
            torch.compile(nested_cat_add, dynamic=True)(*ins2).view(torch.int16),
            nested_cat_add(*ins2).view(torch.int16),
        )

    @requires_gpu()
    def test_default_is_unchanged(self):
        _, code = run_and_get_code(
            torch.compile(nested_cat_add, dynamic=True), *_inputs()
        )
        self.assertEqual(set(_ks_types("\n".join(code)).values()), {"i64"})

    @requires_gpu()
    @config.patch({"triton.narrow_proven_size_args": True})
    def test_other_kernel_is_unchanged(self):
        def three(q, k, va, vb):
            return torch.cat([q, k, va + vb], -1)

        ins = _inputs(widths=(2048, 256, 256, 256))
        _, code = run_and_get_code(torch.compile(three, dynamic=True), *ins)
        self.assertNotIn("i32", set(_ks_types("\n".join(code)).values()))

    @requires_gpu()
    def test_recognizer_rejects_mutations(self):
        _, code = run_and_get_code(
            torch.compile(nested_cat_add, dynamic=True), *_inputs()
        )
        args, body = _kernel("\n".join(code))
        self.assertEqual(
            narrowing.canonical_sha256(args, body), narrowing._PROVED_CAT6_SHA256
        )
        mutations = [
            (
                "tmp32 = tmp30 + tmp31",
                "tmp32 = tmp30 + tmp31\ntmp99 = ks1 * ks2 < ks3",
            ),  # extra size use
            (
                "in_ptr0 + (ks1*x1 + (x0))",
                "in_ptr0 + (ks1*x1 + (x0) + ks2)",
            ),  # different index
            (
                "tmp41 = (ks0).to(tl.int32)",
                "tmp41 = (ks0).to(tl.int64)",
            ),  # different cast
        ]
        for old, new in mutations:
            self.assertIn(old, body)
            mutated = body.replace(old, new, 1)
            try:
                digest = narrowing.canonical_sha256(args, mutated)
            except narrowing._Unsupported:
                continue
            self.assertNotEqual(digest, narrowing._PROVED_CAT6_SHA256)


if __name__ == "__main__":
    run_tests()
