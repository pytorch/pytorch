# Owner(s): ["module: inductor"]
import unittest

import torch
import torch.nn.functional as F

from torch.testing._internal.common_utils import (
    IS_CI,
    TestCase,
    run_tests,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


@unittest.skipIf(not HAS_CUDA, "CUDA required")
@skipIfRocm
class TestInt8WeightOnlyLinearStride(TestCase):
    """
    Regression tests for gh-204778:
    _weight_int8pack_mm must not ignore activation strides.
    """

    def _make_non_contiguous_bf16(self):
        # shape (17, 8, 128), strides (1024, 1, 8) — last-dim stride == 8
        return (
            torch.arange(17 * 128 * 8, device="cuda")
            .reshape(17, 128, 8)
            .remainder(17)
            .to(torch.bfloat16)
            .transpose(1, 2)
        )

    def test_non_contiguous_input_correctness(self):
        weight = torch.eye(64, 128, device="cuda", dtype=torch.int8)
        x = self._make_non_contiguous_bf16()
        scale = torch.ones(64, device="cuda", dtype=torch.bfloat16)

        def fn(x, w, s):
            return F.linear(x, w.to(x.dtype)) * s

        compiled_fn = torch.compile(fn, fullgraph=True)
        eager_out = fn(x, weight, scale)
        compiled_out = compiled_fn(x, weight, scale)

        self.assertTrue(
            torch.allclose(eager_out.float(), compiled_out.float(), atol=1e-2),
            f"Max diff: {(eager_out - compiled_out).abs().max().item():.4f}",
        )

    def test_contiguous_input_unaffected(self):
        weight = torch.eye(64, 128, device="cuda", dtype=torch.int8)
        x = self._make_non_contiguous_bf16().contiguous()
        scale = torch.ones(64, device="cuda", dtype=torch.bfloat16)

        def fn(x, w, s):
            return F.linear(x, w.to(x.dtype)) * s

        compiled_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(
            torch.allclose(
                fn(x, weight, scale).float(),
                compiled_fn(x, weight, scale).float(),
                atol=1e-2,
            )
        )

    def test_reshape_alone_is_correct(self):
        x = self._make_non_contiguous_bf16()
        reshape_fn = torch.compile(lambda t: t.reshape(-1, 128), fullgraph=True)
        self.assertTrue(
            torch.allclose(x.reshape(-1, 128), reshape_fn(x)),
            "reshape/layout path broken independently of matmul",
        )

    def test_linear_without_scale(self):
        weight = torch.eye(64, 128, device="cuda", dtype=torch.int8)
        x = self._make_non_contiguous_bf16()

        def fn(x, w):
            return F.linear(x, w.to(x.dtype))

        compiled_fn = torch.compile(fn, fullgraph=True)
        self.assertTrue(
            torch.allclose(
                fn(x, weight).float(),
                compiled_fn(x, weight).float(),
                atol=1e-2,
            )
        )

    def test_different_batch_sizes(self):
        weight = torch.eye(64, 128, device="cuda", dtype=torch.int8)
        scale = torch.ones(64, device="cuda", dtype=torch.bfloat16)

        def fn(x, w, s):
            return F.linear(x, w.to(x.dtype)) * s

        compiled_fn = torch.compile(fn, fullgraph=True)

        for batch in (1, 4, 32):
            x = (
                torch.randn(batch, 128, 8, device="cuda", dtype=torch.bfloat16)
                .transpose(1, 2)
            )
            self.assertTrue(
                torch.allclose(
                    fn(x, weight, scale).float(),
                    compiled_fn(x, weight, scale).float(),
                    atol=1e-2,
                ),
                f"Failed for batch={batch}",
            )


if __name__ == "__main__":
    run_tests()

