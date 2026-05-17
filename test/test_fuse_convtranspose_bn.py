# Owner(s): ["module: nn"]
"""
Tests for fuse_conv_bn_weights / fuse_conv_bn_eval with ConvTranspose2d/1d
and groups > 1 (issue #180995).

Before the fix, fuse_conv_bn_weights with transpose=True incorrectly broadcast
the BN scale using shape [1, out_channels, 1, 1], but ConvTranspose weight dim 1
is out_channels/groups — causing shape mismatches or silently wrong results when
groups > 1.
"""

import copy
import itertools
import unittest

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFuseConvTransposeBN(TestCase):
    """fuse_conv_bn_weights / fuse_conv_bn_eval with transpose=True."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_convtranspose2d_bn(self, C, groups, bias, seed=0):
        torch.manual_seed(seed)
        conv = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1,
                                  groups=groups, bias=bias)
        bn = nn.BatchNorm2d(C)
        # Give BN non-trivial running stats to exercise the fusion math
        bn.running_mean = torch.randn(C)
        bn.running_var = torch.rand(C) + 0.1
        conv.eval(); bn.eval()
        return conv, bn

    def _make_convtranspose1d_bn(self, C, groups, bias, seed=0):
        torch.manual_seed(seed)
        conv = nn.ConvTranspose1d(C, C, kernel_size=3, padding=1,
                                  groups=groups, bias=bias)
        bn = nn.BatchNorm1d(C)
        bn.running_mean = torch.randn(C)
        bn.running_var = torch.rand(C) + 0.1
        conv.eval(); bn.eval()
        return conv, bn

    # ------------------------------------------------------------------
    # Correctness: fuse_conv_bn_eval with ConvTranspose2d
    # ------------------------------------------------------------------
    def test_convtranspose2d_groups1_correctness(self):
        conv, bn = self._make_convtranspose2d_bn(8, groups=1, bias=False)
        x = torch.randn(4, 8, 16, 16)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused = fuse_conv_bn_eval(conv, bn, transpose=True)
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                        f"max_err={( y_ref - y_fused).abs().max():.2e}")

    def test_convtranspose2d_groups2_correctness(self):
        conv, bn = self._make_convtranspose2d_bn(8, groups=2, bias=False)
        x = torch.randn(4, 8, 16, 16)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused = fuse_conv_bn_eval(conv, bn, transpose=True)
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                        f"max_err={( y_ref - y_fused).abs().max():.2e}")

    def test_convtranspose2d_groups4_correctness(self):
        conv, bn = self._make_convtranspose2d_bn(8, groups=4, bias=False)
        x = torch.randn(4, 8, 16, 16)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused = fuse_conv_bn_eval(conv, bn, transpose=True)
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                        f"max_err={( y_ref - y_fused).abs().max():.2e}")

    def test_convtranspose2d_with_bias_correctness(self):
        # Also works when conv already has a bias
        for groups in (1, 2, 4):
            with self.subTest(groups=groups):
                conv, bn = self._make_convtranspose2d_bn(8, groups=groups, bias=True)
                x = torch.randn(4, 8, 16, 16)
                with torch.no_grad():
                    y_ref = bn(conv(x))
                fused = fuse_conv_bn_eval(conv, bn, transpose=True)
                with torch.no_grad():
                    y_fused = fused(x)
                self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                                f"groups={groups} max_err={( y_ref - y_fused).abs().max():.2e}")

    # ------------------------------------------------------------------
    # Correctness: fuse_conv_bn_eval with ConvTranspose1d
    # ------------------------------------------------------------------
    def test_convtranspose1d_groups1_correctness(self):
        conv, bn = self._make_convtranspose1d_bn(8, groups=1, bias=False)
        x = torch.randn(4, 8, 32)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused = fuse_conv_bn_eval(conv, bn, transpose=True)
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                        f"max_err={( y_ref - y_fused).abs().max():.2e}")

    def test_convtranspose1d_groups2_correctness(self):
        conv, bn = self._make_convtranspose1d_bn(8, groups=2, bias=False)
        x = torch.randn(4, 8, 32)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused = fuse_conv_bn_eval(conv, bn, transpose=True)
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                        f"max_err={( y_ref - y_fused).abs().max():.2e}")

    # ------------------------------------------------------------------
    # fuse_conv_bn_weights: direct API tests
    # ------------------------------------------------------------------
    def test_fuse_conv_bn_weights_transpose_groups_sweep(self):
        for C, groups in [(8, 1), (8, 2), (8, 4), (16, 4), (32, 8)]:
            with self.subTest(C=C, groups=groups):
                torch.manual_seed(groups)
                conv = nn.ConvTranspose2d(C, C, 3, padding=1, groups=groups, bias=False).eval()
                bn = nn.BatchNorm2d(C).eval()
                bn.running_mean = torch.randn(C)
                bn.running_var = torch.rand(C) + 0.1
                x = torch.randn(2, C, 8, 8)
                with torch.no_grad():
                    y_ref = bn(conv(x))
                fused_w, fused_b = fuse_conv_bn_weights(
                    conv.weight, None,
                    bn.running_mean, bn.running_var, bn.eps,
                    bn.weight, bn.bias, transpose=True,
                )
                fused = copy.deepcopy(conv)
                fused.weight = fused_w
                fused.bias = fused_b
                with torch.no_grad():
                    y_fused = fused(x)
                max_err = (y_ref - y_fused).abs().max().item()
                self.assertLess(max_err, 1e-5,
                                f"C={C} groups={groups} max_err={max_err:.2e}")

    def test_fuse_conv_bn_weights_non_square_kernel(self):
        # 3D ConvTranspose (non-square kernel) with groups > 1
        C, groups = 8, 4
        torch.manual_seed(0)
        conv = nn.ConvTranspose2d(C, C, (3, 5), padding=(1, 2), groups=groups, bias=False).eval()
        bn = nn.BatchNorm2d(C).eval()
        bn.running_mean = torch.randn(C)
        bn.running_var = torch.rand(C) + 0.1
        x = torch.randn(2, C, 8, 8)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused_w, fused_b = fuse_conv_bn_weights(
            conv.weight, None,
            bn.running_mean, bn.running_var, bn.eps,
            bn.weight, bn.bias, transpose=True,
        )
        fused = copy.deepcopy(conv)
        fused.weight = fused_w
        fused.bias = fused_b
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5))

    # ------------------------------------------------------------------
    # requires_grad is preserved
    # ------------------------------------------------------------------
    def test_requires_grad_preserved(self):
        cases = itertools.product([True, False], [True, False])
        for w_rg, b_rg in cases:
            with self.subTest(w_rg=w_rg, b_rg=b_rg):
                conv = nn.ConvTranspose2d(8, 8, 3, padding=1, groups=2, bias=True)
                bn = nn.BatchNorm2d(8)
                conv.weight.requires_grad = w_rg
                conv.bias.requires_grad = b_rg
                fused_w, fused_b = fuse_conv_bn_weights(
                    conv.weight, conv.bias,
                    bn.running_mean, bn.running_var, bn.eps,
                    bn.weight, bn.bias, transpose=True,
                )
                self.assertEqual(fused_w.requires_grad, w_rg)
                self.assertEqual(fused_b.requires_grad, b_rg)

    # ------------------------------------------------------------------
    # Dtype preservation
    # ------------------------------------------------------------------
    def test_dtype_float64_preserved(self):
        conv = nn.ConvTranspose2d(8, 8, 3, padding=1, groups=4, bias=False).double().eval()
        bn = nn.BatchNorm2d(8).double().eval()
        x = torch.randn(2, 8, 8, 8, dtype=torch.float64)
        with torch.no_grad():
            y_ref = bn(conv(x))
        fused = fuse_conv_bn_eval(conv, bn, transpose=True)
        self.assertEqual(fused.weight.dtype, torch.float64)
        with torch.no_grad():
            y_fused = fused(x)
        self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-10))

    # ------------------------------------------------------------------
    # Non-affine BN (weight=None, bias=None)
    # ------------------------------------------------------------------
    def test_non_affine_bn_groups(self):
        for groups in (1, 2, 4):
            with self.subTest(groups=groups):
                torch.manual_seed(groups)
                conv = nn.ConvTranspose2d(8, 8, 3, padding=1, groups=groups, bias=False).eval()
                bn = nn.BatchNorm2d(8, affine=False).eval()
                bn.running_mean = torch.randn(8)
                bn.running_var = torch.rand(8) + 0.1
                x = torch.randn(2, 8, 8, 8)
                with torch.no_grad():
                    y_ref = bn(conv(x))
                fused = fuse_conv_bn_eval(conv, bn, transpose=True)
                with torch.no_grad():
                    y_fused = fused(x)
                self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5),
                                f"groups={groups} max_err={( y_ref - y_fused).abs().max():.2e}")

    # ------------------------------------------------------------------
    # fuse_conv_bn_eval raises for non-eval modules
    # ------------------------------------------------------------------
    def test_raises_if_conv_in_train_mode(self):
        conv = nn.ConvTranspose2d(4, 4, 3, padding=1, groups=2)
        bn = nn.BatchNorm2d(4).eval()
        with self.assertRaises(AssertionError):
            fuse_conv_bn_eval(conv, bn, transpose=True)

    def test_raises_if_bn_in_train_mode(self):
        conv = nn.ConvTranspose2d(4, 4, 3, padding=1, groups=2).eval()
        bn = nn.BatchNorm2d(4)
        with self.assertRaises(AssertionError):
            fuse_conv_bn_eval(conv, bn, transpose=True)

    # ------------------------------------------------------------------
    # Regular Conv (transpose=False) still works after refactor
    # ------------------------------------------------------------------
    def test_regular_conv_unaffected(self):
        for groups in (1, 2, 4):
            with self.subTest(groups=groups):
                torch.manual_seed(0)
                conv = nn.Conv2d(8, 8, 3, padding=1, groups=groups).eval()
                bn = nn.BatchNorm2d(8).eval()
                x = torch.randn(2, 8, 8, 8)
                with torch.no_grad():
                    y_ref = bn(conv(x))
                fused = fuse_conv_bn_eval(conv, bn, transpose=False)
                with torch.no_grad():
                    y_fused = fused(x)
                self.assertTrue(torch.allclose(y_ref, y_fused, atol=1e-5))


if __name__ == "__main__":
    run_tests()
