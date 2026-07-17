# Owner(s): ["module: dsl-native-ops"]

import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._native.ops.cross_entropy.helion_impl import (
    _B200_PRETUNED_SHAPES,
    _cross_entropy_cond,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, skipIfNoHelionDSL, TestCase


_SHAPE_CONFIG_INDEX = {
    (2048, 32000): 2,
    (4096, 32000): 2,
    (8192, 32000): 2,
    (8192, 128000): 0,
    (16384, 128000): 0,
    (32768, 128000): 0,
    (2048, 128256): 1,
    (4096, 128256): 4,
    (8192, 128256): 4,
    (16384, 128256): 0,
    (2048, 129280): 1,
    (4096, 129280): 0,
    (8192, 129280): 1,
    (2048, 151936): 1,
    (4096, 151936): 1,
    (8192, 151936): 1,
    (2048, 152064): 3,
    (4096, 152064): 3,
    (8192, 152064): 0,
    (1024, 256000): 6,
    (2048, 256000): 5,
}


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoHelionDSL
class TestHelionCrossEntropy(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.cuda.get_device_capability() != (10, 0):
            raise unittest.SkipTest("B200/GB200 sm100 required")

    def _inputs(self, device, requires_grad=False):
        logits = torch.randn(
            4096,
            32000,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        labels = torch.randint(32000, (4096,), device="cuda", dtype=torch.int64)
        return logits, labels

    def _autograd_cuda_registration(self):
        return next(
            line
            for line in torch._C._dispatch_dump_table(
                "aten::cross_entropy_loss"
            ).splitlines()
            if line.startswith("AutogradCUDA:")
        )

    def test_aot_config_for_all_shapes(self, device):
        from torch._native.ops.cross_entropy._helion_aot_helion_kernel_cuda_sm100 import (
            key_cross_entropy,
        )

        self.assertEqual(_B200_PRETUNED_SHAPES, frozenset(_SHAPE_CONFIG_INDEX))
        for shape, expected in _SHAPE_CONFIG_INDEX.items():
            with self.subTest(shape=shape):
                logits = torch.empty(shape, device="meta", dtype=torch.bfloat16)
                self.assertEqual(key_cross_entropy(logits), expected)

    def test_condition_accepts_pretuned_contract(self, device):
        logits, labels = self._inputs(device)
        self.assertTrue(_cross_entropy_cond(logits, labels))

    def test_correctness_optimized_path(self, device):
        logits, labels = self._inputs(device)
        self.assertTrue(_cross_entropy_cond(logits, labels))
        actual = F.cross_entropy(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "requires at least 2 visible CUDA devices",
    )
    def test_non_current_device(self, device):
        if torch.cuda.get_device_capability(1) != (10, 0):
            self.skipTest("second device must be B200/GB200 sm100")
        old_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            logits = torch.randn(4096, 32000, device="cuda:1", dtype=torch.bfloat16)
            labels = torch.randint(32000, (4096,), device="cuda:1")
            self.assertTrue(_cross_entropy_cond(logits, labels))
            actual = F.cross_entropy(logits, labels)
            with torch.backends.python_native.helion.disabled():
                expected = F.cross_entropy(logits, labels)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, torch.device("cuda:1"))
            self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)
        finally:
            torch.cuda.set_device(old_device)

    def test_cuda_graph_capture_falls_through(self, device):
        logits, labels = self._inputs(device)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            F.cross_entropy(logits, labels)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = F.cross_entropy(logits, labels)
        graph.replay()
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    def test_disable_restores_autograd_registration(self, device):
        self.assertIn("fallthrough", self._autograd_cuda_registration())
        with torch.backends.python_native.helion.disabled():
            self.assertNotIn("fallthrough", self._autograd_cuda_registration())
        self.assertIn("fallthrough", self._autograd_cuda_registration())

    def test_operation_filter_restores_autograd_registration(self, device):
        pn = torch.backends.python_native
        with pn.operations_disabled("cross_entropy_loss"):
            self.assertNotIn("fallthrough", self._autograd_cuda_registration())
        self.assertIn("fallthrough", self._autograd_cuda_registration())

    def test_dispatch_filter_restores_autograd_registration(self, device):
        pn = torch.backends.python_native
        pn.disable_dispatch_keys("CUDA")
        try:
            self.assertNotIn("fallthrough", self._autograd_cuda_registration())
        finally:
            pn.enable_dispatch_keys("CUDA")
        self.assertIn("fallthrough", self._autograd_cuda_registration())

    def test_jit_disabled_enable_does_not_install_fallthrough(self, device):
        script = textwrap.dedent(
            """\
            import torch

            torch.backends.python_native.helion.enable()
            line = next(
                line
                for line in torch._C._dispatch_dump_table(
                    "aten::cross_entropy_loss"
                ).splitlines()
                if line.startswith("AutogradCUDA:")
            )
            print(line)
            """
        )
        env = dict(os.environ)
        env["TORCH_DISABLE_NATIVE_JIT"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertNotIn("fallthrough", result.stdout)

    def test_condition_rejects_unsupported_contracts(self, device):
        logits, labels = self._inputs(device)
        with self.subTest("weight"):
            self.assertFalse(
                _cross_entropy_cond(
                    logits,
                    labels,
                    torch.ones(32000, device=device, dtype=torch.bfloat16),
                )
            )
        with self.subTest("reduction"):
            self.assertFalse(_cross_entropy_cond(logits, labels, reduction=0))
        with self.subTest("ignore_index"):
            self.assertFalse(_cross_entropy_cond(logits, labels, ignore_index=-1))
        with self.subTest("label_smoothing"):
            self.assertFalse(_cross_entropy_cond(logits, labels, label_smoothing=0.1))
        with self.subTest("requires_grad"):
            self.assertFalse(_cross_entropy_cond(logits.requires_grad_(), labels))
        logits.requires_grad_(False)
        with self.subTest("target_dtype"):
            self.assertFalse(_cross_entropy_cond(logits, labels.to(torch.int32)))
        with self.subTest("target_layout"):
            labels_storage = torch.empty(8192, device=device, dtype=torch.int64)
            self.assertFalse(_cross_entropy_cond(logits, labels_storage[::2]))
        with self.subTest("cow"):
            self.assertFalse(_cross_entropy_cond(logits._lazy_clone(), labels))
        with self.subTest("shape"):
            small_logits = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
            small_labels = torch.randint(128, (32,), device=device)
            self.assertFalse(_cross_entropy_cond(small_logits, small_labels))
        with self.subTest("performance_gate"):
            gated_logits = torch.randn(2048, 32000, device=device, dtype=torch.bfloat16)
            gated_labels = torch.randint(32000, (2048,), device=device)
            self.assertFalse(_cross_entropy_cond(gated_logits, gated_labels))
        with self.subTest("architecture"):
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                self.assertFalse(_cross_entropy_cond(logits, labels))

    def test_correctness_with_ignored_labels(self, device):
        logits, labels = self._inputs(device)
        labels[::17] = -100
        self.assertFalse(_cross_entropy_cond(logits, labels))
        actual = F.cross_entropy(logits, labels)
        with torch.backends.python_native.helion.disabled():
            expected = F.cross_entropy(logits, labels)
        self.assertEqual(actual, expected, rtol=1e-2, atol=1e-2)

    def test_all_labels_ignored(self, device):
        logits, labels = self._inputs(device)
        labels.fill_(-100)
        self.assertFalse(_cross_entropy_cond(logits, labels))
        actual = F.cross_entropy(logits, labels)
        self.assertTrue(torch.isnan(actual))

    def test_invalid_target_raises(self, device):
        logits, labels = self._inputs(device)
        labels[0] = 32000
        self.assertFalse(_cross_entropy_cond(logits, labels))

        script = textwrap.dedent(
            """\
            import torch
            import torch.nn.functional as F

            logits = torch.randn(4096, 32000, device="cuda", dtype=torch.bfloat16)
            labels = torch.zeros(4096, device="cuda", dtype=torch.int64)
            labels[0] = 32000
            F.cross_entropy(logits, labels)
            torch.cuda.synchronize()
            """
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("device-side assert triggered", result.stderr)

    def test_requires_grad_falls_through(self, device):
        logits, labels = self._inputs(device, requires_grad=True)
        self.assertFalse(_cross_entropy_cond(logits, labels))
        loss = F.cross_entropy(logits, labels)
        self.assertIsNotNone(loss.grad_fn)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_forward_ad_falls_through(self, device):
        logits, labels = self._inputs(device)
        tangent = torch.randn_like(logits)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(logits, tangent)
            self.assertFalse(_cross_entropy_cond(dual, labels))
            output = F.cross_entropy(dual, labels)
            _, output_tangent = torch.autograd.forward_ad.unpack_dual(output)
            self.assertIsNotNone(output_tangent)


instantiate_device_type_tests(TestHelionCrossEntropy, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
