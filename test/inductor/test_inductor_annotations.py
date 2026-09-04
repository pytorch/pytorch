# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.inductor_utils import HAS_TRITON


class InductorAnnotationTestCase(TestCase):
    hw_classification = HardwareClassification.CUDA

    def get_code(self, device):
        def f(a, b):
            return a + b, a * b

        a = torch.randn(5, device=device)
        b = torch.randn(5, device=device)
        f_comp = torch.compile(f)

        _, code = run_and_get_code(f_comp, a, b)
        return code[0]

    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_no_annotations(self, device):
        code = self.get_code(device)

        self.assertTrue("from torch.cuda import nvtx" not in code)
        self.assertTrue("training_annotation" not in code)

    @inductor_config.patch(annotate_training=True)
    @unittest.skipIf(not HAS_TRITON, "requires triton")
    def test_training_annotation(self, device):
        code = self.get_code(device)

        self.assertTrue("from torch.cuda import nvtx" in code)
        self.assertTrue(
            code.count("training_annotation = nvtx._device_range_start('inference')")
            >= 1
        )
        self.assertTrue(code.count("nvtx._device_range_end(training_annotation)") >= 1)


instantiate_device_type_tests(InductorAnnotationTestCase, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
