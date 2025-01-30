# Owner(s): ["module: inductor"]
import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.triton_utils import requires_cuda


class InductorAnnotationTestCase(TestCase):
    def get_code(self):
        def f(a, b):
            return a + b, a * b

        a = torch.randn(5, device="cuda")
        b = torch.randn(5, device="cuda")
        f_comp = torch.compile(f)

        _, code = run_and_get_code(f_comp, a, b)
        return code[0]

    @requires_cuda
    def test_no_annotations(self):
        code = self.get_code()

        self.assertTrue("from torch.cuda import nvtx" not in code)
        self.assertTrue("training_annotation" not in code)

    @inductor_config.patch(annotate_training=True)
    @requires_cuda
    def test_training_annotation(self):
        code = self.get_code()

        self.assertTrue("from torch.cuda import nvtx" in code)
        self.assertEqual(
            code.count("training_annotation = nvtx._device_range_start('inference')"), 1
        )
        self.assertEqual(code.count("nvtx._device_range_end(training_annotation)"), 1)


if __name__ == "__main__":
    run_tests()
