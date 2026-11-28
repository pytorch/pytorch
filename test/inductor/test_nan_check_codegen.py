# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import requires_gpu


@requires_gpu()
class TestNanCheckCodegen(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("cpp_wrapper", (False, True))
    def test_nan_checks_identify_kernel_and_buffer(self, device, cpp_wrapper):
        x = torch.ones(8, device=device)
        y = torch.ones(8, device=device)
        with config.patch(nan_asserts=True, cpp_wrapper=cpp_wrapper):
            compiled = torch.compile(lambda x, y: x / y)
            result, code = run_and_get_code(compiled, x, y)
        self.assertEqual(result, x / y)
        self.assertRegex("\n".join(code), r"triton_poi_fused_div_\d+: buf\d+")
        with self.assertRaisesRegex(
            (AssertionError, RuntimeError), r"triton_poi_fused_div_\d+: buf\d+"
        ):
            compiled(x, torch.zeros_like(y))


instantiate_device_type_tests(
    TestNanCheckCodegen, globals(), allow_xpu=True, except_for="cpu"
)

if __name__ == "__main__":
    run_tests()
