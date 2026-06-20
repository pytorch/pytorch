# Owner(s): ["module: ci"]
# TEMPORARY: demo for the test-diff CI comment (testintro). DELETE before landing.
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestIntroDemo(TestCase):
    def test_plain(self):
        self.assertTrue(True)

    @parametrize("x", [1, 2])
    def test_param(self, x):
        self.assertTrue(True)


class TestIntroDemoDevice(TestCase):
    def test_any(self, device):
        self.assertTrue(True)

    @onlyCUDA
    def test_cuda_only(self, device):
        self.assertTrue(True)


instantiate_parametrized_tests(TestIntroDemo)
instantiate_device_type_tests(TestIntroDemoDevice, globals())

if __name__ == "__main__":
    run_tests()
