# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)

from test_utils import DummyPrivateUse1Module


class TestPrivateUse1(TestCase):
    def test_lazy_init(self):
        """
        Validate that no PrivateUse1 calls are made during `import torch` call
        """

        # The Privateuse1 backend module must be registered before calling
        # device-related APIs. The PrivateUse1 backend have the same behavior
        # as CUDA or XPU in lazy_init call
        with self.assertRaises(ModuleNotFoundError):
            print(torch.rand(1, 4).to('privateuseone:0'))

        torch._register_device_module("privateuseone", DummyPrivateUse1Module)
        with self.assertRaises(NotImplementedError):
            print(torch.rand(1, 4).to('privateuseone:0'))


if __name__ == "__main__":
    run_tests()
