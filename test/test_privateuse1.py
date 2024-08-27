# Owner(s): ["module: PrivateUse1"]

from test_utils import DummyPrivateUse1Module

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPrivateUse1(TestCase):
    def test_lazy_init(self):
        """
        Validate that no PrivateUse1 calls are made during `import torch` call
        """

        # The PrivateUse1 backend have the same behavior as CUDA or XPU
        # in lazy_init call. Its module must be registered before calling
        # device-related APIs. See: https://github.com/pytorch/pytorch/pull/121379
        with self.assertRaises(ModuleNotFoundError):
            print(torch.rand(1, 4).to("privateuseone:0"))

        torch._register_device_module("privateuseone", DummyPrivateUse1Module)
        msg = r"This could be because the operator doesn't exist for this backend"
        with self.assertRaisesRegex(NotImplementedError, msg):
            print(torch.rand(1, 4).to("privateuseone:0"))


if __name__ == "__main__":
    run_tests()
