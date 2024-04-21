# Owner(s): ["module: inductor"]
import random
import string
import sys
import unittest

import torch
import torch._dynamo
import torch.utils.cpp_extension

try:
    from extension_backends.triton.device_interface import DeviceInterface
    from extension_backends.triton.extension_codegen_backend import (
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .extension_backends.triton.device_interface import DeviceInterface
    from .extension_backends.triton.extension_codegen_backend import (
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

from torch._C import FileCheck
from torch._dynamo import device_interface
from torch._inductor import metrics
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.utils import get_triton_code
from torch.testing._internal.common_utils import IS_MACOS

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase


def mock_triton_hash_with_backend(*args, **kwargs):
    # Generate a random string of length 64. Used to mock the triton_hash_with_backend function
    # since we don't have a triton backend
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=64))


class TritonExtensionBackendTests(TestCase):
    """
    Test creating a backend for inductor with Triton scheduling.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_open_device_registration(self):
        register_backend_for_device("cpu", ExtensionScheduling, ExtensionWrapperCodegen)
        register_device_op_overrides("cpu", CPUDeviceOpOverrides())
        device_interface.register_interface_for_device("cpu", DeviceInterface)

        self.assertTrue(get_scheduling_for_device("cpu") == ExtensionScheduling)
        self.assertTrue(
            get_wrapper_codegen_for_device("cpu") == ExtensionWrapperCodegen
        )
        self.assertTrue(
            device_interface.get_interface_for_device("cpu") == DeviceInterface
        )

        device = torch.device("cpu")
        x = torch.empty(2, 16).fill_(1).to(device)

        def foo(x):
            return torch.sin(x) + x.min()

        metrics.reset()
        opt_fn = torch.compile(foo)

        # Since we don't have a triton backend, we need to mock the triton_hash_with_backend
        # function
        with unittest.mock.patch(
            "torch.utils._triton.triton_hash_with_backend",
            new=mock_triton_hash_with_backend,
        ):
            code = get_triton_code(opt_fn, x)

        FileCheck().check("import triton").check("@triton.jit").check(
            "tl_math.sin"
        ).check("device_str='cpu'").run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests()
