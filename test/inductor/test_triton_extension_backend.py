# Owner(s): ["module: inductor"]
import random
import string
import sys
import unittest

import torch
import torch._dynamo
import torch.utils.cpp_extension


try:
    from extension_backends.triton.device_interface import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend  # noqa: B950
        DeviceInterface,
    )
    from extension_backends.triton.extension_codegen_backend import (  # @manual=fbcode//caffe2/test/inductor/extension_backends:extension_codegen_backend  # noqa: B950
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
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS


try:
    from .test_extension_backend import BaseExtensionBackendTests
except ImportError:
    from test_extension_backend import BaseExtensionBackendTests

try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise


TestCase = test_torchinductor.TestCase


def mock_triton_hash_with_backend(*args, **kwargs):
    # Generate a random string of length 64. Used to mock the triton_hash_with_backend function
    # since we don't have a triton backend
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=64))


@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class TritonExtensionBackendTests(BaseExtensionBackendTests):
    """
    Test creating a backend for inductor with Triton scheduling.
    """

    def test_open_device_registration(self):
        torch._register_device_module("privateuseone", self.module)
        register_backend_for_device(
            "privateuseone", ExtensionScheduling, ExtensionWrapperCodegen
        )
        register_device_op_overrides("privateuseone", CPUDeviceOpOverrides())
        device_interface.register_interface_for_device("privateuseone", DeviceInterface)

        self.assertEqual(
            get_scheduling_for_device("privateuseone"), ExtensionScheduling
        )
        self.assertEqual(
            get_wrapper_codegen_for_device("privateuseone"), ExtensionWrapperCodegen
        )
        self.assertEqual(
            device_interface.get_interface_for_device("privateuseone"), DeviceInterface
        )

        device = torch.device("privateuseone")
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
        ).check("device_str='privateuseone'").run(code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests()
