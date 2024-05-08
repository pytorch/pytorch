# Owner(s): ["module: functorch"]
import unittest

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
)


class TestTorchbind(TestCase):
    def setUp(self):
        super().setUp()
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        elif IS_SANDCASTLE or IS_FBCODE:
            torch.ops.load_library(
                "//caffe2/test/cpp/jit:test_custom_class_registrations"
            )
        elif IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
            torch.ops.load_library(str(lib_file_path))
        else:
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    def test_torchbind_inductor(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.b = torch.randn(2, 3)

            def forward(self, x):
                x = x + self.b
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = M()
        orig_res = m(torch.ones(2, 3))

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, (torch.ones(2, 3),), strict=False)
        compiled = torch._inductor.compile(ep.module(), (torch.ones(2, 3),))

        new_res = compiled(torch.ones(2, 3))
        self.assertTrue(torch.allclose(orig_res, new_res))


if __name__ == "__main__":
    run_tests()
