
# Owner(s): ["module: functorch"]
import unittest

import torch
import torch._dynamo
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.aot_autograd import aot_export_module
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, _get_torch_cuda_version


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestWithEffects(TestCase):
    def setUp(self):
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

    def test_print(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten.print("moo")
                return (x + x,)

        inputs = (torch.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        FileCheck().check_count(
            "torch.ops.aten.print.default", 1, exactly=True
        ).run(gm.code)

        # With functionalization, it should appear wrapped with tokenize()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        FileCheck().check_count(
            "torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops.aten.print.default, 'moo')", 1, exactly=True
        ).run(gm.code)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

    def test_torchbind_custom_op(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return (x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x),)

        with enable_torchbind_tracing():
            gm, gs = aot_export_module(M(), (torch.ones(2, 3),), trace_joint=False)

        FileCheck().check_count(
            "torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, _tensor_constant0, arg1_1)", 1, exactly=True
        ).run(gm.code)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)


if __name__ == '__main__':
    run_tests()
