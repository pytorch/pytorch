# Owner(s): ["oncall: export"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch.export import export
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestSafeguard(TestCase):
    # If the autograd state doesn't change, dynamo eliminates autograd state manager op and later export can succeed.
    # Otherwise, autograd can be preserved in the produced gragh, and export will fail.
    def test_global_autograd(self):
        class F1(torch.nn.Module):
            def forward(self, a):
                with torch.no_grad():
                    b = a + a
                return b

        f1 = F1()

        class F2(torch.nn.Module):
            def forward(self, a):
                with torch.enable_grad():
                    b = a + a
                return b

        f2 = F2()

        class F3(torch.nn.Module):
            def forward(self, a):
                with torch.set_grad_enabled(False):
                    b = a + a
                return b

        f3 = F3()

        class F4(torch.nn.Module):
            def forward(self, a):
                with torch.set_grad_enabled(True):
                    b = a + a
                return b

        f4 = F4()

        a = torch.randn(10)
        with torch.no_grad():
            export(f1, (a,))
            export(f2, (a,))
            export(f3, (a,))
            export(f4, (a,))

        with torch.enable_grad():
            export(f2, (a,))
            export(f4, (a,))

            with self.assertRaisesRegex(
                RuntimeError, "Encountered autograd state manager op.*"
            ):
                export(f1, (a,))

            with self.assertRaisesRegex(
                RuntimeError, "Encountered autograd state manager op.*"
            ):
                export(f3, (a,))

    def test_tensor_autograd(self):
        # dynamo errors when Tensor.requires_grad_ change the autograd state
        class F1(torch.nn.Module):
            def forward(self, a):
                a.requires_grad_(True)
                b = a + a
                return b

        f1 = F1()

        # dynamo errors when Tensor.requires_grad_ change the autograd state
        class F2(torch.nn.Module):
            def forward(self, a):
                a.requires_grad_(False)
                b = a + a
                return b

        f2 = F2()

        # dynamo always errors on Tensor.requires_grad
        class F3(torch.nn.Module):
            def forward(self, a):
                a.requires_grad = False
                b = a + a
                return b

        f3 = F3()

        export(f1, (torch.randn(10, requires_grad=True),))
        export(f2, (torch.randn(10, requires_grad=False),))

        with self.assertRaises(RuntimeError):
            export(f1, (torch.randn(10, requires_grad=False),))
        with self.assertRaises(RuntimeError):
            export(f2, (torch.randn(10, requires_grad=True),))
        with self.assertRaises(RuntimeError):
            export(f3, (torch.randn(10, requires_grad=False),))

    def test_global_autograd_exempt_predispatch(self):
        class F1(torch.nn.Module):
            def forward(self, a):
                with torch.no_grad():
                    b = a + a
                return b

        f1 = F1()

        class F2(torch.nn.Module):
            def forward(self, a):
                with torch.enable_grad():
                    b = a + a
                return b

        f2 = F2()

        class F3(torch.nn.Module):
            def forward(self, a):
                with torch.set_grad_enabled(False):
                    b = a + a
                return b

        f3 = F3()

        class F4(torch.nn.Module):
            def forward(self, a):
                with torch.set_grad_enabled(True):
                    b = a + a
                return b

        f4 = F4()

        a = torch.randn(10)

        from torch.export._trace import _export

        with torch.no_grad():
            _export(f1, (a,), pre_dispatch=True)
            _export(f2, (a,), pre_dispatch=True)
            _export(f3, (a,), pre_dispatch=True)
            _export(f4, (a,), pre_dispatch=True)

        with torch.enable_grad():
            _export(f1, (a,), pre_dispatch=True)
            _export(f2, (a,), pre_dispatch=True)
            _export(f3, (a,), pre_dispatch=True)
            _export(f4, (a,), pre_dispatch=True)


if __name__ == "__main__":
    run_tests()
