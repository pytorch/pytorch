# Owner(s): ["module: dynamo"]
import torch
from torch.export import export
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSafeguard(TestCase):
    def test_grad_mode_unsupported(self):
        def f1(a):
            with torch.no_grad():
                b = a + a
            return b

        def f2(a):
            with torch.enable_grad():
                b = a + a
            return b

        def f3(a):
            with torch.set_grad_enabled(False):
                b = a + a
            return b

        def f4(a):
            with torch.set_grad_enabled(True):
                b = a + a
            return b

        a = torch.randn(10)
        with torch.no_grad():
            export(f1, (a,))
            export(f3, (a,))

            with self.assertRaisesRegex(
                RuntimeError, "Encountered autograd state manager op.*"
            ):
                export(f2, (a,))

            with self.assertRaisesRegex(
                RuntimeError, "Encountered autograd state manager op.*"
            ):
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


if __name__ == "__main__":
    run_tests()
