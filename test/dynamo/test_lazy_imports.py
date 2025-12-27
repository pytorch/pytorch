# Owner(s): ["module: dynamo"]

import subprocess
import sys
import textwrap

from torch.testing._internal.common_utils import run_tests, TestCase


class TestLazyImports(TestCase):
    def test_sympy_not_imported_with_torch(self):
        code = textwrap.dedent("""
            import sys
            import torch

            if 'sympy' in sys.modules:
                print('FAIL: sympy was imported with torch')
                sys.exit(1)
            else:
                print('PASS: sympy not imported with torch')
                sys.exit(0)
        """)

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"sympy should not be imported with torch.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}",
        )

    def test_sympy_imported_when_symbolic_shapes_used(self):
        code = textwrap.dedent("""
            import sys
            import torch
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            if 'sympy' not in sys.modules:
                print('FAIL: sympy was not imported after using symbolic shapes')
                sys.exit(1)
            else:
                print('PASS: sympy imported when symbolic shapes used')
                sys.exit(0)
        """)

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"sympy should be imported when symbolic shapes are used.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}",
        )

    def test_lazy_sympy_module(self):
        code = textwrap.dedent("""
            import sys
            from torch.utils._sympy.lazy import sympy, is_sympy_loaded

            if is_sympy_loaded():
                print('FAIL: sympy loaded before first access')
                sys.exit(1)

            _ = sympy.Symbol('x')

            if not is_sympy_loaded():
                print('FAIL: sympy not loaded after first access')
                sys.exit(1)

            print('PASS: lazy sympy module works correctly')
            sys.exit(0)
        """)

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"Lazy sympy module should work correctly.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}",
        )


if __name__ == "__main__":
    run_tests()
