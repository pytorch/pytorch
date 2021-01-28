import torch
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TestPythonBindings\n\n"
        "instead."
    )


class TestPythonBindings(JitTestCase):
    def test_cu_get_functions(self):
        @torch.jit.script
        def test_get_python_cu_fn(x: torch.Tensor):
            return 2 * x

        cu = torch.jit._state._python_cu
        self.assertTrue(
            "test_get_python_cu_fn" in (str(fn.name) for fn in cu.get_functions())
        )

    def test_cu_create_function(self):
        @torch.jit.script
        def fn(x: torch.Tensor):
            return 2 * x

        cu = torch._C.CompilationUnit()
        cu.create_function("test_fn", fn.graph)

        inp = torch.randn(5)

        self.assertEqual(inp * 2, cu.find_function("test_fn")(inp))
        self.assertEqual(cu.find_function("doesnt_exist"), None)
        self.assertEqual(inp * 2, cu.test_fn(inp))
        with self.assertRaises(AttributeError):
            cu.doesnt_exist(inp)
