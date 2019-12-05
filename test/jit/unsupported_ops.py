import os
import sys
from textwrap import dedent

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit_utils import JitTestCase, execWrapper

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# NOTE: FIXING FAILING TESTS
# If you are seeing a test failure from this file, congrats, you improved
# parity between JIT and Python API. Before you fix the test, you must also update
# the corresponding section in documentation that states the unsupported behavior.

class TestUnsupportedOps(JitTestCase):
    def test_factory_ops_requires_grad_fail(self):
        # Keyword argument {name} unknown is a JIT-only error message,
        # so these functions are succeeding in eager and failing in JIT

        # Complete issue and set of ops is https://github.com/pytorch/pytorch/issues/30761
        # only testing some because they should be fixed all at once

        with self.assertRaisesRegex(Exception, "Keyword argument requires_grad unknown"):
            def foo():
                return torch.ones([2], requires_grad=True)
            foo()
            torch.jit.script(foo)
        self.assertTrue(thrown)

        with self.assertRaisesRegex(Exception, "Keyword argument requires_grad unknown"):
            def foo():
                return torch.randn([2], requires_grad=True)
            foo()
            torch.jit.script(foo)

        with self.assertRaisesRegex(Exception, "Keyword argument requires_grad unknown"):
            def foo():
                return torch.zeros([2], requires_grad=True)
            foo()
            torch.jit.script(foo)



    def test_tensor_options_behavior_mismatch(self):
        # Any schema declaration which contains a non-optional (ScalarType dtype, Layout layout, Device device)
        # tuple is implicitly made to be optional for pytorch eager code. This makes the schema incorrect for JIT / C++ api.

        # Complete issue and set of ops is https://github.com/pytorch/pytorch/issues/30763
        # only testing one here because they should be fixed all at once

        with self.assertRaisesRegex(Exception, "Argument layout not provided."):
            def foo(x):
                return torch.ones_like(x, dtype=torch.double)
            foo(torch.tensor([2.]))
            print(torch.jit.script(foo).graph)


    def test_python_layer_ops_mismatches(self):
        def randint():
            return torch.randint(3, 5, (3,))[0]

        self.assertNotEqual(randint().dtype, torch.jit.script(randint)().dtype)

    def test_ops_bound_in_functional(self):
        ops_bound_in_functional = "lu_unpack", "unique", "lu"

        tensor = torch.tensor([2])
        funcs_template = dedent('''
        def func():
            return torch.{op}()
        ''')
        for op in ops_bound_in_functional:
            funcs_str = funcs_template.format(op=op)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            f = scope['func']
            with self.assertRaisesRegex(Exception, "Unknown builtin op"):
                cu = torch.jit.CompilationUnit(funcs_str)

        def fn():
            a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
            b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
            return torch.cdist(a, b, compute_mode="use_mm_for_euclid_dist")
        fn()
        with self.assertRaisesRegex(Exception, "Expected a value of type"):
            torch.jit.script(fn)

        def norm():
            c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
            return torch.norm(c, p="fro")

        norm()
        with self.assertRaisesRegex(Exception, "Expected a value of type"):
            torch.jit.script(norm)

        def unique_consec():
            x = torch.tensor([1])
            return torch.unique_consecutive(x, return_inverse=False, return_counts=True, dim=0)

        self.assertNotEqual(unique_consec(), torch.jit.script(unique_consec)())

        def tensordot():
            a = torch.arange(60.).reshape(3, 4, 5)
            b = torch.arange(24.).reshape(4, 3, 2)
            torch.tensordot(a, b, dims=([1, 0], [0, 1]))

        tensordot()
        with self.assertRaisesRegex(Exception, "Argument dims_self"):
            torch.jit.script(tensordot)

    def test_ops_bound_on_tensor(self):
        ops_not_bound_on_tensor = "retain_grad", "is_shared", "share_memory_", "lu", \
            "__reversed__", "register_hook", "lu", "refine_names", "rename_", "_update_names", "__contains__", \
            "apply_", "ndimension", "nelement", "numpy", "tolist", "bfloat16", "bool", "new_ones"

        tensor = torch.tensor([2])
        funcs_template = dedent('''
        def func(x):
            return x.{op}()
        ''')
        tensor = torch.tensor([2])
        for op in ops_not_bound_on_tensor:
            funcs_str = funcs_template.format(op=op)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            # attr exists in eager
            getattr(tensor, op)
            with self.assertRaisesRegex(Exception, "nonexistent attribute"):
                cu = torch.jit.CompilationUnit(funcs_str)

    def test_properties_not_bound_on_tensor(self):
        properties = "T", "_version", "grad_fn", "is_leaf", \
            "output_nr", "name", "ndim", "names"

        tensor = torch.tensor([2])
        funcs_template = dedent('''
        def func(x):
            return x.{property}
        ''')

        tensor = torch.tensor([2])
        for property in properties:
            funcs_str = funcs_template.format(property=property)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            # attr exists in eager
            f = scope['func']
            f(tensor)
            thrown = False
            try:
                torch.jit.CompilationUnit(funcs_str)
            except Exception:
                thrown = True
            self.assertTrue(thrown, property)
