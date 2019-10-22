import unittest
import os
import sys

import torch
import torch.nn as nn

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from jit_utils import JitTestCase

class TestProperty(JitTestCase):
    def test_basic(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = 0

            @property
            def x_and_1(self):
                return self.x + 1

            def forward(self, new_x):
                # type: (int) -> int
                self.x = new_x
                return self.x_and_1

        m = M()
        self.checkModule(m, (0,),)

    def test_inheritance_style(self):
        """
        Just make sure things work in the old inheritance-style ScriptModule creation
        """
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__()
                self.x = 0

            @property
            def x_and_1(self):
                return self.x + 1

            @torch.jit.script_method
            def forward(self, new_x):
                # type: (int) -> int
                self.x = new_x
                return self.x_and_1

        m = M()
        self.assertEqual(m(0), 1)
        self.assertEqual(m(5), 6)

    def test_property_as_tuple(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = 0

            @property
            def tuple_of_xs(self):
                return self.x, self.x

            def forward(self, new_x):
                # type: (int) -> int
                self.x = new_x
                foo, bar = self.tuple_of_xs
                return foo + bar

        m = M()
        self.checkModule(m, (0,),)

    def test_tensor_method(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = torch.rand(2, 3)

            @property
            def prop(self):
                return self.x

            def forward(self):
                return self.prop.add(torch.rand(2, 3))

        m = M()
        self.checkModule(m, ())

    def test_tensor_slice(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = torch.rand(2, 3)

            @property
            def prop(self):
                return self.x

            def forward(self):
                return self.prop[:]

        m = M()
        self.checkModule(m, ())

    def test_list_len(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = [1, 2, 3]

            @property
            def prop(self):
                return self.x

            def forward(self):
                # type () -> int
                return len(self.prop)

        m = M()
        self.checkModule(m, ())

    @unittest.skip("TODO list comprehension code kind of hairy")
    def test_list_comprehension(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = [1, 2, 3]

            @property
            def prop(self):
                return self.x

            def forward(self):
                # type () -> int
                a = [x + 5 for x in self.prop]
                return a

        m = M()
        self.checkModule(m, ())

    def test_property_setters_deleters(self):
        """
        Test that @property with setters and deleters are disallowed.
        We can allow them in the future, it's just not implemented yet.
        """
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = 0

            @property
            def x_and_1(self):
                return self.x + 1

            @x_and_1.setter
            def x_and_1(self):
                pass

            def forward(self, new_x):
                # type: (int) -> int
                self.x = new_x
                return self.x_and_1

        with self.assertRaisesRegex(RuntimeError, 'setter'):
            torch.jit.script(M())

        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = 0

            @property
            def x_and_1(self):
                return self.x + 1

            @x_and_1.deleter
            def x_and_1(self):
                pass

            def forward(self, new_x):
                # type: (int) -> int
                self.x = new_x
                return self.x_and_1

        with self.assertRaisesRegex(RuntimeError, 'deleter'):
            torch.jit.script(M())


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")
