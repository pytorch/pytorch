import torch
import torch.nn as nn
import unittest
from torch.testing._internal.jit_utils import JitTestCase

from torch.testing import FileCheck
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_utils import set_default_dtype

from torch.jit._recursive import wrap_cpp_module
from typing import Any
from itertools import product

import io

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

TEST_CUDA = torch.cuda.is_available()
TEST_ROCM = torch.cuda.is_available() and torch.version.hip is not None
TEST_CUDNN = False
if TEST_CUDA and not TEST_ROCM:  # Skip ROCM
    torch.ones(1).cuda()  # initialize cuda context
    TEST_CUDNN = TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1., device=torch.device('cuda:0')))

class TestFreezing(JitTestCase):
    def test_freeze_module(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.a = 1                      # folded
                self.b = 1.2                    # folded
                self.c = "hello"                # folded
                self.c2 = "hi\xA1"              # not folded
                self.d = [1, 1]                 # folded
                self.e = [1.0, 1.1]             # folded
                self.f = ["hello", "world"]     # folded
                self.f2 = [(1, "Over \u0e55\u0e57 57")]
                self.g = ([1, 2], 3.2, "4.4", torch.tensor([5.5], requires_grad=True))     # folded
                self.h = {"layer" : [torch.tensor([7.7], requires_grad=True)]}
                self.h2 = {"layer\xB1" : [torch.tensor([8.8], requires_grad=True)]}
                self.t = torch.tensor([1.2, 2.4], requires_grad=True)  # folded
                self.ts = [torch.tensor([1.0, 2.0], requires_grad=True), torch.tensor([3.0, 4.0], requires_grad=True)]  # folded
                self.tt = [[torch.tensor([3.3, 2.3], requires_grad=True), None]]

            def forward(self, x):
                return str(self.a) + str(self.b) + self.c + self.c2 + str(self.d) + \
                    str(self.e) + str(self.f) + str(self.f2) + str(self.g) +        \
                    str(self.h) + str(self.h2) + str(self.t) + str(self.ts) + str(self.tt)


        m = torch.jit.script(M())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        m._c = torch._C._freeze_module(m._c)
        buffer = io.BytesIO()
        torch.jit.save(m._c, buffer)
        buffer.seek(0)
        m2 = torch.jit.load(buffer)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     tt = ...
        #   }
        #   ...
        # }
        self.assertFalse(m2._c.hasattr('a'))
        self.assertFalse(m2._c.hasattr('b'))
        self.assertFalse(m2._c.hasattr('c'))
        self.assertFalse(m2._c.hasattr('c2'))
        self.assertFalse(m2._c.hasattr('d'))
        self.assertFalse(m2._c.hasattr('e'))
        self.assertFalse(m2._c.hasattr('f'))
        self.assertFalse(m2._c.hasattr('f2'))
        self.assertFalse(m2._c.hasattr('g'))
        self.assertFalse(m2._c.hasattr('h'))
        self.assertFalse(m2._c.hasattr('h2'))
        self.assertFalse(m2._c.hasattr('t'))
        self.assertFalse(m2._c.hasattr('ts'))
        self.assertFalse(m2._c.hasattr('tt'))
        output_f = m2.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_submodule(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                return self.a + self.b

        class SubModule2(nn.Module):
            def __init__(self):
                super(SubModule2, self).__init__()
                self.a = 12
                self.b = 2

            def forward(self, x):
                self.b = 30
                return self.a + self.b

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule2()
                self.a = 3
                self.b = 4

            def forward(self, x):
                self.b = 20
                return self.sub1(x) + self.a + self.b + self.sub2(x)

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch.jit.freeze(m)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     sub2 = ...
        #      b =
        #   }
        #   ...
        #   submodule {
        #     module m {
        #       attributes {
        #         sub2 = ...
        #         b =
        #       }
        #       ...
        #     }
        #   }
        # }
        mf = mf._c
        self.assertFalse(mf.hasattr('sub1'))
        self.assertFalse(mf.hasattr('a'))
        self.assertTrue(mf.hasattr('b'))
        self.assertTrue(mf.hasattr('sub2'))
        self.assertTrue(mf.sub2.hasattr('b'))   # verify b is preserved in sub2
        self.assertFalse(mf.sub2.hasattr('a'))  # verify a is removed in sub2
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                return self.a * self.b + x

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub = SubModule()

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(20, 20)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr('a'))
        self.assertFalse(mf.hasattr('b'))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nested_fork(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                return self.a * self.b + x

        class SubModule2(nn.Module):
            def __init__(self):
                super(SubModule2, self).__init__()
                self.sub = SubModule()
                self.c = torch.ones(20, 20)

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                return y_hat + y + self.c

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub = SubModule2()
                self.d = 1

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                self.d = 2
                return y_hat * y + self.d

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(20, 20)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr('a'))
        self.assertFalse(mf.hasattr('b'))
        self.assertFalse(mf.hasattr('c'))
        self.assertTrue(mf.hasattr('d'))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)


    def test_freeze_module_with_fork2(self):
        @torch.jit.script
        def foo(x):
            return x * 2

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                fut = torch.jit._fork(foo, self.a)
                y_hat = foo(self.b)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.a = ...
        #     self.b = ..
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. both 'a' and 'b' are preserved.
        self.assertTrue(mf.hasattr('a'))
        self.assertFalse(mf.hasattr('b'))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork_calling_module_method(self):
        @torch.jit.script
        def foo(x, y):
            return x * y

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            @torch.jit.export
            def foo(self, x):
                return x * self.a

            @torch.jit.export
            def bar(self, x):
                return x * self.b

            def forward(self, x):
                fut = torch.jit._fork(self.foo, self.b)
                y_hat = self.bar(self.a)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.b = ..
        #   }
        #   ...
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. 'b' is preserved.
        self.assertFalse(mf.hasattr('a'))
        self.assertTrue(mf.hasattr('b'))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_sharedclasstype(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self. b

            @torch.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        class SubModule2(nn.Module):
            def __init__(self):
                super(SubModule2, self).__init__()
                self.sub = SubModule()
                self.b = torch.tensor([3.3])

            def forward(self, x):
                y = self.sub.modify_b(x)
                return y + self.b

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub1 = SubModule()  # sub1 and sub2.sub shared same class type.
                self.sub2 = SubModule2()
                self.a = torch.tensor([4.4])

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z + self.a

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Checking if  Frozen module looks as  below
        # module mf {
        #   attributes {
        #     sub1 = ...
        #     sub2 = ...
        #   }
        #   ...
        #   submodules {
        #     module sub1 {
        #       attributes {
        #         a = ...
        #         b = ...
        #       }
        #       ...
        #     }
        #     module sub2 {
        #       attributes {
        #         sub = ...
        #       }
        #       ...
        #       submodule {
        #         module sub {
        #           attributes {
        #             a = ...
        #             b = ...
        #           }
        #           ...
        #         }
        #       }
        #     }
        #   }
        # }

        self.assertTrue(mf.hasattr('sub1'))
        self.assertTrue(mf.sub1.hasattr('a'))
        self.assertTrue(mf.sub1.hasattr('b'))
        self.assertFalse(mf.hasattr('a'))
        self.assertTrue(mf.hasattr('sub2'))
        self.assertTrue(mf.sub2.hasattr('sub'))
        self.assertFalse(mf.sub2.hasattr('b'))
        self.assertTrue(mf.sub2.sub.hasattr('a'))
        self.assertTrue(mf.sub2.sub.hasattr('b'))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nestedaliasing(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] = 10
                return self. b

            @torch.jit.export
            def modify_b(self, x):
                self.b[0] = 20
                return self.a
        Sub = SubModule()

        class SubModule2(nn.Module):
            def __init__(self):
                super(SubModule2, self).__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                return self.sub.a

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = torch.jit.script(TestModule())
        m.eval()
        mf = torch._C._freeze_module(m._c)
        self.assertTrue(mf.hasattr('sub1'))
        self.assertTrue(mf.sub1.hasattr('a'))
        self.assertFalse(mf.sub1.hasattr('b'))
        self.assertTrue(mf.hasattr('sub2'))
        self.assertTrue(mf.sub2.hasattr('sub'))
        self.assertTrue(mf.sub2.sub.hasattr('a'))  # Freezing detects that self.sub2.sub.a and self.sub1.a are alias
        self.assertFalse(mf.sub2.sub.hasattr('b'))
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    # FIXME: JIT is not honoring aliasing. 'Sub' module is copied. As a result
    # Eager and Script modules produce different output.
    def test_freeze_module_with_nestedaliasingscalar(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = 1.1
                self.b = 2.2

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a = 10.0
                return self. b

            @torch.jit.export
            def modify_b(self, x):
                self.b = 20.0
                return self.a
        Sub = SubModule()

        class SubModule2(nn.Module):
            def __init__(self):
                super(SubModule2, self).__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                return self.sub.a

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z
        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c)
        self.assertTrue(mf.hasattr('sub1'))
        self.assertTrue(mf.sub1.hasattr('a'))
        self.assertFalse(mf.sub1.hasattr('b'))
        # sub2 is fully folded becasue self.sub1 and self.sub2.sub are not alias (Scripting bug)
        self.assertFalse(mf.hasattr('sub2'))
        input = torch.randn(2, 2)
        output = m.forward(input)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        # Should be equal
        self.assertNotEqual(output, output_s)
        self.assertEqual(output_s, output_f)


    def test_freeze_module_with_preserve_sub_module(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                return self.a

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)
        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c, ["sub1"])

        # Test that 'sub1' is preserved entirely and 'sub2' is completely folded
        self.assertTrue(mf.hasattr('sub1'))
        self.assertTrue(mf.sub1.hasattr('a'))
        self.assertTrue(mf.sub1.hasattr('b'))
        self.assertFalse(mf.hasattr('sub2'))
        input = torch.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module_and_mutation(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                self.a[0] = 3.3
                return self.a

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)
        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c, ["sub1"])

        # Test that be both sub1 and sub1 are preserved and 'b' is preserved
        # even if it is not used. To fulfill user request to preserve 'sub1'
        self.assertTrue(mf.hasattr('sub1'))
        self.assertTrue(mf.sub1.hasattr('a'))
        self.assertTrue(mf.sub1.hasattr('b'))
        self.assertTrue(mf.hasattr('sub2'))
        self.assertTrue(mf.sub2.hasattr('a'))
        self.assertTrue(mf.sub2.hasattr('b'))
        input = torch.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)


    def test_freeze_module_with_helperfunction(self):
        class SubModule(nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                return self.a + self.b

        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.sub = SubModule()
                self.a = 3
                self.b = 4

            def forward(self, x):
                self.b = 20
                return self._forward(x) + self.a + self.b

            def _forward(self, x):
                return self.sub(x)
        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        mf = torch._C._freeze_module(m._c)
        self.assertFalse(mf.hasattr('sub'))
        self.assertFalse(mf.hasattr('a'))
        self.assertTrue(mf.hasattr('b'))
        with self.assertRaisesRegex(AttributeError, "TestModule \(.*\) does not have a field with name '_forward'"):  # noqa: W605
            mf._forward(x)

    def test_freeze_module_with_inplace_mutable(self):
        class FreezeMe(torch.jit.ScriptModule):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = [11, 22]

            @torch.jit.script_method
            def forward(self, x):
                for i in range(3):
                    self.a.append(i)
                return self.a

        m = FreezeMe()
        m.eval()
        m_f = torch._C._freeze_module(m._c)
        self.assertTrue(m_f.hasattr('a'))
        m.forward(torch.tensor([3]))
        out = m_f.forward(torch.tensor([5]))
        expected = [11, 22, 0, 1, 2, 0, 1, 2]
        self.assertEqual(out, expected)

    # Mutable attributes
    def test_freeze_module_with_mutable_list(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = [1, 2]

            def forward(self, x):
                return self.a

        m = FreezeMe()
        m.eval()
        m.a.append(3)
        m_s = torch.jit.script(m)
        v = m_s.a
        v.append(4)
        m_s.a = v
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # Post-freezing mutating m_s.a  does not affect m_f (m_f has its own copy).
        v = m_s.a
        v.append(5)
        m_s.a = v
        self.assertFalse(m_f.hasattr('a'))
        out = m_f.forward(torch.tensor([5]))
        expected = [1, 2, 3, 4]
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_dict(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = {"layer" : "4"}

            def forward(self, x):
                return self.a

            @torch.jit.export
            def modify_a(self, x):
                self.a["layer"] = self.a["layer"] + "1"
                return self.a

        m = FreezeMe()
        m.eval()
        m.a["layer2"] = "3"
        m_s = torch.jit.script(m)
        t = torch.tensor(5)
        m_s.modify_a(t)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        m.a["layer2"] += "2"
        m_s.modify_a(t)
        self.assertFalse(m_f.hasattr('a'))
        out = m_f.forward(t)
        expected = {"layer" : "411", "layer2" : "3"}
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1., 2., 3.])

            def forward(self, x):
                return self.a

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.a[1] += 3.0
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # Post-freezing tensor attribute mutations affect m_f.
        # FIXME: deep copy all folded attributes so that m_f has full ownership.
        m_s.a[0] += 5.0
        self.assertFalse(m_f.hasattr('a'))
        out = m_f.forward(torch.tensor([5]))
        expected = [6., 5., 3.]
        self.assertEqual(out, expected)

    def test_freeze_module_with_tuple(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = (torch.tensor([1, 2, 3, 4, 5, 6]), "hi")

            def forward(self, x):
                if (x[0] == 2.0):
                    self.a[0][0] = 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([2.0])
        expected = m_s.forward(inp)
        m_s.a[0][0] = 1
        m_f = torch._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr('a'))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])

            def forward(self, x):
                x = self.a.view(2, 3)
                x[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr('a'))
        m_f.a[0] -= 10
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_list(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = [torch.tensor([1, 2, 3, 4, 5, 6])]

            def forward(self, x):
                self.a[0][1] += 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0][1] -= 10
        m_f = torch._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr('a'))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = self.a.view(2, 3)

            def forward(self, x):
                self.b[1] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr('a'))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = torch.tensor(51)  # 1+2+3+14+15+16
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr2(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = {"layer" : ([self.a.view(2, 3), torch.tensor([10])], 20)}
                self.c = ([self.a.view(2, 3), torch.tensor([10])], 20)
                self.d = (self.a.view(2, 3), 20)

            def forward(self, x):
                self.d[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        with self.assertRaisesRegex(RuntimeError, "module contains attributes values that overlaps"):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_tensor_attr3(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                self.a[1] += 10
                return self.b[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr('a'))
        self.assertTrue(m_f.hasattr('b'))
        out = m_f.forward(inp)
        expected += 10  # account for  self.a += 10.
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr4(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                self.b[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0] -= 10
        with self.assertRaisesRegex(RuntimeError, "module contains attributes values that overlaps"):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_overlapping_attrs(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6])

        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.b = [a.view(3, 2), torch.tensor([10])]
                self.c = (20, a.view(2, 3))

            def forward(self, x):
                self.b[0][0] += 10
                return self.c[1].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        a[0] -= 10
        with self.assertRaisesRegex(RuntimeError, "module contains attributes values that overlaps"):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_attr(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = self.a
                self.c = (self.a, 10)

            def forward(self, x):
                self.b[1] += 10
                return str(self.a) + str(self.c)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # FIXME: It should be assertTrue. Currently scripting is making a copy for setting self.b (see #33034)
        self.assertFalse(m_f.hasattr('a'))
        self.assertFalse(m_f.hasattr('c'))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m_s.forward(inp)
        self.assertEqual(out, expected)

    # Check attribute a is preserved. Alias analysis detects that 'a' has output writers.
    # In this example, 'a' is not mutated. However, we do not track which sub
    # values of a composite ivalue is mutated.
    def test_freeze_module_with_aliased_attr2(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                v = self.a
                self.b = (v, [12])
                v2 = self.b[1]
                v2.append(7)
                return str(v) + str(v2)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr('a'))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_attr3(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                v = self.a
                v2 = (v, [12])
                v3 = v2[0]
                v3.append(7)
                return str(self.a)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr('a'))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_return_self(self):
        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.a = torch.tensor([1., 2., 3.])

            def forward(self, x):
                return self

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        with self.assertRaisesRegex(RuntimeError, "attempted to freeze a module that return itself"):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_return_sub_module(self):

        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)

            def forward(self, x):
                return self.conv1

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr('conv1'))

    def test_freeze_module_no_forward(self):

        class FreezeMe(nn.Module):
            def __init__(self):
                super(FreezeMe, self).__init__()
                self.lin = nn.Linear(10, 1)

            @torch.jit.export
            def foo(self, x):
                return self.lin(x)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c, preservedAttrs=['foo'])
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))


    def test_freeze_module_in_training_mode(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = nn.functional.log_softmax(x, dim=1)
                return output

        model = torch.jit.script(Net())
        model.train()
        mTrain_freezed = torch._C._freeze_module(model._c)
        # verify mTrain_freezed looks exactly as:
        # module {
        #   attributes {
        #     conv1 = ...
        #     conv2 = ...
        #     dropout1 = ...
        #     dropout2 = ...
        #     fc1 = ...
        #     fc2 = ...
        #   }
        #   ...
        #   submodules {
        #     module conv1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module conv2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module dropout1 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module dropout2 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module fc1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module fc2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        self.assertFalse(mTrain_freezed.hasattr('training'))
        self.assertTrue(mTrain_freezed.hasattr('conv1'))
        self.assertFalse(mTrain_freezed.conv1.hasattr('training'))
        self.assertTrue(mTrain_freezed.conv1.hasattr('weight'))
        self.assertTrue(mTrain_freezed.conv1.hasattr('bias'))
        self.assertTrue(mTrain_freezed.hasattr('conv2'))
        self.assertFalse(mTrain_freezed.conv2.hasattr('training'))
        self.assertTrue(mTrain_freezed.conv2.hasattr('weight'))
        self.assertTrue(mTrain_freezed.conv2.hasattr('bias'))
        self.assertTrue(mTrain_freezed.hasattr('dropout1'))
        self.assertTrue(mTrain_freezed.dropout1.hasattr('training'))
        self.assertTrue(mTrain_freezed.hasattr('dropout2'))
        self.assertTrue(mTrain_freezed.dropout2.hasattr('training'))
        self.assertTrue(mTrain_freezed.hasattr('fc1'))
        self.assertTrue(mTrain_freezed.fc1.hasattr('weight'))
        self.assertTrue(mTrain_freezed.fc1.hasattr('bias'))
        self.assertTrue(mTrain_freezed.hasattr('fc2'))
        self.assertTrue(mTrain_freezed.fc2.hasattr('weight'))
        self.assertTrue(mTrain_freezed.fc2.hasattr('bias'))
        model.eval()
        mEval_freezed = torch._C._freeze_module(model._c)
        self.assertFalse(mEval_freezed.hasattr('conv1'))
        self.assertFalse(mEval_freezed.hasattr('conv2'))
        self.assertFalse(mEval_freezed.hasattr('dropout1'))
        self.assertFalse(mEval_freezed.hasattr('training'))
        self.assertFalse(mEval_freezed.hasattr('fc1'))
        self.assertFalse(mEval_freezed.hasattr('dropout2'))
        self.assertFalse(mEval_freezed.hasattr('fc2'))
        with self.assertRaisesRegex(AttributeError, "does not have a field with name 'state_dict'"):
            print(mEval_freezed.state_dict())
        buffer = io.BytesIO()
        torch.jit.save(mEval_freezed, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        FileCheck().check_not('GetAttr[name=') \
                   .run(m._c._get_method('forward').graph)
        m2 = torch._C._freeze_module(model._c, preserveParameters=True)
        self.assertTrue(m2.hasattr('conv1'))
        self.assertTrue(m2.hasattr('conv2'))
        self.assertFalse(m2.hasattr('dropout1'))
        self.assertFalse(m2.hasattr('training'))
        self.assertTrue(m2.hasattr('fc1'))
        self.assertFalse(m2.hasattr('dropout2'))
        self.assertTrue(m2.hasattr('fc2'))

    def test_freeze_module_detach_gradient(self):
        mod = nn.Conv2d(8, 3, 4, 2, 1)
        self.assertTrue(mod.weight.requires_grad)
        smod = torch.jit.script(mod)
        smod.eval()
        fmod = torch._C._freeze_module(smod._c)
        self.assertTrue(mod.weight.requires_grad)
        self.assertTrue(smod.weight.requires_grad)
        self.assertFalse(fmod.hasattr('weight'))
        inp = torch.ones(1, 8, 32, 32)
        out1 = fmod.forward(inp)
        # FIXME: frozen module mutated from outside (original module).
        with torch.no_grad():
            smod.weight[0, 0, 0, 0] += 100.0
        out2 = fmod.forward(inp)
        out3 = smod(inp)
        self.assertNotEqual(out1, out2)
        self.assertEqual(out2, out3)

    def test_freeze_module_with_user_preserved_attr(self):
        class Module(nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["a"])
        # Attribute "a" is preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))

    def test_freeze_module_with_user_preserved_method(self):
        class Module(nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        # Both attribute "a" and method "modify_a" are preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))
        input = torch.randn(2, 2)
        expected = m.forward(input)
        out = fm.forward(input)
        self.assertEqual(out, expected)

    def test_freeze_module_with_user_preserved_method2(self):
        class Module(nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                self.b += 10
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b + self.a

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        FileCheck().check('prim::GetAttr[name="a"]').run(fm.forward.graph)
        FileCheck().check('prim::GetAttr[name="b"]').run(fm.modify_a.graph)

    @skipIfNoFBGEMM
    def test_module_with_shared_type_instances(self):
        class Child(nn.Module):
            def __init__(self):
                super(Child, self).__init__()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)

            def forward(self, x):
                x = self.conv1(x)
                return x

        class Parent(nn.Module):
            def __init__(self):
                super(Parent, self).__init__()
                self.quant = torch.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)
                self.child = Child()
                self.child2 = Child()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.child2(x)
                x = self.dequant(x)
                return x

        def _static_quant(model):
            qModel = torch.quantization.QuantWrapper(model)
            qModel.qconfig = torch.quantization.default_qconfig
            torch.quantization.prepare(qModel, inplace=True)
            qModel(torch.rand(4, 1, 4, 4, dtype=torch.float32))
            torch.quantization.convert(qModel, inplace=True)
            return model

        with override_quantized_engine('fbgemm'):
            data = torch.randn(4, 1, 4, 4, dtype=torch.float32)
            m = Parent().to(torch.float32)
            m = _static_quant(m)
            m = torch.jit.script(m)
            m.eval()
            torch._C._jit_pass_inline(m.graph)
            m_frozen = wrap_cpp_module(torch._C._freeze_module(m._c))
            # Earlier bug resulted in _packed_params set to false.
            FileCheck().check_not('_packed_params = False').run(m_frozen._c.dump_to_str(True, True, False))

            m_res = m(data)
            # It used to segfault while running frozen module.
            m_frozen_res = m_frozen(data)
            self.assertEqual(m_res, m_frozen_res)

    def test_module_getattr_indirection(self):
        @torch.jit.script
        class ValHolder(object):
            def __init__(self, val: int):
                self.val: int = val

        class Mod(nn.Module):
            def __init__(self):
                super(Mod, self).__init__()
                self.mod1 = ValHolder(1)
                self.mod2 = ValHolder(2)

            def forward(self, cond: bool):
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        mod = Mod()
        mod.eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        mod_eager = Mod()
        self.assertEqual(mod_eager(True), frozen_mod(True))
        self.assertEqual(mod_eager(False), frozen_mod(False))

    def test_freeze_module_with_non_static_module_container_index(self):
        """
        Test that Modules containing non-static ModuleDict or ModuleList
        indexing cannot be frozen.
        """
        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                pass

        class ImplementsInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        class ModWithDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.d = torch.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = torch.jit.script(ModWithDict())
        m.eval()
        with self.assertRaisesRegex(RuntimeError, "Freezing modules containing prim::ModuleContainerIndex is not supported"):
            mf = torch._C._freeze_module(m._c)

        class ModWithList(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        m = torch.jit.script(ModWithList())
        m.eval()
        with self.assertRaisesRegex(RuntimeError, "Freezing modules containing prim::ModuleContainerIndex is not supported"):
            mf = torch._C._freeze_module(m._c)

    def test_freeze_non_module_class_getattr(self):
        class BoxCoder(object):
            def __init__(self, bbox_xform_clip):
                # type: (float) -> None
                self.bbox_xform_clip = bbox_xform_clip

            def decode(self, input):
                return input * self.bbox_xform_clip

        class MyModule(torch.nn.Module):
            __annotations__ = {
                'box_coder': BoxCoder,
            }

            def __init__(self):
                super(MyModule, self).__init__()
                self.box_coder = BoxCoder(50.)

            def forward(self, input):
                return self.box_coder.decode(input)

        model = MyModule()
        model.eval()
        script_model = torch.jit.freeze(torch.jit.script(model))
        inp = torch.randn([4, 4])
        output_eager = model(inp)
        self.assertEqual(model(inp), script_model(inp))
        FileCheck().check_not("GetAttr").run(script_model.graph)

class TestFrozenOptimizations(JitTestCase):
    def setUp(self):
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)

    def tearDown(self):
        torch.set_default_dtype(self.default_dtype)

    def test_conv_bn_folding(self):
        conv_bias = [True, False]
        module_pairs = [(nn.Conv1d, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d), (nn.Conv3d, nn.BatchNorm3d)]
        use_tracing = [True, False]

        for use_bias, modules, tracing in product(conv_bias, module_pairs, use_tracing):
            class ConvBN(torch.nn.Module):
                def __init__(self, in_channels, out_channels, **kwargs):
                    super(ConvBN, self).__init__()
                    self.conv = modules[0](in_channels, out_channels, bias=use_bias, **kwargs)
                    self.bn = modules[1](out_channels, eps=0.001)

                def forward(self, x):
                    x = self.conv(x)
                    return self.bn(x)

            mod_eager = ConvBN(3, 32, kernel_size=3, stride=2).eval()
            inps = [4, 3, 4]
            if modules[0] == nn.Conv2d:
                inps.append(inps[-1])
            if modules[0] == nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps)

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            self.run_pass("peephole", scripted_mod.graph)
            self.run_pass("constant_propagation", scripted_mod.graph)

            FileCheck().check("conv").check("batch").run(scripted_mod.graph)
            # successfully no-ops with non-const inputs
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            FileCheck().check("conv").check("aten::batch_norm").run(scripted_mod.graph)

            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_conv_bn", scripted_mod.graph)
            FileCheck().check("conv").check_not("aten::batch_norm").run(scripted_mod.graph)

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))


    def test_conv_add_folding(self):

        @torch.no_grad()
        def test_conv_fusion(use_bias, module, tracing, op, scalar, add_tensor, expect_success):

            class ConvOp(torch.nn.Module):
                __constants__ = ['use_scalar']

                def __init__(self, in_channels, out_channels, tensor=None, **kwargs):
                    super(ConvOp, self).__init__()
                    self.conv = module(in_channels, out_channels, bias=use_bias, **kwargs)
                    self.conv2 = module(in_channels, out_channels, bias=use_bias, **kwargs)
                    self.use_scalar = scalar
                    tensor_size = [1 for _ in range(self.conv.weight.ndim)]
                    tensor_size[1] = self.conv.weight.size(0)
                    self.tensor = add_tensor if add_tensor is not None else torch.rand(tensor_size)
                    self.op = op

                def forward(self, x):
                    x = self.conv(x)
                    if self.use_scalar:
                        return self.op(x, 2.)
                    else:
                        return self.op(x, self.tensor)

            mod_eager = ConvOp(3, 32, kernel_size=3, stride=2).eval()

            inps = [4, 3, 4]
            if module == nn.Conv2d:
                inps.append(inps[-1])
            if module == nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])


            inp = torch.rand(inps)

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp,))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            self.run_pass("inline", scripted_mod.graph)
            op_str = "aten::" + op.__name__

            FileCheck().check("conv").check(op_str).run(scripted_mod.graph)
            # successively no-ops with non-const inputs
            self.run_pass("fold_frozen_conv_mul_or_div", scripted_mod.graph)
            self.run_pass("fold_frozen_conv_add_or_sub", scripted_mod.graph)
            FileCheck().check("conv").check(op_str).run(scripted_mod.graph)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("fold_frozen_conv_mul_or_div", scripted_mod.graph)
            self.run_pass("fold_frozen_conv_add_or_sub", scripted_mod.graph)

            if expect_success:
                FileCheck().check("conv").check_not(op_str).run(scripted_mod.graph)
            else:
                FileCheck().check("conv").check(op_str).run(scripted_mod.graph)

            self.assertEqual(mod_eager(inp), scripted_mod(inp))
            self.assertEqual(mod_eager(inp), scripted_mod(inp))

        conv_bias = [True, False]
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        use_tracing = [False, True]
        use_scalar = [False, True]
        ops = [torch.add, torch.sub, torch.mul, torch.div]

        for use_bias, module, tracing, pytorch_op, scalar in product(conv_bias, modules, use_tracing, ops, use_scalar):
            test_conv_fusion(use_bias, module, tracing, pytorch_op, scalar, add_tensor=None, expect_success=True)


        for use_bias, pytorch_op in product(conv_bias, ops):
            # broadcasting add
            test_conv_fusion(use_bias, nn.Conv2d, False, pytorch_op, False,
                             add_tensor=torch.rand(32, 1, 32), expect_success=False)

            # broadcasting add
            test_conv_fusion(use_bias, nn.Conv2d, False, pytorch_op, False, add_tensor=torch.rand(1, 1), expect_success=True)

            # add with different dtype
            test_conv_fusion(use_bias, nn.Conv2d, False, pytorch_op, False,
                             add_tensor=torch.rand(1).to(torch.int), expect_success=False)

    def test_optimize_freeze_module(self):
        in_channels, out_channels = 3, 32
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)
        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
        mod = torch.nn.Sequential(conv, bn)
        # set optimize to False here, by default freezing runs optimize_frozen_module
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize_numerics=False)
        # inspect frozen mod
        FileCheck().check("batch_norm").run(frozen_mod.graph)
        torch.jit.optimize_frozen_module(frozen_mod)
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

        # optimize_frozen_module should be run
        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()))
        FileCheck().check_not("batch_norm").run(frozen_mod.graph)

    def test_freeze_remove_dropout(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        mod = torch.jit.script(Net())
        # inspect mod
        torch._C._jit_pass_inline(mod.graph)
        FileCheck().check("aten::dropout").run(mod.graph)
        frozen_mod = torch.jit.freeze(mod.eval())
        FileCheck().check_not("aten::dropout").run(frozen_mod.graph)

        input = torch.randn(2)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_remove_feature_dropout(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.dropout = nn.Dropout2d(0.5)

            def forward(self, x):
                return self.dropout(x)

        mod = torch.jit.script(Net().eval())
        # inspect mod
        torch._C._jit_pass_inline(mod.graph)
        FileCheck().check("aten::feature_dropout").run(mod.graph)
        frozen_mod = torch.jit.freeze(mod)
        FileCheck().check_not("aten::feature_dropout").run(frozen_mod.graph)

        input = torch.randn(2, 2)
        output_s = mod.forward(input)
        output_f = frozen_mod.forward(input)
        self.assertEqual(output_s, output_f)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_conv_to_mkldnn(self):
        with set_default_dtype(torch.float):
            for module, trace in product([nn.Conv2d, nn.Conv3d], [False, True]):
                mod = module(3, 32, kernel_size=3, stride=2).eval()
                inps = [4, 3, 4]
                if module == nn.Conv2d:
                    inps.append(inps[-1])
                if module == nn.Conv3d:
                    inps.append(inps[-1])
                    inps.append(inps[-1])

                inp = torch.rand(inps)
                if trace:
                    scripted_mod = torch.jit.script(mod)
                else:
                    scripted_mod = torch.jit.trace(mod, (inp,))

                self.run_pass("inline", scripted_mod.graph)

                FileCheck().check("conv").run(scripted_mod.graph)
                # successfully no-ops with non-const inputs
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check_not("to_mkldnn").run(scripted_mod.graph)

                scripted_mod = torch.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check("to_mkldnn").check("aten::conv").check("to_dense").run(scripted_mod.graph)

                self.assertEqual(mod(inp), scripted_mod(inp))
                self.assertEqual(mod(inp), scripted_mod(inp))

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_linear_to_mkldnn(self):

        with set_default_dtype(torch.float):
            # make sure mkldnn handles broadcast rules
            inp_shapes = [[20], [20, 20], [1, 20, 20]]
            for inp_shape in inp_shapes:
                mod = nn.Linear(20, 30).eval()
                scripted_mod = torch.jit.script(mod)
                inp = torch.rand(inp_shape)

                self.run_pass("inline", scripted_mod.graph)
                FileCheck().check("aten::linear").run(scripted_mod.graph)
                # successfully no-ops with non-const inputs
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check_not("ConvertToMKLDNN").run(scripted_mod.graph)

                scripted_mod = torch.jit.freeze(scripted_mod)
                self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
                FileCheck().check("to_mkldnn").check("aten::linear").check("to_dense").run(scripted_mod.graph)

                self.assertEqual(mod(inp), scripted_mod(inp))
                self.assertEqual(mod(inp), scripted_mod(inp))

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_collapse_adjacent_conversions(self):

        with set_default_dtype(torch.float):
            mod = nn.Sequential(nn.Linear(20, 20), nn.Linear(20, 20)).eval()
            scripted_mod = torch.jit.script(mod)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            FileCheck().check("to_mkldnn").check("aten::linear").check("aten::linear").check("to_dense").run(scripted_mod.graph)
            FileCheck().check_count("to_mkldnn", 1, exactly=True).run(scripted_mod.graph)

            inp = torch.rand([20, 20])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

            # testing unsupported behavior
            class Add(nn.Module):
                def __init__(self, tensor):
                    super().__init__()
                    self.tensor = tensor

                def forward(self, x):
                    return x + self.tensor

            def test_unsupported(module, preserved_attrs=None):
                mod = torch.jit.freeze(torch.jit.script(module.eval()), preserved_attrs)
                self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
                FileCheck().check("to_mkldnn").check("linear").check("to_dense").check("add").run(mod.graph)

            lin = nn.Linear(20, 20)
            # Scalar-Tensor not supported
            test_unsupported(nn.Sequential(lin, Add(.5)))
            # # 0-dim not supported
            test_unsupported(nn.Sequential(lin, Add(torch.tensor(.5))))
            # tensor of unknown dtype (getAttr node here) not supported
            test_unsupported(nn.Sequential(lin, Add(torch.tensor([20]))), ['1'])

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_mkldnn_fuser_broadcasting(self):
        class Add(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                return x + self.tensor

        with set_default_dtype(torch.float):
            mod = nn.Sequential(nn.Linear(20, 20), Add(torch.rand([20]))).eval()
            scripted_mod = torch.jit.script(mod)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            FileCheck().check("prim::BroadcastMKLDNNTensors").run(scripted_mod.graph)
            inp = torch.rand([20, 20])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

            # for good measure, check that broadcasting does not work without this op
            # so we can remove the op if it ever gets supported
            with self.assertRaisesRegex(RuntimeError, ""):
                torch.rand([20, 20]).to_mkldnn() + torch.rand([20]).to_mkldnn()

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_mkldnn_inplace_removal(self):
        class AddMul(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.tensor = tensor

            def forward(self, x):
                return x.add_(self.tensor).div_(self.tensor) - 4

        with set_default_dtype(torch.float):
            mod = nn.Sequential(nn.Linear(20, 20), AddMul(torch.rand([20]))).eval()
            scripted_mod = torch.jit.script(mod)
            scripted_mod = torch.jit.freeze(scripted_mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", scripted_mod.graph)
            FileCheck().check("aten::to_mkldnn").check_not("aten::add_").check("aten::div_").run(scripted_mod.graph)
            inp = torch.rand([20, 20])
            self.assertEqual(scripted_mod(inp), mod(inp))
            self.assertEqual(scripted_mod(inp), mod(inp))

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    @skipIfNoTorchVision
    def test_maxpool_mkldnn(self):
        with set_default_dtype(torch.float):
            model = torchvision.models.resnet18()
            sub_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            mod = torch.jit.freeze(torch.jit.script(sub_model.eval()))
            N, C, H, W, = 10, 3, 224, 224
            inp = torch.randn(N, C, H, W)
            self.run_pass("convert_frozen_ops_to_mkldnn", mod.graph)
            FileCheck().check("max_pool").check("to_dense").run(mod.graph)
            FileCheck().check_count("to_dense", 1, exactly=True).run(mod.graph)
            self.assertEqual(mod(inp), sub_model(inp))

    @unittest.skipIf(torch._C.has_mkldnn, "Testing no mkldnn")
    def test_conv_to_mkldnn_no_mkldnn(self):
        # test no error when mkldnn not available
        with set_default_dtype(torch.float):
            mod = torch.jit.script(nn.Conv2d(3, 32, kernel_size=3, stride=2).eval())
            frozen = torch.jit.freeze(mod)
            self.run_pass("convert_frozen_ops_to_mkldnn", frozen.graph)
            inp = torch.rand([4, 3, 4, 4])
            self.assertEqual(frozen(inp), mod(inp))

    @unittest.skipIf(not TEST_CUDNN, "requires CUDNN")
    def test_freeze_conv_relu_fusion(self):
        conv_bias = [True, False]
        conv_ops = [nn.Conv2d]
        add_z = [True, False]
        use_tracing = [True, False]
        for use_bias, conv, add_z, tracing in product(conv_bias, conv_ops, add_z, use_tracing):
            class Net(nn.Module):
                def __init__(self, in_channels, out_channels, **kwargs):
                    super(Net, self).__init__()
                    self.conv = conv(in_channels, out_channels, bias=use_bias, **kwargs)
                    self.relu = nn.ReLU(inplace=True)
                    self.add_z = add_z

                def forward(self, x):
                    z = self.conv(x)
                    out = self.conv(x)
                    if self.add_z:
                        out += z
                    out = self.relu(out)
                    return out

            mod_eager = Net(3, 6, kernel_size=3, stride=2).eval().cuda()

            inps = [5, 3, 4]
            if conv == nn.Conv2d:
                inps.append(inps[-1])
            if conv == nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])
            inp = torch.rand(inps).cuda()

            if tracing:
                scripted_mod = torch.jit.trace(mod_eager, (inp))
            else:
                scripted_mod = torch.jit.script(mod_eager)

            frozen_mod = torch.jit.freeze(scripted_mod)
            FileCheck().check("aten::relu").run(frozen_mod.graph)
            self.run_pass("fuse_frozen_conv_add_relu", frozen_mod.graph)
            if add_z:
                FileCheck().check("aten::cudnn_convolution_add_relu").run(frozen_mod.graph)
            else:
                FileCheck().check("aten::cudnn_convolution_relu").run(frozen_mod.graph)

            self.assertEqual(mod_eager(inp), frozen_mod(inp))
