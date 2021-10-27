# Owner(s): ["oncall: jit"]

import torch
import torch.jit.frontend
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing import FileCheck, make_tensor

class TestFrontend(JitTestCase):

    def test_instancing_error(self):
        @torch.jit.ignore
        class MyScriptClass(object):
            def unscriptable(self):
                return "a" + 200


        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, x):
                return MyScriptClass()

        with self.assertRaises(torch.jit.frontend.FrontendError) as cm:
            torch.jit.script(TestModule())

        checker = FileCheck()
        checker.check("Cannot instantiate class")
        checker.check("def forward")
        checker.run(str(cm.exception))

    def test_python_frontend(self):
        def fn(x, y, z):
            q = None
            q = x + y - z.sigmoid()
            print(q)
            w = -z
            if not x and not y and z:
                m = x if not z else y
            while x < y > z:
                q = x
            assert 1 == 1, "hello"
            return x

        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        self.assertExpected(str(ast))

    def test_python_frontend_source_range(self):
        def fn():
            raise Exception("hello")
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        FileCheck().check("SourceRange at:") \
                   .check("def fn():") \
                   .check("~~~~~~~~~") \
                   .check('raise Exception("hello")') \
                   .check('~~~~~~~~~~~~~~~~~ <--- HERE') \
                   .run(str(ast.range()))

    def test_python_frontend_py3(self):
        def fn():
            raise Exception("hello")
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        self.assertExpected(str(ast))

    def test_dict_expansion_raises_error(self):
        def fn(self):
            d = {"foo": 1, "bar": 2, "baz": 3}
            return {**d}

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError,
                                    "Dict expansion "):
            torch.jit.script(fn)

    def test_for_else(self):
        def fn():
            c = 0
            for i in range(4):
                c += 10
            else:
                print("In else block of for...else")

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, "else branches of for loops aren't supported"):
            torch.jit.script(fn)

    def test_kwargs_error_msg(self):
        def other(**kwargs):
            print(kwargs)

        def fn():
            return other()

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'variable number'):
            torch.jit.script(fn)

        def another_other(*args):
            print(args)

        def another_fn():
            return another_other()

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'variable number'):
            torch.jit.script(another_fn)

    def test_kwarg_expansion_error(self):
        @torch.jit.ignore
        def something_else(h, i):
            pass

        def fn(x):
            something_else(**x)

        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, "keyword-arg expansion is not supported"):
            torch.jit.script(fn)
