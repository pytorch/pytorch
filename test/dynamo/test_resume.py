# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case


def fn_creator():
    var1 = 1

    def fn(x):
        x = x + 1
        var2 = 1
        torch._dynamo.graph_break()
        x = x + var1

        def inner_fn():
            return var2

        return x

    return fn


class ResumeFunctionTests(torch._dynamo.test_case.TestCase):
    def test_freevars(self):
        fn = fn_creator()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(10))
        codes = [v for k, v in list(globals().items()) if k.startswith("__resume_at")]
        self.assertEqual(len(codes), 1)
        # co_freevars of resume functions, are sorted concatenation of the original function's co_freevars and co_cellvars
        self.assertEqual(codes[0].co_freevars, ("var1", "var2"))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
