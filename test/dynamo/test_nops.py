# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import eval_frame
from torch._dynamo.hooks import Hooks


c = 10


def fn1(a, b):
    return a + b - c


def fn2(a, b):
    x = 0
    y = 1

    def modify():
        nonlocal x
        x += a + b + c

    for _ in range(2):
        modify()

    return x + y


def fn3():
    yield 1
    yield 2


with_debug_nops = eval_frame._optimize_catch_errors(
    torch._dynamo.testing.debug_insert_nops, Hooks(None, None)
)


class NopTests(torch._dynamo.test_case.TestCase):
    @with_debug_nops
    def test1(self):
        self.assertEqual(fn1(1, 2), -7)
        self.assertEqual(fn1(1, 2), -7)

    @with_debug_nops
    def test2(self):
        self.assertEqual(fn2(1, 2), 27)
        self.assertEqual(fn2(1, 2), 27)

    @with_debug_nops
    def test3(self):
        t = fn3()
        self.assertEqual(next(t), 1)
        self.assertEqual(next(t), 2)
        self.assertRaises(StopIteration, lambda: next(t))

    def test_extended_args(self):
        too_many_adds = "+".join(["a", "b"] * 256)
        source = (
            f"lambda a, b: ({too_many_adds}+a if a.sum() > 0 else {too_many_adds} - b)"
        )
        fn = eval(source)
        a = torch.ones(1)
        b = torch.ones(1)
        fn = with_debug_nops(fn)
        self.assertEqual(fn(a, b).sum(), 513)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
