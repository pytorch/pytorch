# Owner(s): ["module: dynamo"]
import itertools
from collections import OrderedDict

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class GeneratorTests(torch._dynamo.test_case.TestCase):
    expected_failures = []

    def run(self, result=None):
        # Override the run method to inject the "expectingFailure" marker
        # when the test case runs.
        marker = "__unittest_expecting_failure__"
        for test_name in dir(self):
            test_method = getattr(self, test_name)
            if test_name.startswith("test_") and not getattr(
                test_method, marker, False
            ):
                getattr(self, test_name).__dict__[marker] = (
                    test_name in self.expected_failures
                )
        return super().run(result=result)

    def setUp(self):
        super().setUp()
        torch._dynamo.config.enable_yield_on_generator = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_yield_on_generator = False

    def test_generator_simple(self):
        def whoo():
            yield 1
            yield 2
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo()
            t = t + next(gen)
            t = t + next(gen)
            t = t + next(gen)
            return t

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 6)

    def test_infinite_generator(self):
        def whoo():
            i = 0
            while True:
                yield i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo()
            t = t + next(gen)
            t = t + next(gen)
            t = t + next(gen)
            return t

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 3)

    def test_infinite_generator_2(self):
        def whoo(t):
            i = 0
            while True:
                yield t + i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(zip(range(3), whoo(t)))

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, list(zip(range(3), whoo(t))))

    def test_infinite_generator_3(self):
        def whoo(i):
            while True:
                yield i

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(zip(range(3), whoo(1)))

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, list(zip(range(3), whoo(1))))

    def test_iter(self):
        def whoo():
            i = 0
            while True:
                yield i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            s = 0
            for i in whoo():
                if i > 5:
                    break
                s += i
            return t + s

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + sum(range(6)))

    def test_graph_break_in_generator(self):
        def whoo():
            yield 1
            torch._dynamo.graph_break()
            yield 2

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            gen = whoo()
            s = next(gen)
            s += next(gen)
            return t + s

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 3)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_in_generator_2(self):
        def whoo(x):
            yield x.sin()
            torch._dynamo.graph_break()
            yield x.cos()

        def call_whoo(x):
            gen = whoo(x)
            sin = next(gen)
            cos = next(gen)
            return sin, cos

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            sin, cos = call_whoo(t)
            return sin + cos

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin() + t.cos())
        self.assertEqual(len(eager.graphs), 1)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_0_: "f32[2]", L_stack0_1_: "f32[2]"):
        l_stack0_0_ = L_stack0_0_
        l_stack0_1_ = L_stack0_1_

        add: "f32[2]" = l_stack0_0_ + l_stack0_1_;  l_stack0_0_ = l_stack0_1_ = None
        return (add,)
""",
        )

    def test_generator_as_argument(self):
        # The inline tracer needs to be kept in sync if an already advanced generator
        # is given to a compiled function.
        def whoo():
            yield 1
            yield 2
            yield 3

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo()
        next(ctx)
        with self.assertRaisesRegex(
            Unsupported, "Generator as graph argument is not supported"
        ):
            fn(t, ctx)

    def test_generator_as_argument_2(self):
        def whoo(x):
            yield x.sin()
            yield x.cos()

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo(t)
        next(ctx)
        with self.assertRaisesRegex(
            Unsupported, "Generator as graph argument is not supported"
        ):
            fn(t, ctx)

    def test_generator_as_argument_3(self):
        # The inline tracer needs to be kept in sync if an already advanced generator
        # is given to a compiled function.
        def whoo():
            yield 1
            yield 2
            yield 3

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo()
        with self.assertRaisesRegex(
            Unsupported, "Generator as graph argument is not supported"
        ):
            fn(t, ctx)

    def test_generator_as_argument_4(self):
        def whoo(x):
            yield x.sin()
            yield x.cos()

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo(t)
        with self.assertRaisesRegex(
            Unsupported, "Generator as graph argument is not supported"
        ):
            fn(t, ctx)

    def test_send(self):
        def double():
            x = yield
            yield x * 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = double()
            next(gen)
            return gen.send(t)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t * 2)

    def test_islice_chain(self):
        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t):
            tmp1 = [t + 1, t + 2]
            tmp2 = [t + 3, t + 4]
            return list(itertools.chain(tmp1, tmp2))

        t = torch.tensor([1.0])
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3, t + 4])

    def test_zip_generator(self):
        def whoo(t):
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return zip(range(3), whoo(t))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, list(y))

    def test_zip_generator_2(self):
        def bar(t, i):
            return t + i

        def whoo(t):
            yield bar(t, 1)
            yield bar(t, 2)
            yield bar(t, 3)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return zip(range(3), whoo(t))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, list(y))

    def test_zip_subgenerator(self):
        def subgen(t):
            yield t + 1
            yield t + 2

        def whoo(t):
            yield from subgen(t)
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return zip(range(3), whoo(t))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, list(y))

    def test_list_zip_generator(self):
        def whoo(t):
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(zip(range(3), whoo(t)))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, y)

    def test_zip_infinite_generator(self):
        def whoo(t):
            i = 0
            while True:
                yield t + i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(zip(range(3), whoo(t)))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, y)

    @parametrize("container", [list, tuple, dict, OrderedDict])
    def test_dict_tuple_list_generator(self, container):
        def whoo(t):
            yield 1, t + 1
            yield 2, t + 2
            yield 3, t + 3

        def fn(t):
            gen = whoo(t)
            return container(gen)

        t = torch.randn(2)
        expected = fn(t)
        got = torch.compile(backend="eager", fullgraph=True)(fn)(t)
        self.assertEqual(expected, got)

    def test_return_generator(self):
        def whoo(t):
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            return gen

        t = torch.tensor([1.0])
        gen = fn(t)
        self.assertEqual(next(gen), torch.tensor([2.0]))

    def test_return_advanced_generator(self):
        def whoo(t):
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            next(gen)
            return gen

        t = torch.tensor([1.0])
        gen = fn(t)
        self.assertEqual(next(gen), torch.tensor([3.0]))

    def test_return_exhaust_generator(self):
        def whoo(t):
            yield t + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            next(gen)
            return gen

        t = torch.tensor([1.0])
        gen = fn(t)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_subgenerator(self):
        def subgen(t):
            yield t + 1
            yield t + 2

        def main_gen(t):
            yield from subgen(t)
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = main_gen(t)
            return list(gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3])

    def test_return_subgenerator(self):
        def subgen(t):
            yield t + 1
            yield t + 2

        def main_gen(t):
            yield from subgen(t)
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = main_gen(t)
            next(gen)
            return gen

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(list(y), [t + 2, t + 3])

    def test_dynamo_disable_generator(self):
        @torch._dynamo.disable
        def main_gen(t):
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = main_gen(t)
            return list(gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3])

    def test_dynamo_disable_sub_generator(self):
        @torch._dynamo.disable
        def subgen(t):
            yield t + 2
            yield t + 3

        def main_gen(t):
            yield t + 1
            yield from subgen(t)

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = main_gen(t)
            return list(gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3])

    def test_graph_break_outside_generator(self):
        def whoo(t):
            yield t + 1
            yield t + 2

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = whoo(t)
            x = next(gen)
            torch._dynamo.graph_break()
            y = next(gen)
            return x + y

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, (t + 1) + (t + 2))

    def test_graph_break_before_calling_generator(self):
        def whoo(t):
            for perm in itertools.product(itertools.permutations((0, 1, 2)), repeat=1):
                yield sum(perm[0])

        def fn(t):
            s = 0
            for b, p in itertools.product(whoo(t), itertools.permutations((4, 5))):
                s += b
            return s

        t = torch.randn(2)
        expected = fn(t)
        got = torch.compile(backend="eager", fullgraph=False)(fn)(t)
        self.assertEqual(expected, got)

    def test_generator_with_side_effects(self):
        i = 0

        def whoo(t):
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            return zip(range(3), gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(i, 0)
        self.assertEqual(list(y), [(0, t), (1, t + 1), (2, t + 2)])
        self.assertEqual(i, 3)

    def test_subgenerator_with_side_effects(self):
        i = 0

        def subgen(t):
            nonlocal i
            i += 1
            yield t
            i += 1
            yield t + 1

        def whoo(t):
            nonlocal i
            yield from subgen(t)
            i += 1
            yield t + 2
            i += 1
            yield t + 3
            i += 1
            yield t + 4

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            return zip(range(3), gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(i, 0)
        self.assertEqual(list(y), [(0, t), (1, t + 1), (2, t + 2)])
        self.assertEqual(i, 3)

    def test_generator_with_side_effects_graph_break(self):
        i = 0

        def whoo(t):
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = whoo(t)
            torch._dynamo.graph_break()
            return list(zip(range(3), gen))

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(i, 3)
        self.assertEqual(y, [(0, t), (1, t + 1), (2, t + 2)])

    def test_generator_with_side_effects_graph_break_2(self):
        i = 0

        def whoo(t):
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j
                torch._dynamo.graph_break()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = whoo(t)
            return list(zip(range(3), gen))

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(i, 3)
        self.assertEqual(y, [(0, t), (1, t + 1), (2, t + 2)])


class GeneratorTestsOldBehavior(GeneratorTests):
    expected_failures = [
        "test_generator_as_argument",
        "test_generator_as_argument_2",
        "test_generator_as_argument_3",
        "test_generator_as_argument_4",
        "test_infinite_generator",
        "test_infinite_generator_2",
        "test_infinite_generator_3",
        "test_iter",
        "test_graph_break_in_generator",
        "test_zip_infinite_generator",
        "test_generator_with_side_effects",
        "test_generator_with_side_effects_graph_break",
        "test_subgenerator_with_side_effects",
        "test_send",
    ]

    def setUp(self):
        super().setUp()
        torch._dynamo.config.enable_yield_on_generator = False

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_yield_on_generator = True


instantiate_parametrized_tests(GeneratorTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
