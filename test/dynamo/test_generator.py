# Owner(s): ["module: dynamo"]
import itertools
import sys
import unittest
from collections import OrderedDict

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class GeneratorTestsBase(torch._dynamo.test_case.TestCaseWithNestedGraphBreaks):
    def setUp(self):
        super().setUp()
        self._old = torch._dynamo.config.enable_faithful_generator_behavior
        torch._dynamo.config.enable_faithful_generator_behavior = True
        self._unittest_old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_faithful_generator_behavior = self._old
        torch._dynamo.config.enable_trace_unittest = self._unittest_old

    def _compile_check(self, fn, args=None, fullgraph=True):
        eager = EagerAndRecordGraphs()
        if args is None:
            args = (torch.randn(2),)
        r = torch.compile(fn, backend=eager, fullgraph=fullgraph)(*args)
        self.assertGreater(len(eager.graphs), 0)
        return r


class GeneratorTests(GeneratorTestsBase):
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
            return list(zip(range(3), whoo(1))), t.sin()

        t = torch.randn(2)
        y, _ = fn(t)
        self.assertEqual(y, list(zip(range(3), whoo(1))))

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

    def test_reconstruct_generator_with_local_var_mutation(self):
        def whoo(t):
            x = 0
            yield t.sin() + x
            x += 1
            yield t.cos() + x
            x += 1
            yield t.tan() + x

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = whoo(t)
            next(gen)
            return t.sin(), gen

        t = torch.randn(2)
        y, g = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(g), [t.cos() + 1, t.tan() + 2])

    def test_reconstruct_generator_with_dict_mutation(self):
        counters.clear()

        def whoo(t, d):
            d[2] = t
            yield t.sin()
            yield t.cos()
            d[3] = t + 1
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, d):
            gen = whoo(t, d)
            next(gen)
            return t.sin(), whoo(t, d)

        t = torch.randn(2)
        d = {1: t}
        fn(t, d)
        self.assertEqual(len(counters["unimplemented"]), 1)
        self.assertIn(
            "Cannot reconstruct a generator with variable mutations",
            next(iter(counters["unimplemented"].keys())),
        )

    def test_reconstruct_generator_with_dict_mutation_before(self):
        def whoo(t, d):
            d[2] = t
            yield t.sin()
            yield t.cos()
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, d):
            gen = whoo(t, d)
            next(gen)
            return t.sin(), gen

        t = torch.randn(2)
        d = {1: t}
        y, g = fn(t, d)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(g), [t.cos(), t.tan()])
        self.assertEqual(d, {1: t, 2: t})

    def test_reconstruct_generator_with_object_mutation(self):
        class Counter:
            def __init__(self):
                self.x = 0

            def incr(self):
                self.x += 1

        def whoo(t, c):
            c.incr()
            yield t.sin()
            yield t.cos()
            c.incr()
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, c):
            gen = whoo(t, c)
            next(gen)
            return t.sin(), gen

        t = torch.randn(2)
        c = Counter()
        fn(t, c)
        self.assertEqual(len(counters["unimplemented"]), 1)
        self.assertIn(
            "Cannot reconstruct a generator with variable mutations",
            next(iter(counters["unimplemented"].keys())),
        )

    def test_reconstruct_generator_with_object_mutation_before(self):
        class Counter:
            def __init__(self):
                self.x = 0

            def incr(self):
                self.x += 1

        def whoo(t, c):
            c.incr()
            yield t.sin()
            yield t.cos()
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, c):
            gen = whoo(t, c)
            next(gen)
            # We should be able to reconstruct the generator as there's no object
            # mutation after the first yield
            return t.sin(), gen

        t = torch.randn(2)
        c = Counter()
        y, g = fn(t, c)
        self.assertEqual(c.x, 1)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(g), [t.cos(), t.tan()])

    def test_graph_break_and_reconstruct_generator(self):
        def whoo(t):
            yield t.sin()
            yield t.cos()
            yield t.tan()

        def g(t):
            torch._dynamo.graph_break()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            gen = whoo(t)
            next(gen)
            g(t)
            return t.sin(), list(gen)

        t = torch.randn(2)
        y, gen = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(gen), [t.cos(), t.tan()])

    def test_graph_break_in_generator_while_reconstructing(self):
        counters.clear()

        def whoo():
            yield 1
            torch._dynamo.graph_break()
            yield 2

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            gen = whoo()
            s = next(gen)
            torch._dynamo.graph_break()
            s += next(gen)
            return t + s

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 3)
        self.assertEqual(len(eager.graphs), 0)

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
            Unsupported, "Detected a method call to a user-defined generator object."
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
            Unsupported, "Detected a method call to a user-defined generator object."
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
            Unsupported, "Detected a method call to a user-defined generator object."
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
            Unsupported,
            "Detected a method call to a user-defined generator object.",
        ):
            fn(t, ctx)

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

        def fn(t):
            return zip(range(3), whoo(t)), t.sin()

        t = torch.randn(2)
        z, _ = self._compile_check(fn, args=(t,))
        self.assertEqual(list(z), list(zip(range(3), whoo(t))))

    @unittest.expectedFailure
    def test_zip_generator_2(self):
        def bar(t, i):
            return t + i

        def whoo(t):
            yield bar(t, 1)
            yield bar(t, 2)
            yield bar(t, 3)

        def fn(t):
            return zip(range(3), whoo(t))

        t = torch.randn(3)
        y = self._compile_check(fn, args=(t,), fullgraph=False)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, list(y))

    def test_zip_subgenerator(self):
        def subgen(t):
            yield t + 1
            yield t + 2

        def whoo(t):
            yield from subgen(t)
            yield t + 3

        def fn(t):
            return zip(range(3), whoo(t)), t.sin()

        t = torch.randn(2)
        z, _ = self._compile_check(fn, args=(t,))
        self.assertEqual(list(z), list(zip(range(3), whoo(t))))

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
        self.assertEqual(list(gen), [t + 1, t + 2, t + 3])

    def test_return_tuple_generator(self):
        def whoo(t):
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            g1, g2 = whoo(t), whoo(t + 1)
            return (g1, g2), t.sin()

        t = torch.randn(2)
        (g1, g2), _ = fn(t)
        self.assertEqual(list(g1), [t.sin(), t.cos()])
        self.assertEqual(list(g2), [(t + 1).sin(), (t + 1).cos()])

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
        self.assertEqual(list(gen), [t + 2, t + 3])

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

    @unittest.expectedFailure
    def test_reconstruct_generator_tensor_mutation(self):
        def whoo(t):
            yield t.sin_()
            yield t.cos_()

        def fn(t):
            gen = whoo(t)
            return gen

        with self.assertRaisesRegex(
            Unsupported,
            "Cannot reconstruct a generator with variable mutations",
        ):
            self._compile_check(fn)

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
        gen = fn(t)
        self.assertEqual(list(gen), [t + 2, t + 3])

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
        counters.clear()
        i = 0

        def whoo(t):
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j

        @torch.compile(backend="eager")
        def fn(t):
            return whoo(t), t.sin()

        t = torch.randn(2)
        fn(t)
        self.assertEqual(len(counters["unimplemented"]), 1)
        entry = next(iter(counters["unimplemented"].items()))
        self.assertIn(
            "Cannot reconstruct a generator with variable mutations.", entry[0]
        )
        self.assertEqual(entry[1], 1)

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

        @torch.compile(backend="eager")
        def fn(t):
            return whoo(t), t.sin()

        t = torch.randn(2)
        gen, y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(len(list(gen)), 5)
        for gb in counters["unimplemented"]:
            if "Cannot reconstruct a generator with variable mutations." in gb:
                break
        else:
            self.assertTrue(False, "expected side effect error; not found")

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
            next(gen)
            return gen, t.sin()

        t = torch.randn(2)
        gen, y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(len(list(gen)), 4)
        found = any(
            "Generator reconstruction with mutations" in msg
            and "Cannot reconstruct a generator with variable mutations" in msg
            for msg in counters["unimplemented"]
        )
        self.assertTrue(found)

    def test_generator_with_side_effects_graph_break_2(self):
        i = 0

        def whoo(t):
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j
                torch._dynamo.graph_break()

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            gen = whoo(t)
            return list(zip(range(3), gen))

        t = torch.randn(2)
        fn(t)
        self.assertEqual(len(eager.graphs), 0)

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    @unittest.expectedFailure
    def test_cleanup_throw(self):
        def nested_generator():
            try:
                yield 1
                yield 2
            except StopIteration:
                return 123  # noqa: B901

        def outer_generator():
            yield from nested_generator()
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = outer_generator()
            next(gen)  # Start the outer generator and enter the nested generato

            i = 0
            try:
                # Force an exception while the generator is running
                i = gen.throw(StopIteration("stop"))
            except RuntimeError:
                pass
            return (i, t.sin())

        t = torch.randn(2)
        i, y = self._compile_check(fn, args=(t,))
        self.assertEqual(i, 3)
        self.assertEqual(y, t.sin())

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

    def test_list_extend(self):
        def f(x):
            y = [1]
            y.extend(y[-1] + z for z in range(3))
            return x + 1, y

        self.assertEqual(
            f(torch.ones(3)),
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(3)),
        )

    def test_deque_extendleft(self):
        import collections

        def f(x):
            y = collections.deque([1])
            y.extendleft(y[0] + z for z in range(3))
            return x + 1, y

        self.assertEqual(
            f(torch.ones(3)),
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(3)),
        )

    @make_dynamo_test
    def test_generator___contains__(self):
        def whoo():
            yield 1
            yield 2

        g = whoo()
        self.assertTrue(1 in g)
        self.assertTrue(2 in g)
        self.assertRaises(StopIteration, next, g)
        self.assertFalse(3 in whoo())

    @make_dynamo_test
    def test_generator___contains___side_effects(self):
        n = 0

        def whoo():
            nonlocal n
            n = 1
            yield 1
            n = 2
            yield 2

        g = whoo()
        self.assertTrue(1 in g)
        self.assertEqual(n, 1)
        self.assertTrue(2 in g)
        self.assertEqual(n, 2)
        self.assertRaises(StopIteration, next, g)
        self.assertFalse(3 in whoo())


class TestGeneratorSend(GeneratorTestsBase):
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

    @parametrize("fullgraph", [True, False])
    def test_send_stop_iteration(self, fullgraph):
        def double():
            x = yield
            yield x * 2

        @torch.compile(backend="eager", fullgraph=fullgraph)
        def fn(t):
            gen = double()
            next(gen)
            a = gen.send(t)
            b = gen.send(t)  # should result in StopIteration
            return a + b

        t = torch.randn(2)
        if fullgraph:
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                fn(t)
        else:
            with self.assertRaises(StopIteration):
                fn(t)


class TestGeneratorClose(GeneratorTestsBase):
    def test_close(self):
        def whoo(t):
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    def test_close_subgen(self):
        z = 0

        def subgen(t):
            nonlocal z
            z = 1
            yield t.sin()
            z = 3
            yield t.cos()

        def whoo(t):
            yield from subgen(t)
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(z, 1)

    def test_close_with_side_effects(self):
        L = []
        z = 0

        def whoo(t):
            nonlocal z
            try:
                L.append(1)
                yield t.sin()
                L.append(2)
                yield t.cos()
            finally:
                L.append(z)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            z = -123
            gen.close()
            L.append(len(L))
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(L, [1, -123, 2])

    def test_close_capture_GeneratorExit_return(self):
        z = 0

        def whoo(t):
            nonlocal z
            try:
                z += 1
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                z += 10
                return t.tan()  # noqa: B901
            finally:
                z += 100

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            y = gen.close()
            return (i, y)

        t = torch.randn(2)
        (i, y) = fn(t)
        self.assertEqual(i, t.sin())
        self.assertEqual(y, t.tan())
        self.assertEqual(z, 111)

    @parametrize("fullgraph", [True, False])
    def test_close_capture_GeneratorExit(self, fullgraph):
        z = 0

        def whoo(t):
            nonlocal z
            try:
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                yield t.tan()
            finally:
                z = 1

        @torch.compile(backend="eager", fullgraph=fullgraph)
        def fn(t):
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)
        if fullgraph:
            # This should actually be RuntimeError("generator ignored GeneratorExit")
            # but Dynamo swallow the exception and raises Unsupported instead
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                fn(t)
        else:
            with self.assertRaisesRegex(
                RuntimeError, "generator ignored GeneratorExit"
            ):
                fn(t)

    def test_close_capture_and_reraise_GeneratorExit(self):
        L = []
        z = 0

        def whoo(t):
            nonlocal z
            try:
                L.append(1)
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                L.append(z)
                z = -1
                raise
            finally:
                L.append(z)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            z = -123
            gen.close()
            L.append(456)
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(L, [1, -123, -1, 456])

    @parametrize("exc", [RuntimeError, AttributeError])
    @make_dynamo_test
    def test_close_capture_and_reraise_exc(self, exc):
        def whoo(t):
            try:
                yield t.sin()
                yield t.cos()
            except GeneratorExit as e:
                raise exc from e
            finally:
                pass

        def fn(t):
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)

        z = 0
        try:
            fn(t)
        except exc:
            z = 1
        finally:
            assert z == 1  # noqa: S101

    def test_close_with_subgen(self):
        L = []
        z = 0

        def subgen(t):
            yield t.sin()
            yield t.cos()

        def whoo(t):
            nonlocal z
            L.append(10)
            yield from subgen(t)
            L.append(20)
            try:
                L.append(1)
                z = 4
                yield t.tan()
            finally:
                L.append(z)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            z = -123
            gen.close()
            L.append(456)
            return i, t.sin()

        t = torch.randn(2)
        y, _ = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(L, [10, 456])
        self.assertEqual(z, -123)

    def test_close_after_close(self):
        z = 0

        def whoo(t):
            nonlocal z
            try:
                z += 1
                yield t.sin()
                yield t.cos()
            finally:
                # finally should only be executed once
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return (i, gen.close())

        t = torch.randn(2)
        (i, y) = fn(t)
        self.assertEqual(i, t.sin())
        self.assertEqual(y, None)
        self.assertEqual(z, 2)

    @parametrize("fullgraph", [True, False])
    def test_next_after_close(self, fullgraph):
        def whoo(t):
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=fullgraph)
        def fn(t):
            gen = whoo(t)
            gen.close()
            a = next(gen)
            return [t.sin(), a]

        t = torch.randn(3)
        if fullgraph:
            with self.assertRaises(Unsupported):
                fn(t)
        else:
            with self.assertRaises(StopIteration):
                fn(t)

    def test_close_after_exception(self):
        def whoo(t):
            raise ValueError("foo")
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            try:
                next(gen)
            except ValueError:
                pass
            b = gen.close()
            return [t.sin(), b]

        t = torch.randn(2)
        y, b = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsNone(b)

    def test_close_handling_finally(self):
        z = 0

        def whoo(t):
            nonlocal z
            try:
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                z += 1
                return t.tan()  # noqa: B901
            finally:
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            next(gen)
            b = gen.close()
            return t.sin(), b

        t = torch.randn(2)
        y, b = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(b, t.tan())
        self.assertEqual(z, 2)


class TestGeneratorThrow(GeneratorTestsBase):
    def test_throw(self):
        def whoo(t):
            try:
                yield t.sin()
            except RuntimeError:
                yield t.cos()

        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(y, t.sin() + t.cos())

    def test_throw_with_finally(self):
        z = 0

        def whoo():
            nonlocal z
            z = 0
            try:
                try:
                    yield 1
                except ValueError:
                    yield 2
                finally:
                    z += 2
            except ValueError:
                z += 33
                yield 4
            finally:
                z += 1
            z += 10

        def f(x):
            gen = whoo()
            next(gen)
            gen.throw(ValueError)
            return x.sin()

        self._compile_check(f)
        self.assertEqual(z, 3)

    def test_throw_without_finally(self):
        z = 0

        def whoo(t):
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
                z += 10
            except RuntimeError:
                z += 100
                yield t.cos()
                z += 1_000
            z += 10_000

        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(y, t.sin() + t.cos())
        self.assertEqual(z, 101)

    def test_throw_no_yield_after_throw(self):
        z = 0

        def whoo(t):
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError:
                z += 10
            finally:
                z += 100

        def fn(t):
            gen = whoo(t)
            a = next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration:
                return a

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(z, 111)
        self.assertEqual(y, t.sin())

    def test_throw_not_catch(self):
        z = 0

        def whoo(t):
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError:
                z += 10
                yield t.cos()
            finally:
                z += 100

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        with self.assertRaises(RuntimeError):
            fn(t)

    def test_throw_raise_difference_exc(self):
        z = 0

        def whoo(t):
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError as e:
                z += 10
                raise RuntimeError from e
            finally:
                z += 100

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(ValueError)
            return a + b

        t = torch.randn(2)
        with self.assertRaises(RuntimeError):
            fn(t)

    def test_throw_yield_finally(self):
        z = 0

        def whoo(t):
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except RuntimeError:
                z += 10
                yield t.cos()
            finally:
                z += 100
                yield t.tan()  # RuntimeError: generator ignored GeneratorExit

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        with self.assertRaises(Unsupported):
            fn(t)

    def test_throw_try_except_finally(self):
        z = 0

        def whoo(t):
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError:
                z += 10
                yield t.cos()
            except RuntimeError:
                z += 100
                yield t.tan()
            finally:
                z += 1000
            z += 10_000

        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(y, t.sin() + t.tan())
        self.assertEqual(z, 1 + 100 + 1000)

    def test_exception_context_with_yield(self):
        def f():
            yield

        def fn(t):
            gen = f()
            gen.send(None)
            try:
                gen.throw(ValueError)
            except ValueError:
                z = 1
            except Exception as e:
                raise AssertionError from e
            assert z == 1  # noqa: S101
            return t.sin()

        self._compile_check(fn)

    def test_return_const_value_in_except_and_finally(self):
        def whoo():
            try:
                yield 1
            except ValueError:
                return 2  # noqa: B901
            finally:
                return 3  # noqa: B012, SIM107, B901

        def fn(t):
            gen = whoo()
            next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                assert e.args[0] == 3  # noqa: S101
            except Exception as e:
                raise AssertionError from e
            return t.sin()

        self._compile_check(fn)

    def test_return_value_in_except_and_finally(self):
        class Foo:
            def __init__(self, x):
                self.x = x

        def whoo():
            try:
                yield 1
            except ValueError:
                return Foo(2)  # noqa: B901
            finally:
                return Foo(3)  # noqa: B012, SIM107, B901

        def fn(t):
            gen = whoo()
            next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                assert e.args[0].x == 3  # noqa: S101
            except Exception as e:
                raise AssertionError from e
            return t.sin()

        self._compile_check(fn)

    def test_return_None_in_except_and_finally(self):
        def whoo():
            try:
                yield 1
            except ValueError:
                return 2  # noqa: B901
            finally:
                return  # noqa: B012, SIM107

        def fn(t):
            gen = whoo()
            next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                assert len(e.args) == 0  # noqa: S101
            except Exception as e:
                raise AssertionError from e
            return t.sin()

        self._compile_check(fn)


instantiate_parametrized_tests(GeneratorTests)
instantiate_parametrized_tests(TestGeneratorSend)
instantiate_parametrized_tests(TestGeneratorClose)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
