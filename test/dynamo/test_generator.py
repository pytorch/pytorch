# Owner(s): ["module: dynamo"]
import itertools
import sys
import types
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


class GeneratorTestsBase(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_load_build_class = True
        self._unittest_old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._unittest_old
        torch._dynamo.config.enable_trace_load_build_class = self._prev

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
        with self.assertRaises(Unsupported):
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
        with self.assertRaises(Unsupported):
            fn(t, ctx)

    def test_generator_as_argument_3(self):
        # An unstarted generator passed as an argument is re-inlined as if its
        # function were called here, so next() works.
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
        self.assertEqual(fn(t, ctx), t + 1)

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
        with self.assertRaises(Unsupported):
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

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw_custom_StopIteration(self):
        class MyStopIteration(StopIteration):
            pass

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
                i = gen.throw(MyStopIteration("stop"))
            except RuntimeError:
                pass
            return (i, t.sin())

        t = torch.randn(2)
        i, y = self._compile_check(fn, args=(t,))
        self.assertEqual(i, 3)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw_subgen_return_value(self):
        # CLEANUP_THROW must resume the delegating generator with the
        # subgenerator's return value (StopIteration.value). When the
        # subgenerator catches the thrown StopIteration and returns, the
        # yield-from expression must evaluate to that returned value. Dynamo
        # currently extracts the value of the *thrown* exception instead.
        def nested_generator():
            try:
                yield 1
                yield 2
            except StopIteration:
                return 123  # noqa: B901

        def outer_generator():
            r = yield from nested_generator()
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = outer_generator()
            next(gen)
            i = gen.throw(StopIteration("stop"))
            return (i, t.sin())

        t = torch.randn(2)
        i, y = fn(t)
        self.assertEqual(i, 123)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw_empty_stopiteration(self):
        def nested_generator():
            yield 1
            yield 2

        def outer_generator():
            yield from nested_generator()
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = outer_generator()
            next(gen)
            gen.throw(StopIteration())
            return t.sin()

        t = torch.randn(2)
        with self.assertRaises(RuntimeError):
            fn(t)

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

    def test_raise_immediately(self):
        # see https://github.com/python/cpython/issues/143493
        @torch.compile(fullgraph=True, backend="eager")
        def f(s):
            return (x for x in s)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "'int' object is not iterable"
        ):
            f(1)

    def test_pep479_raise_stopiteration_in_body(self):
        # PEP 479: a StopIteration that escapes a generator body is converted
        # to RuntimeError. On 3.12+ the compiler emits CALL_INTRINSIC_1 3 for
        # this; on earlier versions Dynamo converts at the generator frame
        # boundary. The branch taken (RuntimeError vs StopIteration) is encoded
        # in the returned tensor so the eager/compiled results can be compared.
        def fn(t):
            def whoo():
                yield t + 1
                raise StopIteration("boom")

            g = whoo()
            next(g)
            try:
                next(g)
            except RuntimeError:
                return t + 1.0
            except StopIteration:
                return t + 2.0
            return t

        t = torch.randn(2)
        self.assertEqual(self._compile_check(fn, args=(t,)), t + 1.0)

    @make_dynamo_test
    def test_pep479_stopiteration_error(self):
        # Ported from CPython test_generators ExceptionTest.test_stopiteration_error.
        # Asserts the RuntimeError message, not just the type.
        def gen():
            raise StopIteration
            yield

        with self.assertRaisesRegex(RuntimeError, "raised StopIteration"):
            next(gen())

    @make_dynamo_test
    def test_pep479_tutorial_stopiteration(self):
        # Ported from CPython test_generators ExceptionTest.test_tutorial_stopiteration.
        def f():
            yield 1
            raise StopIteration
            yield 2  # never reached

        g = f()
        self.assertEqual(next(g), 1)
        with self.assertRaisesRegex(RuntimeError, "raised StopIteration"):
            next(g)

    def test_pep479_stopiteration_from_inner_next(self):
        # Tutorial PEP 479 case: StopIteration leaking from an inner next()
        # inside the generator body is also converted to RuntimeError.
        def fn(t):
            def inner():
                yield t + 1

            def whoo():
                it = inner()
                while True:
                    yield next(it)

            g = whoo()
            next(g)
            try:
                next(g)
            except RuntimeError:
                return t + 1.0
            except StopIteration:
                return t + 2.0
            return t

        t = torch.randn(2)
        self.assertEqual(self._compile_check(fn, args=(t,)), t + 1.0)

    @unittest.skipIf(
        sys.version_info >= (3, 12),
        "pre-3.12 converts StopIteration at the generator frame boundary",
    )
    @make_dynamo_test
    def test_pep479_exception_stack_balanced(self):
        # The pre-3.12 StopIteration -> RuntimeError conversion must not leave
        # the StopIteration on the exception stack. Otherwise a later bare
        # `raise` re-raises the leftover StopIteration instead of reporting that
        # there is no active exception.
        def gen():
            yield 1
            raise StopIteration

        g = gen()
        next(g)
        try:
            next(g)  # StopIteration escapes the body -> RuntimeError
        except RuntimeError:
            pass
        with self.assertRaisesRegex(RuntimeError, "No active exception to reraise"):
            raise  # noqa: PLE0704


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

    def test_yield_from_return_value(self):
        # `yield from` evaluates to the subgenerator's return value, which the
        # SEND opcode extracts from the StopIteration raised on completion.
        def subgen(t):
            yield t.sin()
            return t.cos()  # noqa: B901

        def outer(t):
            r = yield from subgen(t)
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(outer(t))

        t = torch.randn(2)
        self.assertEqual(fn(t), [t.sin(), t.cos()])

    def test_yield_from_return_none(self):
        # A subgenerator that falls off the end returns None (StopIteration
        # with no value); `yield from` must evaluate to None.
        def subgen(t):
            yield t.sin()

        def outer(t):
            r = yield from subgen(t)
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(outer(t))

        t = torch.randn(2)
        out = fn(t)
        self.assertEqual(out[0], t.sin())
        self.assertIsNone(out[1])

    def test_send_through_yield_from(self):
        # Values sent into the delegating generator are forwarded to the
        # subgenerator: SEND with a non-None value routes through `send`.
        def subgen():
            x = yield 10
            y = yield x + 1
            return y * 100  # noqa: B901

        def outer():
            r = yield from subgen()
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = outer()
            a = gen.send(None)
            b = gen.send(7)
            c = gen.send(3)
            return [a, b, c, t.sin()]

        t = torch.randn(2)
        out = fn(t)
        self.assertEqual(out[:3], [10, 8, 300])
        self.assertEqual(out[3], t.sin())

    def test_nested_yield_from_return_value(self):
        # Return values propagate through a chain of `yield from` delegations.
        def leaf(t):
            yield t.sin()
            return t.cos()  # noqa: B901

        def mid(t):
            r = yield from leaf(t)
            return r + 1  # noqa: B901

        def top(t):
            r = yield from mid(t)
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(top(t))

        t = torch.randn(2)
        self.assertEqual(fn(t), [t.sin(), t.cos() + 1])

    def test_yield_from_iterable_return_none(self):
        # `yield from` over a non-generator iterable yields its items via the
        # SEND tp_iternext path and evaluates to None.
        def outer(t):
            r = yield from [t.sin(), t.cos()]
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            return list(outer(t))

        t = torch.randn(2)
        out = fn(t)
        self.assertEqual(out[0], t.sin())
        self.assertEqual(out[1], t.cos())
        self.assertIsNone(out[2])


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

    def test_untrack_generator_after_hop(self):
        # Regression: speculate_subgraph clones SideEffects and leaves the clone
        # as the live instance, so the set of open generators must be owned by
        # OutputGraph, not SideEffects. torch.cond's branch speculation swaps in
        # a clone (which never carried local_generators); exhausting a generator
        # created before the cond then untracked it on the clone, raising
        # "ValueError: list.remove(x): x not in list".
        # See https://github.com/pytorch/pytorch/pull/157149
        def whoo(t):
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            a = next(gen)
            # cond speculates both branches, swapping SideEffects to a clone
            b = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
            acc = a + b
            # Exhaust the generator after the swap: untrack must not crash
            for x in gen:
                acc = acc + x
            return acc

        t = torch.randn(2)
        y = fn(t)
        ref = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
        self.assertEqual(y, t.sin() + ref + t.cos())

    def test_close_open_generator_after_hop(self):
        # An open (non-exhausted) generator created before a HOP must still be
        # closed at compile_subgraph time after SideEffects is swapped, so its
        # finally block runs. Reads the tracking list from OutputGraph.
        z = 0

        def whoo(t):
            nonlocal z
            try:
                yield t.sin()
                yield t.cos()
            finally:
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            a = next(gen)
            b = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
            return a + b  # gen left open; finally must run via close

        t = torch.randn(2)
        y = fn(t)
        ref = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
        self.assertEqual(y, t.sin() + ref)
        self.assertEqual(z, 1)

    def test_close_open_generator_fast_path(self):
        # An open generator whose only pending work is a finally block must be
        # closed at compile_subgraph time even when the frame is eligible for
        # the fast path (single frame, all-tensor stack, empty side effects).
        # Returning an input tensor keeps side effects empty -- a freshly
        # produced tensor (e.g. t.sin()) is tracked as a new mutation and would
        # divert to the slow path, which closes generators unconditionally.
        # The fast-path guard `not self.local_generators` at output_graph.py is
        # what forces close_local_generators to run here so the finally fires.
        z = 0

        def whoo(t):
            nonlocal z
            try:
                yield t
                yield t
            finally:
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            gen = whoo(t)
            next(gen)  # start generator; finally now pending
            return t  # return input tensor -> fast-path eligible

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t)
        self.assertEqual(z, 1)

    def test_close_replays_new_object_attr_mutation(self):
        # A generator closed at frame exit may, in its finally block, act on a
        # new object reachable only through the generator's frame. Its attribute
        # mutations (AttributeMutationNew) must survive prune_dead_object_new so
        # close() can replay the finally; otherwise the mutation is lost.
        class C:
            pass

        def fn(t):
            marker = []

            def gen():
                obj = C()
                obj.data = marker  # AttributeMutationNew on a new object
                try:
                    yield t.sin()
                finally:
                    obj.data.append(1)  # replayed when the generator is closed

            g = gen()
            next(g)
            return t.sin(), marker

        t = torch.randn(3)
        self.assertEqual(fn(t), self._compile_check(fn, args=(t,)))

    def test_deep_tee_buffer_no_recursion_error(self):
        # VariableTracker.visit walks generator/iterator state during compile.
        # A long itertools.tee buffer is a deeply chained structure; visit uses
        # an explicit worklist so it must not overflow the Python stack.
        n = 2000  # exceeds the default recursion limit

        def fn(t):
            a, b = itertools.tee(iter(range(n)))
            list(a)  # advance one side so the shared tee buffer grows to n
            return t + 1

        t = torch.randn(3)
        self._compile_check(fn, args=(t,))


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
            except StopIteration as e:
                if len(e.args) > 0:
                    raise AssertionError(
                        "Expected StopIteration with no arguments"
                    ) from e
                return a
            raise AssertionError("Expected StopIteration")

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


class TestGeneratorPEP(GeneratorTestsBase):
    # Ported from CPython Lib/test/test_generators.py `pep_tests` doctest block.

    @make_dynamo_test
    def test_resume_running_generator(self):
        # A generator cannot be resumed while it is actively running.
        def g():
            i = next(me)
            yield i

        me = g()
        self.assertRaisesRegex(ValueError, "generator already executing", next, me)

    @make_dynamo_test
    def test_return_is_not_stopiteration(self):
        # return simply exits; it is not caught by a bare except.
        def f1():
            try:
                return
            except:  # noqa: E722
                yield 1

        self.assertEqual(list(f1()), [])

        # a raised StopIteration is caught by a bare except, like any exception.
        def f2():
            try:
                raise StopIteration
            except:  # noqa: E722
                yield 42

        self.assertEqual(list(f2()), [42])

    @make_dynamo_test
    def test_exception_propagation(self):
        def f():
            return 1 // 0

        def g():
            yield f()  # the zero division exception propagates
            yield 42  # and we'll never get here

        k = g()
        self.assertRaises(ZeroDivisionError, next, k)
        self.assertRaises(StopIteration, next, k)  # cannot be resumed

    @make_dynamo_test
    def test_try_except_finally(self):
        def f():
            try:
                yield 1
                try:
                    yield 2
                    1 // 0
                    yield 3  # never get here
                except ZeroDivisionError:
                    yield 4
                    yield 5
                    raise
                except:  # noqa: E722
                    yield 6
                yield 7  # the "raise" above stops this
            except:  # noqa: E722
                yield 8
            yield 9
            try:
                x = 12  # noqa: F841
            finally:
                yield 10
            yield 11

        self.assertEqual(list(f()), [1, 2, 4, 5, 8, 9, 10, 11])

    @unittest.expectedFailure
    @make_dynamo_test
    def test_recursive_inorder_tree(self):
        class Tree:
            def __init__(self, label, left=None, right=None):
                self.label = label
                self.left = left
                self.right = right

            def __iter__(self):
                return inorder(self)

        def tree(lst):
            n = len(lst)
            if n == 0:
                return []
            i = n // 2
            return Tree(lst[i], tree(lst[:i]), tree(lst[i + 1 :]))

        def inorder(t):
            if t:
                for x in inorder(t.left):
                    yield x
                yield t.label
                for x in inorder(t.right):
                    yield x

        t = tree("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.assertEqual("".join(t), "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class TestGeneratorCoroutine(GeneratorTestsBase):
    # Ported from CPython Lib/test/test_generators.py `coroutine_tests` doctest
    # block. Cases relying on stdout capture were rewritten to record into a
    # list and assert; cases relying on gc finalization, traceback-level or
    # gi_frame introspection, and the deprecated 3-arg throw() were omitted.

    @make_dynamo_test
    def test_send_into_started_generator(self):
        sent = []

        def f():
            sent.append((yield 1))
            yield 2

        g = f()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.send(42), 2)
        self.assertEqual(sent, [42])

    @make_dynamo_test
    def test_send_non_none_to_just_started(self):
        def f():
            yield 1
            yield 2

        self.assertRaisesRegex(
            TypeError,
            "can't send non-None value to a just-started generator",
            f().send,
            "foo",
        )

    @make_dynamo_test
    def test_bare_yield_yields_none(self):
        def f():
            yield

        self.assertEqual(list(f()), [None])

    @make_dynamo_test
    def test_yield_in_generator_expression(self):
        def f():
            list(i for i in [(yield 26)])  # noqa: C400

        self.assertIsInstance(f(), types.GeneratorType)

    @make_dynamo_test
    def test_augmented_assignment_coroutine(self):
        def coroutine(seq):
            count = 0
            while count < 200:
                count += yield
                seq.append(count)

        seq = []
        c = coroutine(seq)
        next(c)
        self.assertEqual(seq, [])
        c.send(10)
        self.assertEqual(seq, [10])
        c.send(10)
        self.assertEqual(seq, [10, 20])
        c.send(10)
        self.assertEqual(seq, [10, 20, 30])

    @make_dynamo_test
    def test_throw_caught_in_loop(self):
        caught = []

        def f():
            while True:
                try:
                    yield
                except ValueError as v:
                    caught.append(str(v))

        g = f()
        next(g)
        g.throw(ValueError)
        self.assertEqual(caught, [""])
        g.throw(ValueError("xyz"))
        self.assertEqual(caught, ["", "xyz"])

    # gen.throw() of a non-exception should raise a catchable TypeError, but
    # Dynamo surfaces it as an uncatchable InternalTorchDynamoError.
    @unittest.expectedFailure
    @make_dynamo_test
    def test_throw_non_exception_is_type_error(self):
        def f():
            while True:
                try:
                    yield
                except ValueError:
                    pass

        g = f()
        next(g)
        self.assertRaises(TypeError, g.throw, "abc")
        self.assertRaises(TypeError, g.throw, 0)
        self.assertRaises(TypeError, g.throw, list)

    @make_dynamo_test
    def test_throw_terminates_generator(self):
        caught = []

        def f():
            while True:
                try:
                    yield
                except ValueError as v:
                    caught.append(str(v))

        g = f()
        next(g)
        self.assertRaises(TypeError, g.throw, TypeError)
        self.assertRaises(StopIteration, g.send, 2)

    @make_dynamo_test
    def test_throw_on_just_opened_generator(self):
        def f():
            yield 1

        self.assertRaisesRegex(ValueError, "7", f().throw, ValueError(7))

    @make_dynamo_test
    def test_close_catches_generator_exit(self):
        log = []

        def f():
            try:
                yield
            except GeneratorExit:
                log.append("exiting")

        g = f()
        next(g)
        g.close()
        self.assertEqual(log, ["exiting"])
        g.close()  # should be a no-op now
        self.assertEqual(log, ["exiting"])

    @make_dynamo_test
    def test_close_on_various_states(self):
        def f():
            yield

        f().close()  # close before opening
        g = f()
        next(g)
        g.close()  # close normally

    @make_dynamo_test
    def test_generator_exit_not_caught_by_except_exception(self):
        log = []

        def f():
            try:
                yield
            except Exception:
                log.append("except")
            finally:
                log.append("finally")

        g = f()
        next(g)
        g.close()
        self.assertEqual(log, ["finally"])

    @make_dynamo_test
    def test_generator_ignored_generator_exit(self):
        def f():
            try:
                yield
            except GeneratorExit:
                yield "foo!"

        g = f()
        next(g)
        self.assertRaisesRegex(RuntimeError, "generator ignored GeneratorExit", g.close)
        g.close()

    @make_dynamo_test
    def test_error_during_close_propagates(self):
        def f():
            try:
                yield
            except GeneratorExit:
                raise TypeError("fie!") from None

        g = f()
        next(g)
        self.assertRaisesRegex(TypeError, "fie!", g.close)

    @make_dynamo_test
    def test_yield_expression_makes_generator(self):
        def f():
            x = yield  # noqa: F841

        self.assertIsInstance(f(), types.GeneratorType)

    @make_dynamo_test
    def test_send_to_subscript_targets(self):
        def f(d):
            d[(yield "a")] = d[(yield "b")] = 27

        data = [1, 2]
        g = f(data)
        self.assertEqual(g.send(None), "a")
        self.assertEqual(data, [1, 2])
        self.assertEqual(g.send(0), "b")
        self.assertEqual(data, [27, 2])
        self.assertRaises(StopIteration, g.send, 1)
        self.assertEqual(data, [27, 27])


class _DelegatingIterator:
    # An iterator (not a generator) implementing the generator protocol, used to
    # exercise `yield from <iterator>` where the subiterator has its own
    # throw()/close(). Defined at module scope so tracing doesn't hit the local
    # class definition (__build_class__) limitation.
    def __init__(self, log):
        self.log = log

    def __iter__(self):
        return self

    def __next__(self):
        return 1

    def send(self, value):
        return 1

    def throw(self, typ, val=None, tb=None):
        self.log.append("iter throw")
        return 2

    def close(self):
        self.log.append("iter close")


class TestSubgeneratorDelegation(GeneratorTestsBase):
    # Delegation semantics for send/throw/close through `yield from`. CPython
    # forwards throw() and close() into the subiterator the delegating
    # generator is suspended on; these tests assert the subgenerator's own
    # handlers/finally run, which distinguishes real forwarding from raising
    # the exception at the outer `yield from` point.

    @make_dynamo_test
    def test_send_into_subgen(self):
        got = []

        def subgen():
            x = yield 1
            got.append(x)
            yield 2

        def outer():
            yield from subgen()

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.send(42), 2)
        self.assertEqual(got, [42])

    @make_dynamo_test
    def test_throw_into_subgen_caught(self):
        log = []

        def subgen():
            try:
                yield 1
            except ValueError:
                log.append("subgen caught")
                yield 2

        def outer():
            yield from subgen()

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.throw(ValueError), 2)
        self.assertEqual(log, ["subgen caught"])

    @make_dynamo_test
    def test_throw_into_subgen_uncaught_runs_finally(self):
        log = []

        def subgen():
            try:
                yield 1
            finally:
                log.append("subgen finally")

        def outer():
            yield from subgen()

        g = outer()
        next(g)
        self.assertRaises(ValueError, g.throw, ValueError)
        self.assertEqual(log, ["subgen finally"])

    @make_dynamo_test
    def test_throw_generator_exit_into_subgen(self):
        # throw(GeneratorExit) through `yield from` closes the subiterator, then
        # raises the GeneratorExit in the outer frame (CPython's `goto
        # throw_here`), rather than throwing into the already-closed subiterator.
        log = []

        def subgen():
            try:
                yield 1
            finally:
                log.append("subgen finally")

        def outer():
            yield from subgen()

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertRaises(GeneratorExit, g.throw, GeneratorExit)
        self.assertEqual(log, ["subgen finally"])

    @make_dynamo_test
    def test_throw_through_nested_yield_from(self):
        log = []

        def leaf():
            try:
                yield 1
            except ValueError:
                log.append("leaf caught")
                yield 2

        def mid():
            yield from leaf()

        def outer():
            yield from mid()

        g = outer()
        next(g)
        self.assertEqual(g.throw(ValueError), 2)
        self.assertEqual(log, ["leaf caught"])

    @make_dynamo_test
    def test_close_runs_subgen_finally(self):
        log = []

        def subgen():
            try:
                yield 1
                yield 2
            finally:
                log.append("subgen finally")

        def outer():
            yield from subgen()

        g = outer()
        next(g)
        g.close()
        self.assertEqual(log, ["subgen finally"])

    @make_dynamo_test
    def test_close_subgen_catches_generator_exit(self):
        log = []

        def subgen():
            try:
                yield 1
            except GeneratorExit:
                log.append("subgen exit")

        def outer():
            yield from subgen()

        g = outer()
        next(g)
        g.close()
        self.assertEqual(log, ["subgen exit"])

    @make_dynamo_test
    def test_close_subgen_ignores_generator_exit(self):
        def subgen():
            try:
                yield 1
            except GeneratorExit:
                yield 2

        def outer():
            yield from subgen()

        g = outer()
        next(g)
        self.assertRaisesRegex(RuntimeError, "generator ignored GeneratorExit", g.close)

    @make_dynamo_test
    def test_throw_into_iterator_subiter(self):
        log = []

        def outer():
            yield from _DelegatingIterator(log)

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.throw(ValueError), 2)
        self.assertEqual(log, ["iter throw"])

    @make_dynamo_test
    def test_close_into_iterator_subiter(self):
        log = []

        def outer():
            yield from _DelegatingIterator(log)

        g = outer()
        next(g)
        g.close()
        self.assertEqual(log, ["iter close"])

    @make_dynamo_test
    @unittest.expectedFailure
    def test_throw_into_iterator_without_throw(self):
        # A plain iterator (no throw method): the exception is raised in the
        # outer frame at the yield-from point (CPython's `goto throw_here`).
        def outer():
            yield from iter([1, 2, 3])

        g = outer()
        next(g)
        self.assertRaises(ValueError, g.throw, ValueError)


instantiate_parametrized_tests(GeneratorTests)
instantiate_parametrized_tests(TestGeneratorSend)
instantiate_parametrized_tests(TestGeneratorClose)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
