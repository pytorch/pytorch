# Owner(s): ["module: dynamo"]
import unittest
from unittest.mock import patch

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import config
from torch._dynamo.testing import unsupported
from torch._dynamo.utils import disable_cache_limit, ifdyn

globalmod = torch.nn.ReLU()


def indirectly_unsupported(a, b):
    c = a + b
    return unsupported(a, c)


class SubGraphTests(torch._dynamo.test_case.TestCase):
    def _common(self, fn, frame_count, op_count):
        torch._dynamo.reset()
        v1 = torch.ones(10)
        v2 = torch.ones(10) * -2.0
        correct1 = fn(v1, v2)
        correct2 = fn(v2, v1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        r1 = opt_fn(v1, v2)
        r2 = opt_fn(v2, v1)
        self.assertTrue(torch._dynamo.testing.same(r1, correct1))
        self.assertTrue(torch._dynamo.testing.same(r2, correct2))
        self.assertEqual(
            cnt.frame_count,
            frame_count,
            f"actual {cnt.frame_count} != expected {frame_count}",
        )
        self.assertEqual(cnt.op_count, op_count)

    def test_control_flow1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            if c1.sum() > c2.sum():
                return c1
            else:
                return c2

        self._common(fn, 1, 5)

    def test_control_flow2(self):
        def fn(a, b):
            if a.sum() > b.sum():
                return 1
            else:
                return 2

        self._common(fn, 1, 3)

    def test_control_flow3(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            m = globalmod
            if c1.sum() > c2.sum():
                return m(c1)
            else:
                return m(c2)

        self._common(fn, 3, 7)

    def test_control_flow4(self):
        def fn(a, b):
            tmp1 = a.sum() > b.sum() and a.sum() > 0
            if tmp1:
                return 1
            else:
                return 2

        self._common(fn, 3, 5)

    def test_control_flow5(self):
        def fn(a, b):
            tmp1 = a.sum() > b.sum() and a.sum() > 0
            tmp2 = a.sum() < b.sum() or b.sum() > 0
            if tmp1 and tmp2:
                return 1, tmp1, tmp2
            else:
                return 2, tmp1, tmp2

        self._common(fn, 6, 13)

    def test_capi_call1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return unsupported(c1, c2)

        self._common(fn, 1, 2)

    def test_capi_call2(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return a - (b - unsupported(c1, c2))

        self._common(fn, 2, 4)

    def test_capi_call3(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return torch._dynamo.testing.unsupported(c1, c2)

        self._common(fn, 1, 2)

    def test_indirect_unsupported1(self):
        def fn(a, b):
            c1 = a - b
            c2 = b - a
            return indirectly_unsupported(c1, c2)

        self._common(fn, 2, 3)

    def test_indirect_unsupported2(self):
        def fn(a, b):
            local_const1 = 7
            local_const2 = 22
            c1 = a - b
            c2 = b - a
            return local_const1 / (local_const2 - indirectly_unsupported(c1, c2))

        self._common(fn, 3, 5)

    def test_indirect_unsupported3(self):
        def fn(a, b):
            args = [a - b, b - a]
            return indirectly_unsupported(*args)

        self._common(fn, 2, 3)

    def test_stack_state1(self):
        def fn(a, b):
            t1 = 1.23 * a
            t2 = 4.56 * a
            c1 = a - b
            c2 = b - a
            return t1 / (t2 - unsupported(c1, c2))

        self._common(fn, 2, 6)

    def test_stack_state2(self):
        def fn(a, b):
            t1 = 1.23 * a
            t2 = 4.56 * a
            c1 = a - b
            c2 = b - a
            return t1 / (t2 - indirectly_unsupported(c1, c2))

        self._common(fn, 3, 7)

    def test_multigraph(self):
        def fn(a, b):
            x = a + b
            x = x / 2.0
            if x.sum() < 0:
                return x * -1.0
            return x

        self._common(fn, 2, 5)

    def test_extended_args(self):
        too_many_adds = "+".join(["a", "b"] * 256)
        source = (
            f"lambda a, b: ({too_many_adds}+a if a.sum() > 0 else {too_many_adds} - b)"
        )
        self._common(eval(source), 3, 1026)

    def test_resume1(self):
        def fn(a, b):
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = unsupported(x, a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 6)

    def test_resume2(self):
        def fn(a, b):
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = indirectly_unsupported(x, a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 3, 7)

    def test_resume3(self):
        def fn(a, b):
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = indirectly_unsupported(x, b=a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 3, 7)

    def test_resume4(self):
        def fn(a, b):
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = indirectly_unsupported(a=x, b=a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 3, 7)

    def test_resume5(self):
        def fn(a, b):
            x = a + b
            x = x / 2.0
            x = x + 2.0
            print(x)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 6)

    def test_start1(self):
        def fn(a, b):
            print(a)
            x = a + b
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 1, 3)

    def test_start2(self):
        def fn(a, b):
            x = indirectly_unsupported(a, b)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 4)

    def test_start3(self):
        def fn(a, b):
            x = unsupported(a, b)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 1, 3)

    def test_start4(self):
        def fn(a, b, check):
            if check:
                return a + b + 10
            else:
                return a + b - 10

        v1 = torch.randn(10)
        v2 = torch.randn(10)
        f = torch.zeros(1, dtype=torch.int32)
        t = torch.ones(1, dtype=torch.int32)
        correct1 = fn(v1, v2, t)
        correct2 = fn(v1, v2, f)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        r1 = opt_fn(v1, v2, t)
        r2 = opt_fn(v1, v2, f)
        self.assertTrue(torch._dynamo.testing.same(r1, correct1))
        self.assertTrue(torch._dynamo.testing.same(r2, correct2))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 4)

    def test_resume_freevars(self):
        c1 = torch.randn(10)
        c2 = torch.randn(10)

        def fn(a, b):
            x = a + b + (c1 - c2)
            x = unsupported(x, x)
            return x + (c1 - c2)

        self._common(fn, 2, 5)

    def test_restore_state(self):
        def fn(a, b):
            len_ = len
            x = a + b
            x = torch.add(unsupported(x, x), 1)
            return a * x + len_(b)

        self._common(fn, 2, 5)

    def test_restore_range(self):
        def fn(a, b):
            x = a + b
            rng = range(3, 8, 2)
            x = unsupported(x, x)
            for i in rng:
                x = x + i
            return x

        # We don't specialize on range with dynamic shapes, which
        # means we fail to unroll the loop.
        # TODO: Consider forcing specialization when we iterate over
        # the loop
        if config.assume_static_by_default:
            self._common(fn, 2, 4)
        else:
            self._common(fn, 2, ifdyn(4, 1))

    def test_restore_range_iter(self):
        def fn(a, b):
            x = a + b
            rng = iter(range(3, 8, 2))
            x = unsupported(x, x)
            x += next(rng)
            return x, list(rng)

        self._common(fn, 2, 2)

    def test_pop_after_resume(self):
        def fn(a, b):
            tmp = [a + 1, b + 2, a + b]
            x = a
            x = unsupported(x, x)
            for i in range(3):
                x += tmp.pop(-1)
            return x

        self._common(fn, 2, 6)

    @disable_cache_limit()
    def test_dynamic_shapes(self):
        if config.assume_static_by_default:
            return unittest.skip("Already covered identically in test_dynamic_kwarg")

        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_static = torch._dynamo.testing.CompileCounter()
        with patch("torch._dynamo.config.dynamic_shapes", False):
            opt_fn = torch._dynamo.optimize(cnt_static)(fn)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        self.assertEqual(cnt_static.frame_count, 10)

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with patch("torch._dynamo.config.dynamic_shapes", True):
            opt_fn = torch._dynamo.optimize(cnt_dynamic)(fn)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # just one graph now rather than 10
        self.assertEqual(cnt_dynamic.frame_count, 1)

    @patch("torch._dynamo.config.dynamic_shapes", True)
    @patch("torch._dynamo.config.assume_static_by_default", False)
    def test_dynamic_getitem(self):
        def fn(a, b):
            return a[b.size(0) - 1]

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        for i in range(3, 12):
            opt_fn(torch.randn(i), torch.randn(i))
        # just one graph
        self.assertEqual(cnt.frame_count, 1)

    @patch("torch._dynamo.config.assume_static_by_default", False)
    def test_dynamic_kwarg(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        start = 2
        end = 12
        steps = end - start
        for i in range(start, end):
            opt_fn(torch.randn(i), torch.randn(i))

        if config.assume_static_by_default:
            # 2 graph breaks - 1 static, 1 made dynamic via automatic
            self.assertEqual(cnt_dynamic.frame_count, 2)
        else:
            # just one graph
            self.assertEqual(cnt_dynamic.frame_count, 1)

    @patch("torch._dynamo.config.assume_static_by_default", False)
    def test_dynamic_duck_size(self):
        def fn(a, b):
            if a.size(0) == b.size(0):
                return a + b
            else:
                return a.sum() + b.sum()

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        x = torch.randn(2)
        y = torch.randn(3)
        self.assertEqual(opt_fn(x, x), fn(x, x))
        self.assertEqual(opt_fn(x, y), fn(x, y))
        self.assertEqual(cnt_dynamic.frame_count, 2)

    def test_dynamic_order_dependence(self):
        def fn(a, b):
            return a.sum() + b.sum()

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        x = torch.randn(2)
        y = torch.randn(3)
        self.assertEqual(opt_fn(x, y), fn(x, y))
        self.assertEqual(opt_fn(x, x), fn(x, x))
        # NB: This COULD validly be 2, but we don't test disjointness in the
        # guards for when x and y didn't duck size together, so we end up
        # with a generic graph that also works when x and y happen to duck
        # size together.
        if config.assume_static_by_default:
            self.assertEqual(cnt_dynamic.frame_count, 2)
        else:
            self.assertEqual(cnt_dynamic.frame_count, 1)

        torch._dynamo.reset()
        cnt_dynamic.frame_count = 0
        self.assertEqual(opt_fn(x, x), fn(x, x))  # this overspecializes!
        self.assertEqual(opt_fn(x, y), fn(x, y))
        self.assertEqual(cnt_dynamic.frame_count, 2)

    def test_dynamic_zero_inference(self):
        def fn(a):
            if a.size(0) != 0:
                return a * 2
            else:
                return a + 1

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt_dynamic, dynamic=True)(fn)
        x = torch.randn(0)
        y = torch.randn(2)
        self.assertEqual(opt_fn(y), fn(y))
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_no_graph_break_on_item(self):
        def fn(a, b):
            x = a + b - 1.5
            x = x.sum()
            x.item()
            x = x / (a + b)
            return x

        self._common(fn, 1, 6)

    @patch.object(torch._dynamo.config, "dynamic_shapes", True)
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_graph_break_on_item(self):
        def fn(a, b):
            x = a + b - 1.5
            x = x.sum()
            x.item()
            x = x / (a + b)
            return x

        self._common(fn, 2, 5)

    def test_resume_paths_join(self):
        def fn(x, c1, c2, c3):
            x = x + 1
            if c1:
                x = x + 2
            x = x + 3
            if c2:
                x = x + 4
            x = x + 5
            if c3:
                x = x + 6
            return x + 7

        v1 = torch.randn(10)
        t = torch.Tensor([True])
        f = torch.Tensor([False])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        for a in (t, f):
            for b in (t, f):
                for c in (t, f):
                    opt_fn(v1, a, b, c)

        # checking here we don't create 2^n graphs
        self.assertEqual(cnt.frame_count, 7)
        self.assertEqual(cnt.op_count, 10)

    def test_resume_with_no_grad1(self):
        def fn(a, b):
            x = a + b
            with torch.no_grad():
                x = x + 1
                x.sum().tolist()  # graph break
                x = x + 2
            x = x + 3
            return x

        self._common(fn, 2, 9)
        torch._dynamo.reset()
        with torch.no_grad():
            self._common(fn, 2, 9)

    def test_resume_with_no_grad2(self):
        def fn(a, b):
            x = a + b
            with torch.no_grad():
                x = x + 1
                x.sum().tolist()  # graph break
                x = x + 2
                x.sum().tolist()  # graph break
                x = x + 3
            x = x + 4
            return x

        self._common(fn, 3, 13)

    def test_resume_with_no_grad3(self):
        def fn(a, b):
            x = a + b
            with torch.no_grad():
                with torch.no_grad():
                    x = x + 1
                    with torch.enable_grad():
                        x.sum().tolist()  # graph break
                        x = x[0] + 2
                    x = x + 3
            x = x + 4
            return x

        self._common(fn, 2, 19)

    def test_resume_tuple_iterator(self):
        def fn(a, b):
            x = a + b
            it = iter(tuple(range(10)))
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            return x

        self._common(fn, 2, 8)

    def test_tuple_iterator_return(self):
        def fn(x):
            it = iter(tuple(range(10)))
            x = x + next(it)
            x = x + next(it)
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            return x, it

        v1 = torch.randn(10)
        v2, it2 = fn(v1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        v3, it3 = opt_fn(v1)
        v4, it4 = opt_fn(v1)
        self.assertEqual(v2.tolist(), v3.tolist())
        self.assertEqual(v2.tolist(), v4.tolist())
        self.assertEqual(list(it2), list(it3))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 6)

    @unittest.skip("not working yet")
    def test_tuple_iterator_mutate(self):
        def fn(x, it):
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            return x

        v1 = torch.randn(10)
        it1 = iter(tuple(range(10)))
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnt)(fn)
        self.assertEqual(opt_fn(v1, it1).tolist(), (v1 + 1 + 2 + 3).tolist())
        self.assertEqual(list(it1), [4, 5, 6, 7, 8, 9])

    def test_enumerate_not_break_graph(self):
        def fn(a, b):
            for i, x in enumerate(a.shape):
                b = b + x
            for i, x in enumerate(b.shape, 8):
                b = b + x * i
            return b

        self._common(fn, 1, 4)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
