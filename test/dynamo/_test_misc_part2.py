# Owner(s): ["module: dynamo"]
# flake8: noqa: C416, F401, F403, F405, F821, F841
# ruff: noqa: C416,F401,F403,F405,F821,F841,PERF102,RSE102,SIM118,TRY002
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


x = None


class MiscTestsPart2:
    def test_check_raises_at_runtime_when_predicate_false_and_message_has_global(self):
        global GLOBAL_INT
        GLOBAL_INT = 1

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{GLOBAL_INT} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            RuntimeError, f"{GLOBAL_INT} is not greater than 3"
        ):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_None(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_None(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, None):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_and_message_has_no_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_raises_at_runtime_when_predicate_false_constant_and_message_has_no_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(3)

        with self.assertRaisesRegex(RuntimeError, "Shape is not greater than 3"):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_false_and_message_has_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(3)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check()"
        ):
            f(x)

    def test_check_assert_error_at_runtime_when_predicate_true_and_message_has_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{x.shape[0]} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Can't extract message from torch._check()"
        ):
            f(x)

    def test_assert(self):
        @torch.compile(backend="eager")
        def fn1(x):
            assert x.shape != x.shape  # noqa: S101

        with self.assertRaises(AssertionError):
            a = torch.randn(10)
            fn1(a)

        def fn2(x):
            assert x.shape == x.shape  # noqa: S101
            return x.abs()

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=1, expected_ops=1)

    # When we unspecialize float, we wobble this test by changing
    # the op count since previously we would just specialize and constant
    # fold floats into the graph, whereas when we unspecialize we will have
    # ops for item, add, and all other tensorified operations. Since this
    # test really isn't testing that, we purposely specialize floats here.
    @torch._dynamo.config.patch(specialize_float=True)
    def test_config_obj(self):
        class Cfg:
            def __init__(self) -> None:
                self.val = 0.5
                self.count = 3

        def fn(x, cfg):
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        cfg1.val = 1.0
        cfg2 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v = opt_fn(v, cfg1)  # 3
        v = opt_fn(v, cfg2)  # 4.5
        cfg2.count = 1
        v = opt_fn(v, cfg2)  # 5
        cfg2.val = 2.0
        v = opt_fn(v, cfg2)  # 7
        self.assertEqual(v[0], 7)
        self.assertEqual(cnts.op_count, 8)

    def test_config_getattr_default(self):
        class Cfg:
            def __init__(self) -> None:
                self.val = 0.5
                self.count = 10

        def fn(x, cfg):
            if getattr(cfg, "just_add_7", False):
                return x + 7
            for i in range(cfg.count):
                x = x + cfg.val
            return x

        cfg1 = Cfg()
        v = torch.zeros(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        cfg1.just_add_7 = True
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        self.assertEqual(opt_fn(v, cfg1)[0], 7)
        cfg1.just_add_7 = False
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(opt_fn(v, cfg1)[0], 5)
        self.assertEqual(cnts.frame_count, 3)

    def test_size_input(self):
        def fn(x, s):
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, v.size())[0, 0], -10)
        self.assertEqual(opt_fn(v, (10, 20))[0, 0], -10)
        self.assertEqual(opt_fn(v, [10, 20])[0, 0], -10)
        # One recompile per differing input type
        self.assertEqual(cnts.frame_count, 3)

    def test_cell_output1(self):
        out = None

        def fn(a, b):
            nonlocal out
            out = a + b * 10

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIsNone(opt_fn(v, v))
        self.assertEqual(out[0], 1100)
        self.assertEqual(cnts.op_count, 2)

    def test_cell_output2(self):
        out = None

        def fn(a, b):
            nonlocal out
            c = unsupported(a, b)
            out = a + b * 10 + c

        v = torch.Tensor([100])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIsNone(opt_fn(v, v))
        self.assertEqual(out[0], 1200)
        self.assertEqual(cnts.op_count, 3)

    def test_return_nested_function(self):
        out = None

        def fn(a, b):
            nonlocal out
            c = a + b
            d = a + 1.0

            def fn2(f: int = 7, g: float = 9.0):
                nonlocal out
                out = a + b * 10
                return c * f - d * g

            return fn2

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn_ret = torch.compile(opt_fn(v1, v2), backend=cnts)
        self.assertEqual(opt_fn_ret(1.5)[0], -459)
        self.assertEqual(out[0], 2100)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 7)

    def test_tensor_dict1(self):
        def fn(inputs):
            return inputs["a"] - inputs["b"] * 1.5

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn({"a": v1, "b": v2})[0], -200)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_dict3(self):
        def fn(inputs_a, inputs_b):
            total = torch.zeros(1)
            input_keys = inputs_a.keys() | inputs_b.keys()
            for k in input_keys:
                if k in inputs_a:
                    total += inputs_a[k]
                if k in inputs_b:
                    total += inputs_b[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(
            opt_fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
            fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
        )
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_tensor_dict2(self):
        def fn1(inputs):
            total = torch.zeros(1)
            for k, v in inputs.items():
                total += v
            return total

        def fn2(inputs):
            total = torch.zeros(1)
            for v in inputs.values():
                total += v
            return total

        def fn3(inputs):
            total = torch.zeros(1)
            for k in inputs.keys():
                total += inputs[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn2({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn3({"a": v1, "b": v2})[0], 300)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_user_code_statically_known(self):
        from torch.fx.experimental.symbolic_shapes import (
            has_static_value,
            statically_known_true,
        )

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            # At this point, this isn't statically known, only the hint says so.
            if statically_known_true(x.shape[0] > 9):
                raise Exception()
            torch._check(x.shape[0] >= 10)
            # But now it is.
            return statically_known_true(x.shape[0] > 9), has_static_value(x.shape[0])

        x = torch.zeros(10)
        torch._dynamo.mark_dynamic(x, 0)
        self.assertEqual(f(x), (True, False))

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def g(x, y):
            n = x.item()
            torch._check(n == 3)
            return has_static_value(4.0), has_static_value(n)

        out = g(torch.tensor([3]), torch.zeros(1))
        self.assertEqual(out, (True, True))

    def test_dictcomp(self):
        def fn1(inputs):
            return {k: v + 1 for k, v in inputs.items()}

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["a"], 101)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})["b"], 201)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_listcomp(self):
        def fn2(inputs):
            return torch.sum(torch.cat([v + 1 for k, v in inputs.items()], 0))

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts)
        self.assertEqual(opt_fn2({"a": v1, "b": v2}), 302)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_is_floating_point(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_floating_point(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_floating_point2(self):
        def fn(a, b):
            x = a + 1.0
            if b.is_floating_point():
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_tensor(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor2(self):
        def fn(x):
            if torch.is_tensor(x):
                return x + 1
            else:
                return torch.ones([2, 3])

        x1 = {"input": torch.rand(2, 3)}
        x2 = torch.rand(2, 3)
        ref1 = fn(x1)
        ref2 = fn(x2)
        opt_fn = torch.compile(fn, backend="eager")
        res1 = opt_fn(x1)
        res2 = opt_fn(x2)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_numel(self):
        def fn(a):
            return (a + a.numel() + torch.numel(a), a + a.nelement())

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=3,
            expected_ops_dynamic=ifdynstaticdefault(3, 4),
        )

    def test_pair(self):
        def fn(a):
            return (
                torch.zeros(torch.nn.modules.utils._pair(a.size()))
                + a
                + torch.ones(torch.nn.modules.utils._ntuple(3)(3)).sum()
            )

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=5,
            expected_ops_dynamic=5,
        )

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple1(self):
        def fn(a, b):
            tmp = MyTuple(a, b, a + b)
            return MyTuple(tmp.a, tmp[1], tmp.ab + b)

        v1 = torch.Tensor([10])
        v2 = torch.Tensor([20])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2).ab, 50)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_namedtuple2(self):
        def fn(packed):
            a, b, c = packed
            if hasattr(packed, "b"):
                b = packed.b + 1
            c = packed[2]
            d = len(packed._fields)
            return a + b + c + d

        v1 = torch.Tensor([1])
        v2 = torch.Tensor([2])
        v3 = torch.Tensor([3])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(MyTuple(v1, v2, v3))[0], 10)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def test_namedtuple3(self):
        def fn(x, packed):
            if isinstance(packed, MyTuple):
                return x + 1
            else:
                return x - 1

        x = torch.rand([2, 3])
        packed = MyTuple(1, 2, 3)
        ref = fn(x, packed)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, packed)
        self.assertTrue(same(ref, res))

    def test_namedtuple_with_custom_getitem(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(my_tuple):
            return my_tuple.a + 1

        class MyTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

            def __getitem__(self, index):
                return MyTuple(a[index], b[index])

        a = torch.randn(2)
        b = torch.randn(2)

        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

        # Test guard evaluation in the second call
        out = f(MyTuple(a, b))
        self.assertTrue(same(a + 1, out))

    def test_namedtuple_source_dynamic_attributes(self):
        class MyNamedTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(tup):
            c = torch.tensor(3.0)
            tup.c = c  # Add dynamic attribute
            return tup

        extended_tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
        result = f(extended_tup)
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple_sourceless_dynamic_attributes(self):
        class MyNamedTuple(typing.NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class MyNamedTupleSubclass(MyNamedTuple):
            pass

        @torch.compile(backend="eager")
        def f():
            # Create namedtuple inside function (sourceless)
            tup = MyNamedTupleSubclass(a=torch.tensor([1.0]), b=torch.tensor(2.0))
            # Add dynamic attribute
            tup.c = torch.tensor(3.0)
            return tup

        result = f()
        # Verify the tuple has the expected structure
        self.assertEqual(result.a, torch.tensor([1.0]))
        self.assertEqual(result.b, torch.tensor(2.0))
        # Verify the dynamic attribute is preserved
        self.assertTrue(hasattr(result, "c"))
        self.assertEqual(result.c, torch.tensor(3.0))

    def test_namedtuple___eq__(self):
        class MyNamedTuple(typing.NamedTuple):
            a: int
            b: int

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            t1 = MyNamedTuple(a=1, b=2)
            t2 = (1, 2)
            return x.sin(), (t1 == t2)

        x = torch.randn(2)
        res = f(x)
        self.assertTrue(res[1])

    def test_structseq1(self):
        def fn(x, y):
            return torch.return_types.max((x, y))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq2(self):
        def fn(x, y):
            return tuple(torch.return_types.qr((2 * x, y - 1)))

        x = torch.randn(3, 2)
        y = torch.randn(2, 4)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    def test_structseq_repr(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            result = torch.max(x, dim=0)
            s = repr(result)
            return result.values

        x = torch.randn(3, 2)

        # Verify that fullgraph=True fails (confirms graph break occurs)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, fullgraph=True, backend="eager")(x)

        # Verify that it works without fullgraph
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        result = opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

    def test_range_input(self):
        def fn(a, rng):
            x = a
            for i in rng:
                x = x + i
            return x

        def fn1(a):
            return fn(a, rng=range(3))

        return torch._dynamo.testing.standard_test(
            self, fn=fn1, nargs=1, expected_ops=3
        )

    def test_range_with_shape(self):
        def fn(a):
            for i in range(1, a.shape[0]):
                a += 1
            return a

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=9,
        )

    def test_range_iter_guards(self):
        @torch.compile(backend="eager")
        def func():
            @torch._dynamo.disable(recursive=False)
            def run(n):
                # For python <= 3.11, list comprehension is implemented by
                # desugaring to:
                # 1. creation of an iterator object
                # 2. calling a new `listcomp` function with (1)
                #
                # In this test we force Dynamo to trace through (2) as the root
                # frame, thereby ensuring we have the right guards for range
                # iterators.
                xs = [torch.ones(1) for i in range(n)]
                return torch.concat(xs)

            return run(2), run(3)

        res2, res3 = func()
        self.assertTrue(same(res2, torch.ones(2)))
        self.assertTrue(same(res3, torch.ones(3)))

    def test_range___iter__(self):
        def func(x):
            it = range(3).__iter__()
            return x + next(it)

        opt_func = torch.compile(func, backend="eager", fullgraph=True)
        x = torch.randn(3)
        self.assertTrue(same(func(x), opt_func(x)))

    def test_range_iter_side_effects(self):
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, it):
            n = next(it)
            return x + n

        it = iter(range(1, 3))
        res = run(torch.zeros(1), it)
        self.assertTrue(same(res, torch.ones(1)))
        self.assertEqual(next(it), 2)

    def test_build_tuple_unpack(self):
        def fn1(a, b, c):
            return a - b / c

        def fn2(a, b, c):
            tmp1 = (a,)
            tmp2 = (b, c)
            args = (*tmp1, *tmp2)
            return fn1(*args)

        def fn3(a, *args):
            return fn1(a, *args)

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=3, expected_ops=2)
        torch._dynamo.testing.standard_test(self, fn=fn3, nargs=3, expected_ops=2)

    def test_list_mul(self):
        def fn(count):
            head_mask = count * [None] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), [None] * 4)
        # TODO: the captured frame here is a bit goofy, because we don't
        # output anything and none of the traced operations have side
        # effects.  Probably need better heuristic for bailing on
        # dynamo if there are no outputs
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_list_slice_mul(self):
        def fn(count):
            a = [1, 2, 3]
            head_mask = count * a[1:] * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), [2, 3] * 4)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_tuple_mul(self):
        def fn(count):
            head_mask = count * (2, 3) * count
            return head_mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(2), (2, 3) * 4)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.frame_count, """0""")
            self.assertExpectedInline(cnts.op_count, """0""")
        else:
            self.assertExpectedInline(cnts.frame_count, """1""")
            self.assertExpectedInline(cnts.op_count, """2""")

    def test_tuple_mul_with_shape(self):
        def fn(a):
            x = a.shape[0]
            y = 2 * (x, 3) * 2
            return a + y[4]

        # expect 3 ops post folding for dynamic case: size, index, add
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=1
        )

    def test_tuple_iadd_with_shape(self):
        def fn(a):
            output = (a + a.shape[0], a - a.shape[0])
            # tuple += tuple
            output += (a - a.shape[0], a + a.shape[0])
            # tuple += constant tuple
            output += (2, 3)
            return output

        # expect 4 add / subs for static
        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=4, expected_ops_dynamic=4
        )

    def test_list_iadd_with_shape(self):
        def fn(a):
            output = [a + a.shape[0], a - a.shape[0]]
            # list += list
            output += [a - a.shape[0], a + a.shape[0]]
            # list += tuple
            output += (a + a.shape[0], a - a.shape[0])
            return output

        # expect 6 add / subs for static

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=6, expected_ops_dynamic=6
        )

    def test_list_iadd_side_effect(self):
        def fn(a, b):
            a += [b]
            torch._dynamo.graph_break()
            return a

        a = [1, 2, 3]
        b = torch.ones(2, 2)

        opt_fn = torch.compile(fn, backend="eager")

        exp = fn(a, b)

        a = [1, 2, 3]
        b = torch.ones(2, 2)
        act = opt_fn(a, b)

        self.assertEqual(exp, act)

    def test_class_binop(self):
        class Foo:
            def __init__(self, x):
                self.x = x

            def __add__(self, other):
                return Foo(self.x + other.x)

        def fn(a, b):
            return a + b

        x = torch.randn(2)
        a, b = Foo(x), Foo(x + 1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(a, b).x, 2 * x + 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        def fn(a, b):
            return a - b

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertRaises(torch._dynamo.exc.Unsupported, opt_fn, a, b)

    def test_user_getattr1(self):
        class MyConfig(dict):
            def __getattr__(self, name):
                return self[name]

        def fn(cfg, x, y):
            return x + y + cfg.offset

        x = torch.randn(10)
        cfg = MyConfig(offset=5)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_user_getattr2(self):
        class MyConfig:
            defined_on_class = 1

            def __init__(self) -> None:
                self.defined_on_object = 2

            def __getattr__(self, name):
                return 3

        def fn(cfg, x):
            return x + cfg.defined_on_class - cfg.defined_on_object + cfg.not_defined

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x), x + 1 - 2 + 3))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_getset_descriptor(self):
        def fn(g, x):
            # Just to make Dynamo not skip the frame
            torch.sin(x)
            return g.__get__(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        g = torch.Tensor.shape

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            res = opt_fn(g, torch.ones(2, 2))

    def test_set_descriptor(self):
        class Field:
            def __set__(self, obj, value):
                obj.__dict__["field"] += value * 2

        class Foo:
            field = Field()

            def __init__(self):
                self.__dict__["field"] = 0

        def fn(x, foo):
            foo.field = 10
            return x + foo.field

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        x = torch.zeros(2)
        foo1, foo2 = Foo(), Foo()

        ref = fn(x, foo1)
        res = opt_fn(x, foo2)
        self.assertEqual(ref, res)
        self.assertEqual(foo1.field, foo2.field)

    def test_dict_with_descriptor(self):
        class MyDescriptor:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get("_value", 0)

            def __set__(self, obj, value):
                obj.__dict__["_value"] = value

        class MyClass:
            prop = MyDescriptor()

            def __init__(self):
                self.prop = 42

        def fn(obj):
            obj.__dict__["extra"] = 99
            return obj.prop + obj.__dict__["extra"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        res = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["_value"], 42)
        self.assertEqual(obj2.__dict__["extra"], 99)

    def test_dict_with_slots(self):
        class SlottedClass:
            __slots__ = ("x",)

            def __init__(self, x):
                self.x = x

        def fn(obj):
            # SlottedClass doesn't have __dict__, so this should fail or be handled
            return obj.x * 2

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedClass(5)
        res = fn(obj)
        out = opt_fn(obj)
        self.assertEqual(res, out)

    def test_dict_with_slots_and_dict(self):
        class SlottedWithDict:
            __slots__ = ("x", "__dict__")

            def __init__(self, x):
                self.x = x

        def fn(obj):
            obj.__dict__["custom"] = 100
            return obj.x + obj.__dict__["custom"]

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = SlottedWithDict(7)
        res = fn(obj)

        obj2 = SlottedWithDict(7)
        out = opt_fn(obj2)
        self.assertEqual(res, out)
        self.assertEqual(obj2.__dict__["custom"], 100)

    def test_dict_descriptor_interaction(self):
        class DescriptorWithDict:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return self
                return obj.__dict__.get("_internal", "default")

            def __set__(self, obj, value):
                obj.__dict__["_internal"] = f"desc_{value}"

        class MyClass:
            desc = DescriptorWithDict()

            def __init__(self):
                pass

        def fn(obj):
            obj.desc = "hello"
            obj.__dict__["_internal"] = "direct"
            return obj.desc

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        obj = MyClass()
        ref = fn(obj)

        obj2 = MyClass()
        out = opt_fn(obj2)
        self.assertEqual(ref, out)
        self.assertEqual(out, "direct")

    def test_mutable_mapping_dict_update(self):
        """Test that MutableMappingVariable handles __dict__ updates correctly."""
        from collections import OrderedDict

        class CustomMapping(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            mapping.__dict__["custom_attr"] = 42
            return mapping.__dict__["custom_attr"]

        mapping = CustomMapping()
        out = fn(mapping)
        self.assertEqual(out, 42)
        self.assertIn("custom_attr", mapping.__dict__)
        self.assertEqual(mapping.__dict__["custom_attr"], 42)

    def test_mutable_mapping_dict_access_pattern(self):
        """Test accessing attributes through __dict__ on MutableMapping subclasses."""
        from collections import OrderedDict

        class TrackedDict(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(d):
            # Pattern: check if attribute exists in __dict__
            if "tracker" not in d.__dict__:
                d.__dict__["tracker"] = []
            d.__dict__["tracker"].append(1)
            return len(d.__dict__["tracker"])

        d = TrackedDict()
        out = fn(d)
        self.assertEqual(out, 1)
        self.assertIn("tracker", d.__dict__)
        self.assertEqual(d.__dict__["tracker"], [1])

    def test_mutable_mapping_lazy_dict_initialization(self):
        """Test lazy initialization pattern with MutableMapping __dict__."""
        from collections import defaultdict

        class LazyMapping(dict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # Lazy initialization in __dict__
            if not hasattr(mapping, "_cache"):
                mapping.__dict__["_cache"] = {}
            mapping.__dict__["_cache"]["key"] = "value"
            return mapping.__dict__["_cache"]["key"]

        m = LazyMapping()
        out = fn(m)
        self.assertEqual(out, "value")
        self.assertIn("_cache", m.__dict__)
        self.assertEqual(m.__dict__["_cache"]["key"], "value")

    def test_mutable_mapping_dict_with_property_setter(self):
        """Test MutableMapping with property setters that access __dict__."""
        from collections import OrderedDict

        class PropertyMapping(OrderedDict):
            def __init__(self):
                super().__init__()
                self.__dict__["_value"] = 0

            @property
            def value(self):
                return self.__dict__.get("_value", 0)

            @value.setter
            def value(self, v):
                self.__dict__["_value"] = v

        @torch.compile(fullgraph=True, backend="eager")
        def fn(m):
            m.value = 100
            return m.value

        m = PropertyMapping()
        out = fn(m)
        self.assertEqual(out, 100)
        self.assertIn("_value", m.__dict__)
        self.assertEqual(m.__dict__["_value"], 100)

    def test_mutable_mapping_dict_multiple_accesses(self):
        """Test multiple accesses and mutations to MutableMapping __dict__."""
        from collections import OrderedDict

        class MultiAccessMapping(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def fn(mapping):
            # Multiple accesses to __dict__
            mapping.__dict__["a"] = 1
            mapping.__dict__["b"] = mapping.__dict__["a"] + 1
            mapping.__dict__["c"] = mapping.__dict__["b"] + 1
            return mapping.__dict__["a"] + mapping.__dict__["b"] + mapping.__dict__["c"]

        m = MultiAccessMapping()
        out = fn(m)
        self.assertEqual(out, 6)  # 1 + 2 + 3
        self.assertIn("a", m.__dict__)
        self.assertIn("b", m.__dict__)
        self.assertIn("c", m.__dict__)
        self.assertEqual(m.__dict__["a"], 1)
        self.assertEqual(m.__dict__["b"], 2)
        self.assertEqual(m.__dict__["c"], 3)

    def test_get_attr_function(self):
        def fn(g, x):
            return g(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        g = torch.Tensor.shape.__get__

        res = opt_fn(g, torch.ones(2, 2))
        exp_res = fn(g, torch.ones(2, 2))
        self.assertEqual(res, exp_res)

    def test_user_getattribute(self):
        class MyObject:
            def __init__(self) -> None:
                self.custom_dict = {"a": torch.rand((2, 2))}
                self.my_number = 42

            def __getattribute__(self, name):
                custom_dict = super().__getattribute__("custom_dict")
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattribute__(name)

            def run(self, x):
                return self.my_number * x + self.a * x

        def fn(obj, x):
            return obj.run(x)

        obj = MyObject()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj, x), fn(obj, x)))

    def test_nn_module_getattr(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.custom_dict = {"queue": [torch.rand((2, 2)) for _ in range(3)]}
                self.other_attr = torch.rand((2, 2))

            def __getattr__(self, name):
                custom_dict = self.custom_dict
                if name in custom_dict:
                    return custom_dict[name]
                return super().__getattr__(name)

            def forward(self, x):
                return x @ self.other_attr + self.queue[-1]

        x = torch.rand((2, 2))
        mod = MyMod()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnts)
        self.assertTrue(same(opt_mod(x), mod(x)))
        self.assertTrue(cnts.frame_count, 1)
        self.assertTrue(cnts.op_count, 2)

    def test_nn_module_getattribute(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.my_number = 42

            def __getattribute__(self, name):
                if name == "special_attr":
                    return torch.tensor([[1, 2], [3, 4]])
                return super().__getattribute__(name)

            def forward(self, x):
                return self.my_number * x + self.special_attr * x

        def fn(mod, x):
            return mod(x)

        mod = MyMod()
        x = torch.rand((2, 2))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(mod, x), fn(mod, x)))

    def test_nn_module_getattribute_simple_delegation(self):
        # Test that nn.Module with __getattribute__ that overrides a
        # single attribute name compiles without graph break.
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = 3.0

            def __getattribute__(self, name):
                if name == "my_scale":
                    return super().__getattribute__("scale")
                return super().__getattribute__(name)

            def forward(self, x):
                return x * self.my_scale

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_nn_module_getattribute_graph_break(self):
        # __getattribute__ that Dynamo cannot trace produces correct results
        # via eager fallback instead of crashing.
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def __getattribute__(self, name):
                if name == "my_attr":
                    return eval("42")  # eval is untraceable
                return super().__getattribute__(name)

            def forward(self, x):
                a = self.my_attr
                return self.linear(x) + a

        mod = MyMod()
        x = torch.randn(2, 4)
        expected = mod(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mod, backend=cnts)
        result = opt_fn(x)
        self.assertTrue(same(result, expected))

    def test_constant_getattr(self):
        # https://github.com/pytorch/pytorch/issues/97480
        def fn():
            return getattr(None, "arg", 3)

        cnt = torch._dynamo.testing.CompileCounter()
        optimized_fn = torch.compile(fn, backend=cnt)
        res = optimized_fn()
        self.assertTrue(same(res, 3))

    def test_user_property(self):
        class MyConfig:
            @property
            def prop5(self):
                return 5

        def fn(cfg, x, y):
            return x + y + cfg.prop5

        x = torch.randn(10)
        cfg = MyConfig()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(cfg, x, x), 2 * x + 5))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_data_access_in_inference_mode(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            y = x.data
            return y

        with torch.inference_mode():
            x = torch.randn(3)
            y = f(x)
        self.assertEqual(y, x)

    def test_dataclass_fields(self):
        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)  # noqa: S101
            assert all(field.default is None for field in class_fields[1:])  # noqa: S101
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none  # noqa: S101

            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        obj1 = MyDataClass(torch.randn(10), torch.randn(10), torch.randn(10))
        obj2 = MyDataClass(torch.randn(10), e=torch.randn(10))
        correct1 = fn(obj1)
        correct2 = fn(obj2)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj1), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj2), correct2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        # guard failure
        obj2.z = True
        self.assertEqual(opt_fn(obj2), -2)

    def test_dataclass_local_hasattr(self):
        cnt = CompileCounter()
        x = torch.randn(10)

        @dataclasses.dataclass
        class MyDataClass:
            a: torch.Tensor
            b: torch.Tensor

        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            obj = MyDataClass(x + 1, x - 1)
            if not hasattr(obj, "a"):
                return -1
            if hasattr(obj, "z"):
                return -2
            return obj

        result = fn()
        self.assertIsInstance(result, MyDataClass)
        self.assertEqual(result.a, x + 1)
        self.assertEqual(result.b, x - 1)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_catch_watchings1(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            with warnings.catch_warnings(record=True):
                return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(cnt.frame_count, 1)

    def test_catch_watchings2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x.sin(), warnings.catch_warnings(record=True)

        x = torch.randn(8)
        _, a = fn(x)
        _, b = fn(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(a, warnings.catch_warnings)
        self.assertIsInstance(b, warnings.catch_warnings)
        self.assertIsNot(a, b)

    def test_tensor_build_list_unpack(self):
        def fn(x):
            # seen in fastNLP_Bert
            return torch.cat([*x], dim=-1)

        val = torch.randn([1, 1, 473, 768])
        correct = fn(val)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(val), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_int_constant(self):
        def fn(x, a, b):
            return x + (a % b)

        args = [torch.randn(10), 4096, np.int64(8)]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, dynamic=True, fullgraph=True)
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertTrue(same(opt_fn(*args), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_numpy_subdtype(self):
        def fn(x, n):
            return np.issubdtype(type(n), np.integer) + x

        args = [torch.randn(10), 4096]
        correct = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn(*args), correct)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_take_along_axis(self):
        def fn(x, i, a):
            return np.take_along_axis(x, i, a)

        def sample_to_args(s):
            args = (s.input, *sample.args)
            return tuple(a.numpy() if isinstance(a, torch.Tensor) else a for a in args)

        samples = list(
            sample_inputs_take_along_dim(
                None, "cpu", torch.float32, requires_grad=False
            )
        )
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        i = 1
        for sample in samples:
            args = sample_to_args(sample)
            if len(args) < 3:
                # if axis is None, second argument is treated as 1d array
                args = (args[0], np.ravel(args[1]), None)
            self.assertEqual(fn(*args), opt_fn(*args))
            self.assertEqual(cnts.frame_count, i)
            i += 1

    def test_numpy_torch_operators(self):
        def fn(op, t1, t2):
            return op(t1, t2)

        from torch._dynamo.variables.builtin import BuiltinVariable

        operators = BuiltinVariable._fx_graph_functions()

        for op, t1_np, t2_np in itertools.product(
            operators, (True, False), (True, False)
        ):
            if op in [operator.eq, operator.ne]:
                # returns equivalent of torch.eq/ne
                continue
            if op is operator.getitem:
                # skip
                # Did you know that tensor[ndarray_of_floats] works?
                continue
            if op is operator.imatmul and (t1_np or t2_np):
                # skip
                # in numpy, in place matmul does not work single
                # dimensional arrays
                continue
            t1 = torch.rand(5)
            if t1_np:
                t1 = t1.numpy()
            t2 = torch.rand(5)
            if t2_np:
                t2 = t2.numpy()
            try:
                # TODO try a bit harder
                result = op(t1, t2)
            except (RuntimeError, TypeError, IndexError):
                continue
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts)
            self.assertEqual(result, opt_fn(op, t1, t2), msg=f"{op=} {t1_np=} {t2_np=}")
            self.assertEqual(cnts.frame_count, 1, msg=f"{op=} {t1_np=} {t2_np=}")
            torch._dynamo.reset()

    def test_numpy_ndarray_graph_break(self):
        def fn(x):
            a = x.numpy()
            b = a.real
            torch._dynamo.graph_break()
            c = np.multiply(b, 2.0)
            return c

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_ndarray_graph_break_with_multiple_outputs(self):
        def fn(x, y):
            a = x.numpy()
            b = y.numpy()
            torch._dynamo.graph_break()
            return np.add(a, 1), np.add(b, 1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_force(self):
        def fn(x):
            return x.numpy(force=False)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(3)
        res = opt_fn(x)
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

        def fn(x):
            return x.numpy(force=True)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(3, requires_grad=True)
        res = opt_fn(x)
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_recompilation_scalar(self):
        def fn(x, a):
            return np.where(x < 0.5, a, x)

        x = np.random.randn(8)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, dynamic=True)

        ref = fn(x, 3)
        res = opt_fn(x, 3)
        self.assertEqual(ref, res)

        ref = fn(x, 4)
        res = opt_fn(x, 4)
        self.assertEqual(ref, res)

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_interacts_with_numpy_ndarray(self):
        def fn(x, y):
            a = x.numpy()
            b = y.numpy()
            c = np.ones_like(a)
            d = np.ones_like(b)
            torch._dynamo.graph_break()
            return np.add(a, c), np.add(b, d)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_ndarray_works_with_builtin_function(self):
        def fn(x):
            v = x.sum() / len(x)
            return v

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        for _ in range(10):
            x = np.random.randn(2, 3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_array_of_arrays(self):
        def fn(x, y):
            return np.array([x, y])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x, y = np.float64(1), np.float64(2)
        res = opt_fn(x, y)
        self.assertEqual(res, np.array([1, 2], dtype=float))
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

        x, y = np.arange(2), np.arange(2) + 2
        res = opt_fn(x, y)
        self.assertEqual(res, np.array([[0, 1], [2, 3]]))
        self.assertEqual(type(res), np.ndarray)
        self.assertEqual(cnts.frame_count, 2)

    def test_numpy_readonly(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            return x

        x = np.broadcast_to(np.arange(3), (2, 3))
        self.assertFalse(x.flags.writeable)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", category=DeprecationWarning)  # from asyncio
            y = fn(x)
        self.assertTrue(y.flags.writeable)  # XXX: differs from numpy

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_numpy_tolist(self):
        def fn(x):
            return x.tolist()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x = np.arange(5)
        r = opt_fn(x)

        self.assertEqual(r, [0, 1, 2, 3, 4])
        self.assertEqual(type(r), list)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_size_attr(self):
        def fn(x):
            return x.size + x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        x = np.arange(5)
        r = opt_fn(x)

        self.assertEqual(r, fn(x))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_no_raise(self):
        def _inf_nan_preprocess(t, t_np):
            t_np = np.nan_to_num(t_np)
            return t, t_np

        def fn():
            # shape, dims format
            test_cases = (
                (3, 3),
                (4, 4),
                (5, 5),
            )

            for shape in test_cases:
                t = torch.randn(shape, dtype=torch.complex64)
                t_np = np.random.randn(*shape).astype(np.complex64)

                _, t_np = _inf_nan_preprocess(t, t_np)
                print(t, t_np)  # Just a side effect so that compilation kicks in

        cnt = CompileCounterWithBackend("inductor")
        fn = torch.compile(fn, backend=cnt)
        fn()
        self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))

    def test_mandelbrot_numpy(self):
        def mandelbrot_numpy(max_iter):
            # Define the boundaries of the complex plane
            xn = 450
            yn = 375
            xmin = -2.25
            xmax = 0.75
            ymin = -1.25
            ymax = 1.25

            # Create the grid of complex numbers
            x_values = np.linspace(xmin, xmax, xn, dtype=np.float64)
            y_values = np.linspace(ymin, ymax, yn, dtype=np.float64)
            rx, iy = np.meshgrid(x_values, y_values, indexing="xy")

            x = rx.copy()
            y = iy.copy()
            mask = np.zeros_like(x)
            for i in range(max_iter):
                x_prev = x
                y_prev = y
                x = x_prev**2 - y_prev**2 + rx
                y = 2 * x_prev * y_prev + iy
                inside = np.sqrt(x**2 + y**2) <= 2
                mask += inside
            return mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(mandelbrot_numpy, backend=cnts, fullgraph=True)
        n_iter = torch._dynamo.config.recompile_limit - 2
        for i in range(n_iter):
            x = i + 3
            ref = mandelbrot_numpy(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
        # We need to specialise the number as it's in a forloop
        self.assertEqual(cnts.frame_count, n_iter)

    def test_numpy_as_global(self):
        global x
        x = np.arange(10)

        @torch.compile(fullgraph=True, backend="eager")
        def fn(y):
            return y + x + x

        r = fn(np.arange(10))
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x * 3)
        del x

    def test_numpy_gt(self):
        x = np.arange(10)

        @torch.compile(backend="eager")
        def fn(y):
            return y >= 3

        r = fn(x)
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r, x >= 3)

    def test_numpy_min(self):
        x = np.arange(10)

        @torch.compile(backend="eager")
        def fn(y):
            return min(y, 3), min(y, y - 1)

        r1, r2 = fn(x)
        self.assertEqual(type(r1), np.ndarray)
        self.assertEqual(type(r2), np.ndarray)
        self.assertEqual(r1, np.minimum(x, 3))
        self.assertEqual(r2, np.minimum(x, x - 1))

    def test_graph_break_correctly_when_passing_numpy_ndarray_to_torch_function(self):
        # from transformers/models/big_bird/modeling_big_bird.py
        def fn(x: int, y: torch.Tensor):
            ndarray_list = [np.ones([2, x])]
            ndarray = np.stack(ndarray_list, axis=0)
            tensor = torch.tensor(ndarray, dtype=torch.long)
            tensor.unsqueeze_(0)
            return tensor + y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for x in range(1, 10):
            y = torch.randn([1, 2, x])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        # It's all traced once with x = 1 and then x = ks0
        # For dynamic it's x=ks0
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(str(cnts.frame_count), """2""")
        else:
            self.assertExpectedInline(str(cnts.frame_count), """2""")

    @skipIfWindows(
        msg="AssertionError: Object comparison failed: dtype('int64') != <class 'int'>"
    )
    def test_numpy_with_builtin_type(self):
        x = np.random.rand(5)

        def fn(x):
            return (x * 5).astype(bool).astype(float).astype(int) + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn(x)
        self.assertEqual(r.dtype, int)
        self.assertEqual(cnts.frame_count, 1)

    def test_with_builtin_type(self):
        x = torch.randn(5)

        def fn(x):
            return (x * 5).to(bool).to(float).to(int) + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn(x)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unique_consecutive(self):
        x = torch.tensor([1, 1, 2, 2, 1, 3])

        def fn(x):
            return torch.unique_consecutive(x)

        expected = fn(x)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        result = opt_fn(x)
        self.assertEqual(result, expected)

    def test_numpy_unique_f16(self):
        def fn():
            x = np.asarray([1, 1, 2, 2, 3], dtype=np.float16)
            return np.unique(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn()
        self.assertEqual(r.dtype, np.float16)
        self.assertEqual(cnts.frame_count, 1)

    def test_numpy_fallback_on_eager(self):
        def fn():
            return np.asarray(["L", "U"])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        r = opt_fn()
        self.assertEqual(cnts.frame_count, 0)  # graph break
        self.assertEqual(r, np.asarray(["L", "U"]))

        # repeat with a different function
        def fn2():
            return np.random.choice(["L", "U"])

        cnts2 = torch._dynamo.testing.CompileCounter()
        opt_fn2 = torch.compile(fn2, backend=cnts2)

        r2 = fn2()
        self.assertEqual(cnts.frame_count, 0)
        if r2 not in ("L", "U"):
            raise AssertionError(f"Expected r2 to be 'L' or 'U', got {r2}")

    def test_trace_ndarray_frame(self):
        def fn(x):
            x = x**2
            print("graph break.")
            return 2 * x

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)

        x = np.arange(8)
        self.assertEqual(fn(x), compiled_fn(x))
        self.assertEqual(counter.frame_count, 2)

    @skipIfWindows(
        msg="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64."
    )
    def test_trace_ndarray_frame_2(self):
        # no tensors/ndarray as inputs in the frame
        def fn(x):
            print("graph break.")
            return 2 * np.arange(x)

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)

        x = 8
        self.assertEqual(fn(x), compiled_fn(x))
        self.assertEqual(counter.frame_count, 1)

    def test_numpy_non_torch_dtype(self):
        # test that we gracefully graph break on dtypes
        # that do not have pytorch equivalents.
        def fn(x):
            return isinstance(x, torch.Tensor)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        # torch does not have the `uint16` dtype
        for x in [np.array([42], dtype=np.uint16), np.uint16(42), np.dtype("uint16")]:
            r = opt_fn(x)

            self.assertEqual(r, False)
            self.assertEqual(cnts.frame_count, 0)  # graph break

    def test_numpy_iter(self):
        # test that iteration over an ndarray produces ndarrays not bare tensors
        def fn(x):
            return [bm for bm in x]

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        proba_map = np.arange(3)[:, None]
        res = opt_fn(proba_map)

        self.assertEqual([type(r) for r in res], [np.ndarray, np.ndarray, np.ndarray])
        self.assertEqual(res, [np.array([0]), np.array([1]), np.array([2])])
        self.assertEqual(cnts.frame_count, 1)

    # cache size limit needs to be larger than the `dtypes` list size
    @torch._dynamo.config.patch(recompile_limit=12)
    def test_dtypes_no_graphbreaks(self):
        dtypes = [
            # floats
            float,
            np.float64,
            "float64",
            np.float32,
            "float32",
            # np.dtype('float64')   # XXX: this is not supported, yet
            # integers
            int,
            "int",
            np.intp,
            np.int32,
            np.uint8,
            # np.dtype('int')       # XXX: as above
        ]

        def fn(dt):
            return np.arange(5, dtype=dt)

        for dtyp in dtypes:
            cnts = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnts)

            val = fn(dtyp)
            opt_val = opt_fn(dtyp)

            self.assertEqual(cnts.frame_count, 1)  # no graph break

    # setting the config value makes the PRNG identical to numpy's
    # NB this may involve a graph break
    @torch._dynamo.config.patch(use_numpy_random_stream=True)
    def test_numpy_random_config_to_numpy(self):
        @torch.compile(backend="eager")
        def fn():
            return np.random.uniform(size=13)

        self.assertEqual(fn().shape, (13,))

    def test_inplace_view_on_graph_input(self):
        # graph break when calling methods with inplace_view tag on graph input
        func_args_map = {
            lambda x: x.resize_(6).mul_(2): torch.ones(4),
            lambda x: x.t_().mul_(2): torch.rand(2, 3),
            lambda x: x.transpose_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.squeeze_().mul_(2): torch.rand(1, 2, 3),
            lambda x: x.unsqueeze_(0).mul_(2): torch.rand(2, 3),
            lambda x: x.resize_as_(torch.rand(200, 300)): torch.rand(2, 3),
            lambda x: x.swapaxes_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.swapdims_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.rename_("N", "C").mul_(2): torch.zeros(2, 3),
            lambda x: x.as_strided_((3, 2), (2, 1)).mul_(2): torch.zeros(2, 3),
            lambda x: x.detach_().mul_(2): torch.zeros(2, 3),
        }
        for func, args in func_args_map.items():
            args_clone = args.clone()
            cnts = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(func, backend=cnts)
            self.assertTrue(same(func(args).shape, opt_f(args_clone).shape))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 1)  # mul_
