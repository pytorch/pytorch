# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: C405,C416,F403,F405,F841,PLR0133,RUF015,SIM113,TRY203,UP032
try:
    from .test_misc import *
except ImportError:
    from test_misc import *


class PythonBuiltinTests(torch._inductor.test_case.TestCase):
    def test_int_shape_binops(self):
        def fn(x):
            # Test reversal by putting int arg first.
            y = 15 - x.shape[0]
            y = 4 + y
            y = 5 * y
            y = 2 % y
            y = 3**y
            y = 10 // y
            y = pow(2, y)
            y = 10 / y
            return x + y

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 9)
        )

    def test_int_shape_inplace_binops(self):
        def fn(x):
            p = x.shape[0]
            # Test reversal by putting constant first
            y = 2
            y += p
            y = 2
            y -= p
            y = 2
            y **= p
            y = 2
            y /= p
            y = 2
            y *= p
            y = 2
            y //= p
            y = 2
            y %= p
            return x + y

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 2)
        )

    def test_int_int_comparisons(self):
        def fn(x):
            if 2 != 2:
                out = 1
            elif 2 < 1:
                out = 1
            elif 1 > 2:
                out = 1
            elif 1 >= 2:
                out = 1
            elif 2 <= 1:
                out = 1
            elif 2 == 2:
                out = 2
            else:
                out = 1
            return x + out

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_int_shape_comparisons(self):
        def fn(x):
            a = x.shape[0]
            # Ensure support for constant on left side
            if 10 != a:
                out = 1
            elif 12 < a:
                out = 1
            elif 2 > a:
                out = 1
            elif 2 >= a:
                out = 1
            elif 12 <= a:
                out = 1
            elif 10 == a:
                out = 2
            else:
                out = 1
            return x + out

        # TODO: Test the guards maybe?
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_builtin_abs(self):
        def fn(x, y):
            return abs(x) + abs(y)

        sample = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for sample in [
            (torch.randn(10, 10), torch.randn(10, 10)),
            (-10, make_tensor(10, dtype=torch.int64, device="cpu")),
            (-0.1, torch.randn(10)),
        ]:
            expect = fn(*sample)
            actual = opt_fn(*sample)
            self.assertEqual(expect, actual)

    def test_builtin_isinstance(self):
        def fn(x):
            t = torch.arange(1, 3)
            a = isinstance(x, torch.Tensor)
            b = isinstance(t, torch.Tensor)
            c = isinstance(x, int)
            d = isinstance(3, int)
            e = isinstance([1, 2, 3], list)
            f = isinstance({"foo": 1, "bar": 2}, dict)
            res = [a, b, c, d, e, f]
            # Can't run yet due to other unimplemented instructions
            # res += [isinstance(torch.nn.LazyLinear(2, 3), torch.nn.Linear)]
            return res

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_empty_list(self):
        def fn(x, ll):
            if len(ll) == 0 and not ll and ll is not None:
                return x + 1

        i = torch.randn(5, 10)
        r1 = fn(i, [])
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i, [])
        r3 = opt_fn(i, ())
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

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

    def test_python_slice(self):
        def f1(input):
            y = 0
            for i, x in enumerate(input[2:], 1):
                y = y + x
            return y

        def f2(input):
            y = 0
            for i, x in enumerate(input.shape[2:], 1):
                y = y + x
            return y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
        res1 = opt_f1([1, 2, 3, 5])
        res2 = opt_f2(torch.rand([2, 3, 4, 5]))

        self.assertEqual(res1, 8)
        self.assertEqual(res2, 9)

    def test_builtin_subclasses_as_method_on_class_type(self):
        class Foo:
            def __init__(self, name):
                self.ame_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Foo):
            def __init__(self, name):  # noqa: B903
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()

        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        subs_of_foo_optim = fn()

        self.assertEqual(len(subs_of_foo_reg), 2)
        self.assertEqual(subs_of_foo_reg, subs_of_foo_optim)

    def test_builtin_subclasses_as_method_on_var(self):
        class Foo:
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Foo " + self.name_

        class Bar(Foo):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Bar " + self.name_

        class Baz(Bar):
            def __init__(self, name):
                self.name_ = name

            def get_name(self):
                return "Baz " + self.name_

        subs_of_foo_reg = Foo.__subclasses__()
        sub_of_foo_subclass_var_reg = subs_of_foo_reg[0].__subclasses__()

        sub_of_foo_subclass_var_optim = []
        counter = CompileCounter()

        @torch._dynamo.optimize_assert(counter)
        def fn():
            return Foo.__subclasses__()

        @torch._dynamo.optimize_assert(counter)
        def fn_single(subs_of_foo_optim):
            return subs_of_foo_optim[0].__subclasses__()

        subs_of_foo_optim = fn()
        sub_of_foo_subclass_var_optim = fn_single(subs_of_foo_optim)

        self.assertEqual(len(sub_of_foo_subclass_var_optim), 1)
        self.assertEqual(sub_of_foo_subclass_var_optim, sub_of_foo_subclass_var_reg)

    def test_builtin_str_on_user_defined_function(self):
        def another_fn():
            pass

        def fn():
            return "another_fn" in str(another_fn)

        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertTrue(opt_fn())

    def test_repeat_interleave_graphbreaks(self):
        def fn_no_breaks(x):
            # no breaks on self_int
            x += 1
            x = torch.repeat_interleave(x, 2, 3)
            x += 1
            return x

        def fn_has_breaks(x):
            # breaks on self_Tensor
            x += 1
            x = torch.repeat_interleave(x, torch.tensor(2), 3)
            x += 1
            return x

        x = torch.randn([4, 16, 1, 64])

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn_no_breaks, backend=cnts)
        opt_fn(x)
        self.assertEqual(cnts.frame_count, 1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn_has_breaks, backend=cnts)
        opt_fn(x)
        self.assertEqual(cnts.frame_count, 2)

    def test_set_discard(self):
        def fn(y):
            x = set(["bar"])
            x.discard("bar")
            x.discard("foo")
            return y + len(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.randn(3)
        self.assertEqual(opt_fn(x), x)
        self.assertEqual(cnts.op_count, 1)

    def test_set_update(self):
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, int_set, int_list):
            int_set.update(map(int, int_list))
            return x + 1

        int_set = set()
        int_list = [1, 2, 1]
        res = run(torch.ones(1), int_set, int_list)
        self.assertTrue(same(res, torch.ones(1) + 1))
        self.assertEqual(int_set, set([1, 2]))
        self.assertEqual(int_list, [1, 2, 1])

    def test_frozenset_torch_func_contains(self):
        funcs = frozenset([torch.add])

        def fn(x, func):
            if func in funcs:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, torch.add)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, torch.mul)
        self.assertEqual(cnts.op_count, 1)

    def test_list_append_return_none(self):
        def fn(x):
            alist = []
            blist = alist.append(x + 1)
            return alist, blist

        x = torch.tensor([2.3])
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)

    def test_nan(self):
        def f(x, n):
            return x * 2 + n

        x = torch.randn(4)
        n = float("nan")

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        opt_f(x, n)
        opt_f(x, n)
        self.assertEqual(cnts.frame_count, 1)

    def test_repr(self):
        class Config:
            def __repr__(self):
                return "Config()"

        def forward(x, config):
            return x * len(repr(config))

        config = Config()
        x = torch.randn(2, 2)

        compiled = torch.compile(forward, fullgraph=True, backend="eager")
        compiled(x, config)

    def test_large_reduction_list(self):
        dtype = torch.float32
        device = "cpu"

        def check_sum_all(tensor: torch.Tensor) -> None:
            pylist = tensor.reshape(-1).tolist()
            self.assertTrue(same(tensor.sum(), torch.tensor(sum(pylist))))

        check_sum_all(torch.randn(200000, dtype=dtype, device=device))

    def test_tuple_from_tuple_iter(self):
        def inner_fn(*args):
            acc = torch.ones(10, 10)
            for arg in args:
                acc.add_(arg)

            return acc

        @torch.compile(backend="eager")
        def fn(inputs, params):
            y = tuple(inputs) + tuple(params)
            return inner_fn(*y)

        inputs = [torch.randn(10, 10) for _ in range(3)]

        fn(inputs, iter(tuple(inputs)))

        def fn(params):
            y = tuple(params)
            return inner_fn(*y)

        opt_fn = torch.compile(fn, backend="eager")
        inputs = [torch.randn(10, 10) for _ in range(3)]
        self.assertTrue(same(fn(iter(tuple(inputs))), opt_fn(iter(tuple(inputs)))))

        # Force recompilation
        inputs = [torch.randn(10, 10) for _ in range(4)]
        self.assertTrue(same(fn(iter(tuple(inputs))), opt_fn(iter(tuple(inputs)))))

    def test_int_list(self):
        # if assume_static_by_default == True: spec int list
        # otherwise: unspec int list
        def fn(x, y):
            return torch.sin(x + y[1] % 2)

        x = torch.randn(6)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        for i in range(10, 25, 3):
            y = [i, i + 1, i + 2]
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))
        if torch._dynamo.config.assume_static_by_default:
            if torch._dynamo.config.automatic_dynamic_shapes:
                self.assertExpectedInline(cnt.frame_count, """2""")
            else:
                self.assertExpectedInline(cnt.frame_count, """5""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_patched_builtin_functions(self):
        import builtins

        # Cache the original builtin function ids
        torch._dynamo.trace_rules._builtin_function_ids()

        class MyClass:
            pass

        builtin_isinstance = builtins.isinstance

        def patched_isinstance(obj, classinfo) -> bool:
            if builtin_isinstance(obj, MyClass):
                return False
            else:
                return builtin_isinstance(obj, classinfo)

        def fn(x, y):
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        y = MyClass()

        try:
            ref = fn(x, y)
            # Monkey patch builtin function
            builtins.isinstance = patched_isinstance
            opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, x + 1))
            self.assertTrue(same(res, x - 1))
        finally:
            builtins.isinstance = builtin_isinstance

        # check recompilation because builtins is now unpatched
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(res, x + 1))

    def test_set_custom_tensor_attribute(self):
        def fn(x):
            x.custom_attr = 3.14
            return x.custom_attr * x

        x = torch.rand((2, 2))
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_ordered_dict_alias_reconstruct(self):
        od = collections.OrderedDict

        def fn():
            d1 = dict()  # noqa: C408
            d1["a"] = 1
            d2 = od(d1)
            d2["b"] = 2
            torch._dynamo.graph_break()
            if isinstance(d2, od):
                return d2["a"] + d2["b"]
            else:
                return 0

        dis.dis(fn)
        self.assertEqual(torch.compile(fn, backend="eager")(), 3)

    def test_int_neg(self):
        def int_neg(a, b):
            x = a.shape[0]
            y = b.shape[0]
            return -x * -y * a * b

        torch._dynamo.testing.standard_test(self, int_neg, 2)

    def test_hash_getitem_slice(self):
        s = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        s2 = GetItemSource(LocalSource("foo"), slice(None, -1, None))
        s3 = GetItemSource(LocalSource("foo"), slice(None, -1, 2))
        some_set = set()

        self.assertTrue(s not in some_set)
        self.assertTrue(s2 not in some_set)
        self.assertTrue(s3 not in some_set)

        some_set.add(s)

        self.assertTrue(s in some_set)
        # s and s2 should hash the  same
        self.assertTrue(s2 in some_set)
        # s3 should be different
        self.assertTrue(s3 not in some_set)

        self.assertTrue(s == s2)
        self.assertTrue(s != s3)

    def test_add_sizes(self):
        def func(x):
            y = x.size()
            return y + y

        eager_out = func(torch.ones(10, 10, 3))
        compile_out = torch.compile(func, backend="eager")(torch.ones(10, 10, 3))
        self.assertTrue(isinstance(compile_out, torch.Size))
        self.assertEqual(eager_out, compile_out)

    def test_list_hasattr1(self):
        def fn(x):
            if hasattr(x, "foo"):
                return x[0] + 1
            return x[0] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = [torch.randn(3)]
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_list_hasattr2(self):
        def fn():
            x = [torch.zeros(3)]
            if hasattr(x, "__len__"):
                return x[0] + 1
            return x[0] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertTrue(same(fn_out, compiled_out))

    def test_tuple_hasattr(self):
        def fn(x):
            if hasattr(x, "foo"):
                return x[0] + 1
            return x[1] - 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = (torch.randn(3), torch.randn(3))
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_simple_set_usage(self):
        def foo(x, y):
            setty = {x, y}
            return setty.pop() * setty.pop()

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        foo(x, y)
        self.assertEqual(counter.frame_count, 1)

    def test_add_to_set(self):
        def foo(x, y):
            setty = set()
            setty.add(x[0])
            setty.add(x[1])
            setty.add(x[2])
            setty.add(y)
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        eager_result = foo([x, x, x, x, y], y)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        result = foo([x, x, x, x, y], y)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_remove_set(self):
        def fn(x):
            set_a = set((4, 5))
            set_a.remove(4)
            return x * len(set_a)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_iter_set(self):
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)
        eager_result = foo([x, x, x, x, y], y)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter, fullgraph=True)
        result = foo([x, x, x, x, y], y)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(result, eager_result)

    def test_set_aliasing_recompiles(self):
        g1 = torch.randn(10)
        g2 = torch.randn(10)
        g3 = torch.randn(10)
        g4 = torch.randn(10)

        def foo(a, b, c):
            myset = {g1, a, b, c}
            return a + len(myset)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        # first call with no aliasing
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 1)

        # no aliasing again
        foo(g3, g2, g4)
        # assert no recompile
        self.assertEqual(counter.frame_count, 1)

        # aliasing changes, we should recompile
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 2)

        # same aliasing, different tensor
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 2)

        # aliasing between global and arg, should recompile again
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 3)

        # Reset
        torch._dynamo.reset()

        # aliasing between global and arg, first call
        foo(g1, g1, g1)
        self.assertEqual(counter.frame_count, 4)

        # same aliasing, different tensor, all local, recompile
        foo(g3, g3, g3)
        self.assertEqual(counter.frame_count, 5)

        # aliasing same tensor, we shouldn't recompile
        foo(g2, g2, g2)
        self.assertEqual(counter.frame_count, 5)

        # No aliasing
        foo(g2, g3, g4)
        self.assertEqual(counter.frame_count, 6)

        # No aliasing again
        foo(g3, g2, g4)
        # assert no recompile
        self.assertEqual(counter.frame_count, 6)

    def test_str_format_return1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            x = torch.sin(img)
            y = f"shape {img.shape[-2:]} batch size {img.shape[0]}"
            return img + x, y

        img1 = torch.randn(1, 1, 8, 8)
        res, msg = fn(img1)
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1")
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str___iter__(self):
        def fn(x):
            s = "a"
            if next(s.__iter__()) == "a":
                return x + 1
            else:
                return x

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_str_format_return2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            x = torch.sin(img)
            y = "shape {} batch size {y:.2f}".format(img.shape[-2:], y=img.shape[0])
            return img + x, y

        img1 = torch.randn(1, 1, 8, 8)
        res, msg = fn(img1)
        self.assertEqual(msg, "shape torch.Size([8, 8]) batch size 1.00")
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str_format_assert1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(img):
            x = torch.sin(img)
            val = x.shape[-2:]
            torch._assert(len(val) == 2, f"shape {img.shape}")
            return img + x

        img1 = torch.randn(1, 1, 8, 8)
        res = fn(img1)
        self.assertEqual(res, img1 + torch.sin(img1))

    def test_str_format_assert2(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(img):
            x = torch.sin(img)
            torch._assert(
                img.shape[-2] == 8 and img.shape[-1] == 16, f"shape {img.shape}"
            )
            return img + x

        img1 = torch.randn(1, 3, 8, 16)
        res = fn(img1)
        self.assertEqual(res, img1 + torch.sin(img1))
        self.assertEqual(cnt.frame_count, 1)

        # trigger a recompile and graph break
        img2 = torch.randn(1, 3, 8, 15)
        self.assertRaises(AssertionError, lambda: fn(img2))

    def test_deque_input(self):
        a = torch.randn([2, 3])
        b = torch.randn([2, 3])
        d1 = collections.deque(["foo", a, b])
        d2 = d1.copy()

        def fn(q):
            a = q.pop()
            b = q.pop()
            return a * b

        eager = fn(d1)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(d2)
        self.assertEqual(d1, d2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_deque_append_left(self):
        d1 = collections.deque(["foo", 10, 10])
        d2 = d1.copy()

        def fn(q, a, b):
            q.appendleft(a)
            q.appendleft(b)
            return q.popleft() * q.popleft()

        a = torch.randn([3, 3])
        b = torch.randn([3, 3])
        eager = fn(d1, a, b)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(d2, a, b)
        self.assertEqual(d1, d2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)
        self.assertTrue(isinstance(compiled, torch.Tensor))

    def test_yield_from(self):
        def yield_from_fn(t_list, k):
            def yield_from_gen(l):
                l2 = [t * k for t in l]
                yield from l2

            return [t * k for t in yield_from_gen(t_list)]

        t_list = [torch.randn([2, 3]) for _ in range(3)]
        eager = yield_from_fn(t_list, 2)
        counter = CompileCounter()
        compiled = torch.compile(yield_from_fn, backend=counter)(t_list, 2)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_in_a_loop(self):
        def gen2():
            yield 1

        def gen1():
            for value in range(5):
                yield from gen2()

        def fn(x):
            c = 0
            for i in gen1():
                c = c + i
            return x + c

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.zeros(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_yield_gen_and_from(self):
        def populate_and_multiply_sequence(n, multiplier):
            # Inline generator
            def tensor_generator():
                for i in range(n):
                    yield torch.tensor([i])

            # Use 'yield from' to iterate over tensors and multiply
            t_list = [tensor * multiplier for tensor in tensor_generator()]

            def yield_from_gen():
                yield from t_list

            return [t for t in yield_from_gen()]

        multiplier = torch.tensor([10])
        eager = populate_and_multiply_sequence(5, multiplier)
        counter = CompileCounter()
        compiled = torch.compile(populate_and_multiply_sequence, backend=counter)(
            5, multiplier
        )
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_yield_from_user_stop_iteration(self):
        class MyIter:
            def __init__(self, seq):
                self.seq = seq
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.index += 1
                if self.index <= len(self.seq):
                    return self.seq[self.index - 1]
                raise StopIteration(self.index)

        def yield_from_iter_fn(seq):
            def gen(seq):
                yield from MyIter(seq)

            return [i for i in gen(seq)]

        seq = [torch.randn([2, 3]) for _ in range(3)]
        eager = yield_from_iter_fn(seq)
        counter = CompileCounter()
        compiled = torch.compile(yield_from_iter_fn, backend=counter)(seq)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 0)

    def test_pep0479_convert_stopiteration(self):
        # https://peps.python.org/pep-0479/
        def generator_with_stop_iteration():
            yield 1
            # Explicitly raising StopIteration inside the generator
            raise StopIteration("StopIteration raised within generator")
            yield 2  # This should never be reached

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                # Try to consume the generator
                gen = generator_with_stop_iteration()
                next(gen)
                next(gen)
            except RuntimeError as e:
                # Check that StopIteration was converted to RuntimeError
                # See STOPITERATION_ERROR opcode in symbolic_convert.py
                return 100
            except StopIteration:
                return 200

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, 100)

    def test_yield_send_to_subgenerator_graph_break(self):
        def subgenerator(tensor):
            multiplier = yield
            yield tensor * multiplier

        def main_generator(t_list):
            for tensor in t_list:
                subgen = subgenerator(tensor)
                next(subgen)
                yield from subgen.send(torch.tensor([10]))

        t_list = [torch.tensor([i]) for i in range(5)]
        eager = list(main_generator(t_list))

        counter = CompileCounter()
        compiled_fn = torch.compile(main_generator, backend=counter)
        compiled = list(compiled_fn(t_list))

        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 0)

    def test_iterator_limit(self):
        def fn(x):
            def gen():
                while True:
                    yield x

            return list(gen())

        x = torch.randn([0, 1, 2, 3, 4, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "infinite generator"
        ):
            compiled_fn(x)

    def test_itertools_islice(self):
        counters.clear()

        def fn(x):
            return itertools.islice(x, 2, 5, 2)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_islice_default_step(self):
        counters.clear()

        def fn(x):
            return itertools.islice(x, 2, 5)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_islice_default_end(self):
        counters.clear()

        def fn(x):
            return itertools.islice(x, 2)

        x = torch.randn([0, 1, 2, 3, 4, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_repeat(self):
        counters.clear()

        def fn(x):
            r = itertools.repeat(100.0, 5)
            for i in r:
                x += i
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat(self):
        counters.clear()

        def fn(x):
            r = itertools.repeat(100.0)
            idx = 0
            for i in r:
                x += i
                idx += 1
                if idx > 10:
                    break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_repeat_mutation(self):
        counters.clear()

        def fn(x):
            r = itertools.repeat(x)
            idx = 0
            for i in r:
                x += i
                i += 1
                idx += 1
                if idx > 10:
                    break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_count(self):
        for args in ([], [10], [5, -1]):
            counters.clear()

            def fn(x):
                r = itertools.count(*args)
                idx = 0
                for i in r:
                    x += i
                    idx += 1
                    if idx > 10:
                        break
                return x

            x = torch.randn([2, 5])
            eager = fn(x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_count_from_uncompiled_region(self):
        counters.clear()
        counter = itertools.count()

        def fn(x):
            return x * (next(counter) + 1)

        x = torch.randn([2, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(compiled_fn(x), x)
        self.assertEqual(compiled_fn(x), x * 2)
        self.assertEqual(next(counter), 2)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_count_already_advanced(self):
        counters.clear()
        counter = itertools.count()
        # Advance the counter before entering the compiled region
        next(counter)  # 0
        next(counter)  # 1

        def fn(x):
            return x * (next(counter) + 1)

        x = torch.randn([2, 5])
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(compiled_fn(x), x * 3)  # next(counter) = 2
        self.assertEqual(compiled_fn(x), x * 4)  # next(counter) = 3
        self.assertEqual(next(counter), 4)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_infinite_cycle(self):
        counters.clear()

        def fn(x):
            for iterator in (
                iter([]),
                iter([10, 11.0]),
                itertools.repeat(-1, 3),
                itertools.count(10),
            ):
                r = itertools.cycle(iterator)
                idx = 0
                x += 1
                for i in r:
                    x += i
                    idx += 1
                    if idx > 10:
                        break
            return x

        x = torch.randn([2, 5])
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_symint_default_sum(self):
        # https://github.com/pytorch/pytorch/issues/110287
        counters.clear()

        def fn(x):
            r = itertools.accumulate([x.size(0), x.size(1)])
            for i in r:
                x *= i
            return x

        x = torch.randn(2, 3)
        eager = fn(x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_default_sum(self):
        counters.clear()

        def fn(a, b, c, d, x):
            l = [a, b, c, d, x]
            for i, t in enumerate(l):
                l[i] = t * x
            return itertools.accumulate(l)

        t_list = [torch.tensor([i + 1]) for i in range(4)]
        x = torch.tensor([[1, 2], [3, 4]])
        eager = fn(*t_list, x)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(*t_list, x)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_builtins(self):
        for builtin_op in [operator.mul, operator.sub, operator.pow]:
            counters.clear()

            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, builtin_op)

            t_list = [torch.tensor([i + 1]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])
            eager = fn(*t_list, x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_accumulate_tensors_kwargs(self):
        from torch._dynamo.utils import counters

        for kwargs in [
            {"func": operator.mul},
            {"initial": 100},
            {"func": operator.sub, "initial": -1},
        ]:
            counters.clear()

            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, **kwargs)

            t_list = [torch.tensor([i + 1]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)
            eager = fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_packaging_version_parse(self):
        from packaging import version

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            x = torch.zeros(1)
            if version.parse(torch.__version__) >= version.parse("2.0.0"):
                return x + 1
            return x

        self.assertEqual(fn().item(), 1)

    def test_itertools_accumulate_tensors_user_defined(self):
        def udo_fn_0(a, b):
            return -1

        rando = random.randint(0, 1)

        def udo_fn_1(a, b):
            return a * rando + b * rando

        seen = []

        def udo_fn_2(a, b):
            seen.append(a)
            seen.append(b)
            return a * len(seen)

        for udo_fn in [udo_fn_0, udo_fn_1, udo_fn_2]:
            counters.clear()
            torch._dynamo.reset()

            def fn(a, b, c, d, x):
                l = [a, b, c, d, x]
                for i, t in enumerate(l):
                    l[i] = t * x
                return itertools.accumulate(l, udo_fn)

            t_list = [torch.tensor([i]) for i in range(4)]
            x = torch.tensor([[1, 2], [3, 4]])
            eager = fn(*t_list, x)

            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled = compiled_fn(*t_list, x)

            self.assertEqual(list(eager), list(compiled))
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_pure_python_accumulate(self):
        def accumulate(iterable, func=lambda x, y: x + y):
            it = iter(iterable)
            try:
                # Initialize the accumulator with the first value from the iterable
                accumulator = next(it)
            except StopIteration:
                # If the iterable is empty, return an empty generator
                return
            yield accumulator

            for element in it:
                accumulator = func(accumulator, element)
                yield accumulator

        def fn(it):
            return accumulate(it)

        t_list = [torch.tensor([i]) for i in range(4)]
        eager = fn(t_list)

        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter)
        compiled = compiled_fn(t_list)

        self.assertEqual(list(eager), list(compiled))
        self.assertEqual(counter.frame_count, 1)

    def test_itertools_groupby_pure_python_default_identify_func(self):
        counters.clear()

        def fn(l):
            return [(k, list(g)) for k, g in itertools.groupby(l)]

        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_groupby_pure_python_key_func(self):
        counters.clear()

        def fn(l):
            return [(k, list(g)) for k, g in itertools.groupby(l, key=operator.neg)]

        l = [1, 2, -2, 3, 4, 4, -4, 0, -2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_itertools_tee(self):
        counters.clear()

        def fn(l):
            a, b = itertools.tee(l)
            return list(a), list(b)

        l = [1, 2, 2, 3, 4, 4, 4, 1, 2]
        eager = fn(l)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled = compiled_fn(l)

        self.assertEqual(eager, compiled)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_list_iterator_contains(self):
        def fn(x):
            it = iter(["my_weight", "not_my_weight"])
            next(it)
            if "my_weight" in it:
                return x + 2
            return x + 1

        x = torch.zeros(3)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)

        self.assertEqual(fn(x), compiled_fn(x))

    def test_conditional_list_comp_in_context(self):
        def fn(inp):
            try:
                return [torch.sin(x) for x in inp if x is not None]
            except Exception:
                pass

        inp = [torch.randn(3, 3) for _ in range(3)] + [None]
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(inp)

    def test_ordered_dict_move_to_end(self):
        d = {
            "foo": 1,
            "bar": 2,
        }

        d = collections.OrderedDict(d)
        d.move_to_end("foo")

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_defaultdict(self):
        d = collections.defaultdict()
        d["foo"] = 1
        d["bar"] = 2

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_hash_hop(self):
        associative_scan = importlib.import_module(
            "torch._higher_order_ops.associative_scan"
        )

        @torch.compile(fullgraph=True, backend="eager")
        def fn(y, s):
            d = dict()
            d[s] = y
            return d[s] + 1.0

        fn(torch.ones(2, 2, device="cpu"), associative_scan.AssociativeScanOp())

    def test_iter_type(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(y):
            x = iter([])
            if isinstance(x, list):
                return y + 1
            else:
                return y + 2

        res = fn(torch.ones(2))
        self.assertEqual(torch.ones(2) + 2, res)

    def test_tuple_class(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)

        r = opt_fn((d1, d2))
        self.assertEqual(r.__class__, tuple)
        r1, r2 = r
        self.assertEqual(r1, torch.ones(2, 2))
        self.assertEqual(r2, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_list_class(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)

        r = opt_fn([d1, d2])
        self.assertEqual(r.__class__, list)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0], torch.ones(2, 2))
        self.assertEqual(r[1], torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_frozenset_of_non_literals(self):
        class Foo:
            pass

        foo = Foo()
        foo.x = 0
        s = frozenset([foo])

        @torch.compile(backend="eager")
        def run(x, s, foo0):
            # Dynamo must have the same representation for `foo0` and `foo1`,
            # otherwise the update to `foo0.x` won't be reflected in the read of
            # `foo1.x`.
            foo1 = list(s)[0]
            foo0.x += 1
            return x + 1, foo1.x

        res = run(torch.ones(1), s, foo)
        self.assertTrue(same(res[0], torch.ones(1) + 1))
        self.assertEqual(res[1], 1)

    def test_ne_operator_with_custom_eq(self):
        class Foo:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return self.x == other.x

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            f1 = Foo(0)
            f2 = Foo(0)
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)

    def test_ne_operator_with_custom_ne(self):
        class Foo:
            def __init__(self, x):
                self.x = x
                self.ne_called = False

            def __ne__(self, other):
                # ne_called attr is later checked to ensure that overridden
                # `__ne__` is traced
                self.ne_called = True
                return not self.__eq__(other)

            def __eq__(self, other):
                return self.x == other.x

        f1 = Foo(0)
        f2 = Foo(0)

        @torch.compile(fullgraph=True, backend="eager")
        def run(x):
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)
        self.assertTrue(f1.ne_called)

    def test_ne_operator_with_custom_graphbreak_eq(self):
        counters.clear()

        class Foo:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                # This allows us to check that Dynamo actually traced into the
                # custom eq method.
                torch._dynamo.graph_break()
                return self.x == other.x

        @torch.compile(backend="eager")
        def run(x):
            f1 = Foo(0)
            f2 = Foo(0)
            # `x + 1` prevents Dynamo from skipping this frame.
            return x + 1, f1 != f2

        _, ne = run(torch.ones(1))
        self.assertFalse(ne)
        self.assertEqual(len(counters["graph_break"]), 1)

    @unittest.skipIf(sys.version_info < (3, 12), "Python 3.12+")
    def test_CALL_INTRINSIC(self):
        from torch.testing._internal.py312_intrinsics import Foo

        Foo.test_default_update(self)

    @unittest.skipIf(sys.version_info < (3, 11), "Python 3.11+")
    def test_RAISE_VARARGS_0(self):
        def foo():
            try:
                raise ValueError
            except:
                raise

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                foo()
            except ValueError:
                return t.sin()
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symint(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randint(10, (1,))

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_tensor(self):
        def f_all(mask):
            return bool((mask == 0).all())

        opt_f_all = torch.compile(f_all, backend="eager", fullgraph=True)

        mask_zeros = torch.zeros(2, 3)
        mask_nonzero = torch.tensor([[0, 1], [0, 0]])

        self.assertEqual(f_all(mask_zeros), opt_f_all(mask_zeros))
        self.assertEqual(f_all(mask_nonzero), opt_f_all(mask_nonzero))

        def f(x):
            return bool(x)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)

        for val in [42, 0, 3.14, 0.0]:
            x = torch.tensor(val)
            self.assertEqual(f(x), opt_f(x))
        non_scalar = torch.tensor([1, 2, 3])
        with self.assertRaises(RuntimeError):
            f(non_scalar)
        with self.assertRaises(RuntimeError):
            opt_f(non_scalar)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symfloat(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(1)

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symbool(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(1) == 1

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
