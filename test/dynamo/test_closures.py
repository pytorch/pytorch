# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: E722,F403,F405,F841
try:
    from .test_misc import *
except ImportError:
    from test_misc import *


class ClosureTests(torch._inductor.test_case.TestCase):
    def test_closure_recompiles(self):
        cnt = CompileCounter()

        def fn(x, other_fn):
            return other_fn(x + 1) - 1

        opt = torch.compile(fn, backend=cnt, fullgraph=True)

        x = torch.randn(8)
        for f in (
            closure_adder(5),
            closure_adder(5),
            closure_adder(torch.randn(8)),
            closure_adder(torch.randn(8)),
        ):
            self.assertEqual(opt(x, f), fn(x, f))

        self.assertEqual(cnt.frame_count, 2)

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

    def test_nested_wraps(self):
        def foo(x, y):
            def add(x, y):
                return x + y

            @functools.wraps(add)
            def wrapped_call(x, y):
                return add(x, y)

            return wrapped_call(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x + y)

        def foo(x, y):
            def nested_call(x, y):
                def mul(x, y):
                    return x * y

                @functools.wraps(mul)
                def double_nested_call(x, y):
                    return mul(x, y)

                return double_nested_call(x, y)

            return nested_call(x, y)

        o = torch.compile(foo, fullgraph=True, backend="eager")(x, y)
        self.assertEqual(o, x * y)

    def test_setattr_mutation1(self):
        class MyObj:  # noqa: B903
            def __init__(self, a, b):
                self.a = a
                self.b = b

        def fn(obj):
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            obj.c = obj.a * obj.b + 4
            obj.b = obj.a * obj.c + 5
            obj.a = obj.b * obj.c + 6
            return obj

        x1 = torch.randn(10)
        x2 = torch.randn(10)
        obj1 = MyObj(x1, x2)
        obj2 = MyObj(x1, x2)
        fn(obj2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIs(opt_fn(obj1), obj1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    def test_setattr_mutation2(self):
        class MyObj:
            def __init__(self, x):
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1.a, obj2.a))
        self.assertTrue(same(obj1.b, obj2.b))
        self.assertTrue(same(obj1.c, obj2.c))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_setattr_mutation3(self):
        # TODO(jansel): dead code eliminate the object creation
        class MyObj:
            def __init__(self, x):
                super().__init__()
                self.a = x + 1
                self.b = x + 2

        def fn(x):
            x = x / 3.0
            obj = MyObj(x)
            obj.c = obj.a * obj.b + 1
            obj.b = obj.a * obj.c + 2
            obj.a = obj.b * obj.c + 3
            return obj.a, obj.b, obj.c

        x1 = torch.randn(10)
        obj2 = fn(x1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        obj1 = opt_fn(x1)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure(self):
        v0 = torch.randn(10)

        def fn1():
            v1 = torch.randn(10)

            def fn2(*args, **kwargs):
                assert len(args) == 1  # noqa: S101
                assert len(kwargs) == 1  # noqa: S101
                v2 = torch.randn(10) + args[0] + kwargs["b"]

                def fn3(v3=torch.randn(10)):
                    def fn4():
                        return v0 + v1 + v2 + v3 + 1

                    return fn4

                return fn3

            return fn2(1, b=2)()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        tmp1 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        tmp2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        self.assertTrue(tmp1().shape, (10,))
        self.assertTrue(same(tmp1(), tmp1()))
        self.assertFalse(same(tmp1(), tmp2()))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    def test_nested_closure_mutation(self):
        def fn1():
            v1 = torch.randn(10)

            def fn2():
                v2 = torch.randn(10)

                def fn3():
                    nonlocal v1, v2
                    v1 += 1
                    v2 += 2
                    return v1 + v2

                return fn3

            rv = fn2()
            rv()
            rv()
            return rv

        torch.manual_seed(9000)
        counter1 = fn1()
        result1 = [counter1(), counter1(), counter1()]

        torch.manual_seed(9000)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch._dynamo.optimize_assert(cnts)(fn1)
        counter2 = torch._dynamo.optimize_assert(cnts)(opt_fn1())
        result2 = [counter2(), counter2(), counter2()]
        result1.append(counter1())
        result2.append(counter2())

        self.assertTrue(same(result1, result2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 11)

    def test_write_to_closures_in_inlining(self):
        out = []
        for use_dynamo in [False, True]:

            def make_counter():
                x = torch.randn(10)

                def counter():
                    nonlocal x
                    x = x + 1
                    return x

                return counter

            torch.manual_seed(0)
            counter = make_counter()
            if not use_dynamo:
                out.append(counter() + counter())
            else:
                cnts = torch._dynamo.testing.CompileCounter()

                @torch.compile(backend=cnts, fullgraph=True)
                def fn(counter):
                    return counter() + counter()

                out.append(fn(counter))
                self.assertEqual(cnts.frame_count, 1)
                self.assertEqual(cnts.op_count, 3)
                self.assertFalse(same(counter() + counter(), out[-1]))

        self.assertTrue(same(out[0], out[1]))

    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell(self):
        cell1 = torch.rand(1).item()
        cell2 = torch.rand(3, 3)

        def indirect():
            return direct()

        def direct():
            def inner():
                return cell1 + 1, cell2 + 3

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts)
        result1, result2 = opt_fn()
        self.assertAlmostEqual(cell1 + 1, result1)
        self.assertTrue(torch.allclose(cell2 + 3, result2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    @torch._dynamo.config.patch(specialize_float=True)
    def test_closure_out_of_scope_cell_with_mutation(self):
        cell1 = torch.rand(1).item()
        orig1 = cell1
        cell2 = torch.rand(3, 3)
        orig2 = cell2.clone()

        def indirect():
            return direct()

        def direct():
            def inner():
                nonlocal cell1, cell2
                x = cell2 + 1
                cell1 += 1
                cell2 += 10
                x = x + cell2
                return cell1, cell2, x

            return inner()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(indirect, backend=cnts, fullgraph=True)
        for i in range(1, 4):
            result1, result2, _ = opt_fn()
            self.assertAlmostEqual(orig1 + 1 * i, result1)
            self.assertTrue(torch.allclose(orig2 + 10 * i, result2))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 3)
            cnts.clear()

    def test_closure_with_mutation_and_graph_break(self):
        def fn():
            x = torch.zeros(1)

            def subfunc():
                x[0] = backup

            if x[0] >= -1e5:
                pass

            backup = 1
            subfunc()
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        expected = fn()
        actual = opt_fn()
        self.assertTrue(same(expected, actual))
        self.assertEqual(cnts.frame_count, 2)

    def test_closure_out_of_scope_cell_with_cond(self):
        # Test closure with out-of-scope cell variable, used in a cond
        # where the two branches read different closure variables
        from functorch.experimental.control_flow import cond

        def g(x):
            return x

        class ModuleCondDeep(torch.nn.Module):
            def forward(self, pred, x):
                return self._indirection(pred, x)

            def _indirection(self, pred, x):
                return self.indirection(pred, x)

            def indirection(self, pred, x):
                def true_fn(y):
                    return y + 2

                def false_fn(y):
                    return y - 2

                def shallow(x):
                    return x * 2

                def deep(x):
                    # y = g(x)
                    y = x
                    return cond(
                        x[0][0] > 0,
                        true_fn,
                        false_fn,
                        [y],
                    )

                return cond(pred, shallow, deep, [x])

        mod = ModuleCondDeep()
        opt_mod = torch.compile(mod, backend="eager")
        inp = torch.randn(3, 3)
        exp1 = mod(torch.tensor(False), inp)
        actual1 = opt_mod(torch.tensor(False), inp)
        exp2 = mod(torch.tensor(True), inp)
        actual2 = opt_mod(torch.tensor(True), inp)
        self.assertTrue(torch.allclose(exp1, actual1))
        self.assertTrue(torch.allclose(exp2, actual2))

    def test_closure_write_across_functions(self):
        z = 1
        k = 2

        def create_fn():
            def fn(x):
                nonlocal k, z
                k = z

            return fn

        def update_z_and_run_fn(fn, x):
            nonlocal z
            z = 3
            fn(x)
            return x.cos()

        @torch.compile(backend="eager")
        def foo(x):
            fn = create_fn()
            return update_z_and_run_fn(fn, x)

        x = torch.randn(1)
        foo(x)
        self.assertEqual(3, z)
        self.assertEqual(3, k)

    def test_free_var_and_local_name_collision(self):
        x = 10

        def make_func():
            def func():
                return x

            return func

        @torch.compile(backend="eager")
        def root(t):
            x = 0
            func = make_func()
            res = func()
            return t + 1, x, res

        res = root(torch.ones(1))
        self.assertTrue(torch.allclose(torch.ones(1) + 1, res[0]))
        self.assertEqual(0, res[1])
        self.assertEqual(10, res[2])

    def test_cell_captured_by_existing_func_but_not_root_frame(self):
        x = torch.ones(1)

        def get_inner():
            def inner():
                return x + x

            # Calling `inner` so Dynamo won't skip this frame.
            return inner(), inner

        @torch.compile(backend="eager")
        def root():
            return get_inner()

        res, inner = root()
        self.assertTrue(torch.allclose(x + x, res))
        self.assertTrue(torch.allclose(inner(), res))

    def test_writes_to_cells_across_frames1(self):
        # This regression test was added when Dynamo accidentally had both
        # unboxed and normal modeling for pre-existing cells, and failed to
        # account for buffered writes when we read from the unboxed value.
        x = 0

        def inc_x():
            nonlocal x
            x += 1

        class MyObj:
            def inc_x_then_return_x(self, fn):
                fn()
                return x

        @torch.compile(backend="eager")
        def root(t):
            obj = MyObj()
            res = obj.inc_x_then_return_x(inc_x)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_writes_to_cells_across_frames2(self):
        # This regression test was added when Dynamo didn't fully account for
        # already established `CellVariable` instance for pre-existing cell,
        # while encountering the same cell again (we should reuse the instance
        # rather than creating a new one). This caused buffered writes to escape
        # the newly created `CellVariable`.
        x = 0

        def inc_x_and_get_x(obj):
            nonlocal x
            x += 1
            return obj.get_x()

        class MyObj:
            def get_x(self):
                return x

        @torch.compile(backend="eager")
        def root(t):
            obj = MyObj()
            res = inc_x_and_get_x(obj)
            return t + 1, res

        res = root(torch.zeros(1))
        self.assertTrue(torch.allclose(res[0], torch.ones(1)))
        self.assertEqual(res[1], 1)
        self.assertEqual(x, 1)

    def test_write_to_cells_with_name_shadowing(self):
        x = 0
        y = x

        def make_x_get_set():
            # NOTE: this `x` is a different cell object than the outer `x`.
            x = y

            def set_x(v):
                nonlocal x
                x = v

            def get_x():
                return x

            return get_x, set_x

        get_x, set_x = make_x_get_set()

        @torch.compile(fullgraph=True, backend="eager")
        def fn(t):
            set_x(42)  # This sets the `x` created within `make_x_get_set`
            res = t + x  # This uses the `x` outside `make_x_get_set`.
            return res

        result = fn(torch.ones(1))
        inner_x = get_x()
        self.assertTrue(torch.allclose(result, torch.ones(1)))
        self.assertEqual(inner_x, 42)

    def test_existing_func_that_creates_capturing_nested_func(self):
        x = 0  # Captured by both `make_get_x` and `root`

        def make_get_x():
            def get_x():
                return x

            return get_x

        @torch.compile(backend="eager", fullgraph=True)
        def root(t):
            get_x = make_get_x()
            res = t + x
            return res, get_x

        res, get_x = root(torch.ones(1))
        self.assertTrue(torch.allclose(res, torch.ones(1)))
        self.assertEqual(0, get_x())
        x += 1
        self.assertEqual(1, get_x())

    def test_nested_optimize_decorator(self):
        cnts2 = torch._dynamo.testing.CompileCounter()
        cnts3 = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.run()
        def fn1(x):
            return torch.sin(x) * 10

        @torch.compile(backend=cnts2, fullgraph=True)
        def fn2(x):
            return fn1(x) + 1

        @torch.compile(backend=cnts3, fullgraph=True)
        def fn3(x):
            return torch.relu(fn2(x))

        fn3(torch.randn(4, 5))
        self.assertEqual(cnts2.frame_count, 0)
        self.assertEqual(cnts3.frame_count, 1)
        self.assertEqual(cnts3.op_count, 4)

    def test_nested_optimize_run(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 1)

        fn(torch.randn(4, 4))
        self.assertEqual(cnts.frame_count, 2)

        # Test that run works on a decorated fn
        fn = torch._dynamo.run(fn)
        fn(torch.randn(4, 4, 4))
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_optimize(self):
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return torch.relu(torch.cos(x) + torch.sin(x))

        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)

        # The first optimize in the nesting should be ignored
        fn2(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 1)
        self.assertEqual(cnts1.frame_count, 0)

        # Since the fn code object is already compiled, calling fn1 should
        # directly call the compiled_fn callable.
        torch._dynamo.run()(fn1)(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 0)

        # Test same behavior by reversing the calls
        torch._dynamo.reset()
        cnts1 = torch._dynamo.testing.CompileCounter()
        cnts2 = torch._dynamo.testing.CompileCounter()
        fn1 = torch.compile(fn, backend=cnts1, fullgraph=True)
        fn2 = torch.compile(fn1, backend=cnts2, fullgraph=True)
        fn1(torch.randn(4))
        self.assertEqual(cnts1.frame_count, 1)
        torch._dynamo.run()(fn2)(torch.randn(4))
        self.assertEqual(cnts2.frame_count, 0)

    def test_inline_func_jump_on_tensor_condition(self):
        def f1(input):
            if input == 0:
                return input + 1
            else:
                return input + 2

        def f2(input):
            return f1(input)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res1 = opt_f2(torch.tensor([1.0]))
        res2 = opt_f2(torch.tensor([0.0]))

        self.assertEqual(res1, 3)
        self.assertEqual(res2, 1)

    def test_inline_list_mutation(self):
        def f1(x):
            x.append(torch.ones(8))
            return x

        def f2():
            x = [torch.ones(6)]
            f1(x)
            return x

        res1 = f2()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_inline_dict_mutation(self):
        def f1(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        def f2():
            d = {"a": torch.ones(5), "b": torch.ones(5)}
            f1(d)
            return d

        res1 = f2()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f2 = torch.compile(f2, backend=cnts)
        res2 = opt_f2()
        self.assertTrue(same(res1, res2))

    def test_inline_local_dict_clear(self):
        def f(d):
            d.clear()
            return d

        inp = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        out = torch.compile(f, backend="eager", fullgraph=True)(inp)
        self.assertEqual(len(out), 0)
        self.assertEqual(len(inp), 0)

    def test_inline_module_attr_dict_clear(self):
        class MyMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

            def forward(self):
                self.a.clear()
                return self.a

        m = MyMod()
        out = torch.compile(m, backend="eager", fullgraph=True)()
        self.assertEqual(len(out), 0)
        self.assertEqual(len(m.a), 0)

    def test_inline_user_defined_dict_attr_clear(self):
        class MyMod:
            def __init__(self) -> None:
                self.a = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}

        def f(obj, inp):
            ret = len(obj.a) + inp
            obj.a.clear()
            return obj.a, ret

        m = MyMod()
        before_len = len(m.a)
        t_inp = torch.ones(1)
        d, ret = torch.compile(f, backend="eager", fullgraph=True)(m, t_inp)
        self.assertEqual(len(m.a), 0)
        self.assertEqual(len(d), 0)
        self.assertEqual(ret, t_inp + before_len)

    def test_recursive_inline_list_mutation(self):
        def f1(x, y):
            x.append(torch.tensor([1.1]))
            y.append(torch.tensor([1.2]))
            return x, y

        def f2(x, y):
            x.append(torch.tensor([2.1]))
            y.append(torch.tensor([2.2]))
            f1(x, y)
            return x, y

        def f3(x):
            x.append(torch.tensor([3.1]))
            y = [torch.tensor([3.2])]
            f2(x, y)
            return x, y

        def f4():
            x = [torch.tensor([4.1])]
            return f3(x)

        res1 = f4()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f4 = torch.compile(f4, backend=cnts)
        res2 = opt_f4()
        self.assertTrue(same(res1, res2))

    def test_replay_side_effects_config(self):
        # Test that replay_side_effects config controls mutation replay
        def fn(x, lst):
            lst.append(x + 1)
            return x * 2

        x = torch.tensor([5.0])

        # Test with replay enabled (default)
        lst_with_replay = []
        opt_fn_with_replay = torch.compile(fn, backend="eager")
        result1 = opt_fn_with_replay(x, lst_with_replay)
        self.assertEqual(len(lst_with_replay), 1)  # Mutation should be replayed
        self.assertTrue(same(result1, x * 2))

        torch._dynamo.reset()

        # Test with replay disabled
        lst_without_replay = []
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="warn"
        ):
            opt_fn_without_replay = torch.compile(fn, backend="eager")
            result2 = opt_fn_without_replay(x, lst_without_replay)
            self.assertEqual(
                len(lst_without_replay), 0
            )  # Mutation should NOT be replayed
            self.assertTrue(same(result2, x * 2))

        torch._dynamo.reset()
        lst_without_replay = []
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            opt_fn_without_replay = torch.compile(fn, backend="eager")
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: [\"L['lst']\"]"
                ),
            ):
                _ = opt_fn_without_replay(x, lst_without_replay)

    def test_replay_side_effects_model_attr(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = 4

            def forward(self, x):
                return x.cos()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = 4
                self.tensor = None
                self.bar = Bar()

            def forward(self, x):
                self.const = 5
                self.tensor = x.sin()
                res = self.bar(x)
                return x.cos() + res.sum() + self.tensor

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            foo = Foo()
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "While compiling, we found certain side effects happened in the model.forward. Here are the list of potential sources you can double check: [\"L['self']\"]"
                ),
            ):
                torch.compile(foo, fullgraph=True, backend="eager")(torch.randn(4, 4))

        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="silent"
        ):
            foo_v2_compile = Foo()
            foo_v2_eager = Foo()
            inp = torch.randn(4, 4)
            res = torch.compile(foo_v2_compile, fullgraph=True, backend="eager")(
                torch.randn(4, 4)
            )
            self.assertEqual(foo_v2_compile.tensor, None)
            self.assertEqual(foo_v2_compile.const, 4)
            self.assertEqual(foo_v2_compile.bar.const, 4)
            same(res, foo_v2_eager(inp))

    def test_replay_side_effects_input_mut(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = 4
                self.tensor = None

            def forward(self, x):
                x.add_(5)
                return x.cos()

        # This is ok because we actually capture the graph which
        # has mutation. In export, we never retrace the actual
        # gm so we won't see any mutation applied to inputs
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="error"
        ):
            foo = Foo()
            torch.compile(foo, fullgraph=True, backend="eager")(torch.randn(4, 4))

    def test_nested_sequential_with(self):
        def fn(x):
            with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(False):
                    x = x + 1
                with torch.set_grad_enabled(True):
                    x = x + 1
                return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try(self):
        def fn(x):
            try:
                try:
                    x = x + 1
                except:
                    pass
                try:
                    try:
                        x = x + 1
                    except:
                        pass
                except:
                    pass
            except:
                pass
            return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try_with(self):
        def fn(x):
            with torch.set_grad_enabled(True):
                try:
                    x = x + 1
                except:
                    pass
                try:
                    with torch.set_grad_enabled(False):
                        x = x + 1
                except:
                    pass
            return x

        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(torch.ones(1)), torch.tensor([3.0]))

    def test_nested_sequential_try_with_graph_break(self):
        def fn(x, n):
            with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(False):
                    x = x + 1
                    torch._dynamo.graph_break()
                try:
                    with torch.set_grad_enabled(False):
                        x = x + 1
                        if n == 0:
                            torch._dynamo.graph_break()
                except:
                    pass
                with torch.set_grad_enabled(False):
                    x = x + 1
                    torch._dynamo.graph_break()
                x = x + 1
            return x

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(opt_fn(torch.ones(1), 0), torch.tensor([5.0]))
        self.assertEqual(counter.frame_count, 1)

        torch._dynamo.reset()
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        self.assertEqual(opt_fn(torch.ones(1), 1), torch.tensor([5.0]))
        self.assertEqual(counter.frame_count, 3)

    def test_inline_dict_function(self):
        def _result_type_dict(dtype):
            return {bool: torch.float32}[dtype]

        @torch.compile(backend="eager")
        def f():
            return torch.ones(3, dtype=_result_type_dict(bool))

        self.assertEqual(f(), torch.ones(3, dtype=torch.float32))

    def test_inline_dict_function_passed_as_arg(self):
        @torch.compile(backend="eager")
        def fn(d, x, y):
            if d[x] is torch.float32:
                return y.cos()
            else:
                return y.sin()

        dd = {bool: torch.float32, int: torch.int64}
        self.assertEqual(fn(dd, bool, torch.ones(4)), torch.ones(4).cos())
        self.assertEqual(fn(dd, int, torch.ones(4)), torch.ones(4).sin())

    def test_nested_function_resuming_with_correct_globals(self):
        # https://github.com/pytorch/pytorch/issues/99665
        try:
            from .utils import outer_func
        except ImportError:
            from utils import outer_func

        def gn(x, y):
            return x + y

        def fn(x, y):
            return outer_func(gn)(x, y)

        x = torch.rand([3])
        y = torch.rand([3])
        opt_fn = torch.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_inline_closure_not_loaded_by_parent(self):
        def outer(a):
            return a + 1

        def indirect(x):
            return direct(x)

        def direct(x):
            def deep2(c):
                return outer(c)

            def deep(c):
                return deep2(c)

            return deep(x)

        x = torch.randn(3)
        eager = indirect(x)
        counter = CompileCounter()
        compiled = torch.compile(indirect, backend=counter)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    def test_inline_closure_returned_by_another_function_and_captures(self):
        x = torch.ones(1)

        def fn():
            def inner():
                return x + 2

            return inner

        @torch.compile(backend="eager")
        def start():
            # Obtain the `inner` function, which holds reference to `x`.
            inner = fn()

            # When we call `inner`, we end up looking up `x` from our inlining
            # tracer, Dynamo must make sure it still has some modeling of `x` at
            # that point.
            res = inner()
            return res

        res = start()
        self.assertEqual(torch.ones(1) * 3, res)

    def test_nested_dataclass_reconstruct(self):
        @dataclasses.dataclass(frozen=True)
        class NestedDataClass:
            x: int = 2

        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            y: torch.Tensor
            ndc: NestedDataClass = NestedDataClass()

        def fn(y):
            dc = TestDataClass(y)
            z = dc.y + dc.ndc.x
            return z, dc

        fn_opt = torch.compile(backend="eager")(fn)
        inps = (torch.ones(2, 2),)
        actual = fn_opt(*inps)
        expected = fn(*inps)

    def test_nested_frozen_dataclass_hashable(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClassInner:
            x: float
            y: float

        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            b: TestDataClassInner
            z: int
            a: int

        def inner_fn(dc, x, y):
            d = {}
            d[dc] = 2
            return dc.b.x + dc.b.y + d[dc] + x + y

        def fn(x, y):
            dc = TestDataClass(b=TestDataClassInner(2.4, 4.4), z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

    def test_return_dict_with_graph_break_and_update(self):
        def create():
            torch._dynamo.graph_break()
            return {0: torch.tensor(3)}

        def fn():
            return {**create()}

        opt_fn = torch.compile(backend="eager")(fn)
        result = opt_fn()
        self.assertIn(0, result)
        self.assertTrue(same(result[0], torch.tensor(3)))

    class AssertNumOutputBackend:
        """
        A backend that checks the number of output for compiled graph, and
        return the graph as is.
        """

        def __init__(self, test_case, expected_num_output: int):
            self.test_case = test_case
            self.expected_num_output = expected_num_output

        def __call__(self, gm: torch.fx.GraphModule, example_inputs):
            outputs = gm(*example_inputs)
            self.test_case.assertEqual(self.expected_num_output, len(outputs))
            return gm

    def test_returning_nested_func_with_captured_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def test():
            x = torch.rand(1)

            def func():
                return x + x

            # Returning `func` forces dynamo to output `x` in the compiled
            # graph, so that we can store it as `func`'s closure. The output of
            # compiled graph would be `(x, x + x)`.
            return func, func()

        test()

    def test_running_nested_func_with_captured_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 1))
        def test():
            x = torch.rand(1)

            def func():
                return x + x

            # `x` is no longer needed after running the compiled graph, so we
            # shouldn't return it. The output of compiled graph would be `(x +
            # x,)`.
            return func()

        test()

    def test_returning_func_with_captured_func_and_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def test():
            x = torch.rand(1)

            def nested():
                return x + x

            def func():
                return nested()

            # Returning `func` forces dynamo to output `x` in the compiled
            # graph, so that we can store it as `func`'s closure. The output of
            # compiled graph would be `(x, x + x)`.
            return func, func()

        test()

    def test_running_func_with_captured_func_and_tensor(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 1))
        def test():
            x = torch.rand(1)

            def nested():
                return x + x

            def func():
                return nested()

            # `x` is no longer needed after running the compiled graph, so we
            # shouldn't return it. The output of compiled graph would be `(x)`.
            return func()

        test()

    def test_escaping_closure_var_with_backward_hook(self):
        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def fn(x):
            temp = x * x
            captured_var = temp + 1

            # This is where the lambda escapes the lifetime of `fn`, so
            # dynamo must generate proper bytecode to update `captured_var`.
            x.register_hook(lambda _: captured_var)

            # The output of compiled graph would be `(x * x, x * x + 1)`.
            return temp

        ones = torch.ones(4, requires_grad=True)
        fn(ones).sum().backward()

    def test_escaping_closure_var_with_nonlocal_var(self):
        nonlocal_fn = None

        @torch.compile(backend=self.AssertNumOutputBackend(self, 2))
        def fn(x):
            temp = x * x
            captured_var = x + 1

            def inner():
                return captured_var

            # This is where `inner` escapes the lifetime of `fn`, so dynamo must
            # generate proper bytecode to update `captured_var`.
            nonlocal nonlocal_fn
            nonlocal_fn = inner

            # The output of compiled graph would be `(x * x, x * x + 1)`.
            return temp

        ones = torch.ones(4, requires_grad=True)
        fn(ones)
        nonlocal_fn()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
