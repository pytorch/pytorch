# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: B021,E711,E721,F403,F405,F841
try:
    from .test_misc import *
except ImportError:
    from test_misc import *

from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size


class GuardTests(torch._inductor.test_case.TestCase):
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_repeat_cat(self):
        def f(x, n):
            m = x.item()
            x = torch.empty(x).repeat(n)  # s0*u0
            return torch.cat([x, x], dim=0)

        fn = torch.compile(f, backend="eager", dynamic=True, fullgraph=True)
        fn(torch.tensor([5]), 5)

    def test_shape_int_inplace_binops(self):
        def fn(x):
            p = x.shape[0]
            p += 2
            p -= 2
            p **= 2
            p /= 2
            p *= 2
            p //= 2
            p %= 2
            return x + p

        torch._dynamo.testing.standard_test(
            self, fn, 1, expected_ops=1, expected_ops_dynamic=ifdynstaticdefault(1, 6)
        )

    def test_shape_int_comparisons(self):
        def fn(x):
            a = x.shape[0]
            # Ensure support for constant on right side
            if a != 10:
                out = 1
            elif a < 2:
                out = 1
            elif a > 12:
                out = 1
            elif a >= 12:
                out = 1
            elif a <= 2:
                out = 1
            elif a == 10:
                out = 2
            else:
                out = 1
            return x + out

        # TODO: Test the guards maybe?
        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_param_shape_binops(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(15))

            def forward(self, x):
                # Test reversal by putting param shape arg first.
                p = self.param.shape[0]
                y = p - x.shape[0]
                y = p + y
                y = p * y
                y = p % y
                y = p**y
                y = p // y
                y = pow(p, y)
                y = p / y
                return x + y

        counts = torch._dynamo.testing.CompileCounter()
        mod = MyModule()
        optimized_mod = torch.compile(mod, backend=counts, fullgraph=True)

        x = torch.randn(3)
        ref = mod(x)
        res = optimized_mod(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """9""")

    def test_compare_shapes_eq(self):
        def compare_shapes(a, b, to_list):
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            if x == y:
                return a + 1
            else:
                return a + 2

        # Test both ListVariable and ShapeVariable
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    def test_compare_shapes_tuple_eq(self):
        def compare_shapes(a, b):
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            if x == y:
                return a + 1
            else:
                return a + 2

        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    def test_compare_shapes_tuple_neq(self):
        def compare_shapes(a, b):
            x = tuple(a.unsqueeze(-1).shape)
            y = tuple(b.unsqueeze(-1).shape)
            if x != y:
                return a + 1
            else:
                return a + 2

        torch._dynamo.testing.standard_test(self, lambda a, b: compare_shapes(a, b), 2)

    def test_compare_shapes_neq(self):
        def compare_shapes(a, b, to_list):
            x = list(a.unsqueeze(-1).shape) if to_list else a.shape
            y = list(b.unsqueeze(-1).shape) if to_list else b.shape
            if x != y:
                return a + 1
            else:
                return a + 2

        # Test both ListVariable and ShapeVariable
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=True), 2
        )
        torch._dynamo.testing.standard_test(
            self, lambda a, b: compare_shapes(a, b, to_list=False), 2
        )

    def test_compare_shapes_with_constant(self):
        def compare_shapes(a):
            x = a.shape
            if x[0] != 3:
                return a * 4
            return a * 3

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(compare_shapes)
        opt_fn(torch.randn([3, 4]))
        opt_fn(torch.randn([4, 3]))
        self.assertIn(
            """tensor 'a' size mismatch at index 0. expected 3, actual 4""",
            guard_failure.reason,
        )

    def test_recompile_message_on_parameter(self):
        def guard_failures(failure):
            self.assertIn("torch._dynamo.config.force_parameter_static_shapes", failure)

        @torch._dynamo.optimize("eager", guard_fail_fn=guard_failures)
        def fn(x):
            return torch.cos(x)

        x1 = torch.nn.Parameter(torch.rand(32, 16))
        x2 = torch.nn.Parameter(torch.rand(8, 4, 3, 3))
        x3 = torch.nn.Parameter(torch.rand(8, 8, 3, 3))
        fn(x1)
        fn(x2)
        fn(x3)

    def test_shape_unpack(self):
        def fn(x):
            a, b = x.size()
            return x * b

        i = torch.randn(5, 10)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i)
        self.assertTrue(same(r1, r2))

    def test_check_compiles_when_predicate_true_and_message_has_no_closure(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_has_no_closure(
        self,
    ):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: "Shape is not greater than 3")
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_constant_and_message_None(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_None(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3)
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

    def test_check_compiles_when_predicate_true_and_message_has_global(self):
        global GLOBAL_INT
        GLOBAL_INT = 1

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            torch._check(x.shape[0] > 3, lambda: f"{GLOBAL_INT} is not greater than 3")
            return x + 1

        x = torch.randn(4)
        torch._dynamo.maybe_mark_dynamic(x, 0)

        y = f(x)
        self.assertEqual(y.shape, x.shape)

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

    def test_id_of_container_as_dict_key(self):
        MY_DICT = {"a": 1, "b": 2}

        def fn(x):
            memo = {}
            memo[id(MY_DICT)] = True
            if id(MY_DICT) in memo:
                return x + 1.0
            return x + 2.0

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_id_of_list_as_dict_key(self):
        MY_LIST = [1.0, 2.0]

        def fn(x):
            memo = {}
            memo[id(MY_LIST)] = True
            if id(MY_LIST) in memo:
                return x + 1.0
            return x + 2.0

        x = torch.randn(4)
        correct = fn(x)
        result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(result, correct)

    def test_global_state_guard_serialization(self):
        GlobalStateGuard = torch._C._dynamo.guards.GlobalStateGuard
        guards = GlobalStateGuard()
        serialized_guards = guards.__getstate__()
        json_guards = json.loads(serialized_guards)

        samples = []
        # Test on non autocast state and autocast cache states.
        self.assertIn("autocast_state", json_guards)
        for key, value in json_guards.items():
            if type(value) is int:
                variant = value + 1
            elif type(value) is bool:
                variant = not value
            elif isinstance(value, dict) and key == "autocast_state":
                variant = value.copy()
                variant["cached_enabled"] = not variant["cached_enabled"]
                continue
            else:
                self.fail(f"Unknown global state type {key}: {value}")
            new_dict = json_guards.copy()
            new_dict[key] = variant
            samples.append(new_dict)

        for sample in samples:
            guards.__setstate__(json.dumps(sample))
            self.assertFalse(guards.check())

        guards.__setstate__(json.dumps(json_guards))
        self.assertTrue(guards.check())

        # Test on autocast states.
        def _test_autocast(dtype):
            with torch.autocast("cpu", dtype):
                guards = GlobalStateGuard()
                serialized_guards = guards.__getstate__()
                json_guards = json.loads(serialized_guards)

                for i, enabled in enumerate(json_guards["autocast_state"]["enabled"]):
                    if enabled:
                        self.assertEqual(
                            type(json_guards["autocast_state"]["dtype"][i]), int
                        )
                        json_guards["autocast_state"]["dtype"][i] += 1
                        guards.__setstate__(json.dumps(json_guards))
                        self.assertFalse(guards.check())

        _test_autocast(torch.float16)
        _test_autocast(torch.float32)
        _test_autocast(torch.float64)
        _test_autocast(torch.bfloat16)

    def test_shape_type(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x + (type(x.shape) == torch.Size)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.zeros(())
        self.assertEqual(opt_fn(x), fn(x))

    def test_id_guarded_class(self):
        class MyClass1:
            pass

        class MyClass2:
            pass

        def fn(x, y):
            return x + id(y) // 100000

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(backend=cnts, fullgraph=True)(fn)
        x = torch.randn(3)
        y = MyClass1
        self.assertEqual(fn(x, y), compiled_fn(x, y))
        self.assertEqual(cnts.frame_count, 1)

        # No recompile if still pass in the original class (MyClass1)
        x = torch.randn(3)
        y = MyClass1
        self.assertEqual(fn(x, y), compiled_fn(x, y))
        self.assertEqual(cnts.frame_count, 1)

        # Have to recompile if pass in new class (MyClass2)
        x = torch.randn(3)
        y = MyClass2
        self.assertEqual(fn(x, y), compiled_fn(x, y))
        self.assertEqual(cnts.frame_count, 2)

    def test_id_guarded_object(self):
        class UserDefinedObject:
            @torch.compile(backend="eager")
            def call(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                else:
                    x = torch.mul(x, 0)
                return x

        # Make sure we do recompile when id(self) is executed on
        # different self objects.
        x = torch.ones(2)
        obj1 = UserDefinedObject()
        obj1_id = id(obj1)
        self.assertEqual(obj1.call(x, obj1_id), torch.ones(2))

        obj2 = UserDefinedObject()
        # if we do not install ID_MATCH: ___check_obj_id(L['self'], xxx) this fails.
        self.assertEqual(obj2.call(x, obj1_id), torch.zeros(2))

    def test_id_guarded_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                else:
                    x = torch.mul(x, 0)
                return x

        cnts = torch._dynamo.testing.CompileCounter()

        # Make sure we do recompile when id(self) is executed on
        # different self objects.
        x = torch.ones(2)
        m1 = M()
        m1_id = id(m1)
        opt_m1 = torch.compile(m1, backend=cnts, fullgraph=True)
        self.assertEqual(opt_m1(x, m1_id), torch.ones(2))
        self.assertEqual(opt_m1(x, m1_id), torch.ones(2))

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

        m2 = M()
        opt_m2 = torch.compile(m2, backend=cnts, fullgraph=True)
        # if we do not install ID_MATCH: ___check_obj_id(L['self'], xxx) this fails.
        self.assertEqual(opt_m2(x, m1_id), torch.zeros(2))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_id_tensor(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.y1 = torch.ones(2)
                self.y2 = torch.zeros(2)
                self.ref_y1_id = id(self.y1)
                self.ref_y2_id = id(self.y2)

            def forward(self, x, ref_id):
                if ref_id == id(self.y1):
                    x = torch.mul(x, self.y1)
                else:
                    x = torch.mul(x, self.y2)
                return x

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.ones(2)
        m = M()
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)

        self.assertEqual(opt_m(x, m.ref_y1_id), torch.ones(2))
        self.assertEqual(cnts.frame_count, 1)

        self.assertEqual(opt_m(x, m.ref_y2_id), torch.zeros(2))
        self.assertEqual(cnts.frame_count, 2)

    def test_id_of_nn_module(self):
        class M(torch.nn.Module):
            def forward(self, x, ref_id):
                self_id = id(self)
                if self_id == ref_id:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        m = M().eval()
        data = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        correct_ref_id = id(m)
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, correct_ref_id)
        # Extra op is the recorded equality test (although once
        # the trace is flattened this is dead!)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.op_count, """2""")
        else:
            self.assertExpectedInline(cnts.op_count, """2""")

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        incorrect_ref_id = id(m) + 1
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, incorrect_ref_id)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnts.op_count, """1""")
        else:
            self.assertExpectedInline(cnts.op_count, """1""")

    def test_raise_on_backend_error(self):
        def my_compiler(gm, _):
            raise RuntimeError("duck!")

        @torch.compile(backend=my_compiler)
        def fn(a, b):
            return a + b / (a - b)

        self.assertRaises(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: fn(torch.randn(10), torch.randn(10)),
        )

    @patch.object(torch._dynamo.config, "error_on_nested_fx_trace", False)
    def test_no_error_on_nested_fx_trace(self):
        input = torch.rand(2, 3)

        def f(x):
            x + x

        real = f(input)

        optimized = torch.compile(f, backend="eager")
        self.assertTrue(same(optimized(input), real))

        # should not error
        gm = torch.fx.symbolic_trace(optimized)
        self.assertTrue(same(gm(input), real))

    def test_guard_failure_fn(self):
        def fn(x, y, k):
            x = x + 1
            y = y + 1
            return x * y * k

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        opt_fn(x, y, 3)
        opt_fn(x2, y2, 5)

        if (
            not torch._dynamo.config.specialize_int
            and not torch._dynamo.config.assume_static_by_default
        ):
            # we didn't actually test guard_failure_fn here but whatever,
            # nice to see no guard failure on the test
            self.assertTrue(guard_failure is None)
        else:
            self.assertTrue(guard_failure is not None)

    def test_guard_failure_fn_shape_control(self):
        def fn(x, y):
            if x.shape[0] < 4:
                if y.shape[0] < 3:
                    return x * y
                else:
                    return x + y
            else:
                return -1

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.randn([5, 5])
        y2 = torch.randn([5, 5])

        opt_fn(x, y)
        opt_fn(x2, y2)

        self.assertTrue(guard_failure is not None)
        first_guard_failure = guard_failure[0].partition("\n")[0]
        self.assertIn(
            """tensor 'x' size mismatch at index 0. expected 2, actual 5""",
            first_guard_failure,
        )

    def test_guard_failure_fn2(self):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x * y

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        x2 = torch.tensor([0.5, 0.5, 1.0])
        y2 = torch.tensor([0.5, 0.5, 0.5])

        opt_fn(x, y)
        opt_fn(x2, y2)

        if torch._dynamo.config.assume_static_by_default:
            self.assertIn(
                """tensor 'x' size mismatch at index 0. expected 2, actual 3""",
                guard_failure[0],
            )
        else:
            self.assertTrue(guard_failure is None)

    def test_guard_failure_fn_tensor_iter(self):
        def fn(x):
            for y in x:
                y.add_(1.0)
            return y

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", nopython=True, guard_fail_fn=guard_failures
        )(fn)

        args1 = torch.randn(10, 10)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertTrue(same(out, opt_out))

        args2 = torch.randn(9, 10)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertTrue(same(out, opt_out))

        # guard is expected for both static and dynamic shapes
        self.assertTrue(guard_failure is not None)
        self.assertIn(
            """size mismatch at index 0. expected 10, actual 9""",
            guard_failure[0],
        )

    def test_no_guard_for_unused_sym_node_fstring(self):
        def fn(x):
            f"{x.shape[0]}"
            return x.sin()

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        args1 = torch.randn(10, 11)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertEqual(out, opt_out)

        # We change x.shape[0] to test whether it's guarded
        args2 = torch.randn(9, 11)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertEqual(out, opt_out)
        self.assertEqual(guard_failure, None)

    def test_guard_sym_node_fstring_when_used(self):
        def fn(x):
            # assign fstring to a variable causes the fstring to be used,
            # which realizes the variable tracker.
            f_str = f"{x.shape[0]}"
            return x.sin(), f_str

        guard_failure = None

        def guard_failures(failure):
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        args1 = torch.randn(10, 11)
        out = fn(args1)
        opt_out = opt_fn(args1)
        self.assertEqual(out, opt_out)

        # We change x.shape[0] to test whether it's guarded
        args2 = torch.randn(9, 11)
        out = fn(args2)
        opt_out = opt_fn(args2)
        self.assertEqual(out, opt_out)
        self.assertTrue(guard_failure is not None)
        self.assertIn("""tensor 'x' size mismatch at index 0""", guard_failure[0])

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_argwhere_with_dynamic_shapes(self):
        def fn(
            tensor: torch.Tensor,
            mapping: torch.Tensor,
        ) -> torch.Tensor:
            xx, yy = torch.meshgrid(mapping, tensor, indexing="ij")
            indices = torch.argwhere(xx == yy)

            mapped_values = torch.zeros_like(tensor)
            mapped_values[indices[:, 1]] = indices[:, 0]

            return mapped_values

        tensor = torch.tensor([1, 2, 3, 5, 6, 7])
        mapping = torch.tensor([0, 3, 4, 5, 7])
        opt = torch.compile(fn, fullgraph=True, backend="eager")
        self.assertEqual(fn(tensor, mapping), opt(tensor, mapping))

    def test_precompile_entry_hit(self):
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            return x + 1

        def injected(x):
            return x + 42

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(),
            injected.__code__,
        )
        self.assertEqual(compiled_fn(*args), injected(*args))
        _reset_precompile_entries(fn.__code__)

        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_entry_miss(self):
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        def fn(x):
            return x + 1

        guard_manager = torch._dynamo.guards.RootGuardManager()
        guard_manager.add_lambda_guard(lambda L: isinstance(L["x"], int), [], None)

        def injected(x):
            return x + 42

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager),
            injected.__code__,
        )
        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_entries(self):
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            return x + 1

        guard_manager_bool = torch._dynamo.guards.RootGuardManager()
        guard_manager_bool.add_lambda_guard(
            lambda L: isinstance(L["x"], bool), [], None
        )

        def injected_bool(x: bool):
            return x + 102

        guard_manager_int = torch._dynamo.guards.RootGuardManager()
        guard_manager_int.add_lambda_guard(lambda L: isinstance(L["x"], int), [], None)

        def injected_int(x: int):
            return x + 42

        guard_manager_tensor = torch._dynamo.guards.RootGuardManager()
        guard_manager_tensor.add_lambda_guard(
            lambda L: isinstance(L["x"], torch.Tensor), [], None
        )

        def injected_tensor(x: torch.Tensor):
            return x + 100

        guard_manager_str = torch._dynamo.guards.RootGuardManager()
        guard_manager_str.add_lambda_guard(lambda L: isinstance(L["x"], str), [], None)

        def injected_str(x: str):
            return x + "1"

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_bool),
            injected_bool.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_int),
            injected_int.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_tensor),
            injected_tensor.__code__,
        )

        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_str),
            injected_str.__code__,
        )

        self.assertEqual(compiled_fn(*args), injected_tensor(*args))
        self.assertEqual(compiled_fn(True), injected_bool(True))
        self.assertEqual(compiled_fn(10), injected_int(10))
        self.assertEqual(compiled_fn("10"), injected_str("10"))
        _reset_precompile_entries(fn.__code__)

        self.assertEqual(compiled_fn(*args), fn(*args))

    def test_precompile_fail_on_recompile(self):
        from torch._C._dynamo.eval_frame import _load_precompile_entry

        @torch.compiler.disable
        def graph(x, s0):
            return x + s0

        def fn(x):
            nonlocal graph  # Forcing fn and injected to have the same closure.
            return x - 1

        def injected(x):
            s0 = call_size(x, 0)
            return graph(x, s0)

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(),
            injected.__code__,
        )
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled_fn(*args), injected(*args))

    def test_shape_and_tuple_equality(self):
        def fn(x, y, t):
            z = x * y
            if x.size() == t:
                return z.cos()
            return z.sin()

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.randn([4, 4]), torch.randn([4, 4]), (4, 4)
        )

    def test_float_speculation_log_divergence(self):
        def fn(x, y, z):
            a = F.interpolate(x, scale_factor=z, mode="bilinear", align_corners=False)
            b = F.interpolate(y, scale_factor=z, mode="bilinear", align_corners=False)
            return a * b

        cnt = CompileCounterWithBackend("inductor")
        fn_opt = torch.compile(fn, backend=cnt)
        y = torch.randn(3, 3, 3, 4)

        self.assertEqual(fn(y, y, 1.0), fn_opt(y, y, 1.0))
        self.assertEqual(fn(y, y, 2.0), fn_opt(y, y, 2.0))

    def test_raise_guard_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_raise_guard_indirect_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def dyn_fn(x):
            if x.shape[0] > 3:
                return x.cos()
            if x.shape[0] < 3:
                return x * 2
            return x.sin()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(dyn_fn, backend="eager")(y)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_empty_tensor(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            n = x.item()
            return torch.empty((n - 1) // 2)

        self.assertEqual(fn(torch.tensor([4])).size(0), 1)
        self.assertEqual(fn(torch.tensor([1])).size(0), 0)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_and_terms(self):
        from torch.fx.experimental.symbolic_shapes import sym_and

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def fn(xs):
            u0, u1 = xs.tolist()
            torch._check(sym_and(u0 >= 3, u0 <= 10, u1 >= 2))

            # test individual checks
            n = 0
            if u0 >= 3:
                n += 1
            if u0 <= 11:
                n += 1
            if u1 >= 1:
                n += 1
            return u0 + u1 + n

        fn(torch.tensor([5, 6]))
        fn(torch.tensor([8, 7]))
        with self.assertRaises(RuntimeError):
            fn(torch.tensor([9, 0]))

    def test_unbacked_2d_expand(self):
        @torch.compile(fullgraph=True, dynamic=True, backend="inductor")
        def func(a, b):
            a.expand(b.shape)
            return a * 10

        a = torch.rand(1, 1)
        b = torch.rand(1, 1)

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)
        func(a, b)
        func(torch.rand(4, 5), torch.rand(4, 5))
        # This does not raise an error right now because of a recompilation.
        # https://github.com/pytorch/pytorch/issues/163785
        # with self.assertRaises(AssertionError):
        #     func(torch.rand(1, 1), torch.rand(2, 1))
        func(torch.rand(1, 1), torch.rand(2, 1))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_sym_constrain_range_on_replaced_unbacked_symbol(self):
        # Tests the following case:
        # Deferred runtime asserts adds sym_constrain_range(u0).
        # However, u0 is replaced with s0 + s1.
        # So, now we have sym_constrain_range(s0 + s1).
        def fn(x, y, z):
            z += 7  # to avoid creating unspecified symbol instead of unbacked symbol
            u0 = z.item()
            s0 = x.size(0)
            s1 = y.size(0)
            torch._check(s0 < 100)
            torch._check(s1 < 100)
            torch._check(u0 == s0 + s1)
            return x, y, z

        inputs = (
            x := torch.randn(16, 10),
            y := torch.randn(16, 10),
            torch.tensor(32 - 7),
        )
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        opt = torch.compile(fn, fullgraph=True)
        opt(*inputs)
        with self.assertRaises(RuntimeError):
            inputs = (
                x := torch.randn(16, 10),
                y := torch.randn(16, 10),
                torch.tensor(32),
            )
            opt(*inputs)

    def test_sym_max_creates_graph_node(self):
        # Test that sym_max creates a graph node and works correctly
        from torch._dynamo.testing import EagerAndRecordGraphs

        def fn(x):
            max_dim = torch.sym_max(x.size(0), x.size(1))
            return x.sum() + max_dim

        backend = EagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)

        x = torch.randn(4, 8)
        result = compiled_fn(x)
        expected = x.sum() + 8
        self.assertEqual(result, expected)

        # Verify sym_max appears in graph
        self.assertGreater(len(backend.graphs), 0)
        graph = backend.graphs[0]
        sym_max_nodes = [n for n in graph.graph.nodes if n.target is torch.sym_max]
        self.assertEqual(len(sym_max_nodes), 1, "sym_max should be in the graph")

    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_mark_dynamic_with_ranges(self):
        y = torch.randn([8, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.mark_dynamic(y, 0, min=2, max=5)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_mark_static(self):
        counter = CompileCounter()

        def my_dyn_fn(x):
            return x.cos()

        y = torch.randn([3])
        torch._dynamo.mark_static(y, 0)
        torch.compile(my_dyn_fn, backend=counter)(y)

        z = torch.randn([4])
        torch.compile(my_dyn_fn, backend=counter)(z)

        self.assertEqual(counter.frame_count, 2)

    def test_no_raise_guard_partial_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch.compile(my_dyn_fn, backend="eager")(y)

    def test_no_raise_guard_partial_constraint_across_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] > 2:
                return z.cos()

            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        torch.compile(my_dyn_fn, backend="eager")(y, y)

    @unittest.expectedFailure
    def test_raise_guard_partial_constraint_across_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            torch._dynamo.graph_break()
            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch.compile(my_dyn_fn, backend="eager")(y, y)
        torch._dynamo.mark_dynamic(y, 0)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            Exception,
        ):
            torch.compile(my_dyn_fn, backend="eager")(y, y)

    def test_raise_guard_partial_constraint_no_graph_break(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x, y):
            z = x * y

            if z.shape[0] == 3:
                return z.cos()

            return x.cos()

        torch._dynamo.mark_dynamic(y, 0)
        with self.assertRaises(ConstraintViolationError):
            torch.compile(my_dyn_fn, backend="eager")(y, y)

    @torch._dynamo.config.patch(force_parameter_static_shapes=True)
    @torch._dynamo.config.patch(force_nn_module_property_static_shapes=True)
    @torch.compiler.config.patch(
        dynamic_sources="L['x'],L['y'],L['self']._modules['y'].x,L['self']._modules['y']._modules['c']._parameters['weight'],L['self']._modules['y']._modules['c']._parameters['bias']"
    )
    def test_dynamic_sources_force_parameter_static_shapes_and_property_static_shapes_override(
        self,
    ):
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        class Y(torch.nn.Module):
            def __init__(self, n_input, n_output):
                super().__init__()
                self.c = torch.nn.Linear(n_input, n_output)
                self.x = n_input

            def forward(self, x):
                return self.c(x) * self.x

        class M(torch.nn.Module):
            def __init__(self, n_input, n_output):
                self.n_input = n_input
                self.n_output = n_output
                super().__init__()
                self.y = Y(n_input, n_output)

            @torch.compile(backend=counter)
            def forward(self, x, y):
                return self.y(x) * y

        model = M(3210, 30)
        model(torch.randn(1, 3210), 2)
        model = M(3211, 30)
        model(torch.randn(1, 3211), 3)
        model = M(3212, 30)
        model(torch.randn(1, 3212), 4)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['self']._modules['inner'].x")
    def test_dynamic_sources_precedence_over_int_specialization(self):
        builder._DYNAMIC_SOURCES = None

        counter = CompileCounter()

        class Model(torch.nn.Module):
            def __init__(self, x) -> None:
                super().__init__()
                self.inner = torch.nn.Linear(10, 10)
                # attach attribute to builtin nn module.
                self.inner.x = x

            @torch.compile(fullgraph=True, backend=counter)
            def forward(self, a):
                return a * self.inner.x

        m1 = Model(50)
        m2 = Model(60)
        with fresh_cache():
            m1(torch.rand(1, 2, 3))
            m2(torch.rand(1, 2, 3))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_int(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return torch.randn(5) * x

        fn(1)
        fn(2)
        fn(3)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_tensor(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return x * x

        fn(torch.randn(2))
        fn(torch.randn(3))
        fn(torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']")
    def test_unbacked_sources_tensor(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return x * x

        fn(torch.randn(0))
        fn(torch.randn(1))
        fn(torch.randn(2))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(unbacked_sources="L['x']")
    def test_unbacked_sources_scalar(self):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(x):
            return x * x

        fn(0)
        fn(1)
        fn(2)

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L['x']")
    def test_dynamic_sources_graph_break(self):
        counter = CompileCounter()

        def foo(x):
            return x * x

        @torch.compile(backend=counter)
        def fn(x):
            x = x * x
            torch._dynamo.graph_break()
            return foo(x)

        fn(torch.randn(2))
        fn(torch.randn(3))
        fn(torch.randn(4))

        # 2 since graph break produces 2 graphs. NB: there are no recompiles
        self.assertEqual(counter.frame_count, 2)

    @torch.compiler.config.patch(dynamic_sources="L['x'], L['y']")
    def test_dynamic_sources_dynamic_override(self):
        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x, y):
            return x * y

        fn(2, torch.randn(2))
        fn(3, torch.randn(3))
        fn(4, torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    @torch.compiler.config.patch(dynamic_sources="L\\['x.*'\\], L\\['y.*'\\]")
    def test_dynamic_sources_dynamic_override_regex(self):
        counter = CompileCounter()

        @torch.compile(dynamic=False, backend=counter)
        def fn(x1, y1):
            return x1 * y1

        fn(2, torch.randn(2))
        fn(3, torch.randn(3))
        fn(4, torch.randn(4))

        self.assertEqual(counter.frame_count, 1)

    def test_cannot_trace_mark_dynamic(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            torch._dynamo.mark_dynamic(x, 0)
            return x * x

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            torch.compile(my_dyn_fn, backend="eager")(y)

    def test_cannot_trace_mark_dynamic_safe_unreached(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x
            print("Running", torch._dynamo.mark_dynamic(x, 0))
            return x * x

        torch.compile(my_dyn_fn, backend="eager")(y)

    def test_anomaly_aot_autograd(self):
        def fail():
            raise AssertionError("fail")

        @allow_in_graph
        def h(a):
            r = a.sum()
            # Trigger an exception in backwards
            r.register_hook(lambda x: fail())
            return r

        @torch.compile(backend="aot_eager")
        def f(a):
            return h(a)

        with (
            warnings.catch_warnings(record=True) as w,
            self.assertRaises(torch._dynamo.exc.BackendCompilerFailed),
        ):
            f(torch.randn(2, 2, requires_grad=True))

        # Suppress unrelated pkg_resources warnings
        self.assertIn("forward call that caused the error", str(w[-1].message))

    def test_py_guards_mark_dynamic(self):
        def my_dyn_fn(a):
            if a.shape[0] > 2:
                return a.cos()
            return a.sin()

        counter = CompileCounter()

        # Run with dynamic
        x0 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x0, 0)
        torch.compile(my_dyn_fn, backend=counter)(x0)
        self.assertEqual(counter.frame_count, 1)

        # Run without dynamic, no recompile
        x = torch.randn([3, 3, 3])
        torch.compile(my_dyn_fn, backend=counter)(x)
        self.assertEqual(counter.frame_count, 1)

        # Mark a new dim, 1, as dynamic
        x1 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x1, 1)
        torch.compile(my_dyn_fn, backend=counter)(x1)
        # Recompile triggered because we marked a new dym as dynamic
        self.assertEqual(counter.frame_count, 2)

        # Reset
        torch._dynamo.reset()
        # Reset counter
        counter = CompileCounter()

        # Run with dynamic 1
        torch.compile(my_dyn_fn, backend=counter)(x1)
        self.assertEqual(counter.frame_count, 1)

        # Run with dynamic 0, not subset
        torch.compile(my_dyn_fn, backend=counter)(x0)
        self.assertEqual(counter.frame_count, 2)

        # Run with dynamic 0, 1, 2, not subset
        x012 = torch.randn([3, 3, 3])
        torch._dynamo.mark_dynamic(x012, 0)
        torch._dynamo.mark_dynamic(x012, 1)
        torch._dynamo.mark_dynamic(x012, 2)
        torch.compile(my_dyn_fn, backend=counter)(x012)
        self.assertEqual(counter.frame_count, 3)

    def test_recompile_on_global_state_change(self):
        last_state = []
        cnt = 0

        def my_compiler(gm, _):
            nonlocal cnt
            cnt += 1
            state = read_state()

            def inner(*args):
                last_state[:] = state
                return gm(*args)

            return inner

        def read_state():
            return [
                torch.is_grad_enabled(),
                torch.are_deterministic_algorithms_enabled(),
                torch._C._get_cublas_allow_tf32(),
            ]

        def write_state(state):
            torch.set_grad_enabled(state[0])
            torch.use_deterministic_algorithms(state[1])
            torch._C._set_cublas_allow_tf32(state[2])

        @torch.compile(backend=my_compiler)
        def fn(x):
            return x + 1

        initial_state = read_state()
        y = torch.randn(10)
        try:
            for round in range(3):
                for i in range(len(initial_state)):
                    new_state = [False] * len(initial_state)
                    new_state[i] = True
                    write_state(new_state)
                    if read_state() != new_state:
                        raise AssertionError(f"Expected read_state() == {new_state}")
                    last_state.clear()
                    fn(y)
                    if last_state != new_state:
                        raise AssertionError(f"Expected last_state == {new_state}")
                    if round == 0:
                        if cnt != i + 1:
                            raise AssertionError(f"Expected cnt == {i + 1}, got {cnt}")
                    else:
                        if cnt != len(initial_state):
                            raise AssertionError(
                                f"Expected cnt == {len(initial_state)}, got {cnt}"
                            )
        finally:
            write_state(initial_state)

    def test_deterministic_algorithms_mutated(self):
        prior = torch.are_deterministic_algorithms_enabled()
        prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        value = None
        warn_only = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            nonlocal warn_only
            value = torch.are_deterministic_algorithms_enabled()
            warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            check_state()
            assert torch.are_deterministic_algorithms_enabled() is True  # noqa: S101
            torch.use_deterministic_algorithms(False, warn_only=False)
            return x + 1

        def run_fn():
            torch.use_deterministic_algorithms(True, warn_only=True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if warn_only is not True:
                raise AssertionError(f"Expected warn_only is True, got {warn_only}")
            if torch.are_deterministic_algorithms_enabled() is not False:
                raise AssertionError(
                    "Expected deterministic algorithms disabled after fn()"
                )
            if torch.is_deterministic_algorithms_warn_only_enabled() is not False:
                raise AssertionError("Expected warn_only disabled after fn()")

        try:
            run_fn()
            value, warn_only = None, None
            run_fn()
            if cnt.frame_count != 1:
                raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")
        finally:
            torch.use_deterministic_algorithms(prior, warn_only=prior_warn_only)

    def test_recompile_on_disable_1(self):
        # fix https://github.com/pytorch/pytorch/issues/157399
        @torch.compile(backend="eager")
        def fn(x):
            @torch._dynamo.disable
            def inner(x):
                return x + 10

            return inner(x) + 1

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            try:
                for i in range(5):
                    fn(torch.rand(2, 3))
            except torch._dynamo.exc.RecompileError as e:
                self.fail("RecompileError raised unexpectedly: " + str(e))

    def test_recompile_on_disable_2(self):
        def outer(x, cond):
            @torch._dynamo.disable()
            def fn0(y):
                return y + 1

            @torch._dynamo.disable()
            def fn1(y):
                return y + 2

            if cond:
                f = fn0
            else:
                f = fn1

            torch._dynamo.graph_break()
            # there will be a resume function here
            return f(x)

    def test_guards_strip_function_call(self):
        from torch._dynamo.guards import strip_function_call

        test_case = [
            ("___odict_getitem(a, 1)", "a"),
            ("a.layers[slice(2)][0]._xyz", "a"),
            ("getattr(a.layers[slice(2)][0]._abc, '0')", "a"),
            ("getattr(getattr(a.x[3], '0'), '3')", "a"),
            ("a.layers[slice(None, -1, None)][0]._xyz", "a"),
            ("a.layers[func('offset', -1, None)][0]._xyz", "a"),
        ]
        # strip_function_call should extract the object from the string.
        for name, expect_obj in test_case:
            self.assertEqual(strip_function_call(name), expect_obj)

    def test_recursion_depth_guards(self):
        @torch.compile(dynamic=True, backend="eager")
        def foo(*args, **kwargs):
            if sum(args) == 0:
                return 0
            return 1

        args = list(range(2000))
        foo(*args)

    @dataclasses.dataclass
    class CSETestCase:
        expr: str
        preface: typing.List[str] = dataclasses.field(default_factory=list)
        expected: typing.Optional[str] = None

    def test_guards_cse_pass_single(self):
        from torch._dynamo.guards import PyExprCSEPass

        testcase = self.CSETestCase
        testcases = [
            # Nothing gets CSE-d, since the only repeated sub-expression is 'x'.
            # i.e. not a node type we are interested on.
            testcase(expr="x[0].a"),
            testcase(expr="x[1].a"),
            testcase(expr="x[2].a"),
            # 'a.b.c' gets CSE-d, since it's a sub-expression used more than 'PyExprCSEPass.USE_THRESHOLD'.
            testcase(
                expr="a.b.c[0].d.e",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e",
            ),
            testcase(expr="a.b.c[1].d.e", expected="_var1[1].d.e"),
            testcase(expr="a.b.c[2].d.e", expected="_var1[2].d.e"),
            # 'm.n[0]' gets CSE-d, since it is a sub-expression used more than 'PyExprCSEPass.USE_THRESHOLD'.
            testcase(
                expr="f(m.n[0], '0').x.y.z",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z",
            ),
            testcase(expr="f(m.n[0], '1').x.y.z", expected="f(_var3, '1').x.y.z"),
            testcase(expr="f(m.n[0], '2').x.y.z", expected="f(_var3, '2').x.y.z"),
            # The whole expression gets CSE-d, as well as all of its sub-expressions.
            testcase(
                expr="self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6",
            ),
            testcase(expr="self.g(a, b).k", expected="_var6"),
            testcase(expr="self.g(a, b).k", expected="_var6"),
        ]
        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            self.assertEqual(preface, t.preface)
            expected = t.expected if t.expected is not None else t.expr
            self.assertEqual(expr, expected)

    def test_guards_cse_pass_multiple(self):
        from torch._dynamo.guards import PyExprCSEPass

        testcase = self.CSETestCase
        testcases = [
            testcase(
                expr="x[0].a < x[1].a * (3 - x[2].a)",
                expected="x[0].a < x[1].a * (3 - x[2].a)",
            ),
            testcase(
                expr="a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
                preface=["_var0 = a.b", "_var1 = _var0.c"],
                expected="_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0",
            ),
            testcase(
                expr="f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
                preface=["_var2 = m.n", "_var3 = _var2[0]"],
                expected="f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512",
            ),
            testcase(
                expr="self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
                preface=["_var4 = self.g", "_var5 = _var4(a, b)", "_var6 = _var5.k"],
                expected="_var6 + (1 - _var6) <= m[0].a + _var6",
            ),
        ]

        csepass = PyExprCSEPass()
        csepass.count([t.expr for t in testcases])

        for t in testcases:
            preface, expr = csepass.replace(t.expr)
            self.assertEqual(preface, t.preface)
            expected = t.expected
            expected = expected if expected is not None else t.expr
            self.assertEqual(expr, expected)

    def test_guard_function_builder_with_cse(self):
        from torch._dynamo.guards import build_guard_function

        exprs = [
            "x[0].a < x[1].a * (3 - x[2].a)",
            "a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0",
            "f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512",
            "self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k",
        ]

        _, pycode = build_guard_function(exprs, "")
        expected = """\
def ___make_guard_fn():
    def guard(L):
        if not (x[0].a < x[1].a * (3 - x[2].a)):
            return False
        _var0 = a.b
        _var1 = _var0.c
        if not (_var1[0].d.e + _var1[1].d.e * _var1[2].d.e > 0):
            return False
        _var2 = m.n
        _var3 = _var2[0]
        if not (f(_var3, '0').x.y.z * f(_var3, '1').x.y.z * f(_var3, '2').x.y.z < 512):
            return False
        _var4 = self.g
        _var5 = _var4(a, b)
        _var6 = _var5.k
        if not (_var6 + (1 - _var6) <= m[0].a + _var6):
            return False
        return True
    return guard
"""

        self.assertEqual(expected, pycode)

    def test_dynamic_one_hot(self):
        def fn(x):
            x = x + 1
            # graph break from data-dependent output shape
            x = torch.nn.functional.one_hot(x)
            x = x + 1
            return x

        inp = torch.arange(20) % 4
        counter = CompileCounter()
        real_out = fn(inp)
        comp_out = torch.compile(fn, backend=counter)(inp)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 2)

    def test_any_all_symnode(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x):
            t = x.size(0) >= 10
            f = x.size(0) >= 100
            if any([]) or any([f]) or any([f, f]):
                return x - 1
            if all([f]) or all([t, f]) or all([f, t]) or all([f, f]):
                return x - 2
            if not (all([]) and all([t]) and all([t, t])):
                return x - 3
            if not (any([t]) and any([t, f]) and any([f, t])):
                return x - 4
            return x + 1

        y1 = torch.randn(16)
        y2 = torch.randn(18)
        self.assertEqual(fn(y1), y1 + 1)
        self.assertEqual(fn(y2), y2 + 1)
        self.assertEqual(cnt.frame_count, 1)
        y3 = torch.randn(5)
        self.assertEqual(fn(y3), y3 - 3)
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_unbacked_symint_split(self):
        @torch.compile(backend="eager")
        def f(lengths, values):
            sizes = lengths.tolist()
            return torch.split(values, sizes)

        f(torch.tensor([2, 3, 4]), torch.randn(9))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_runtime_assert_replacement(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            z = y.item()
            torch._check(z == 3)
            return x + z

        fn(torch.randn(4), torch.tensor([3]))
        self.assertRaises(RuntimeError, lambda: fn(torch.randn(4), torch.tensor([4])))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cat_unbacked(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            z = y.item()
            return torch.cat([x, torch.ones(z)])

        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([0]))
        )
        self.assertRaises(
            RuntimeError, lambda: fn(torch.randn(2, 3), torch.tensor([1]))
        )

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_aot_autograd_propagate_unbacked_symints_shape(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return torch.nonzero(x)

        f(torch.tensor([1, 0, 3, 2, 0]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked(self):
        class SillyCat(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x0, x1, i):
                ctx.save_for_backward(i)
                return torch.cat([x0, x1])

            @staticmethod
            def backward(ctx, grad_out):
                (i,) = ctx.saved_tensors
                i0, i1 = i.tolist()
                g_x0, g_x1 = grad_out.split([i0, i1])
                return g_x0, g_x1, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            i0, i1 = i.tolist()
            x0, x1 = x.split([i0, i1])
            return SillyCat.apply(x0, x1, i)

        f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_validate_outputs_unbacked_by_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_validate_outputs_unbacked",
                "(Tensor a, Tensor b) -> (Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo_validate_outputs_unbacked", "cpu", lib=lib)
            @torch.library.register_fake(
                "mylib::foo_validate_outputs_unbacked", lib=lib
            )
            def foo_impl(x, y):
                return torch.cat([x, y])

            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x, i):
                i0, i1 = i.tolist()
                x0, x1 = x.split([i0, i1])
                return torch.ops.mylib.foo_validate_outputs_unbacked(x0, x1)

            f(torch.randn(9, requires_grad=True), torch.tensor([3, 6]))

    def test_shape_env_no_recording(self):
        main = ShapeEnv(should_record_events=False)

        # The main ShapeEnv should have no event recorded.
        self.assertEqual(len(main.events), 0)

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] == 3 (call evaluate_expr)
        #   - +1 guard entry
        #   - +1 replacement entry
        size = r[0]
        bool(size[0] == 3)

        # The main ShapeEnv should remain with no event recorded.
        self.assertEqual(len(main.events), 0)

        if torch.fx.experimental.validator.translation_validation_enabled():
            from torch.fx.experimental.symbolic_shapes import (
                CURRENT_NODE_KEY,
                SHAPEENV_EVENT_KEY,
            )

            # Check that we don't store any recording metadata on nodes
            # from the symbolic shape FX graph.
            for n in main.graph.nodes:
                self.assertFalse(SHAPEENV_EVENT_KEY in n.meta)
                self.assertFalse(CURRENT_NODE_KEY in n.meta)

    def _replay_and_check(self, shape_env: ShapeEnv):
        if shape_env.should_record_events:
            replayed = replay_shape_env_events(shape_env.events)
            shape_env.check_equal(replayed)

    def test_shape_env_equal_empty(self):
        main, other = ShapeEnv(), ShapeEnv()
        main.check_equal(other)
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_constructor(self):
        main, other = ShapeEnv(allow_scalar_outputs=False), ShapeEnv()
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> settings: values don't match.
  >  Left: ShapeEnvSettings(allow_scalar_outputs=False, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
  > Right: ShapeEnvSettings(allow_scalar_outputs=True, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_create_symbolic_sizes_strides_storage_offset(self):
        main, other = ShapeEnv(), ShapeEnv()
        main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> backed_var_to_val: values don't match.
  >  Left: {s44: 2, s93: 3}
  > Right: {}
==> name_to_node: values don't match.
  >  Left: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {}
==> source_to_symbol: values don't match.
  >  Left: {x.size()[0]: x.size()[0], x.size()[1]: x.size()[1], x.storage_offset(): x.storage_offset(), x.stride()[0]: x.stride()[0], x.stride()[1]: x.stride()[1]}
  > Right: {}
==> source_to_var: values don't match.
  >  Left: {x.size()[0]: s93, x.size()[1]: s44}
  > Right: {}
==> unique_ids: values don't match.
  >  Left: {44, 93}
  > Right: {}
==> val_to_var: values don't match.
  >  Left: {2: s44, 3: s93}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
  > Right: {}
==> var_to_sources: values don't match.
  >  Left: {s44: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=1)], s93: [TensorPropertySource(base=ConstantSource(source_name='x'), prop=<TensorProperty.SIZE: 0>, idx=0)]}
  > Right: {}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_unbacked(self):
        main, other = ShapeEnv(), ShapeEnv()
        main.create_unbacked_symint()
        main.create_unbacked_symfloat()
        main.create_unbacked_symbool()
        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> name_to_node: values don't match.
  >  Left: {u0, u1, zuf0}
  > Right: {}
==> unbacked_symfloat_counter: values don't match.
  >  Left: 1
  > Right: 0
==> unbacked_symint_counter: values don't match.
  >  Left: 2
  > Right: 0
==> var_to_range: values don't match.
  >  Left: {u0: VR[-int_oo, int_oo], u1: VR[0, 1], zuf0: VR[-oo, oo]}
  > Right: {}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_divisible(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] % 3 == 0 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 divisible entry
        size = r[0]
        bool(size[0] % 3 == 0)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {(Mod(s93, 3)) < 0: False, (Mod(s93, 3)) <= 0: True, 0 < (Mod(s93, 3)): False, 0 <= (Mod(s93, 3)): True, Eq(0, Mod(s93, 3)): True, Eq(Mod(s93, 3), 0): True, Ne(0, Mod(s93, 3)): False, Ne(Mod(s93, 3), 0): False}
  > Right: {}
==> divisible: values don't match.
  >  Left: {Mod(s93, 3)}
  > Right: {}
==> guards: values don't match.
  >  Left: [Eq(Mod(s93, 3), 0)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_replacement(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] == 3 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 replacement entry
        size = r[0]
        bool(size[0] == 3)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {False: False, True: True}
  > Right: {}
==> guards: values don't match.
  >  Left: [Eq(s93, 3)]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, eq, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> replacements: values don't match.
  >  Left: {s93: 3}
  > Right: {}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[3, 3]}
  > Right: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_evaluate_expr_refinement(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_symbolic_sizes_strides_storage_offset on both of them.
        r = main.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )
        other.create_symbolic_sizes_strides_storage_offset(
            torch.randn(3, 2), ConstantSource("x")
        )

        # Create a guard: size[0] >= 3 (only in the main ShapeEnv)
        #   - +1 guard entry
        #   - +1 var_to_guard entry
        #   - Change: var_to_range
        size = r[0]
        bool(size[0] >= 3)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {3 <= s93: True, s93 < 3: False}
  > Right: {}
==> guards: values don't match.
  >  Left: [s93 >= 3]
  > Right: []
==> name_to_node: values don't match.
  >  Left: {_assert, ge, x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
  > Right: {x_size_0_, x_size_1_, x_storage_offset, x_stride_0_, x_stride_1_}
==> var_to_range: values don't match.
  >  Left: {s44: VR[2, int_oo], s93: VR[3, int_oo]}
  > Right: {s44: VR[2, int_oo], s93: VR[2, int_oo]}
""",
        )
        self._replay_and_check(main)

    @onlyIfTranslationValidation
    def test_shape_env_equal_runtime_assert(self):
        main, other = ShapeEnv(), ShapeEnv()

        # Call create_unbacked_symint on both of them.
        r = main.create_unbacked_symint()
        other.create_unbacked_symint()

        # Create a runtime assert: r % 3 == 0 (only in the main ShapeEnv)
        #   - +1 deferred_runtime_asserts entry
        #   - Change: num_deferred_runtime_asserts
        expect_true(r % 3 == 0)

        self.assertExpectedRaisesInline(
            NotEqualError,
            lambda: main.check_equal(other),
            """\
ShapeEnv not equal: field values don't match:

==> axioms: values don't match.
  >  Left: {(PythonMod(u0, 3)) < 0: False, (PythonMod(u0, 3)) <= 0: True, 0 < (PythonMod(u0, 3)): False, 0 <= (PythonMod(u0, 3)): True, Eq(0, PythonMod(u0, 3)): True, Eq(PythonMod(u0, 3), 0): True, Ne(0, PythonMod(u0, 3)): False, Ne(PythonMod(u0, 3), 0): False}
  > Right: {}
==> deferred_runtime_asserts: values don't match.
  >  Left: {u0: [Eq(PythonMod(u0, 3), 0)]}
  > Right: {}
==> name_to_node: values don't match.
  >  Left: {_assert, eq, mod, u0}
  > Right: {u0}
==> num_deferred_runtime_asserts: values don't match.
  >  Left: 1
  > Right: 0
""",
        )
        self._replay_and_check(main)

    def test_shape_env_recorded_function_fallback(self):
        # Make sure the record/replay mechanism for ShapeEnv will fallback
        # if no ShapeEnv instance is found.
        constrain_range(5, min=2, max=10)
        constrain_unify(5, 5)

        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: _constrain_range_for_size(5, min=2, max=10),
            """can only constrain range for SymInt""",
        )

    def test_compilation_metrics_size_limit(self):
        def fn1(x):
            return x.relu()

        def fn2(x):
            return x.cos()

        def fn3(x):
            return x.sin()

        def fn4(x):
            return x.exp()

        import contextlib

        @contextlib.contextmanager
        def metrics_limit_ctx():
            try:
                torch._dynamo.utils.set_compilation_metrics_limit(3)
                yield
            finally:
                torch._dynamo.utils.set_compilation_metrics_limit(
                    torch._dynamo.utils.DEFAULT_COMPILATION_METRICS_LIMIT
                )

        x = torch.rand((4, 4))
        torch._dynamo.reset()
        torch.compile(fn1, backend="eager")(x)
        torch.compile(fn2, backend="eager")(x)
        torch.compile(fn3, backend="eager")(x)
        torch.compile(fn4, backend="eager")(x)

        with metrics_limit_ctx():
            torch._dynamo.utils.clear_compilation_metrics()
            torch._dynamo.reset()
            self.assertEqual(0, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn1, backend="eager")(x)
            self.assertEqual(1, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn2, backend="eager")(x)
            self.assertEqual(2, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn3, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))
            torch.compile(fn4, backend="eager")(x)
            self.assertEqual(3, len(torch._dynamo.utils.get_compilation_metrics()))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_check_simplification(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            u0, u1 = x.tolist()
            torch._check((2 * u0) // (u0 + u1) != 0)
            if (2 * u0) // (u0 + u1) == 0:
                return torch.tensor(True)
            else:
                return torch.tensor(False)

        fn(torch.tensor([3, 3]))

    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_mark_unbacked_strict(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            return torch.mul(x, y)

        x = torch.ones(5, 5)
        torch._dynamo.decorators.mark_unbacked(x, 0, strict=True)
        torch._dynamo.decorators.mark_unbacked(x, 1, strict=True)
        y = torch.randn(5, 5)

        with self.assertRaisesRegex(RuntimeError, "specialized"):
            fn(x, y)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_infer_unbacked_size_gt_zero(self):
        # This code, in fact, does NOT work in eager
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = torch.zeros(x.item())
            if y.size(0) < 0:
                assert False  # noqa: B011, S101
            return y

        self.assertEqual(fn(torch.tensor([0])), torch.zeros(0))

    @torch.fx.experimental._config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_strict_mode(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            if x.shape[0] == 5:
                return torch.randn(5)
            return torch.mul(x, y)

        x = torch.ones(5, 5)
        torch._dynamo.decorators.mark_unbacked(x, 0)
        torch._dynamo.decorators.mark_unbacked(x, 1)
        y = torch.randn(5, 5)
        with self.assertRaisesRegex(
            RuntimeError, "Could not guard on data-dependent expression"
        ):
            fn(x, y)

    def test_guard_size_oblivious_backed(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            y = x.size(0)
            # This doesn't actually do anything
            if guard_size_oblivious(y == 0):
                return torch.randn(1)
            else:
                return torch.randn(2)

        # Should not fail in either case
        self.assertEqual(f(torch.randn(0)).shape, (1,))
        self.assertEqual(f(torch.randn(2)).shape, (2,))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_hasattr_nn_module_guard(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.nn.Linear(3, 3)

            def forward(self, x):
                if hasattr(self, "a"):
                    return self.a(x)
                else:
                    return x

        m = M()
        x = torch.randn(3, 3)
        ref = m(x)

        opt_m = torch.compile(backend="eager")(m)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_compare_tensor_with_none(self):
        @torch.compile(backend="eager")
        def f(x):
            return torch.tensor(x == None)

        res = f(torch.tensor(1))
        self.assertEqual(torch.tensor(False), res)

    def test_guard_filter_fn_by_id(self):
        def guard_filter_fn(entries):
            return [entry.guard_type != "ID_MATCH" for entry in entries]

        @torch.compile(
            fullgraph=True,
            options={"guard_filter_fn": guard_filter_fn},
            backend="eager",
        )
        def fn(x):
            return id(x)

        inputs = (torch.randn(3, 2),)
        fn(*inputs)

        inputs_1 = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs_1), id(inputs[0]))

    def test_guard_filter_fn_by_is_global(self):
        def guard_filter_fn(entries):
            return [not entry.is_global for entry in entries]

        global GLOBAL_INT

        @torch.compile(
            fullgraph=True,
            options={"guard_filter_fn": guard_filter_fn},
            backend="eager",
        )
        def fn(x):
            return x + GLOBAL_INT

        GLOBAL_INT = 1
        fn(torch.randn(3, 2))

        GLOBAL_INT = 2
        inputs = (torch.randn(3, 2),)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs), inputs[0] + 1)

    def test_guard_filter_fn_by_name_and_value(self):
        def guard_filter_fn(entries):
            return [
                not (entry.name == "y" and entry.value is None) for entry in entries
            ]

        @torch.compile(
            fullgraph=True,
            options={"guard_filter_fn": guard_filter_fn},
            backend="eager",
        )
        def fn(x, y):
            if y is not None:
                x += y
            return x

        fn(torch.randn(3, 2), None)

        inputs = (torch.randn(3, 2), torch.tensor(1))
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(fn(*inputs), inputs[0])

    def test_guard_filter_inbuilt_nn_modules(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x)

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.norm.eps = 1e-02
        # Since the guards are skipped on inbuilt nn modules, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_nn_modules(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 2
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x) + self.c

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.c = 3
        # Since the guards are skipped on all nn modules, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_tensors(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 2.0
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x) + self.c

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            },
        )

        x = torch.rand(4, 8)
        opt_mod(x)

        mod.c = 3.0
        # Since the guards are skipped on all tensors
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_filter_globals(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 2
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, x):
                return self.norm(x) + self.c + GLOBAL_INT

        mod = Mod()
        opt_mod = torch.compile(
            mod,
            backend="eager",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )

        global GLOBAL_INT
        GLOBAL_INT = 1
        x = torch.rand(4, 8)
        opt_mod(x)

        GLOBAL_INT = 2
        # Since the guards are skipped on globals, we should not recompile
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            opt_mod(x)

    def test_guard_string_escaped(self):
        d = {frozenset({0}): {frozenset({0}): 1}}

        @torch.compile(backend="eager")
        def f(x):
            return x + d[frozenset({0})][frozenset({0})]

        x = torch.ones(3)
        self.assertEqual(x + 1, f(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
