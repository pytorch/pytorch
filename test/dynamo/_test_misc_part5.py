# Owner(s): ["module: dynamo"]
# flake8: noqa: F403, F405, F841
# ruff: noqa: F403,F405,F841,PGH004
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPart5:
    def test_class_has_instancecheck_method(self):
        class A:
            pass

        class ExampleMeta(type):
            def __instancecheck__(cls, instance):
                return True

        class B(metaclass=ExampleMeta):
            pass

        def fn(x, obj):
            if isinstance(obj, B):
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        obj = A()
        ref = fn(x, obj)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, obj)
        self.assertTrue(same(ref, res))

    def test_custom_instancecheck_does_not_cause_extra_init(self):
        # When __new__ returns an object whose type is not a subclass of cls,
        # CPython's type.__call__ skips __init__. The polyfill
        # instantiate_user_defined_class_object must match this behavior even
        # when the metaclass defines a custom __instancecheck__ that would
        # return True for isinstance().
        class Meta(type):
            def __instancecheck__(cls, instance):
                return isinstance(instance, Base) and instance.tag == cls._tag

        class Base:
            def __init__(self, tag="default"):
                self.tag = tag

        class Child(Base, metaclass=Meta):
            _tag = "child"

            def __new__(cls):
                # Returns a Base (not a Child), like ByteStorage.__new__
                return Base(tag="child")

        def fn():
            obj = Child()
            return obj.tag

        ref = fn()
        self.assertEqual(ref, "child")

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn()
        self.assertEqual(res, "child")

    def test_custom_instancecheck_init_not_called(self):
        class AlwaysTrueMeta(type):
            def __instancecheck__(cls, instance):
                return True

        class Child(metaclass=AlwaysTrueMeta):
            def __new__(cls):
                return object()

            def __init__(self):
                raise AssertionError("should NOT be called")

        def fn():
            return Child()

        ref = fn()
        self.assertIsInstance(ref, object)

        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn()
        self.assertIsInstance(res, object)

    def test_variable_tracker_recursively_contains(self):
        # VariableTracker.recursively_contains should be updated correctly when mutation happens
        def fn(x):
            data = [[None] * 3] * 3
            for i in range(3):
                if i == 0:
                    data[0][i] = x
                else:
                    data[0][i] = data[0][i - 1] + 1
            return data[0][-1]

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_disable_flag(self):
        cnt = torch._dynamo.testing.CompileCounter()

        with patch.dict(os.environ, {"TORCH_COMPILE_DISABLE": "1"}):

            def fn(x, y):
                x = x + 1
                y = y + 1

            opt_fn = torch.compile(backend=cnt)

        self.assertEqual(cnt.frame_count, 0)

    def test_is_compiling(self):
        def f1():
            if torch._dynamo.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f2():
            if torch._utils.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f3():
            if torch.compiler.is_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        def f4():
            if torch.compiler.is_dynamo_compiling():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        for f in [f1, f2, f3, f4]:
            opt_f = torch.compile(f, backend="eager")

            self.assertEqual(f(), torch.zeros(2, 2))
            self.assertEqual(opt_f(), torch.ones(2, 2))

    def test_is_exporting_not_true_during_compile(self):
        def f():
            if torch.compiler.is_exporting():
                return torch.ones(2, 2)
            else:
                return torch.zeros(2, 2)

        opt_f = torch.compile(f, backend="eager")

        self.assertEqual(f(), torch.zeros(2, 2))
        self.assertEqual(opt_f(), torch.zeros(2, 2))

    def test_torch_generator_set_state(self):
        def fn():
            default_state = torch.default_generator.get_state()
            x = torch.rand([2, 3])
            if default_state.dtype != "float32":
                x = x * 2
            torch._dynamo.graph_break()
            torch.default_generator.set_state(default_state)
            y = torch.rand([2, 3])
            return x, y

        opt_fn = torch.compile(fn, backend="eager")
        x, y = opt_fn()
        self.assertEqual(x, y * 2)

    def test_torch_distributions_lazy_property(self):
        def fn(x):
            return torch.distributions.Categorical(probs=x).entropy()

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand([4, 4])
        self.assertEqual(opt_fn(x), fn(x))

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

    def test_restore_graphstate(self):
        # This function does some guard accumulation,
        # and then rolls back due to control flow.
        # The idea is that if one were printing guards as they appear,
        # they would see this insert a guard that does not show up in the final set of
        # guards as we rolled back from it.
        def nested_fn(s):
            if x[0] < 10:
                return s * s
            return s

        def fn(x, y):
            x = x + 1
            y = nested_fn(y)
            y = y + 10
            return x * y

        all_guards = []

        def guard_export_print(guards):
            nonlocal all_guards
            all_guards.extend(guards)

        opt_fn = torch._dynamo.optimize("eager", guard_export_fn=guard_export_print)(fn)

        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([1.0, 1.0])
        opt_fn(x, y)

        for guard in all_guards:
            # This guard was created
            self.assertTrue(guard.name != "nested_fn.__closure__[0].cell_contents")

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "Test requires CUDA or XPU.")
    def test_symint_as_device_kwarg_non_strict_export(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                # -2 to make device id 0 for easier testing on CI
                return torch.ones(10, device=x.size(0) - 2)

        x = torch.randn(2)
        m = Mod()
        d1 = torch.export.Dim("d1", max=2048)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, r"Constraints violated \(d1\)"
        ):
            ep = torch.export.export(
                m, (x,), dynamic_shapes={"x": {0: d1}}, strict=False
            )

    def test_call_parent_non_class_methods_from_child(self):
        class A:
            a = 4

            def add(self, x):
                return x + 10

            def mul(self, x):
                return x * 0.1

        class B(A):
            coeff = 4

            def add(self, x):
                return x + 20

            @classmethod
            def cube(cls, x):
                return cls.coeff * x * x * x

            def mul(self, x):
                return super().mul(x) * x * 0.2

        class C(B):
            def add(self, x):
                b = super().cube(x)
                c = A.add(self, x)
                d = B.mul(self, x)
                e = super(B, self).add(x)
                f = super().a * x
                return b + c + d + e + f

        x = torch.rand(4)
        fn = C().add
        ref = fn(x)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnt.frame_count, 1)

        # Check recompilation
        A.a = 5
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        # Ensure that super guard checks are working as expected
        res = opt_fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_builder_for_class_with_metaclass(self):
        class ExampleMeta(type):
            pass

        class MyClass(metaclass=ExampleMeta):
            pass

        def fn(x, y):
            if isinstance(y, MyClass):
                return x + 1
            else:
                return x - 1

        x = torch.rand([4, 4])
        y = MyClass()
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

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

    def test_torch_package_working_with_trace(self):
        # from torch._dynamo.test_case import run_tests

        inputs = [torch.randn([2, 2]), torch.randn([2, 2])]

        optimized_model = torch.compile(
            MyPickledModule(torch.randn([2, 2])), backend="eager"
        )
        from torch import package

        tmp_root = tempfile.gettempdir()
        path = os.path.join(tmp_root, "MyPickledModule.pt")
        package_name = "MyPickledModule"
        resource_name = "MyPickledModule.pkl"

        model = MyPickledModule(torch.randn([2, 2]))

        with package.PackageExporter(path) as exp:
            exp.extern("**")
            exp.save_pickle(package_name, resource_name, model)

        imp = package.PackageImporter(path)
        loaded_model = imp.load_pickle(package_name, resource_name)

        optimized_loaded_model = torch.compile(loaded_model, backend="eager")(*inputs)

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

    def test_fail_on_recompile_error_message(self):
        from torch._C._dynamo.eval_frame import (
            _load_precompile_entry,
            _reset_precompile_entries,
        )

        def fn(x):
            return x + 1

        guard_manager_bool = torch._dynamo.guards.RootGuardManager()
        guard_manager_bool.add_lambda_guard(
            lambda L: isinstance(L["x"], bool), ["isinstance(L['x'], bool)"], None
        )

        def injected_bool(x: bool):
            return x + 102

        args = (torch.randn(3, 2),)

        compiled_fn = torch.compile(fn, backend="eager")
        _load_precompile_entry(
            fn.__code__,
            torch._dynamo.guards.GuardManagerWrapper(guard_manager_bool),
            injected_bool.__code__,
        )

        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                with self.assertRaisesRegex(
                    RuntimeError, "Failed on the following precompiled guards:"
                ):
                    compiled_fn(*args)
        finally:
            _reset_precompile_entries(fn.__code__)

    def test_shape_and_tuple_equality(self):
        def fn(x, y, t):
            z = x * y
            if x.size() == t:
                return z.cos()
            return z.sin()

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.randn([4, 4]), torch.randn([4, 4]), (4, 4)
        )

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

    # specifically test for tensor.attribute -> torch.something()
    def test_real_imag_tensor_attribute(self):
        def fn(x, y):
            a = x.real
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_cast(self):
        from typing import cast

        def fn(x):
            return cast(torch.Tensor, torch.add(x, 1.0))

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(2, 2))
        res = opt_fn(torch.ones(2, 2))

        self.assertTrue(same(ref, res))

    def test_cast_with_different_module_types(self):
        # typing.cast works correctly when used in a mixin pattern with
        # different module types, producing correct results without
        # graph breaks.
        from typing import cast

        class Mixin:
            def get_self_as_module(self):
                return cast(torch.nn.Module, self)

        class ModuleA(Mixin, torch.nn.Module):
            def forward(self, x):
                self.get_self_as_module()
                return x + 1

        class ModuleB(Mixin, torch.nn.Module):
            def forward(self, x):
                self.get_self_as_module()
                return x + 2

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(mod, x):
            mod.get_self_as_module()
            return x + 1

        x = torch.randn(4)
        ref_a = fn.__wrapped__(ModuleA(), x)
        ref_b = fn.__wrapped__(ModuleB(), x)
        res_a = fn(ModuleA(), x)
        res_b = fn(ModuleB(), x)

        self.assertEqual(ref_a, res_a)
        self.assertEqual(ref_b, res_b)
        self.assertEqual(cnt.frame_count, 2)

    def test_cast_fullgraph_with_non_tensor(self):
        # Verify typing.cast works with non-tensor values under fullgraph=True
        from typing import cast

        def fn(x):
            val = cast(int, x.shape[0])
            return x + val

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(3, 4))
        res = opt_fn(torch.ones(3, 4))

        self.assertTrue(same(ref, res))

    def test_cast_no_recompile_after_graph_break(self):
        # In FSDP, cast(nn.Module, self) can be called after a
        # graph break. Without the polyfill + skip_code fix, PEP 523 compiles
        # typing.cast as a standalone frame with TYPE_MATCH guards on val,
        # causing recompilation when different module types pass through.
        # https://github.com/pytorch/pytorch/blob/0feb90404fbeb9b1594ae194f8fd47bbe7f5f245/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L376
        from typing import cast

        from torch._dynamo.utils import counters

        counters.clear()

        class Base(torch.nn.Module):
            def get_state(self):
                torch._dynamo.decorators.graph_break()
                return cast(torch.nn.Module, self)

        class ModuleA(Base):
            pass

        class ModuleB(Base):
            pass

        cnt = torch._dynamo.testing.CompileCounter()
        a, b = ModuleA(), ModuleB()

        @torch.compile(backend=cnt)
        def fn(mod, x):
            mod.get_state()
            return x + 1

        x = torch.randn(4)
        fn(a, x)
        fn(b, x)
        self.assertEqual(cnt.frame_count, 1)
        # 5 frames: fn (x2), get_state before graph_break (x2),
        # get_state resume after graph_break (x1, no recompile).
        # Without skip_code, typing.cast would add 2 more frames (7 total).
        self.assertEqual(counters["frames"]["total"], 5)

    def test_T_tensor_attribute(self):
        def fn(x, y):
            a = x.T
            return torch.add(a, y)

        x = torch.rand((4, 4))
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_recursive_tensor_attribute(self):
        def fn(x, y):
            a = x.real.T
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_object_attribute(self):
        # user-defined functions which are object's attributes are not converted to bound methods
        def my_add(*args):
            a, b = args
            return a + b

        class MyClass:
            def __init__(self, func):
                self.add = func

        obj = MyClass(my_add)

        def fn(x):
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_assigning_function_to_class_attribute(self):
        # user-defined functions which are class's attributes are converted to bound methods
        def my_add(*args):
            obj, a, b = args
            return obj.x + a + b

        class MyClass:
            add = my_add

            def __init__(self, x):
                self.x = x

        obj = MyClass(0.5)

        def fn(x):
            return obj.add(x, 2)

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_tagging_tensors_simple(self):
        def foo(x, y):
            return x * y, x, y

        a = torch.randn([3, 3])
        a.tag = "a"
        b = torch.randn([3, 3])
        b.tag = "b"

        exported = torch._dynamo.export(foo)(a, b)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["a", "b"])

    def test_tagging_tensors_mix_used_unused_structure(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
        state = [
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
        ]
        i = torch.tensor(
            [
                [0.0313, -0.1487, -0.3846, -0.5321],
                [-1.7073, 1.3331, -0.0890, -1.4935],
                [-0.8314, -0.1862, -0.5935, 1.5232],
            ]
        )

        mems.tag = "MEMS"
        i.tag = "FOO"
        state[0].tag = "STATE_0"
        state[1].tag = "HMMM"

        exported = torch._dynamo.export(pre_attention_state_ops)(i, mems, state)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["STATE_0", "HMMM"])

    def test_get_custom_tensor_attribute(self):
        def fn(x):
            return x.custom_attr * x

        x = torch.rand((2, 2))
        x.custom_attr = 3.14
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_set_custom_tensor_attribute(self):
        def fn(x):
            x.custom_attr = 3.14
            return x.custom_attr * x

        x = torch.rand((2, 2))
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_unhandled_exception_in_dynamo(self):
        # traceback.format_exc() approximates an unhandled exception
        def f(a):
            a += 1
            raise RuntimeError("smoge")
            return a

        opt_fn = torch.compile(f, backend="eager")
        try:
            opt_fn(torch.ones(2))
        except RuntimeError as e:
            self.assertIn("smoge", traceback.format_exc())

    def test_unhandled_exception_in_dynamo2(self):
        # segfaults in python 3.11 if shadow frame is freed improperly
        from torch.testing import make_tensor

        def fn():
            # test that the errors are the same for dense and sparse versions
            def test1(*, is_sparse):
                # shapes must be compatible for matrix multiplication
                a = make_tensor((2, 3), dtype=torch.float32, device="cpu")
                if is_sparse:
                    a_sparse = a.to_sparse_csr()
                    return torch.addmm(a, a_sparse, a)
                else:
                    return torch.addmm(a, a, a)

            try:
                test1(is_sparse=False)
            except RuntimeError as msg:
                try:
                    test1(is_sparse=True)
                except RuntimeError as msg2:
                    raise RuntimeError("smoge")

        opt_fn = torch.compile(fn, backend="eager")
        try:
            opt_fn()
        except RuntimeError:
            self.assertIn("smoge", traceback.format_exc())

    def test_variable_access_in_exception(self):
        def fn():
            x = torch.ones(1)
            try:
                raise RuntimeError("bad")
            except RuntimeError:
                x += 1
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), torch.tensor([2.0]))

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

    def test_sparse_output_inductor_should_break(self) -> None:
        # See https://github.com/pytorch/pytorch/issues/164823
        # We want consistent semantics here
        def forward(x: torch.Tensor) -> torch.Tensor:
            x_sparse = x.to_sparse()
            return x_sparse * 2

        test_tensor = torch.randn(10, 10)
        pt = forward(test_tensor)
        aot_eager = torch.compile(forward, backend="aot_eager")(test_tensor)
        self.assertEqual(pt, aot_eager)
        inductor = torch.compile(forward, backend="inductor")(test_tensor)

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

    # NOTE this test can be removed once multiline errors are in Python.
    # See https://github.com/python/cpython/issues/106922
    # Covered by test_logging.py:test_trace_call* tests in 3.13+
    @skipIfNotPy311
    @unittest.skipIf(sys.version_info >= (3, 13), "feature landed in 3.13")
    def test_get_instruction_source_311(self):
        def f():
            # flake8: noqa
            # fmt: off
            # test binary ops
            a = ( b   )   +   c
            a = (a + b) // (c - d)
            a = b    \
         +\
               c  # test
            a = (
                (b  # test +
                    )  \
                # +
            << (

                c  # test
                \
            )  # test
            )

            # test slice
            a = bbb   [  ccc    ]
            b = bbbbb \
                [  ccc # test

                 + ddd  \

                ] # test
            a = bbb[ccc][ddd][eee]

            # test nested and multiline function calls
            a = g(g(g(b)))
            a = g(h(
                g(b),
                c
            ))

            # test chained function calls
            a = (g(x).y)(
                z
            )(1)(2)

            # test unicode (match traceback behavior)
            a = ("🔥🔥🔥" +
                + "🔥🔥") + b

        from torch._dynamo.utils import get_instruction_source_311

        if sys.version_info >= (3, 12):
            # Offsets changed in 3.12, e.g. due to removal of PRECALL inst
            offsets = (3, 11, 15, 19, 23, 29, 35, 44, 53, 65)
        else:
            offsets = (3, 11, 15, 19, 23, 29, 35, 46, 58, 74)
        insts = list(dis.get_instructions(f))
        # uncomment to determine offsets
        # print(*enumerate(insts), sep="\n")
        all_sources = "\n".join(
            get_instruction_source_311(f.__code__, insts[offset]) for offset in offsets
        )
        self.assertExpectedInline(
            all_sources,
            """\
            a = ( b   )   +   c
                ~~~~~~~~~~^~~~~

            a = (a + b) // (c - d)
                ~~~~~~~~^^~~~~~~~~

            a = b    \\
                ~~~~~~
         +\\
         ^~
               c  # test
               ~

                (b  # test +
                ~~~~~~~~~~~~
                    )  \\
                    ~~~~
                # +
                ~~~
            << (
            ^^~~


                c  # test
                ~~~~~~~~~
                \\
                ~
            )  # test
            ~

            a = bbb   [  ccc    ]
                ~~~~~~^^^^^^^^^^^

            b = bbbbb \\
                ~~~~~~~
                [  ccc # test
                ^^^^^^^^^^^^^


                 + ddd  \\
                 ^^^^^^^^


                ] # test
                ^

            a = bbb[ccc][ddd][eee]
                ~~~~~~~~^^^^^

            a = g(g(g(b)))
                  ~^^^^^^

            a = g(h(
                  ~^
                g(b),
                ^^^^^
                c
                ^
            ))
            ^

            a = (g(x).y)(
                ~~~~~~~~~
                z
                ~
            )(1)(2)
            ~^^^
""",
        )
        # test unicode (since assertExpectedInline doesn't support unicode)
        op_offset = 74 if sys.version_info >= (3, 12) else 84
        self.assertEqual(
            get_instruction_source_311(f.__code__, insts[op_offset]),
            """\
            a = ("🔥🔥🔥" +
                ~~~~~~~~
                + "🔥🔥") + b
                ~~~~~~~~^~~
""",
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

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_symint_copy_into_unbacked_slice(self):
        @torch.compile(backend="eager")
        def fn(a, x):
            u0 = torch.tensor(x[0].to(torch.int64).item()).item()
            B, H, T, D = a.shape
            a_padding = torch.zeros((B, H, u0, D), dtype=torch.float64)
            b = torch.cat([a, a_padding], dim=2)
            c = torch.randn(B, H, 152, D)
            b[:, :, :152, :] = c
            return b

        x = torch.tensor([0])
        torch._dynamo.decorators.mark_unbacked(x, 0)
        a = torch.zeros((1, 16, 152, 96))

        # Previously would crash with guard on data dependent error
        fn(a, x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_fold_nontrivial_product_modulo(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            u0, u1 = x.tolist()
            # The condition should fold to true.
            if ((u0 + 10) * (u0 + 10)) % (u0 + 10) == 0:
                return torch.tensor(True)
            return torch.tensor(False)

        res = f(torch.tensor([20, 21]))
        self.assertEqual(torch.tensor(True), res)

    # Translation validation changes the exception type, don't run with it
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

    def test_tolist(self):
        # This should compile with no faluire.
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(fullgraph=False, backend=cnt)
        def func(a):
            a = a * 100
            u0, u1, u2, u3, u4 = a.tolist()
            return a * u0 * u1

        func(torch.tensor([1, 2, 3, 4, 5]))
        self.assertEqual(cnt.frame_count, 2)

    # Sadly, this does not throw - we do not prop correctly across the graph break
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
