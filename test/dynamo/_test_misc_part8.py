# Owner(s): ["module: dynamo"]
# flake8: noqa: B001, E711, E722, F403, F405, F841, W605
# ruff: noqa: E711,F403,F405,F841,PIE804,RSE102,RUF015,TRY002,TRY203,W605
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPart8:
    def test_frozen_dict(self):
        # A pattern from StableDiffusion
        class FrozenDict(collections.OrderedDict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                for key, value in self.items():
                    setattr(self, key, value)

                self.__frozen = True

            def __delitem__(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
                )

            def setdefault(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
                )

            def pop(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
                )

            def update(self, *args, **kwargs):
                raise Exception(
                    f"You cannot use ``update`` on a {self.__class__.__name__} instance."
                )

            def __setattr__(self, name, value):
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setattr__(name, value)

            def __setitem__(self, name, value):
                if hasattr(self, "__frozen") and self.__frozen:
                    raise Exception(
                        f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
                    )
                super().__setitem__(name, value)

        d = {"a": 1}
        frozen_d = FrozenDict(d)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            dict(frozen_d).items()
            return torch.sin(x)

        fn(torch.randn(4))

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

    def test_namedtuple_class(self):
        import collections

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            updated_x = []
            for v in x:
                updated_x.append(v + 1)
            return x.__class__(*updated_x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        d1 = torch.zeros(2, 2)
        d2 = torch.ones(2, 2)
        point = collections.namedtuple("Point", ["x", "y"])
        p = point(d1, d2)

        r = opt_fn(p)
        self.assertEqual(r.__class__, point)
        self.assertEqual(r.x, torch.ones(2, 2))
        self.assertEqual(r.y, torch.ones(2, 2) + 1)
        self.assertEqual(cnts.frame_count, 1)

    def test_getattrvariable_as_python_constant(self):
        from torch._dynamo.variables.misc import GetAttrVariable

        @torch.compile(backend="eager")
        def fn(x, rand1):
            random.Random().setstate(rand1.getstate())
            return x + rand1.random()

        def get_rng():
            rand1 = random.Random(1)
            orig_random = rand1.random
            rand1.random = lambda: orig_random()
            return rand1

        x = torch.randn(3, 3)
        expected = fn.__wrapped__(x, get_rng())

        with patch.object(GetAttrVariable, "as_python_constant", autospec=True) as po:
            actual = fn(x, get_rng())

        self.assertEqual(expected, actual)
        self.assertGreater(po.call_count, 0)

    def test_data_ptr_graph_break_builtin(self):
        def f(a, b):
            # builtin + not implemented for DataPtrVariable
            return a.data_ptr() + b.data_ptr()

        a = torch.randn(4)
        b = torch.randn(5)

        # make sure there is a graph break
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(a, b)

        torch._dynamo.reset()

        expected = f(a, b)
        actual = torch.compile(f, backend="eager")(a, b)

        self.assertEqual(expected, actual)

    def test_data_ptr_graph_break_aten(self):
        def f(a):
            # torch.add not implemented for DataPtrVariable
            return torch.add(a, a.data_ptr())

        a = torch.randn(4)

        counters.clear()

        expected = f(a)
        actual = torch.compile(f, backend="eager")(a)

        self.assertEqual(expected, actual)
        self.assertTrue(len(counters["graph_break"]) > 0)
        counters.clear()

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

    def test_compare_tensor_with_none(self):
        @torch.compile(backend="eager")
        def f(x):
            return torch.tensor(x == None)

        res = f(torch.tensor(1))
        self.assertEqual(torch.tensor(False), res)

    def test_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo:
            x: int

        @torch.compile(backend="eager", fullgraph=True)
        def run(x, foo0):
            if dataclasses.is_dataclass(foo0):
                foo1 = dataclasses.replace(foo0, **{"x": 1})
                return x + 1, foo1
            return x + 2, foo0

        res, foo = run(torch.zeros(1), Foo(0))
        self.assertTrue(res, torch.ones(1))
        self.assertEqual(foo.x, 1)

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

    def test_overridden_getattribute(self):
        class Bar:
            def __init__(self, v):
                self.v = v

        class Foo:
            attribute_map = {}

            def __init__(self):
                self.attribute_map = {
                    "a_premap": "a",
                }
                # `bar` attribute requires propagating sources correctly through
                # object.__getattribute__
                self.bar = Bar(5)

            def __setattr__(self, key, value):
                if key in super().__getattribute__("attribute_map"):
                    key = super().__getattribute__("attribute_map")[key]
                super().__setattr__(key, value)

            def __getattribute__(self, key):
                if key == "sentinel":
                    raise AttributeError()
                if key != "attribute_map" and key in super().__getattribute__(
                    "attribute_map"
                ):
                    key = super().__getattribute__("attribute_map")[key]
                return super().__getattribute__(key)

            def __getattr__(self, key):
                if key == "sentinel":
                    return 5
                raise AttributeError()

        def get_foo():
            f = Foo()
            f.a_premap = 2
            f.b = 3
            return f

        def fn(x, f):
            return x * f.a_premap * f.a * f.b * f.sentinel * f.bar.v

        x = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, get_foo()), opt_fn(x, get_foo()))

    def test_dunder_weakref(self):
        class Foo:
            pass

        def fn(x):
            foo = Foo()
            # tests isgetsetdescriptor
            if foo.__weakref__:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

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

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_builtin_bool_on_symint(self):
        def f(x):
            return bool(x.item())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randint(10, (1,))

        ref = f(x)
        res = opt_f(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_new_tensor_break(self):
        a = torch.tensor([1, 0, 0, 5])

        cases = {
            "scalar": lambda a: a.new_tensor([a.nonzero().squeeze(-1).numel()]),
            "multi": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([n, n + 1, n * 2]),
            )[-1],
            "mixed_shape": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([n * a.shape[0], n + a.shape[0], a.shape[0] - n]),
            )[-1],
            "nested": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([[n, n + 1], [n * 2, n - 1]]),
            )[-1],
            "with_zero": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([0, n, n * n]),
            )[-1],
        }

        for name, fn in cases.items():
            with self.subTest(case=name):
                self.assertEqual(
                    torch.compile(fn, fullgraph=True, backend="eager")(a),
                    fn(a),
                )

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

    def test_builtin_complex(self):
        def f(x):
            c = (
                complex(),
                complex(1),
                complex(2, 3),
                complex(imag=2),
                complex(real=1),
                complex(imag=1, real=2),
                complex("1+2j"),
                complex(1, 2).conjugate(),
            )
            return [x + z for z in c]

        x = torch.randn(1)
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_builtin_complex_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(*args, **kwargs):
            return torch.tensor(complex(*args, **kwargs))

        self.assertRaises(Unsupported, f, 1, 1, 1)
        self.assertRaises(Unsupported, f, 1, 1, fake_arg=1)
        self.assertRaises(Unsupported, f, fake_arg=1)
        self.assertRaises(Unsupported, f, [])
        self.assertRaises(Unsupported, f, "1 + j")

    def test_builtin_class_method_constant_fold(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return (
                bool.__new__(bool),
                bool.__new__(bool, 1),
                bool.__new__(bool, 0),
                bool.from_bytes(b"\x00" * 8, "big"),
                bool.from_bytes(b"abcd", "little"),
                int.__new__(int),
                int.__new__(int, 42),
                int.from_bytes(b"\x00\x03", "big"),
                int.from_bytes(b"\xff", byteorder="big", signed=True),
                float.fromhex("0x1.ffffp10"),
                float.hex(1.5),
            )

        res = fn()
        self.assertIs(res[0], False)
        self.assertIs(res[1], True)
        self.assertIs(res[2], False)
        self.assertIs(res[3], False)
        self.assertIs(res[4], True)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 42)
        self.assertEqual(res[7], 3)
        self.assertEqual(res[8], -1)
        self.assertEqual(res[9], float.fromhex("0x1.ffffp10"))
        self.assertEqual(res[10], "0x1.8000000000000p+0")

    def test_guard_string_escaped(self):
        d = {frozenset({0}): {frozenset({0}): 1}}

        @torch.compile(backend="eager")
        def f(x):
            return x + d[frozenset({0})][frozenset({0})]

        x = torch.ones(3)
        self.assertEqual(x + 1, f(x))

    def test_compiled_class_graph_break(self):
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=False)
        def f(x):
            x += 1

            class C:
                pass

            return x.sin()

        x = torch.randn(3)
        f(x)
        self.assertEqual(counter.frame_count, 2)

    def test_debugmode(self):
        # Test that DebugMode works
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("alias_op(Tensor x) -> (Tensor, Tensor)")
            lib.impl(
                "alias_op",
                lambda x: (x.view_as(x), x.view_as(x)),
                "CompositeExplicitAutograd",
            )
            lib.impl("alias_op", lambda x: (x.view_as(x), x.view_as(x)), "Meta")

            def fn(x):
                aliased, _ = torch.ops.mylib.alias_op(x)
                return aliased + 1

            x = torch.randn(10, 10)
            compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
            with torch._functorch.config.patch(
                check_custom_op_aliasing=True,
                error_on_custom_op_aliasing=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "The output of this custom operator \(1\) must not also be an input",
                ):
                    _ = compiled_fn(x)
                # Shouldn't error here because we already invoked once
                _ = compiled_fn(x)

                compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "The output of this custom operator \(1\) must not also be an input",
                ):
                    _ = compiled_fn(x)

    def test_debugmode_warns_outside_ci(self):
        # Test that DebugMode emits warnings (not errors) when error_on_custom_op_aliasing=False
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("alias_op2(Tensor x) -> (Tensor, Tensor)")
            lib.impl(
                "alias_op2",
                lambda x: (x.view_as(x), x.view_as(x)),
                "CompositeExplicitAutograd",
            )
            lib.impl("alias_op2", lambda x: (x.view_as(x), x.view_as(x)), "Meta")

            def fn(x):
                aliased, _ = torch.ops.mylib.alias_op2(x)
                return aliased + 1

            x = torch.randn(10, 10)
            compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor")
            # Use error_on_custom_op_aliasing=False to emit warnings instead of errors
            with (
                torch._functorch.config.patch(
                    check_custom_op_aliasing=True, error_on_custom_op_aliasing=False
                ),
                warnings.catch_warnings(record=True) as w,
            ):
                warnings.simplefilter("always")
                _ = compiled_fn(x)
                aliasing_warnings = [
                    x for x in w if "may not alias any inputs" in str(x.message)
                ]
                self.assertEqual(len(aliasing_warnings), 1)
                msg = str(aliasing_warnings[0].message)
                self.assertEqual(
                    msg,
                    "mylib::alias_op2 (with implementation in ???): "
                    "The output of this custom operator (1) must not also be an input "
                    "to this custom operator and (2) may not alias any inputs to this "
                    "custom operator or other returns. The most common way to trigger "
                    "this error is if we have y = custom_op(x) and y and x are the same "
                    "Tensor. Please instead return a clone of the offending output "
                    "tensor(s) (e.g. return x.clone()) or refactor the custom operator "
                    "to not return y. This is deprecated and will become an error in PyTorch 2.12.",
                )

    def test_make_contiguous_strides_for_under_compile(self):
        # is_nested_int and sym_max must be traceable under Dynamo.
        from torch._prims_common import make_contiguous_strides_for

        def fn(x):
            strides = make_contiguous_strides_for(x.shape)
            return x.as_strided(x.shape, strides)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
        x = torch.randn(4, 8)
        result = compiled_fn(x)
        self.assertEqual(result, x)

        x2 = torch.randn(7, 8)
        result2 = compiled_fn(x2)
        self.assertEqual(result2, x2)

    def test_requires_grad_changes_dynamo_graph(self):
        # requires_grad_() on a graph input graph-breaks, so no fullgraph
        def fn(x):
            x.requires_grad_()
            if x.requires_grad:
                return x * 2
            return x + 1

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn)
        result = opt_fn(x)
        self.assertEqual(result, x * 2)

    def test_requires_grad_backward_outside_compile(self):
        # requires_grad_() on a graph input graph-breaks, but eager fallback
        # produces correct results.
        def fn(x):
            x.requires_grad_()
            return (x * 2).sum()

        x_ref = torch.randn(3, 3)
        x_test = x_ref.clone()

        fn(x_ref).backward()
        torch.compile(fn)(x_test).backward()

        self.assertEqual(x_ref.grad, x_test.grad)

    def test_detach_inplace_on_intermediate_updates_metadata(self):
        def fn(x):
            y = x * 2
            y.detach_()
            return y + 1, y.requires_grad, y.grad_fn is None

        x = torch.randn(3, 3, requires_grad=True)
        ref = fn(x.clone())
        result = torch.compile(fn, backend="eager", fullgraph=True)(x.clone())

        self.assertEqual(ref, result)
        self.assertFalse(result[1])
        self.assertTrue(result[2])

    def test_requires_grad_on_intermediate(self):
        def fn(x):
            y = x * 2
            y.requires_grad_()
            return y

        x = torch.randn(3, 3)

        # fullgraph=True should error with actionable message
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"requires_grad_\(\)(.|\n)*\.detach\(\)",
        ):
            torch.compile(fn, fullgraph=True)(x)

        # Without fullgraph, falls back to eager and is correct
        result = torch.compile(fn)(x)
        self.assertTrue(result.requires_grad)
        self.assertEqual(fn(x), result)

    def test_requires_grad_on_intermediate_derived_returned(self):
        def fn(x):
            y = x * 2
            y.requires_grad_()
            return y * 3

        x = torch.randn(3, 3)

        # Derived tensor also loses requires_grad — should error with message
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"requires_grad_\(\)(.|\n)*\.detach\(\)",
        ):
            torch.compile(fn, fullgraph=True)(x)

        # Without fullgraph, falls back to eager and is correct
        result = torch.compile(fn)(x)
        ref = fn(x)
        self.assertTrue(result.requires_grad)
        self.assertEqual(ref, result)

    def test_requires_grad_on_intermediate_partial_graph(self):
        # When requires_grad_() on a source-less intermediate leaks as output,
        # Dynamo should restart and graph break at requires_grad_(), capturing
        # ops before it in a compiled graph (partial acceleration).
        def fn(x):
            a = x.sin()
            b = a.cos()
            b.requires_grad_()
            return b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        x = torch.randn(3, 3)
        result = torch.compile(fn, backend=backend)(x)
        self.assertEqual(result, fn(x))
        self.assertTrue(result.requires_grad)
        # The graph should capture the ops before requires_grad_()
        self.assertEqual(len(backend.graphs), 1)
        # Dynamic shapes adds shape guards to the graph, skip the exact check
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    a = l_x_.sin();  l_x_ = None
    b = a.cos();  a = None
    return (b,)""",
            )

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_on_intermediate_not_returned(self):
        def fn(x):
            y = x * 2
            y.requires_grad_()
            loss = (y * 3).sum()
            loss.backward()
            return y.grad

        x = torch.randn(3, 3)

        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_backward_grad_used_in_compute(self):
        # Use the grad result in further computation within compile
        def fn(x):
            y = x * 2
            y.requires_grad_()
            loss = (y**2).sum()
            loss.backward()
            return y.grad * 2 + 1

        x = torch.randn(3, 3)

        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_chunked_loss_backward(self):
        # Mirrors the TxtUnembedding pattern: forward compute, detach, make
        # new leaf, chunked loss with per-chunk backward, then propagate
        # accumulated grad back to the original input via h.backward().
        def fn(x, targets):
            # Forward computation before detach (e.g. transformer layers)
            h = x * 2 + 1
            x_detached = h.detach().requires_grad_()
            chunksz = x_detached.shape[0] // 2
            total_loss = torch.tensor(0.0)
            for start in range(0, x_detached.shape[0], chunksz):
                chunk = x_detached[start : start + chunksz]
                chunk_targets = targets[start : start + chunksz]
                logits = chunk @ torch.eye(chunk.shape[-1])
                loss = torch.nn.functional.cross_entropy(logits, chunk_targets)
                loss.backward()
                total_loss = total_loss + loss.detach()
            # Propagate chunked grad back through the forward computation
            h.backward(x_detached.grad)
            return x.grad, total_loss

        x_ref = torch.randn(4, 8, requires_grad=True)
        targets = torch.randint(0, 8, (4,))

        x_test = x_ref.clone().detach().requires_grad_(True)
        ref_grad, ref_loss = fn(x_ref, targets)
        compiled_grad, compiled_loss = torch.compile(fn, fullgraph=True)(
            x_test, targets
        )
        self.assertEqual(ref_grad, compiled_grad)
        self.assertEqual(ref_loss, compiled_loss)
        # Verify grad propagated to the input
        self.assertEqual(x_ref.grad, x_test.grad)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_backward_and_return_detached(self):
        # Returning a detached version of the tainted tensor is safe — detach()
        # strips requires_grad so AOTAutograd functionalization can't lose anything.
        def fn(x):
            y = x * 2
            y.requires_grad_()
            out = y * 3
            loss = out.sum()
            loss.backward()
            return y.grad, out.detach()

        x = torch.randn(3, 3)

        ref_grad, ref_out = fn(x.clone())
        compiled_grad, compiled_out = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref_grad, compiled_grad)
        self.assertEqual(ref_out, compiled_out)
        self.assertFalse(compiled_out.requires_grad)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_metadata_checks(self):
        # After requires_grad_() on an intermediate, requires_grad and is_leaf
        # should report correctly and be usable in control flow.
        def fn(x):
            y = x * 2
            y.requires_grad_()
            if y.requires_grad and y.is_leaf:
                loss = (y * 3).sum()
                loss.backward()
                return y.grad
            return y

        x = torch.randn(3, 3)
        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_side_effect_global(self):
        # requires_grad_() on intermediate, then store grad in a global
        saved = {}

        def fn(x):
            y = x * 2
            y.requires_grad_()
            loss = (y**2).sum()
            loss.backward()
            saved["grad"] = y.grad
            return y.grad.clone()

        x = torch.randn(3, 3)
        ref = fn(x.clone())
        saved_ref = saved["grad"].clone()
        saved.clear()

        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)
        self.assertEqual(saved_ref, saved["grad"])

    def test_import_user_defined_module(self):
        # testcase for https://github.com/pytorch/pytorch/issues/177682
        # Bad import result for types.ModuleType subclass in sys.modules
        class _ConfigModule(types.ModuleType):
            x = 1

        _ConfigModule.__module__ = __name__
        sys.modules["my_config"] = _ConfigModule("my_config")

        def fn():
            import my_config  # noqa: F401

            return torch.tensor(1)

        compilefn = torch.compile(fn, fullgraph=True, backend="eager")

        ret1 = fn()
        ret2 = compilefn()
        self.assertEqual(ret1, ret2)
