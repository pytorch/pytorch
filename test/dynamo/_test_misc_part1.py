# Owner(s): ["module: dynamo"]
# flake8: noqa: F403, F405, F841
# ruff: noqa: F403,F405,F841,PLR0133
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPart1:
    def test_get_cache_entry(self):
        def f(x):
            return x + 1

        torch.compile(f, backend="eager")(torch.randn(5, 5, 5))
        entries = _debug_get_cache_entry_list(f)
        self.assertTrue(len(entries) > 0)

        def g(x):
            return x + 2

        entries = _debug_get_cache_entry_list(g)
        self.assertTrue(len(entries) == 0)

        try:
            _debug_get_cache_entry_list(1)
        except TypeError as e:
            self.assertIn("expected a code object!", str(e))

        # test get cache entry on skipped code object
        def h(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        torch.compile(h, backend="eager")(torch.randn(3, 3))

        entries = _debug_get_cache_entry_list(torch._dynamo.graph_break)
        self.assertEqual(len(entries), 0)

    def test_boolarg(self):
        def boolarg(aa, bb, flag):
            if flag:
                return aa - bb
            else:
                return bb - aa

        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        correct1 = boolarg(a, b, True)
        correct2 = boolarg(a, b, False)
        correct3 = boolarg(a, b, None)
        counter = CompileCounter()
        opt_boolarg = torch._dynamo.optimize_assert(counter)(boolarg)
        val1 = opt_boolarg(a, b, True)
        val2 = opt_boolarg(a, b, False)
        val3 = opt_boolarg(a, b, None)
        val4 = opt_boolarg(a, b, True)
        self.assertTrue(same(val1, correct1))
        self.assertTrue(same(val2, correct2))
        self.assertTrue(same(val3, correct3))
        self.assertTrue(same(val4, correct1))
        self.assertEqual(counter.frame_count, 3)

    @unittest.skipIf(not TEST_CUDA, "cuda needed")
    def test_assume_32_bit_indexing(self):
        @torch.compile(backend="inductor")
        def func(a, b):
            # Multiple concat operations
            x = torch.concat([a, b], dim=0)
            y = torch.concat([a, b], dim=1)

            # Reshape to create indexing patterns
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            # Take the smaller one and expand
            min_size = min(x_flat.shape[0], y_flat.shape[0])
            x_trunc = x_flat[:min_size]
            y_trunc = y_flat[:min_size]

            # Combine and compute
            result = (x_trunc + y_trunc) * 10

            # Cumulative operations create complex indexing
            cumsum = result.cumsum(dim=0)

            return cumsum.sum()

        a = torch.rand(100, 30, device="cuda")
        b = torch.rand(100, 30, device="cuda")

        torch._dynamo.decorators.mark_unbacked(a, 0)
        torch._dynamo.decorators.mark_unbacked(a, 1)
        torch._dynamo.decorators.mark_unbacked(b, 0)
        torch._dynamo.decorators.mark_unbacked(b, 1)

        source_code = run_and_get_code(func, a, b)[1]
        # Check that int64 indexing is used (either 1D [:] or 2D [:, None] form)
        self.assertTrue(
            "tl.arange(0, XBLOCK)[:].to(tl.int64)" in str(source_code)
            or "tl.arange(0, XBLOCK)[:, None].to(tl.int64)" in str(source_code)
        )
        # Check that 32-bit indexing is NOT used
        self.assertFalse(
            "tl.arange(0, XBLOCK)[:]\n" in str(source_code)
            and ".to(tl.int64)" not in str(source_code)
        )

        torch._dynamo.reset()

        with torch._inductor.config.patch(assume_32bit_indexing=True):
            source_code = run_and_get_code(func, a, b)[1]
            # Check that int64 indexing is NOT used when assume_32bit_indexing=True
            self.assertFalse(
                "tl.arange(0, XBLOCK)[:].to(tl.int64)" in str(source_code)
                or "tl.arange(0, XBLOCK)[:, None].to(tl.int64)" in str(source_code)
            )

    def test_dynamo_side_effect(self):
        class GlobalContext:
            def __init__(self):
                self._tensors = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class Module(torch.nn.Module):
            def forward(self, x):
                with GlobalContext() as ctx:
                    z = x + 1
                    ctx._tensors["6"] = x + 2
                return z, ctx

        mod = Module()
        inp = torch.randn(4, 4)
        with torch._dynamo.config.patch(
            replay_side_effects=False, side_effect_replay_policy="warn"
        ):
            val = torch.compile(mod, backend="eager")(inp)
            # Verify new object is properly initialized
            self.assertIn("6", val[1]._tensors)
            self.assertEqual(val[1]._tensors["6"], inp + 2)

    def test_dynamo_inside_custom_op(self):
        cnt = torch._dynamo.testing.InductorAndRecordGraphs()
        cnt1 = torch._dynamo.testing.InductorAndRecordGraphs()

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x) -> Tensor")

            def inner(x):
                return x.sin().cos()

            def foo_impl(x):
                return torch.compile(inner, fullgraph=True, dynamic=True, backend=cnt)(
                    x
                )

            m.impl("foo", foo_impl, "CompositeExplicitAutograd")

            @torch.compile(fullgraph=True, dynamic=True, backend=cnt1)
            def f(x):
                return torch.ops.mylib.foo.default(x)

            x = torch.randn(3)
            res = f(x)
            res1 = f(x)
            res2 = f(x)
            expected = x.sin().cos()
            self.assertEqual(res, expected)
            self.assertEqual(res1, expected)
            self.assertEqual(res2, expected)
            self.assertTrue(len(cnt.inductor_graphs), 1)
            self.assertTrue(len(cnt1.inductor_graphs), 1)
            self.assertExpectedInline(
                str(cnt.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=0] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%arg1_1,), kwargs = {})
    %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%sin,), kwargs = {})
    return (cos,)""",
            )
            self.assertExpectedInline(
                str(cnt1.inductor_graphs[0].graph).strip(),
                """\
graph():
    %arg0_1 : [num_users=0] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %foo : [num_users=1] = call_function[target=torch.ops.mylib.foo.default](args = (%arg1_1,), kwargs = {})
    return (foo,)""",
            )

    def test_compile_non_infra_inside_compile(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(func, backend=backend, fullgraph=True)(
                    *args, **kwargs
                )
                return out

        x = torch.randn(5)
        with YoloMode():
            out = torch.add(x, x)

        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_empty(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

        self.assertEqual(out.sum().item(), 5.0)
        self.assertEqual(len(backend.graphs), 0)

    def test_compile_non_infra_empty_with_disalloed_dispatch_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.mul.Tensor(args[0], args[1])

        x = torch.ones(5)
        with YoloMode():
            out = torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_multiple(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                def random_fn(func, *args, **kwargs):
                    return func(*args, **kwargs)

                random_fn(func, *args, **kwargs)
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 0)
        self.assertEqual(len(backend.graphs), 0)

    def test_compile_non_infra_multiple_compile_internal(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

            @classmethod
            def ignore_compile_internals(cls):
                return True

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

            @classmethod
            def ignore_compile_internals(cls):
                return True

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 1)
        self.assertEqual(len(backend.graphs), 1)

    def test_compile_non_infra_multiple_compile_internal_mixed(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        class YoloMode2(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(
                    lambda x, y: func(x, y), backend=backend3, fullgraph=True
                )(*args, **kwargs)
                return out

            @classmethod
            def ignore_compile_internals(cls):
                return True

        class YoloMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(torch.add, backend=backend2, fullgraph=True)(
                    args[0], args[1]
                )
                return out

        x = torch.ones(5)
        with YoloMode(), YoloMode2():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 0)
        self.assertEqual(len(backend.graphs), 0)

        torch._dynamo.reset()

        backend3 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        with YoloMode2(), YoloMode():
            torch.compile(
                lambda x, y: torch.add(x, y), fullgraph=True, backend=backend
            )(x, x)

        self.assertEqual(len(backend2.graphs), 1)
        self.assertEqual(len(backend3.graphs), 1)
        self.assertEqual(len(backend.graphs), 0)

    @torch._dynamo.config.patch(inline_torch_dispatch_torch_compile=False)
    def test_compile_non_infra_disabled_config(self):
        """Test that setting inline_torch_dispatch_torch_compile=False reverts to old behavior."""
        from torch.utils._python_dispatch import TorchDispatchMode

        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        # When the config is False, torch.compile inside __torch_dispatch__ should
        # be skipped (old behavior) because we are inside a dispatch mode.
        class YoloMode(TorchDispatchMode):
            @classmethod
            def _should_skip_dynamo(cls):
                return False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = torch.compile(func, backend=backend, fullgraph=True)(
                    *args, **kwargs
                )
                return out

        x = torch.randn(5)
        with YoloMode():
            out = torch.add(x, x)

        # With the config disabled, compile should be skipped, so 0 graphs captured
        self.assertEqual(len(backend.graphs), 0)

    @torch._dynamo.config.patch(accumulated_recompile_limit=1)
    def test_dynamo_disabled_in_custom_op_kernels(self):
        counters.clear()

        @torch.library.custom_op("mylib::foo9", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            torch._dynamo.graph_break()
            return x.clone()

        foo.register_fake(torch.clone)

        @torch.compile(backend="eager")
        def f(x):
            return foo._opoverload(x)

        x = torch.randn(2)
        f(x)
        x = torch.randn(3)
        # Recompile hits the cache size limit, which will cause Dynamo to
        # recurse into the frames. The only frame is the implementation
        # of foo. If Dynamo was not turned off correctly, then
        # we'll see a graph break
        f(x)
        self.assertEqual(len(counters["graph_break"]), 0)

        counters.clear()

        called = 0

        # test register_kernel
        @foo.register_kernel("cpu")
        def _(x):
            nonlocal called
            called += 1
            torch._dynamo.graph_break()
            return x.clone()

        f(x)
        self.assertEqual(called, 1)
        self.assertEqual(len(counters["graph_break"]), 0)

        # test torch.library.register_kernel
        counters.clear()
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo2(Tensor x) -> Tensor")

            @torch.library.register_fake("mylib::foo2", lib=m)
            def _(x):
                return x.clone()

            @torch.library.register_kernel("mylib::foo2", "cpu", lib=m)
            def _(x):
                torch._dynamo.graph_break()
                return x.clone()

            @torch.compile(backend="eager")
            def g(x):
                return torch.ops.mylib.foo2.default(x)

            x = torch.randn(2)
            g(x)  # compiles
            x = torch.randn(3)
            g(x)  # dynamo falls back on the outermost frame
            self.assertEqual(len(counters["graph_break"]), 0)

    def test_invalid_args_builtin(self):
        @torch.compile(backend="eager")
        def fn(x):
            x = x.sin()
            if isinstance(x, torch.Tensor, invalid=True):
                x = x.sin()
            return x

        with self.assertRaises(TypeError):
            fn(torch.randn(16))

    def test_scalar_device_movement(self):
        if not torch._dynamo.config.assume_static_by_default:
            self.skipTest("Doesn't work with symints")

        def add_fn(a, b, out):
            res = torch.add(a, b, out=out)
            return res

        res = add_fn(2, 3, torch.tensor(0.0))
        add_fn = torch.compile(add_fn, backend="eager", fullgraph=True)
        res_compiled = add_fn(2, 3, torch.tensor(0.0))
        self.assertEqual(res, res_compiled)

    def test_callpacked(self):
        def call_packed(args):
            a, b, c = args
            return a - b * c

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        correct = call_packed([a, b, c])
        opt_call_packed = torch._dynamo.optimize_assert(counter)(call_packed)
        val1 = opt_call_packed([a, b, c])
        val2 = opt_call_packed((a, b, c))
        val3 = opt_call_packed([a, b, c])
        val4 = opt_call_packed((a, b, c))
        self.assertTrue(same(val1, correct))
        self.assertTrue(same(val2, correct))
        self.assertTrue(same(val3, correct))
        self.assertTrue(same(val4, correct))
        self.assertEqual(counter.frame_count, 2)

    def test_raises(self):
        def fn(a, b, c, cls):
            x = a + b - c * 10
            raise cls(str(x))

        counter = CompileCounter()
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        c = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaises(AssertionError, lambda: opt_fn(a, b, c, AssertionError))
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    def test_module_not_callable(self):
        def fn(x):
            return torch.fft(x)

        counter = CompileCounter()
        a = torch.randn(10, 10)
        opt_fn = torch.compile(fn, backend=counter)
        self.assertRaisesRegex(
            TypeError, "'module' object is not callable", lambda: opt_fn(a)
        )

    def test_inplace(self):
        def inplace1(a, b):
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o

        torch._dynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    def test_inplace_desugaring(self):
        def inplace_on_literals(y):
            x0 = 1
            x0 += y
            x1 = 1
            x1 -= y
            return x0, x1

        torch._dynamo.testing.standard_test(
            self, inplace_on_literals, 1, expected_ops=2
        )

    def test_unpack4(self):
        def unpack4(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.size()
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self,
            unpack4,
            2,
            expected_ops=5,
        )

    def test_unpack5(self):
        def unpack5(a, b):
            a = a[:5, :]
            b = b[:5, :]
            x, y = a.shape
            o = torch.empty((x, y))
            o.copy_(a / b)
            return o

        torch._dynamo.testing.standard_test(
            self,
            unpack5,
            2,
            expected_ops=5,
        )

    def test_matmul1(self):
        def matmul_op1(a, b):
            return a @ b

        # TODO(jansel): FX doesn't support this, should add upstream support
        torch._dynamo.testing.standard_test(self, matmul_op1, 2, expected_ops=1)

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

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_ops_are_allowed(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::bar",
                "(Tensor x) -> Tensor",
                lib=lib,
                tags=(torch.Tag.pt2_compliant_tag,),
            )
            torch.library.impl(
                "mylib::bar", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            if torch.Tag.pt2_compliant_tag not in torch.ops.mylib.bar.default.tags:
                raise AssertionError("Expected pt2_compliant_tag in bar.default.tags")

            def f(x):
                return torch.ops.mylib.bar(x)

            overload = torch.ops.mylib.bar.default

            def g(x):
                return overload(x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            optimized_f = torch.compile(f, backend=counts, fullgraph=True)
            _ = optimized_f(x)

            optimized_g = torch.compile(f, backend=counts, fullgraph=True)
            _ = optimized_g(x)

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_non_pt2_compliant_ops_graph_break(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define("mylib::bar2", "(Tensor x) -> Tensor", lib=lib)
            torch.library.impl(
                "mylib::bar2", "CompositeImplicitAutograd", torch.sin, lib=lib
            )
            if torch.Tag.pt2_compliant_tag in torch.ops.mylib.bar2.default.tags:
                raise AssertionError(
                    "Expected pt2_compliant_tag not in bar2.default.tags"
                )

            def f(x):
                return torch.ops.mylib.bar2(x)

            overload = torch.ops.mylib.bar2.default

            def g(x):
                return overload(x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                optimized_f = torch.compile(f, backend=counts, fullgraph=True)
                y = optimized_f(x)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                optimized_g = torch.compile(f, backend=counts, fullgraph=True)
                y = optimized_g(x)

    @torch._dynamo.config.patch(only_allow_pt2_compliant_ops=True)
    def test_pt2_compliant_overload(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::bar3.tensor",
                "(Tensor x) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::bar3.int", "(Tensor x, int dim) -> Tensor", lib=lib
            )

            torch.library.impl(
                "mylib::bar3.tensor",
                "CompositeImplicitAutograd",
                torch.sin,
                lib=lib,
            )
            torch.library.impl(
                "mylib::bar3.int", "CompositeImplicitAutograd", torch.sum, lib=lib
            )

            def f(x):
                return torch.ops.mylib.bar3(x)

            def g(x):
                return torch.ops.mylib.bar3(x, 1)

            def h(x):
                return torch.ops.mylib.bar3(x, x, x)

            x = torch.randn(3)

            counts = torch._dynamo.testing.CompileCounter()
            optimized_f = torch.compile(f, backend=counts, fullgraph=True)
            optimized_g = torch.compile(g, backend=counts, fullgraph=True)
            optimized_h = torch.compile(h, backend=counts, fullgraph=True)

            # No error: the overload is PT2 compliant
            optimized_f(x)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "not PT2 compliant"
            ):
                y = optimized_g(x)

            # graph break on incorrect parsing
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "failed to"):
                y = optimized_h(x)

    def test_user_defined_setattr1(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.y = obj.x + 1

        obj = UserDefineSetAttr()
        with patch.object(UserDefineSetAttr, "setup", True):
            obj.x = torch.randn(8)
        fn(obj)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertEqual(obj.y, obj.x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    def test_user_defined_setattr2(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            obj = UserDefineSetAttr()
            obj.x = x
            obj.y = obj.x + 1
            return obj

        x = torch.randn(8)
        obj = fn(x)
        with patch.object(UserDefineSetAttr, "setup", True):
            self.assertIs(obj.x, x)
            self.assertEqual(obj.y, x + 1)
        self.assertEqual(obj.__dict__.keys(), {"pfx_x", "pfx_y"})

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_repeat_cat(self):
        def f(x, n):
            m = x.item()
            x = torch.empty(x).repeat(n)  # s0*u0
            return torch.cat([x, x], dim=0)

        fn = torch.compile(f, backend="eager", dynamic=True, fullgraph=True)
        fn(torch.tensor([5]), 5)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_cond_runtime_assert_generation(self):
        def fn(x):
            y = x.nonzero()  # unbacked binding u0
            torch._check(y.shape[0] % 4 == 0)

            return torch.randn(y.shape[0])

        @torch.compile(dynamic=True, backend="aot_eager")
        def foo(x):
            b = torch.cond(
                pred=(x.shape[0] % 4 == 0),
                true_fn=lambda: fn(x),
                false_fn=lambda: fn(x),
            )

            return b

        foo(torch.randn(4, 4))
        with self.assertRaisesRegex(
            RuntimeError, "Runtime assertion failed for expression Eq(Mod(u1, 4), 0)*"
        ):
            foo(torch.randn(5, 5))

    def test_tensor_setattr_getset_descriptor(self):
        # Tensor attribute `real` has special getter/setter for complex dtype.
        def f(x):
            x.real = 10
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=False)
        x = torch.ones(5, dtype=torch.cfloat)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_newly_constructed_tensor_attr_mutation(self):
        def f(x):
            y = x + 10
            y.grad = x
            y.foo = 42
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)
        self.assertEqual(res.grad, ref.grad)
        self.assertEqual(res.foo, ref.foo)

    def test_input_tensor_custom_attr_mutation(self):
        def f(x, flag):
            x.offloading_activation = flag
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(x.offloading_activation)

    def test_intermediate_tensor_custom_attr_mutation(self):
        def f(x, flag):
            y = x + 1
            y.offloading_activation = flag
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(res.offloading_activation)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "requires Hopper+ (SM >= 9.0) for TMA",
    )
    @unittest.skipIf(
        not torch.utils._triton.has_triton()
        or not hasattr(__import__("triton"), "set_allocator"),
        "requires triton with set_allocator support",
    )
    def test_triton_set_allocator_no_graph_break(self):
        """set_allocator inside torch.compile does not graph break and
        replays correctly at runtime (including cache hits)."""
        import triton
        import triton.language as tl
        from triton.runtime._allocation import NullAllocator

        @triton.jit
        def tma_copy_kernel(
            x_ptr,
            out_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            pid = tl.program_id(0)
            desc = tl.make_tensor_descriptor(
                x_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            block = tl.load_tensor_descriptor(desc, [pid * BLOCK_M, 0])
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            tl.store_tensor_descriptor(out_desc, [pid * BLOCK_M, 0], block)

        M, N, BLOCK_M, BLOCK_N = 128, 64, 64, 64

        def run_kernel(x):
            out = torch.empty_like(x)
            tma_copy_kernel[(M // BLOCK_M,)](
                x,
                out,
                M,
                N,
                x.stride(0),
                x.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            return out

        x = torch.randn(M, N, device="cuda")

        from contextlib import contextmanager

        from triton.runtime._allocation import _allocator

        @contextmanager
        def triton_allocator(allocator):
            prev = _allocator.get()
            triton.set_allocator(allocator)
            try:
                yield
            finally:
                triton.set_allocator(prev)

        def fn_with_set_allocator(x):
            triton.set_allocator(
                lambda size, alignment, stream: torch.empty(
                    size, device="cuda", dtype=torch.int8
                )
            )
            return run_kernel(x)

        opt_fn = torch.compile(
            fn_with_set_allocator, backend="aot_eager", fullgraph=True
        )

        # set_allocator inside compiled region does NOT graph break
        with triton_allocator(NullAllocator()):
            out = opt_fn(x)
            self.assertEqual(out, x)

            # Verify set_allocator replays on cache hit (not just tracing)
            triton.set_allocator(NullAllocator())
            out2 = opt_fn(x)
            self.assertEqual(out2, x)

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

    def test_generate_trivial_abstract_impl(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_generate_trivial_abstract_impl",
                "(Tensor x, Tensor[] y, Tensor(a!)? z, SymInt w) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl(
                "mylib::foo_generate_trivial_abstract_impl", "cpu", lib=lib
            )
            @torch._dynamo.disable
            def foo_impl(x, y, z, w):
                x + y[0] + w
                return

            def f(x, y, z, w):
                return torch.ops.mylib.foo_generate_trivial_abstract_impl(x, y, z, 2)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            w = torch.randn(3)
            args = (x, y, z, w)

            output = torch.compile(f, backend="eager", fullgraph=True)(*args)
            self.assertEqual(output, None)

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

    def test_user_defined_binop(self):
        class MyClass:
            def __init__(self, value):
                self.value = value

            def __radd__(self, other):
                return self.value + other

        def fn(x, c):
            y = x.shape[0] + c
            return x + y

        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=counts)

        x = torch.randn(3)
        c = MyClass(4)
        ref = fn(x, c)
        res = opt_fn(x, c)

        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(counts.op_count, """1""")
        else:
            self.assertExpectedInline(counts.op_count, """2""")

    def test_user_defined_iter(self):
        class Mod:
            def __init__(self) -> None:
                self.a = [torch.randn(2, 2), torch.randn(2, 2)]

            def __iter__(self):
                return iter(self.a)

        def f(mod):
            ret = []
            for x in mod:
                ret.append(x + 1)
            return ret

        mod = Mod()
        counts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=counts, fullgraph=True)
        ref = f(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        res = opt_fn(mod)
        self.assertTrue(same(ref, res))
        self.assertEqual(counts.frame_count, 1)

        mod.a.append(torch.randn(2, 2))
        # `for x in mod` is inlined, where iter(m.a) creates a guard on the list length of m.a
        # Mutating length of mod.a causes a re-compilation.
        ref2 = f(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        res2 = opt_fn(mod)
        self.assertTrue(same(ref2, res2))
        self.assertEqual(counts.frame_count, 2)

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

    def test_os_environ_get(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            if os.environ.get("OS_ENVIRON_TEST") == "1":
                return x + 1
            else:
                return x - 1

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, x + 1)
            self.assertEqual(cnts.frame_count, 1)
            os.environ["OS_ENVIRON_TEST"] = "0"
            res2 = fn(x)
            self.assertEqual(res2, x - 1)
            # Ensure re-compile if os.environ items updated
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_os_environ_set_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=False)
        def fn(x):
            x = x + 1
            os.environ["OS_ENVIRON_TEST"] = "0"
            return torch.sin(x)

        x = torch.ones(2, 3)
        try:
            original = os.environ.get("OS_ENVIRON_TEST", None)

            os.environ["OS_ENVIRON_TEST"] = "1"
            res1 = fn(x)
            self.assertEqual(res1, torch.sin(x + 1))
            self.assertEqual(os.environ["OS_ENVIRON_TEST"], "0")
            # Ensure we graph break on os.environ.__setitem__
            self.assertEqual(cnts.frame_count, 2)
        finally:
            if original is None:
                del os.environ["OS_ENVIRON_TEST"]
            else:
                os.environ["OS_ENVIRON_TEST"] = original

    def test_sys_modules(self):
        def fn(x, y):
            mod_a = sys.modules.get("aaaaaaaa")
            assert mod_a is None  # noqa: S101
            assert "bbbbbbbb" not in sys.modules  # noqa: S101

            assert "operator" in sys.modules  # noqa: S101
            operator = sys.modules["operator"]
            builtins = sys.modules.get("builtins")
            operator2 = sys.modules.get("cccccccc", operator)

            return operator.add(x, y), operator2.neg(builtins.abs(x))

        torch._dynamo.testing.standard_test(self, fn, 2, expected_ops=3)

        x = torch.randn(10, 10)
        _, guards = torch._dynamo.export(fn, x, x)
        guard_code = []
        for guard in guards:
            if guard.code_list:
                guard_code += guard.code_list

        # Filter out id-matches that won't reproduce run to run
        guard_code = filter(
            lambda line: "id" not in line and "lookup_backend" not in line,
            guard_code,
        )
        guard_code_str = "\n".join(guard_code)

        # Make sure that the dict_contains are present in the order of added
        self.assertExpectedInline(
            guard_code_str,
            """\
L['x'].size()[1] == L['x'].size()[0]
L['x'].storage_offset() == 0
2 <= L['x'].size()[0]
utils_device.CURRENT_DEVICE == None
str(L['x'].dtype) == 'torch.float32'
str(L['x'].device) == 'cpu'
L['x'].requires_grad == False
L['x'].ndimension() == 2
hasattr(L['x'], '_dynamo_dynamic_indices') == False
L['x'] is L['y']
not ___dict_contains('aaaaaaaa', G['sys'].modules)
not ___dict_contains('bbbbbbbb', G['sys'].modules)
___dict_contains('operator', G['sys'].modules)
not ___dict_contains('cccccccc', G['sys'].modules)""",
        )

    def test_fold(self):
        def fn(a):
            return a + math.sqrt(63)

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

    def test_getattr_dict(self):
        def fn(x):
            from torch.masked.maskedtensor._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

            return x * len(_MASKEDTENSOR_FUNCTION_TABLE)

        i = torch.randn(5)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(i)
        self.assertEqual(r1, r2)

    def test_tensor_hasattr(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            if hasattr(x, "test"):
                return x + 2
            else:
                return x + 1

        self.assertEqual(torch.ones(2, 2) + 1, fn(torch.ones(2, 2)))

        inp = torch.ones(2, 2)
        inp.test = None
        self.assertEqual(torch.ones(2, 2) + 2, fn(inp))

    def test_tensor_call_obj_hasattr_view(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            output3 = getattr(x, "view", None)(10)
            return output3

        x = torch.randn(10)
        self.assertEqual(x.view(10), fn(x))

    def test_mro_type_tensor_no_source(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            z = []
            input_type = type(torch.ones(2, 2))
            for cls in input_type.__mro__:
                z.append(cls.__name__)

            return x, input_type, z

        inp = torch.ones(2, 2)
        fn(inp)

    def test_tensor_dynamic_method(self):
        def add_one(x):
            return x + 1

        t = torch.nn.Parameter(torch.ones(1))
        t.add_one = add_one

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            return t.add_one(t) + x

        result = fn(torch.ones(1))
        self.assertEqual(torch.ones(1) + 2, result)

    def test_known_tensor_methods_traced(self):
        # Verify that known tensor methods (in all_tensor_attrs) are still
        # traced into the graph via the generic proxy path.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x.abs().cos()

        result = fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_tensor_subclass_method_traced(self):
        # Methods defined on the actual tensor class (including dynamically
        # added ones) should be proxied through the generic call_method path,
        # not graph-broken.  This validates that the guard uses the concrete
        # class_type rather than the static all_tensor_attrs dict.
        def _dynamo_test_method(self):
            return self + 1

        with unittest.mock.patch.object(
            torch.Tensor, "_dynamo_test_method", _dynamo_test_method, create=True
        ):
            cnt = CompileCounterWithBackend("eager")

            @torch.compile(backend=cnt)
            def fn(x):
                y = x._dynamo_test_method()
                return y + 1

            result = fn(torch.randn(4))
            self.assertEqual(cnt.frame_count, 1)
            # Verify _dynamo_test_method appears as a call_method in the FX graph
            call_method_targets = [
                n.target for n in cnt.graphs[0].graph.nodes if n.op == "call_method"
            ]
            self.assertIn("_dynamo_test_method", call_method_targets)

    def test_unknown_tensor_method_graph_break(self):
        # Truly unknown methods raise AttributeError during tracing at
        # LOAD_ATTR time (dynamic_getattr), ensuring dynamo does not
        # silently proxy them into the compiled graph.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            y = x._nonexistent_test_method_xyz()
            return y + 1

        with self.assertRaises(AttributeError):
            fn(torch.randn(4))

    def test_shape_unpack(self):
        def fn(x):
            a, b = x.size()
            return x * b

        i = torch.randn(5, 10)
        r1 = fn(i)
        opt_fn = torch.compile(fn, backend="eager")
        r2 = opt_fn(i)
        self.assertTrue(same(r1, r2))

    def test_typing_dict(self):
        def fn(d):
            return d[T]

        d = {T: torch.randn(3)}
        r1 = fn(d)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r2 = opt_fn(d)
        self.assertEqual(r1, r2)

    def test_tensor__iter__(self):
        def fn(x):
            it = x.__iter__()
            for y in it:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_iter(self):
        def fn(x):
            for y in x:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_share_memory(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 64
                self.num_layers = 2

            def forward(self, x):
                batch_size = x.size(0)
                h = torch.zeros(
                    self.num_layers, batch_size, self.hidden_size
                ).share_memory_()
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                return x + h.sum() + c.sum()

        model = Model()
        x = torch.randn(4, 10)
        expected = model(x)
        compiled_model = torch.compile(model, fullgraph=False, backend="eager")
        actual = compiled_model(x)
        self.assertEqual(expected, actual)

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

    def test_min_max_over_iterable(self):
        def get_test_fn(func):
            def _fn(a, b, func=func):
                # try all of list, iterator, tuple, vararg.
                lst = [a.shape[0] + 1, 8, a.shape[0]]
                x = func(lst)
                y = func(iter(lst))
                z = func(tuple(lst))
                w = func(*lst)
                return a + (x + y + z + w)

            return _fn

        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=min),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 7),
        )
        torch._dynamo.testing.standard_test(
            self,
            get_test_fn(func=max),
            2,
            expected_ops=1,
            expected_ops_dynamic=ifdynstaticdefault(1, 7),
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_bound_shape_checks(self):
        def f1(x, y):
            b = x.item()
            torch._check(b >= 0)
            torch._check(b < y.shape[0])
            return y[:b]

        fn1 = torch.compile(f1, fullgraph=True, backend="eager")
        fn1(torch.tensor(4), torch.ones(10))

        def f2(x, index):
            idx = index.item()
            torch._check(idx >= 0)
            torch._check(idx < x.size(0))
            return x[idx]

        A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index = torch.tensor(1, dtype=torch.int64)
        fn2 = torch.compile(f2, fullgraph=True, backend="eager")
        fn2(A, index)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_arange_length_with_float32_dtype(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            y = x.item()
            r = torch.arange(y, dtype=torch.float32)

            if r.size(0) == y:
                return r + 1

            return r

        x = torch.tensor([300])
        r = f(x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check_symbolic_shape_rel(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(x.shape[0] == 1)
            torch._check(x.shape[0] != 2)
            torch._check(x.shape[0] >= 0)
            torch._check(x.shape[0] > 0)
            torch._check(x.shape[0] < 4)
            torch._check(x.shape[0] <= 3)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_torch_check_nonnegative(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            # Cannot conditional on unbacked SymInt
            if y == 0:
                assert False  # noqa: B011, S101
            else:
                return torch.arange(0, y)

        self.assertRaises(torch._dynamo.exc.UserError, lambda: f(torch.tensor([3])))

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
