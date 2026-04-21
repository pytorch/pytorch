# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: F403,F405,F841,PGH004
try:
    from .test_misc import *
except ImportError:
    from test_misc import *

from torch._dynamo.eval_frame import _debug_get_cache_entry_list


class CompileSemanticsTests(torch._inductor.test_case.TestCase):
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
            out = torch.compile(torch.add, backend=backend, fullgraph=False)(x, x)
        self.assertEqual(out.sum().item(), 5.0)
        self.assertEqual(len(backend.graphs), 0)

        with YoloMode():
            with self.assertRaisesRegex(RuntimeError, "found no compiled frames"):
                torch.compile(torch.add, backend=backend, fullgraph=True)(x, x)

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
                out = torch.compile(func, backend=backend, fullgraph=False)(
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

    def test_fold(self):
        def fn(a):
            return a + math.sqrt(63)

        torch._dynamo.testing.standard_test(self, fn, 1, expected_ops=1)

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

    def test_optimize_on_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def custom_member(self):
                # Just for checking that Dynamo returned mod object can redirect
                # to this method
                pass

            def forward(self, x):
                return self.relu(x)

        cnts1 = torch._dynamo.testing.CompileCounter()
        mod = MockModule()
        optimized_mod = torch.compile(mod, backend=cnts1, fullgraph=True)

        a = torch.randn(10)
        ref = mod(a)
        res = optimized_mod(a)

        optimized_mod.custom_member()

        self.assertTrue(same(ref, res))

    @skipIfWindows(
        msg="TODO(xuhancn): confirm, AssertionError: tensor([0.0290, 0.4019, 0.2598, 0.3666]) is not None"
    )
    def test_release_input_memory(self):
        x = torch.rand([4])
        x_ref = weakref.ref(x)

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def foo(x):
            return x + x

        out = foo(x)
        self.assertTrue(same(out, x + x))
        del x
        self.assertIs(x_ref(), None)

    @skipIfWindows(
        msg="TODO: (xuhancn) conform, AssertionError: Linear(in_features=10, out_features=10, bias=True) is not None"
    )
    def test_release_module_memory(self):
        mod = torch.nn.Linear(10, 10)
        x = torch.rand([10, 10])
        mod_weight_ref = weakref.ref(mod.weight)
        mod_ref = weakref.ref(mod)

        # Modules that are passed into torch._dynamo optimized functions
        # will normally be held onto through the generated GraphModule,
        # which contains the modules. remove the reference in this backend
        # and test that no additional references are being held.
        class NoLeakBackend:
            def __call__(self, gm: torch.fx.GraphModule, example_inputs):
                gm.mod = None

                def foo(*args, **kwargs):
                    return (1,)

                return foo

        no_leak_backend = NoLeakBackend()

        @torch.compile(backend=no_leak_backend)
        def foo(mod, x):
            return mod(x)

        foo(mod, x)
        del mod
        del x
        self.assertIsNone(mod_ref(), None)
        self.assertIsNone(mod_weight_ref(), None)

    @skipIfWindows(msg="TODO: (xuhancn) conform, AssertionError: False is not true")
    def test_release_scope_memory(self):
        def inner(y):
            y

        inner = torch.compile(inner, backend="eager")

        p_ref = None

        x = torch.randn((10, 10))
        inner(x)

        p_ref = weakref.ref(x)
        self.assertTrue(p_ref() is not None)
        del x
        self.assertTrue(p_ref() is None)

    def test_update_locals_and_stack_uses_shared_cache(self):
        def fn(x):
            perm = [0, 3, 5]
            perm = list(range(min(perm))) + perm
            perm.extend(i for i in range(x.dim()) if i not in perm)
            return perm

        x = torch.rand([2, 2, 2, 2, 2, 2])
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_side_effects_codegen_update_mutated(self):
        # codegen to update mutated variables with side effect
        # should after stack value's codegen
        def f1(x):
            alist = [x]
            alist.append(x + 1)
            alist[0].sum().item()  # graph break
            res = alist.pop()
            res.sum().item()  # graph break
            return res

        def f2(a, b):
            d = {"a": a + 1, "b": b + 2}
            x = d.pop("b")
            x.sum().item()  # graph break
            y = d["a"] + x
            y.sum().item()  # graph break
            d["c"] = y
            return d

        x = torch.rand([2, 3])
        a = torch.rand([5, 6])
        b = torch.rand([5, 6])
        res11 = f1(x)
        res21 = f2(a, b)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_f1 = torch.compile(f1, backend=cnts)
        opt_f2 = torch.compile(f2, backend=cnts)
        res12 = opt_f1(x)
        res22 = opt_f2(a, b)
        self.assertTrue(same(res11, res12))
        self.assertTrue(same(res21, res22))

    def _optimize_then_check_exp(
        self, foo, args, cnt, exp_out, exp_frame_count, exp_n_cached_backend
    ):
        opt_out = torch.compile(foo, backend=cnt)(*args)
        self.assertEqual(exp_out, opt_out)
        self.assertEqual(cnt.frame_count, exp_frame_count)

    def test_backend_match_guard(self):
        x = torch.randn([3, 4])

        def foo(x):
            return x.sin() + x.cos()

        def foo_graph_break(x):
            a = x.sin()
            torch._dynamo.graph_break()
            b = x.cos()
            return a + b

        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # We intentionally don't reset dynamo for each backend so that we can test
        # 1. dynamo doesn't recompile when backend stays the same, i.e. frame_count doesn't increase
        # 2. dynamo recompiles when backend changes, i.e. frame_count is non-zero for next backend
        def test_recompile(foo, *, exp_frame_count):
            eager_result = foo(x)
            for i, backend in enumerate(backends):
                cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
                # Run opt_f multiple times to make sure dynamo doesn't recompile.
                # Specifically, frame_count doesn't increase
                # the number of cached backends is i + 2 because we have the optimizing backend + None
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )
                self._optimize_then_check_exp(
                    foo, (x,), cnt, eager_result, exp_frame_count, i + 2
                )

        test_recompile(foo, exp_frame_count=1)
        torch._dynamo.reset()
        test_recompile(foo_graph_break, exp_frame_count=2)

    def test_multithread_compile_dynamic(self):
        def f(x):
            comptime.assert_static(x.shape[0])
            return x * x

        def _do_test(func):
            success = True

            def run(offset):
                for i in range(20):
                    print(func(torch.randn(i * 2 + offset)))

            t1 = threading.Thread(target=run, args=[0])
            t2 = threading.Thread(target=run, args=[1])

            def exc_hook(x):
                nonlocal success
                success = False

            try:
                threading.excepthook = exc_hook
                t1.start()
                t2.start()

                t1.join()
                t2.join()
            finally:
                threading.excepthook = threading.__excepthook__
            self.assertTrue(success)

        _do_test(torch.compile(f, backend="eager", dynamic=False))
        torch._dynamo.reset()

        f_opt = torch.compile(f, backend="eager")

        def g(x):
            with torch._dynamo.config.patch(
                automatic_dynamic_shapes=False, assume_static_by_default=True
            ):
                f_opt(x)

        _do_test(g)

    def test_backend_match_guard_multi_threads(self):
        x = torch.randn([3, 4])

        def foo(x):
            return x.sin() + x.cos()

        def compile_then_check_exp(foo, args, cnt, eager_result, exp_frame_count):
            for i in range(3):
                opt_out = torch.compile(foo, backend=cnt)(*args)
                self.assertEqual(opt_out, eager_result)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            thread_success[threading.current_thread()] = True

        eager_record_backend = torch._dynamo.testing.EagerAndRecordGraphs()
        backends = [eager_record_backend, "eager"]

        # Test dynamo recompiles but only caches a single backend for each thread
        eager_result = foo(x)
        # cnt and None
        exp_frame_count = 1
        threads = []
        thread_success = {}
        for i, backend in enumerate(backends):
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            thread = threading.Thread(
                target=compile_then_check_exp,
                args=(
                    foo,
                    (x,),
                    cnt,
                    eager_result,
                    exp_frame_count,
                ),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        self.assertEqual(len(thread_success), len(threads))

    def test_dynamo_min_operator_with_shape(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x, a):
            return min(x.shape[0], a)

        result = f(torch.ones(6), 3)
        self.assertEqual(result, 3)

    def test_cond(self):
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch.compile(f, backend="eager")
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.cos(torch.tensor([0.25, 0.25])), a))
        b = opt_fn(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), b))

    def test_cond_with_quantization(self):
        from functorch.experimental.control_flow import cond

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                example_inputs = (torch.randn(5, 5),)
                self.model = torch.nn.Linear(5, 5)
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )

            def forward(self, pred, x):
                def true_fn(x):
                    return x.sin() + self.quantized_model(x)

                def false_fn(x):
                    return x.cos() + self.model(x)

                return cond(pred, true_fn, false_fn, [x])

        module = MyModule()
        opt_m = torch.compile(module, backend="eager", fullgraph=True)
        x = torch.rand((5, 5))
        pred = torch.tensor(True)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))
        pred = torch.tensor(False)
        self.assertTrue(same(module(pred, x), opt_m(pred, x)))

    def test_map_with_quantization(self):
        from functorch.experimental.control_flow import map

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                example_inputs = (torch.randn(5, 5),)
                self.model = torch.nn.Linear(5, 5)
                self.quantized_model = prepare_qat_fx(
                    self.model, qconfig_dict, example_inputs=example_inputs
                )

            def forward(self, x):
                def body(x):
                    return x.sin() + self.quantized_model(x)

                return map(body, x)

        module = MyModule()
        opt_m = torch.compile(module, backend="eager", fullgraph=True)
        x = torch.rand((5, 5))
        self.assertTrue(same(module(x), opt_m(x)))

    def test_cond_side_effects(self):
        from functorch.experimental.control_flow import cond

        c = 0

        def true_fn(x):
            return x - c

        def false_fn(x):
            return x + c

        def f(pred, x):
            nonlocal c
            c = 1
            return cond(pred, true_fn, false_fn, [x])

        opt_fn = torch.compile(f, backend="eager")
        c = 0
        a = opt_fn(torch.tensor(False), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([1.25, 1.25]), a))

    def test_map_side_effects(self):
        from functorch.experimental.control_flow import map

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.tensor(1)

            def forward(self, xs):
                def body(x):
                    self.w += 1
                    return x

                return map(body, xs)

        mod = Module()

        error_message = r"Higher Order Operator: torch\.ops\.higher_order\.map_impl"

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError, error_message
        ):
            opt_fn = torch.compile(mod, backend="eager", fullgraph=True)
            opt_fn(torch.randn(3, 2))

    def test_cond_nested(self):
        from functorch.experimental.control_flow import cond

        def true_fn_nested(x):
            return x * 10

        def false_fn_nested(x):
            return x * -1

        def true_fn(pred2, x):
            return x.sin()

        def false_fn(pred2, x):
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        def f(pred, pred2, x):
            return cond(pred, true_fn, false_fn, [pred2, x])

        cc = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(f, backend=cc)
        true_true_sin = opt_fn(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        true_false_sin = opt_fn(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        false_true_sum_mult = opt_fn(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        false_false_sum_neg = opt_fn(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x
        self.assertTrue(cc.frame_count, 2)

    def test_cond_export(self):
        from functorch.experimental.control_flow import cond

        def true_fn_nested(x):
            return x * 10

        def false_fn_nested(x):
            return x * -1

        def true_fn(pred2, x):
            return x.sin()

        def false_fn(pred2, x):
            return x + cond(pred2, true_fn_nested, false_fn_nested, [x])

        def f(pred, pred2, x):
            return cond(pred, true_fn, false_fn, [pred2, x])

        graph, guard = torch._dynamo.export(f)(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        true_true_sin = graph(
            torch.tensor(True), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_true_sin))

        true_false_sin = graph(
            torch.tensor(True), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(same(torch.sin(torch.tensor([0.25, 0.25])), true_false_sin))

        false_true_sum_mult = graph(
            torch.tensor(False), torch.tensor(True), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([2.75, 2.75]), false_true_sum_mult)
        )  # * 10 then add x

        false_false_sum_neg = graph(
            torch.tensor(False), torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        self.assertTrue(
            same(torch.tensor([0.0, 0.0]), false_false_sum_neg)
        )  # * -1 then add x

    def test_cond_export_single_arg(self):
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            return x.clone()

        def false_fn(x):
            return x.sin()

        def f(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        graph, guard = torch._dynamo.export(f)(
            torch.tensor(False), torch.tensor([0.25, 0.25])
        )
        true_mirror = graph(torch.tensor(True), torch.tensor([0.25, 0.25]))
        self.assertTrue(same(torch.tensor([0.25, 0.25]), true_mirror))
        true_mirror_2 = graph(torch.tensor(True), torch.tensor([0.33, 0.33, 0.33]))
        self.assertTrue(same(torch.tensor([0.33, 0.33, 0.33]), true_mirror_2))

        false_sin = graph(torch.tensor(False), torch.tensor([0.5, 0.5]))
        self.assertTrue(same(torch.sin(torch.tensor([0.5, 0.5])), false_sin))

    def test_duplicate_graph_break_log(self):
        torch._logging.set_logs(graph_breaks=True)

        @torch.compile(backend="eager")
        def f1(a, b):
            try:
                f2(a, b)
            finally:
                pass

        def f2(a, b):
            c = a + b
            print("break")
            return a + b + c

        @torch.compile(backend="eager")
        def g1(a, b):
            try:
                g2(a, b)
            finally:
                pass

        def g2(a, b):
            c = a + b
            print("break")
            return a + b + c

        def count_graph_break_msgs(msgs):
            return sum("Graph break in user code" in msg for msg in msgs)

        with (
            self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log,
            torch._dynamo.config.patch(verbose=True),
        ):
            f1(torch.randn(10), torch.randn(10))
            self.assertGreater(count_graph_break_msgs(log.output), 1)

        with (
            self.assertLogs(logger="torch._dynamo", level=logging.DEBUG) as log,
            torch._dynamo.config.patch(verbose=False),
        ):
            g1(torch.randn(10), torch.randn(10))
            self.assertEqual(count_graph_break_msgs(log.output), 1)

        # reset logging state
        torch._logging.set_logs()

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_repro_graph_breaks_in__get_item_by_idx(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = torch.nn.Sequential(
                    torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
                )

            def forward(self, x):
                return self.mod[0](x)

        m = Mod()
        graph, _ = torch._dynamo.export(m)(torch.randn(3, 3))

    def test_error_on_nested_fx_trace(self):
        input = torch.rand(2, 3)

        def f(x):
            x + x

        real = f(input)

        optimized = torch.compile(f, backend="eager")
        self.assertTrue(same(optimized(input), real))

        with self.assertRaisesRegex(RuntimeError, "Detected that you are using FX"):
            gm = torch.fx.symbolic_trace(optimized)

    def test_not_dynamic_scope(self):
        def f(y):
            x = 1

            def g():
                x = 2
                return lambda: x

            return y + g()()

        input = torch.zeros(1)
        real = f(input)
        optimized = torch.compile(f, backend="eager")
        opt = optimized(input)
        self.assertTrue(same(opt, real))

    def test_inference_mode(self):
        @torch.inference_mode()
        def func(x, y):
            return x.add(1.0) + y

        x = torch.ones(4, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        ref = func(x, y)
        opt_func = torch.compile(func, backend="eager")

        x1 = torch.ones(4, requires_grad=True)
        res = opt_func(x1, y)
        self.assertTrue(same(ref, res))
        self.assertTrue(same(x, x1))

    def test_inference_mode_param(self):
        def fn(x):
            p = torch.nn.Parameter(x, requires_grad=False)
            return x * p

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        with torch.inference_mode():
            x = torch.rand(4)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_unpack_tensor_shape_mismatch(self):
        @torch.compile(backend="eager")
        def f1(x):
            a, b = x
            return torch.sin(a + b)

        x = torch.tensor(2.0)
        with self.assertRaisesRegex(AssertionError, "Can't unpack scalar tensors"):
            f1(x)

        x = torch.tensor([2.0])
        with self.assertRaisesRegex(
            AssertionError, "Can't unpack a tensor of 1 rows into a tuple of 2 elements"
        ):
            f1(x)

        @torch.compile(backend="eager")
        def f2(x):
            (a,) = x
            return torch.sin(a + 1)

        x = torch.tensor(2.0)
        with self.assertRaisesRegex(AssertionError, "Can't unpack scalar tensors"):
            f2(x)

        x = torch.tensor([2.0])
        self.assertTrue(same(f2(x), torch.sin(x[0] + 1)))

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

    def test_torch_compile_ctx_on_forward_and_training_step(self):
        class MyModel(torch.nn.Module):
            def forward(self): ...

            def training_step(self):
                self()

        model = MyModel()
        compiled_model = torch.compile(model, backend="eager")

        model.forward = compiled_model.dynamo_ctx(model.forward)
        model.training_step = compiled_model.dynamo_ctx(model.training_step)

        model.training_step()

    def test_error_on_recompile(self):
        @torch.compile(backend="eager")
        def fn(a, b):
            return a + b

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            with self.assertRaises(torch._dynamo.exc.RecompileError):
                fn(torch.rand(2, 3), torch.rand(2, 3))
                fn(torch.rand(2, 3), (1, 2, 3))

    def test_dynamo_compiling_fake_tensor_to_vararg_int(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                # use numpy int so it's wrapped as fake tensor in dynamo
                shape = np.int_(16)
                # test shape as fake tensor, which param type is
                # Sequence[Union[_int, SymInt]]
                return x.reshape(shape)

        x = torch.rand([4, 4])
        model = MyModule()
        orig_out = model(x)
        opt_model = torch.compile(MyModule(), backend="eager")
        opt_out = opt_model(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_compile_with_userland_fake_tensor_mode(self):
        # Test that torch.compile works when called inside a user's FakeTensorMode.
        # The user's fake tensors should be "refakified" to Dynamo's fake mode.
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            model = torch.nn.Linear(4, 4)
            inp = torch.rand(4, 4)
            loss = torch.compile(model, backend="aot_eager")(inp).sum()
            loss.backward()

    @skipIfWindows(
        msg="TypeError: sequence item 0: expected str instance, NoneType found"
    )
    def test_funcname_cache(self):
        src = """\
import torch
if True:
    test = 3

class AAA:
    class DUMMY:
        class DUMMY2:
            pass

    def dummy(self):
        def dummy2():
            pass
    class BBB:
        @staticmethod
        def CCC():
            class DDD:
                if True:
                    @staticmethod
                    def EEE():
                        x = [torch.ones(3, 3) for _ in range(5)]
                        return x
            return DDD
def fn():
    return 3
"""
        with WritableTempFile(mode="w") as f:
            f.write(src)
            f.flush()
            from torch._dynamo.funcname_cache import get_funcname

            names = [get_funcname(f.name, i + 1) for i in range(src.count("\n") + 1)]

        self.assertExpectedInline(
            "\n".join(names),
            """\




AAA
AAA.DUMMY
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.DUMMY.DUMMY2
AAA.dummy
AAA.dummy.dummy2
AAA.dummy.dummy2
AAA.BBB
AAA.BBB
AAA.BBB.CCC
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC.DDD.EEE
AAA.BBB.CCC
fn
fn
""",
        )

    def test_dynamo_reset_clears_cache(self):
        """Test that dynamo bytecode cache is freed
        when dynamo reset is called
        """

        def fn(x):
            return torch.sin(x)

        opt_fn = torch.compile(backend="eager")(fn)
        opt_fn(torch.randn(3, 3))

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 1)

        torch._dynamo.reset()
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c2), 0)

    def test_312_binary_slice_with_graph_break1(self):
        l1 = torch.nn.Linear(5, 5)
        l2 = torch.nn.Linear(5, 5)

        def fn(x):
            # causes a graph break with items in the stack
            n = torch.nn.Sequential(l1, l2)
            out = n[1:](x)
            return out

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(5, 5))

    def test_312_binary_slice_with_graph_break2(self):
        class Foo:
            def __setitem__(self, key, val):
                pass

            def __getitem__(self, key):
                torch._dynamo.graph_break()
                return 1

        foo = Foo()

        def fn(x):
            # graph break in a STORE_SLICE instruction
            foo[:] = x
            # graph break in BINARY_SLICE with has_backedge check
            x = x + foo[:]
            if x is None:
                x = x + 1
            else:
                x = x + 1
            return x

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(5, 5))

    def test_load_fast_and_clear_graph_break(self):
        # Can result in a segfault in 3.12+ if LOAD_FAST_AND_CLEAR
        # is not handled properly in a graph break
        def fn():
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            torch._dynamo.graph_break()
            out = torch.cat([torch.randn(r, 5) for r in range(3)])
            return out

        self.assertEqual(torch.compile(fn, backend="eager")().shape, (3, 5))

    def test_raises_importerror1(self):
        @torch.compile(backend="eager")
        def fn(x):
            try:
                import some_module_that_surely_does_not_exist

                return
            except ImportError:
                pass
            return x.sin()

        x = torch.randn(8)
        self.assertEqual(fn(x), x.sin())

    def test_raises_importerror2(self):
        @torch.compile(backend="eager")
        def fn(x):
            import some_module_that_surely_does_not_exist

            return x + 1

        x = torch.randn(8)
        with self.assertRaises(ImportError):
            fn(x)

    def test_dynamo_cache_move_to_front(self):
        def fn(x, const):
            return x + const

        # dynamic=False forces Dynamo to recompile
        opt_fn = torch.compile(fn, backend="eager", dynamic=False)

        inp = torch.randn(3, 3)

        # NOTE: assumes that each cache entry is guarded
        # on unique Mod instance
        opt_fn(inp, 1)
        opt_fn(inp, 2)
        opt_fn(inp, 3)

        c1 = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(c1), 3)

        # move cache entry to front
        opt_fn(inp, 2)
        c2 = _debug_get_cache_entry_list(fn.__code__)
        self.assertIs(c1[1], c2[0])

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

    def test_builtin_constant_fold_str_conversions(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            s = hex(255) + oct(8) + bin(3) + ascii("hello") + format(42, "x")
            return x + len(s)

        x = torch.randn(4)
        res = fn(x)
        expected = hex(255) + oct(8) + bin(3) + ascii("hello") + format(42, "x")
        self.assertEqual(res, x + len(expected))

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

    def test_constant_subclass_guard_recompiles(self):
        class MyInt(int):
            def __eq__(self, other):
                raise RuntimeError("should not be called during guard check")

        class MyFloat(float):
            def __eq__(self, other):
                raise RuntimeError("should not be called during guard check")

        class MyStr(str):
            def __eq__(self, other):
                raise RuntimeError("should not be called during guard check")

        cnt = torch._dynamo.testing.CompileCounter()

        # int subclass
        @torch.compile(backend=cnt)
        def f(x, y):
            return x + y

        r1 = f(torch.tensor(1), MyInt(5))
        self.assertEqual(r1.item(), 6)
        self.assertEqual(cnt.frame_count, 1)

        r2 = f(torch.tensor(1), MyInt(10))
        self.assertEqual(r2.item(), 11)
        self.assertEqual(cnt.frame_count, 2)

        r3 = f(torch.tensor(1), MyInt(5))
        self.assertEqual(r3.item(), 6)
        self.assertEqual(cnt.frame_count, 2)

        # float subclass
        cnt.clear()

        @torch.compile(backend=cnt)
        def g(x, y):
            return x + y

        r4 = g(torch.tensor(1.0), MyFloat(2.5))
        self.assertEqual(r4.item(), 3.5)
        self.assertEqual(cnt.frame_count, 1)

        r5 = g(torch.tensor(1.0), MyFloat(3.5))
        self.assertEqual(r5.item(), 4.5)
        self.assertEqual(cnt.frame_count, 2)

        r6 = g(torch.tensor(1.0), MyFloat(2.5))
        self.assertEqual(r6.item(), 3.5)
        self.assertEqual(cnt.frame_count, 2)

        # str subclass
        cnt.clear()

        @torch.compile(backend=cnt, fullgraph=True)
        def h(x, s):
            return x + len(s)

        r7 = h(torch.tensor(1), MyStr("abc"))
        self.assertEqual(r7.item(), 4)
        self.assertEqual(cnt.frame_count, 1)

        r8 = h(torch.tensor(1), MyStr("abcde"))
        self.assertEqual(r8.item(), 6)
        self.assertEqual(cnt.frame_count, 2)

        r9 = h(torch.tensor(1), MyStr("abc"))
        self.assertEqual(r9.item(), 4)
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
