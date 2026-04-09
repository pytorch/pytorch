# Owner(s): ["module: dynamo"]
# flake8: noqa: F403,F405

try:
    from ._higher_order_ops_test_utils import *
except ImportError:
    from _higher_order_ops_test_utils import *


class HigherOrderOpTests(torch._dynamo.test_case.TestCaseWithNestedGraphBreaks):
    def _assert_wrap_fallback(self, func, args, setup=lambda: None):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        setup()
        expected = func(*args)
        setup()
        result = torch.compile(func, backend=cnt, fullgraph=False)(*args)
        num_graph_breaks = len(counters["graph_break"].keys())
        self.assertGreater(num_graph_breaks, 0)

        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(node.target is wrap)

        self.assertEqual(result, expected)

    def _test_wrap_simple(
        self,
        func,
        args_generator,
        expected_num_wrap_args,
        expected_opcount=2,
        return_graph=False,
    ):
        # Given a `func` that has a single call to `wrap`,
        # we check that:
        # - there are no graph breaks
        # - eager vs torch.compile has the same result (correctness)
        # - other compilation metrics, e.g, # of ops in the dynamo captured graph,
        #   the wrap has the expected number of args, etc
        #
        # we have one or multiple runs through with each of the args from args_generator,
        # and we will check:
        # - correctness and no graph breaks for every run
        # - other compilation metrics only for the first run, since automatic_dynamic_shapes
        #   may compile another dynamic version graph for the later runs
        graph = None
        for i, args in enumerate(args_generator):
            backend = EagerAndRecordGraphs()
            cnt = CompileCounterWithBackend(backend)
            expected = func(*args)
            result = torch.compile(func, fullgraph=True, backend=cnt)(*args)
            # check correctness and no graph breaks
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(len(backend.graphs), 1)
            # check other compilation metrics
            if i == 0:
                self.assertEqual(cnt.op_count, expected_opcount)
                graph = backend.graphs[0]
                wrap_node = find_first_node(graph, wrap)
                self.assertEqual(len(wrap_node.args), expected_num_wrap_args)
        # We always return/check the graph from the first run if return_graph = True
        if return_graph:
            return normalize_gm(graph.print_readable(print_output=False))

    def test_error_message_sane(self):
        foo = []

        def inner(x):
            foo.append(x)
            return x.clone()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(inner, x)

        x = torch.randn(3)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "HOP: Unsafe side effect",
        ):
            f(x)

    def test_no_freevars(self):
        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_enum_arg(self):
        class SomeEnum(enum.Enum):
            A = 0
            B = 1

        def g(x, val):
            if val == SomeEnum.A:
                return torch.sin(x)
            return torch.cos(x)

        def f(x, val):
            return wrap(g, x, val)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x, SomeEnum.A)), arg_count)

    def test_return_captured_var(self):
        freevar = torch.randn(3)

        def test(x):
            return freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.

        # when testing with dynamic shape, symbols are lifted as input
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 1)

    def test_return_captured_vars(self):
        freevar1 = torch.randn(3)
        freevar2 = torch.randn(3)

        def test(x):
            return freevar1, freevar2, freevar1

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 1)

    def test_return_captured_var_used_multiple_times(self):
        freevar = torch.randn(3)

        def test(x):
            y = x + freevar
            return y, freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 2)

    def test_capture_untracked_global(self):
        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_allow_python_side_effects_utility(self):
        from torch._dynamo.utils import (
            _disable_side_effect_safety_checks_for_current_subtracer,
        )
        from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper

        def wrapper(fn):
            return fn

        count = 0

        def does_side_effect(x):
            nonlocal count
            count += 1
            return x.sin()

        def does_side_effect_wrapped(*args, **kwargs):
            return _disable_side_effect_safety_checks_for_current_subtracer(
                does_side_effect, *args, **kwargs
            )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return dynamo_bypassing_wrapper(wrapper, does_side_effect_wrapped, x)

        x = torch.tensor(1.0)
        fn(x)

        def inner_does_side_effect(x):
            nonlocal count
            count += 1
            return x

        # Test that any nested HOPs are unaffected
        def outer(x):
            return dynamo_bypassing_wrapper(wrapper, inner_does_side_effect, x)

        def outer_wrapped(*args, **kwargs):
            return _disable_side_effect_safety_checks_for_current_subtracer(
                outer, *args, **kwargs
            )

        @torch.compile(backend="eager", fullgraph=True)
        def fn_nested(x):
            return dynamo_bypassing_wrapper(wrapper, outer_wrapped, x)

        x = torch.tensor(1.0)
        with self.assertRaisesRegex(RuntimeError, "HOP: Unsafe side effect"):
            fn_nested(x)

    def test_symint_input(self):
        def f(x):
            i = x.size(0)
            return wrap(lambda x, i: x.view(i), x, i)

        x = torch.randn(3, 1)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
        )

    def test_symint_in_slice(self):
        def f(x):
            i = x.size(0) - 2
            j = x.size(1) - 3
            k = x.size(2)
            return wrap(lambda x: x[:i, :j, k:], x)

        x = torch.randn(3, 4, 5)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            # 3 basic symbols and 2 compound symbols
            ifdynstaticdefault(2, 7),
            # 2 more sym expression computation
            expected_opcount=ifdynstaticdefault(2, 4),
        )

    def test_wrap_pytree_args_nested(self):
        def f(x, y, z):
            def fn(d):
                return d["x"].sin() + d["y"][0].cos() - d["y"][1][2].sin()

            return wrap(fn, d)

        x = torch.tensor(1.5)
        y = torch.tensor(2.0)
        z = torch.tensor(3.0)
        d = {"x": x, "y": (y, [x, y, z])}

        def my_args_generator(t):
            yield t
            yield t[0] + 0.1, t[1], t[2]
            yield t[0], t[1] + 0.1, t[2]

        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator((x, y, z)),
            4,
            return_graph=True,
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_d_x_: "f32[]", L_d_y_0_: "f32[]", L_d_y_1_2_: "f32[]"):
        l_d_x_ = L_d_x_
        l_d_y_0_ = L_d_y_0_
        l_d_y_1_2_ = L_d_y_1_2_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);  wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None
        getitem: "f32[]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_d_x_: "f32[]", l_d_y_0_: "f32[]", l_d_y_1_2_: "f32[]"):
            sin: "f32[]" = l_d_x_.sin();  l_d_x_ = None
            cos: "f32[]" = l_d_y_0_.cos();  l_d_y_0_ = None
            add: "f32[]" = sin + cos;  sin = cos = None
            sin_1: "f32[]" = l_d_y_1_2_.sin();  l_d_y_1_2_ = None
            sub: "f32[]" = add - sin_1;  add = sin_1 = None
            return (sub,)
""",  # NOQA: B950
        )

    def test_wrap_pytree_args_with_symint_constant(self):
        def f(x, y):
            i = x.size(0)
            return wrap(lambda t: t[0].view(t[2]) + t[1], (x, y, i))

        x = torch.randn(3, 1)
        y = 0.5
        actual_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
            return_graph=True,
        )
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 1]"):
            view: "f32[3]" = l_x_.view(3);  l_x_ = None
            add: "f32[3]" = view + 0.5;  view = None
            return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_);  wrap_body_0 = s77 = l_x_ = None
        getitem: "f32[s77]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77, 1]"):
            view: "f32[s77]" = l_x_.view(s77);  l_x_ = s77 = None
            add: "f32[s77]" = view + 0.5;  view = None
            return (add,)
""",
            )

    def test_wrap_pytree_kwargs(self):
        def f(x, y, z):
            def fn(*, x, y, z):
                z1, _ = z
                return (x * 2) + y + z1

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        def my_args_generator(t):
            yield t
            x1 = t[0] + 0.1
            y1 = t[1] + 0.1
            yield (x1, y1, (x1, y1))
            x2 = t[0] + 0.2
            y2 = t[0] + 0.2
            yield (x2, y2, (x2, y2))

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, my_args_generator((x, y, (x, y))), arg_count)

    def test_wrap_pytree_args_not_const_symint_tensor(self):
        class MyClass:
            def __init__(self, x):
                self.val = x

        def f(x, y):
            return wrap(lambda z: z[0].sin() * z[1].val.cos(), (x, y))

        x = torch.tensor(1.2)
        y = MyClass(torch.tensor(3.4))
        self._test_wrap_simple(f, [(x, y)], 3)

    def test_capture_constants(self):
        x = torch.randn(3, 3)

        def fn(x, y, z):
            if z:
                return x + y
            return x * y

        def f(x, y, z):
            return wrap(fn, x, y, z)

        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

        # Ensure that we recompile here
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

    def test_capture_untracked_global_nested(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        x = torch.randn(3)
        result = f(x)

        self.assertEqual(result, x + global_var)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            def g(x):
                return wrap(lambda x: x + y, x)

            # when testing with dynamic shape, a symbol is lifted as input
            arg_count = ifdynstaticdefault(3, 4)
            self._test_wrap_simple(g, default_args_generator((x,)), arg_count)
            return g(x)

        f(x, y)

    def test_capture_tracked(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: x + y, x)

        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_capture_tracked_nested(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_inlined_functions(self):
        def g(x, y):
            return x + y

        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_same_freevar_twice(self):
        free = torch.randn(3)

        def g(x):
            y = free.sin()
            z = free.cos()
            return y, z

        def f(x):
            return wrap(g, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count, 3)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True,
    )
    def test_unbacked_symbol_closure(self):
        def f(x):
            c = x.sum().item()

            def g(x):
                def k(x):
                    return x + c

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        out_graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 4, return_graph=True
        )

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        item: "Sym(zuf0)" = sum_1.item();  sum_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, item);  wrap_body_1 = s77 = l_x_ = item = None
        getitem: "f32[s77]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", item: "Sym(zuf0)"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_, item);  wrap_body_0 = s77 = l_x_ = item = None
            getitem: "f32[s77]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", item: "Sym(zuf0)"):
                add: "f32[s77]" = l_x_ + item;  l_x_ = item = None
                return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        item: "Sym(zuf0)" = sum_1.item();  sum_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, item);  wrap_body_1 = l_x_ = item = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", item: "Sym(zuf0)"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, item);  wrap_body_0 = l_x_ = item = None
            getitem: "f32[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", item: "Sym(zuf0)"):
                add: "f32[3]" = l_x_ + item;  l_x_ = item = None
                return (add,)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_tensor_with_unbacked_shape_closure(self):
        def f(x):
            c = x.nonzero()

            def g(x):
                def k(x):
                    return x.sin(), c.sin()

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(4, 5)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0
        expected_op_count = ifdynstaticdefault(9, 7)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77]"):
        l_x_ = L_x_

        c: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, sym_size_int_1, c);  wrap_body_1 = s77 = l_x_ = sym_size_int_1 = c = None
        getitem: "f32[s77]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", u0: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_, u0, c);  wrap_body_0 = s77 = l_x_ = u0 = c = None
            getitem: "f32[s77]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", u0: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[s77]" = l_x_.sin();  l_x_ = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (sin, sin_1)
""",
            )
        else:
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        c: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int_1 <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, sym_size_int_1, c);  wrap_body_1 = l_x_ = sym_size_int_1 = c = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", u0: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, u0, c);  wrap_body_0 = l_x_ = u0 = c = None
            getitem: "f32[3]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", u0: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[3]" = l_x_.sin();  l_x_ = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (sin, sin_1)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
        capture_scalar_outputs=True,
    )
    def test_tensor_to_list_closure(self):
        def f(x):
            li = x.tolist()

            def g(x):
                def k(x):
                    return li[0] + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.tensor([1, 2, 3], dtype=torch.int16)
        arg_count = ifdynstaticdefault(3, 3)
        out_graph = self._test_wrap_simple(f, ((x,),), arg_count, 4, return_graph=True)

        # tolist will specialize on input shapes, so dynamic and static tests
        # have the same graph
        self.assertExpectedInline(
            out_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "i16[3]"):
        l_x_ = L_x_

        getitem = l_x_[0]
        item: "Sym(u0)" = getitem.item();  getitem = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, item, l_x_);  wrap_body_1 = item = l_x_ = None
        getitem_3: "i16[3]" = wrap[0];  wrap = None
        return (getitem_3,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, item: "Sym(u0)", l_x_: "i16[3]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, item, l_x_);  wrap_body_0 = item = l_x_ = None
            getitem: "i16[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, item: "Sym(u0)", l_x_: "i16[3]"):
                add: "i16[3]" = item + l_x_;  item = l_x_ = None
                return (add,)
""",
        )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_tensor_and_unbacked_symbol_closure(self):
        def f(x):
            c = x.nonzero()
            sz = c.size(0)

            def g(x):
                def k(x):
                    return x.sin() + sz, c.sin()

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(4, 5)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0
        expected_op_count = ifdynstaticdefault(9, 7)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        # Note that u0 is accessed from sz and the shape of c
        # We cached via the symbol u0 and de-duplicate them.
        if not check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        c: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, sym_size_int, c);  wrap_body_1 = l_x_ = sym_size_int = c = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", size: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, size, c);  wrap_body_0 = l_x_ = size = c = None
            getitem: "f32[3]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", size: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[3]" = l_x_.sin();  l_x_ = None
                add: "f32[3]" = sin + size;  sin = size = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (add, sin_1)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_concat_unbacked_shape_tensor(self):
        def f(x, y):
            c = x.nonzero()
            d = y.nonzero()
            cat = torch.cat((c, d))

            def g(x):
                def k(x):
                    return cat.sum() + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        y = torch.randn(3)
        arg_count = ifdynstaticdefault(5, 6)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0 and u1
        expected_op_count = ifdynstaticdefault(15, 11)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        if not check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]", L_y_: "f32[3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        c: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int_2: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_2 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int_2 <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        d: "i64[u1, 1]" = l_y_.nonzero();  l_y_ = None
        sym_size_int_3: "Sym(u1)" = torch.ops.aten.sym_size.int(d, 0)
        ge_1: "Sym(u1 >= 0)" = sym_size_int_3 >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_2 = None
        le_1: "Sym(u1 <= 3)" = sym_size_int_3 <= 3
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u1 <= 3 on node 'le_1'");  le_1 = _assert_scalar_default_3 = None

        cat: "i64[u0 + u1, 1]" = torch.cat((c, d));  c = d = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, sym_size_int_2, sym_size_int_3, cat, l_x_);  wrap_body_1 = sym_size_int_2 = sym_size_int_3 = cat = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, u0: "Sym(u0)", u1: "Sym(u1)", cat: "i64[u0 + u1, 1]", l_x_: "f32[3]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, u0, u1, cat, l_x_);  wrap_body_0 = u0 = u1 = cat = l_x_ = None
            getitem: "f32[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, u0: "Sym(u0)", u1: "Sym(u1)", cat: "i64[u0 + u1, 1]", l_x_: "f32[3]"):
                sum_1: "i64[]" = cat.sum();  cat = None
                add: "f32[3]" = sum_1 + l_x_;  sum_1 = l_x_ = None
                return (add,)
""",
            )

    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        dynamic_shapes=True,
    )
    def test_lift_tensors_with_shared_symbols(self):
        def f(x, y):
            def g(x):
                def k(x):
                    return x @ y

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            6,
            2,
            return_graph=True,
        )
        self.assertExpectedInline(
            out_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]", s94: "Sym(s94)", L_y_: "f32[s27, s94]"):
        l_x_ = L_x_
        l_y_ = L_y_

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, s27, l_x_, s94, l_y_);  wrap_body_1 = s77 = s27 = l_x_ = s94 = l_y_ = None
        getitem: "f32[s77, s94]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", l_x_: "f32[s77, s27]", s94: "Sym(s94)", l_y_: "f32[s27, s94]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, s27, l_x_, s94, l_y_);  wrap_body_0 = s77 = s27 = l_x_ = s94 = l_y_ = None
            getitem: "f32[s77, s94]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", l_x_: "f32[s77, s27]", s94: "Sym(s94)", l_y_: "f32[s27, s94]"):
                matmul: "f32[s77, s94]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
                return (matmul,)
""",
        )

    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        dynamic_shapes=True,
        capture_dynamic_output_shape_ops=True,
    )
    def test_lift_tensors_with_compound_expressions(self):
        def f(x, y):
            x = x.view(-1, 2)
            c = y.nonzero()
            d = torch.concat((x, c))

            def g(x):
                def k(x):
                    return d.sum() + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        f(x, y)

        if not check_dynamic_shape_capture():
            out_graph = self._test_wrap_simple(
                f,
                default_args_generator((x, y)),
                6,
                9,
                return_graph=True,
            )
            self.assertExpectedInline(
                out_graph,
                """\
    class GraphModule(torch.nn.Module):
        def forward(self, s0: "Sym(s0)", s1: "Sym(s1)", L_x_: "f32[s0, s1]", s2: "Sym(s2)", L_y_: "f32[s1, s2]"):
            l_x_ = L_x_
            l_y_ = L_y_

            x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = l_x_.view(-1, 2);  l_x_ = None

            c: "i64[u0, 2]" = l_y_.nonzero();  l_y_ = None

            sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
            _check_is_size = torch._check_is_size(sym_size_int_1);  _check_is_size = None

            ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

            d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = torch.concat((x, c));  c = None

            wrap_body_1 = self.wrap_body_1
            wrap = torch.ops.higher_order.wrap(wrap_body_1, sym_size_int_1, s1, s0, d, x);  wrap_body_1 = sym_size_int_1 = s1 = s0 = d = x = None
            getitem: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_1(torch.nn.Module):
            def forward(self, u0: "Sym(u0)", s1: "Sym(s1)", s0: "Sym(s0)", d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]", x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]"):
                wrap_body_0 = self.wrap_body_0
                wrap = torch.ops.higher_order.wrap(wrap_body_0, u0, s1, s0, d, x);  wrap_body_0 = u0 = s1 = s0 = d = x = None
                getitem: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = wrap[0];  wrap = None
                return (getitem,)

            class wrap_body_0(torch.nn.Module):
                def forward(self, u0: "Sym(u0)", s1: "Sym(s1)", s0: "Sym(s0)", d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]", x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]"):
                    sum_1: "f32[]" = d.sum();  d = None
                    add: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = sum_1 + x;  sum_1 = x = None
                    return (add,)
    """,
            )

    def test_register_subclass(self):
        from torch._higher_order_ops.cond import cond_op
        from torch.testing._internal.two_tensor import TwoTensor

        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.randn(3)
        t = TwoTensor(a, b)

        prev_impl = cond_op.python_key_table.pop(TwoTensor, None)
        cond_op._dispatch_cache.clear()

        def restore_twotensor_impl():
            cond_op.python_key_table.pop(TwoTensor, None)
            if prev_impl is not None:
                cond_op.python_key_table[TwoTensor] = prev_impl
            cond_op._dispatch_cache.clear()

        self.addCleanup(restore_twotensor_impl)

        with self.assertRaisesRegex(
            NotImplementedError,
            "no rule registered for HOP cond and subclass .*TwoTensor'>",
        ):
            res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))

        called = 0

        # Using cond.py_impl
        @cond_op.py_impl(TwoTensor)
        def _(pred, true_fn, false_fn, operands):
            nonlocal called
            called += 1
            assert len(operands) == 1  # noqa: S101
            a = cond_op(pred, true_fn, false_fn, (operands[0].a,))
            b = cond_op(pred, true_fn, false_fn, (operands[0].b,))
            return TwoTensor(a, b)

        res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))
        self.assertEqual(res.a, torch.sin(a))
        self.assertEqual(res.b, torch.sin(b))
        self.assertEqual(called, 1)

    def test_register_mode(self):
        from torch._higher_order_ops.cond import cond_op

        torch_dispatch_called = 0

        class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nonlocal torch_dispatch_called
                torch_dispatch_called += 1
                return func(*args, **kwargs)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with self.assertRaisesRegex(
            NotImplementedError,
            "no rule registered for HigherOrderOperator cond and mode .*MyMode",
        ):
            with MyMode():
                res = cond_op(pred, torch.sin, torch.cos, (a,))

        py_impl_called = 0

        # Using cond.py_impl
        @cond_op.py_impl(MyMode)
        def _(mode, pred, true_fn, false_fn, operands):
            nonlocal py_impl_called
            py_impl_called += 1
            return cond_op(pred, true_fn, false_fn, operands)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with MyMode():
            res = cond_op(pred, torch.sin, torch.cos, (a,))
        self.assertEqual(res, a.sin())

    def test_capture_value_created_in_subgraph(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 1)

        # No changes to args of outer wrap
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # z was lifted to arg of inner wrap
        body_function = getattr(gm, wrap_node.args[0].name)
        # addition + wrap + getitem
        self.assertEqual(op_count(body_function), 3)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # Innermost body function: z was also lifted to arg
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_side_effect_set_new_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()

        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()
            global_obj.foo = nn.Parameter(torch.tensor(4.0))

        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()
            global_obj.foo = torch.tensor(4.0)

        def f(x):
            def h(x):
                def g(x):
                    del global_obj.foo
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_new_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                global_module.foo = nn.Parameter(x + 1)
                return x.clone()

            y = wrap(g, x)
            return y + global_module.foo

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                global_module.existing = nn.Parameter(torch.tensor(4.0))
                return global_module(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                del global_module.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_mutate_global_num(self):
        def setup():
            global global_num
            global_num = 3.14

        def f(x):
            def g(x):
                global global_num
                global_num = global_num + 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_num_builtin(self):
        def setup():
            global global_num
            global_num = 3.14

        def f(x):
            def g(x):
                global global_num
                global_num += 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor(self):
        def setup():
            global global_var
            global_var = torch.ones(3)

        def f(x):
            def g(x):
                global global_var
                global_var = global_var + 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor_builtin(self):
        def setup():
            global global_var
            global_var = torch.ones(3)

        def f(x):
            def g(x):
                global global_var
                global_var += 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_list(self):
        def setup():
            global global_list
            global_list = []

        def f(x):
            def g(x):
                val = x + 1
                global_list.append(val)
                return global_list[-1]

            y = wrap(g, x)
            z = y + global_list[-1]
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_nonlocal_num(self):
        def f(x):
            def h(x):
                val = 1

                def g(x):
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()

                def g(x):
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_existing_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()
                obj.val = 3

                def g(x):
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_del_existing_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()
                obj.val = 3

                def g(x):
                    del obj.val
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_module(self):
        def h(x):
            obj = MyModule()

            def g(x):
                obj.val = x.dim()
                return x.clone()

            y = wrap(g, x)
            z = y + obj.val
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_set_existing_attr_nonlocal_module(self):
        def h(x):
            obj = MyModule()

            def g(x):
                obj.existing = nn.Parameter(torch.tensor(3.14))
                return obj(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_del_existing_attr_nonlocal_module(self):
        def h(x):
            obj = MyModule()

            def g(x):
                del obj.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_mutate_nonlocal_tensor(self):
        def f(x):
            def h(x):
                val = torch.tensor(1.0)

                def g(x):
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_num_builtin(self):
        def f(x):
            def h(x):
                val = 1

                def g(x):
                    nonlocal val
                    val += 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_tensor_builtin(self):
        def f(x):
            def h(x):
                val = torch.tensor(1.0)

                def g(x):
                    nonlocal val
                    val += 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_nonlocal_list_append_graph_break(self):
        def g(x):
            y = []

            def f(k):
                m = k + 1
                y.append(m)
                return k

            wrap(f, x)
            return y[0]

        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_nested_nonlocal_list_append_graph_break(self):
        def g(x):
            def h(x):
                y = []

                def f(k):
                    m = k + 1
                    y.append(m)
                    return k

                wrap(f, x)
                return y[0]

            return h(x)

        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_local_list_append_no_graph_break(self):
        def g(x):
            def f(k):
                y = []
                y.append(k + 1)
                return y[0]

            return wrap(f, x)

        x = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(g, default_args_generator((x,)), arg_count)

    def test_wrap_kwarg(self):
        def f(x, y):
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_int(self):
        def f(x, y):
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = 8

        arg_count = (
            ifdynstaticdefault(2, 3) + 1
            if check_dynamic_shape_capture()
            else ifdynstaticdefault(2, 3)
        )
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_all_kwarg(self):
        def f(y, x):
            return wrap(lambda x, y: (x * 2) + y, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_only(self):
        def f(x, y):
            def fn(*, x, y):
                return (x * 2) + y

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_default(self):
        def f(x, y):
            def fn(*, x, y, z=8):
                return (x * 2) + y + z

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_default_if_branch(self):
        def f(x, y):
            def fn(*, x, y, z=None):
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_recompile(self):
        def f(x, y, z=None):
            def fn(*, x, y, z=None):
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        counters.clear()
        opt = torch.compile(f, backend="eager", fullgraph=True)
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        # verify that we `don't` recompile
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        output = opt(x, y, 8)
        self.assertEqual(counters["stats"]["calls_captured"], 4)
        self.assertEqual(output, 2 * x)

    def test_wrap_kwarg_default_else_branch(self):
        def f(x, y, z):
            def fn(*, x, y, z=None):
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x, y, 8)), arg_count)

    def test_map_subgraph_name_is_valid(self):
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        def map_f(xs, y):
            def inner(x, y):
                def inner2(x, y):
                    return x + y

                return control_flow.map(inner2, x, y)

            return control_flow.map(inner, xs, y)

        graphs = self._check_map_graph_and_extract(map_f, (xs, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_xs_ : torch.Tensor, L_y_ : torch.Tensor):
    l_xs_ = L_xs_
    l_y_ = L_y_
    map_body_1 = self.map_body_1
    map_impl = torch.ops.higher_order.map_impl(map_body_1, [l_xs_], [l_y_]);  map_body_1 = l_xs_ = l_y_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor, l_y_ : torch.Tensor):
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [child], [l_y_]);  map_body_0 = child = l_y_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )

    def test_map_multi_return(self):
        def f(x):
            return control_flow.map(lambda x: (x.sin(), x.sin()), x)

        x = torch.randn(3)
        graphs = self._check_map_graph_and_extract(f, (x,))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    getitem = map_impl[0]
    getitem_1 = map_impl[1];  map_impl = None
    return (getitem, getitem_1)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor):
    child_1 = child.sin()
    child_2 = child.sin();  child = None
    return (child_1, child_2)""",
            )

    def test_map_pytree_return(self):
        def _construct_pytree(a):
            return (
                a.clone(),
                [[[a.clone()]]],
                a.clone(),
                (a.clone(), (a.clone(),), a.clone()),
                {"a": a.clone()},
            )

        def f(x):
            def inner_f(xs):
                return _construct_pytree(xs)

            return control_flow.map(inner_f, x)

        x = torch.randn(3)
        graphs = self._check_map_graph_and_extract(f, (x,))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    getitem = map_impl[0]
    getitem_1 = map_impl[1]
    getitem_2 = map_impl[2]
    getitem_3 = map_impl[3]
    getitem_4 = map_impl[4]
    getitem_5 = map_impl[5]
    value = map_impl[6];  map_impl = None
    return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, value)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor):
    child_1 = child.clone()
    child_2 = child.clone()
    child_3 = child.clone()
    child_4 = child.clone()
    child_5 = child.clone()
    child_6 = child.clone()
    child_7 = child.clone();  child = None
    return (child_1, child_2, child_3, child_4, child_5, child_6, child_7)""",
            )

    def test_map_kwargs(self):
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return control_flow.map(lambda x: x.sin(), x=x)

        x = torch.randn(3)
        self.assertRaises(TypeError, lambda: f(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_map_symint_input(self):
        def fn(x, y):
            def inner(x, y):
                return torch.sin(x + y)

            return control_flow.map(inner, x, y.size(0))

        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor, const_unused : int):
    add = child + 3;  child = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

    def test_map_lowers_to_graph(self):
        def fn(x, y):
            def inner(x, y):
                return torch.sin(x + y)

            return control_flow.map(inner, x, y.size(0))

        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor, const_unused : int):
    add = child + 3;  child = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

    def test_map_example_value_metadata_consistent_with_eager(self):
        from torch._higher_order_ops.map import map_dense

        backend = EagerAndRecordGraphs()

        def inner(x):
            return x.sin(), x.cos().T, x.sin().view(-1)

        rand_44 = torch.randn(4, 4)
        inps = [
            torch.randn(3),
            torch.randn(3, 4),
            torch.randn(3, 4, 5, requires_grad=True),
            torch.randn(3, 4, 5, requires_grad=True).permute((2, 0, 1)),
            torch.randn(3, 4, 5, requires_grad=True).detach(),
            torch.randn(3, 4, 5, requires_grad=True).narrow(1, 1, 2),
            rand_44.T,
            rand_44[::2],
            rand_44[::2, ::2],
            rand_44[1::3, 1::3],
            rand_44[1::3, 1::2].T,
            rand_44.unsqueeze(1),
            rand_44.squeeze(0),
            rand_44.reshape(2, 8),
        ]
        for x in inps:
            compiled_ret = torch.compile(  # noqa: F841
                control_flow.map, backend=backend, fullgraph=True
            )(inner, x)
            eager_sin, eager_transpose, eager_view = map_dense(inner, (x,), ())

            map_node = next(
                node
                for node in backend.graphs[0].graph.nodes
                if node.op == "call_function" and "map" in node.name
            )

            fake_sin, fake_transpose, fake_view = map_node.meta["example_value"]

            def _check_size_stride_contiguous(x, y):
                self.assertEqual(y.size(), x.size())
                self.assertEqual(y.stride(), x.stride())
                self.assertEqual(y.requires_grad, x.requires_grad)
                self.assertEqual(x.is_contiguous(), True)
                self.assertEqual(y.is_contiguous(), True)

            _check_size_stride_contiguous(eager_sin, fake_sin)
            _check_size_stride_contiguous(eager_transpose, fake_transpose)
            _check_size_stride_contiguous(eager_view, fake_view)

            torch._dynamo.reset()
            backend.graphs.clear()

    def test_cond_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            def true_fn(pred2, x, y):
                return x + y

            def false_fn(pred2, x, y):
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        result = cond_f(pred, pred2, xs, y)
        self.assertEqual(result, xs + y)

        cond_gm = backend.graphs[0]
        name_set = set()
        name_set.update(name for name, _ in cond_gm.named_modules())
        self.assertEqual(
            name_set,
            {
                "",
                "cond_true_1",
                "cond_false_1",
                "cond_false_1.cond_false_0",
                "cond_false_1.cond_true_0",
            },
        )

    @torch._dynamo.config.patch(
        assume_static_by_default=True,
        dynamic_shapes=True,
    )
    def test_cond_graph_break_in_one_branch(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x):
                def true_fn(x):
                    self.buffer += 1
                    return self.buffer.sum() + x.sum()

                def false_fn(x):
                    return (x - 1).sum()

                return control_flow.cond(x.sum() > 4, true_fn, false_fn, [x])

        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True)
        mod_for_eager = Foo()

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_eager(torch.ones(6, 4))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_compile(torch.ones(3, 4))

    def test_cond_free_variable_in_both_branches(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = torch.ones(4, 4)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x, y):
                def true_fn(x):
                    return x.sum() + self.buffer.sum() + z.sum()

                def false_fn(x):
                    return x.sum() - z.sum() - self.buffer.sum()

                return control_flow.cond(y, true_fn, false_fn, [x])

        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=True
        )
        mod_for_eager = Foo()

        self.assertEqual(
            mod_for_compile(torch.tensor(True), torch.tensor(5)),
            mod_for_eager(torch.tensor(True), torch.tensor(5)),
        )

        for node in backend.graphs[0].graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                _, _, _, operands = node.args
                # Since we compile with dynamic, each branch takes 4 inputs (buffer, x, z, s1)
                self.assertEqual(len(operands), 4)
            if node.op == "get_attr":
                if str(node.target) in ("cond_true_0, cond_false_0"):
                    num_placeholders = len(
                        [
                            node
                            for node in getattr(
                                backend.graphs[0], str(node.target)
                            ).graph.nodes
                            if node.op == "placeholder"
                        ]
                    )
                    self.assertEqual(num_placeholders, 4)

    def _check_cond_graph_and_extract(self, fn, args):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        gm = backend.graphs[0]
        graph = gm.code.strip()
        true_graph = gm.cond_true_0.code.strip()
        false_graph = gm.cond_false_0.code.strip()
        return (graph, true_graph, false_graph)

    def _check_map_graph_and_extract(self, fn, args):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        gm = backend.graphs[0]
        graph = gm.code.strip()
        subgraphs = []
        for module_name in gm._modules:
            subgraphs.append(getattr(gm, module_name).code.strip())
        return (graph, *subgraphs)

    def test_cond_branches_no_arguments(self):
        def fn(x):
            def true_fn():
                return torch.sin(x)

            def false_fn():
                return torch.cos(x)

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, ())

        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sum_1 = l_x_.sum()
    gt = sum_1 > 0;  sum_1 = None
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, (l_x_,));  gt = cond_true_0 = cond_false_0 = l_x_ = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                true_graph,
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    sin = torch.sin(l_x__1);  l_x__1 = None
    return (sin,)""",
            )
            self.assertExpectedInline(
                false_graph,
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    cos = torch.cos(l_x__1);  l_x__1 = None
    return (cos,)""",
            )

    def test_cond_branches_no_arguments_no_closure(self):
        def fn(x):
            def true_fn():
                return torch.ones(3, 4)

            def false_fn():
                return torch.ones(3, 4).sin()

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, ())

        self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sum_1 = l_x_.sum();  l_x_ = None
    gt = sum_1 > 0;  sum_1 = None
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, ());  gt = cond_true_0 = cond_false_0 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                true_graph,
                """\
def forward(self):
    ones = torch.ones(3, 4)
    return (ones,)""",
            )
            self.assertExpectedInline(
                false_graph,
                """\
def forward(self):
    ones = torch.ones(3, 4)
    sin = ones.sin();  ones = None
    return (sin,)""",
            )

    def test_cond_side_effect_in_one_branches(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = [torch.ones(4, 4)]

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, y, x):
                def true_fn(x):
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x.sum() + z[-1].sum()

                def false_fn(x):
                    return x.sum() - z[0].sum()

                return control_flow.cond(y, true_fn, false_fn, [x])

        mod_for_eager = Foo()
        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=False
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_eager(torch.tensor(True), torch.tensor(5))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_compile(torch.tensor(True), torch.tensor(5))

    def test_cond_with_constant_pred(self):
        def test(pred, x):
            def true_fn(x):
                return x

            def false_fn(x):
                return -x

            return control_flow.cond(pred, true_fn, false_fn, [x])

        opt_test = torch.compile(test, backend="eager")
        inp = torch.ones(3, 3)
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))

    def test_map_graph_break(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, xs):
                def body(x):
                    self.w += 1
                    return x

                return control_flow.map(body, xs)

        mod = Module()

        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))

    def test_map_side_effect(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = [torch.ones(6, 4)]

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, xs):
                def body(x):
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x + z[-1].sum()

                return control_flow.map(body, xs)

        mod = Module()

        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))

    def test_wrap_subgraph_name_is_valid(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        wrap_gm = backend.graphs[0]
        names = set()
        names.update(mod_name for mod_name, _ in wrap_gm.named_modules())
        self.assertEqual(
            names,
            {
                "",
                "wrap_body_2",
                "wrap_body_2.wrap_body_1",
                "wrap_body_2.wrap_body_1.wrap_body_0",
            },
        )

    def test_wrap_allow_local_assign_in_body_fn(self):
        def f(arg1, arg2):
            def inner_f(arg1, arg2):
                a = arg1
                b = arg2
                ret = []
                for x in a:
                    ret.append(x + 1)
                for x in b:
                    ret.append(x + 1)
                return ret

            return wrap(inner_f, arg1, arg2)

        x = torch.ones(3)

        def my_args_generator():
            yield [x], [x.sin()]
            yield (x,), (x.sin(),)

        arg_count = ifdynstaticdefault(3, 4)
        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator(),
            arg_count,
            3,
            return_graph=True,
        )

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_arg1_0_: "f32[3]", L_arg2_0_: "f32[3]"):
        l_arg1_0_ = L_arg1_0_
        l_arg2_0_ = L_arg2_0_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_arg1_0_, l_arg2_0_);  wrap_body_0 = l_arg1_0_ = l_arg2_0_ = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[3]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_arg1_0_: "f32[3]", l_arg2_0_: "f32[3]"):
            add: "f32[3]" = l_arg1_0_ + 1;  l_arg1_0_ = None

            add_1: "f32[3]" = l_arg2_0_ + 1;  l_arg2_0_ = None
            return (add, add_1)
""",
        )

    def test_capture_global_num(self):
        def f(x):
            return wrap(lambda x: x + global_num, x)

        x = torch.zeros([])
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_capture_global_num_adds_guard(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(lambda x: x + global_num, x)

        global global_num
        x = torch.zeros([])
        result = f(x)
        self.assertEqual(result, x + global_num)

        global_num = torch.randn([]).item()
        result = f(x)
        self.assertEqual(result, x + global_num)

    def test_capture_input_num(self):
        def f(x, y):
            return wrap(lambda x: x + y, x)

        x = torch.zeros([])
        y = 3.14
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, default_args_generator((x, y)), 2)

    def test_side_effect_in_body(self):
        counters.clear()
        backend = EagerAndRecordGraphs()

        x = torch.randn([])
        y = torch.randn([])

        def inner(x):
            nonlocal y
            y = x
            return x.clone()

        @torch.compile(backend=backend)
        def f(x):
            return wrap(inner, x)

        f(x)
        self.assertEqual(y, x)
        assert_dict_matches_regex(
            self,
            dict(counters["graph_break"]),
            {"HOP: Unsafe side effect": 1},
        )

    def test_fallback_on_graph_break_simple(self):
        # In the future, there should be a per-HigherOrderOperator switch
        # on whether or not to fallback or raise a loud error.
        # For now we just fallback by default.
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            return wrap(inner, x)

        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_fallback_on_graph_break_complicated(self):
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            y = x.sin()
            y = y * global_var
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            x = x.clone()
            result = wrap(inner, x)
            return result.clone()

        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 2)

    def test_modules(self):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: mod(x), x)

        result = f(x)

        self.assertEqual(result, mod(x))
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        # 3 args - 1 for input, and other 2 for the weight and bias
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        linear_node = find_first_node(body_function, torch._C._nn.linear)
        self.assertTrue(linear_node is not None)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_flat_list_output(self):
        def f(x):
            return wrap(lambda x: [torch.sin(x), torch.cos(x)], x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, expected_opcount=3
        )

    def test_support_float_in_output(self):
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: [1, torch.sin(x), 2.0], x)

        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [1, torch.sin(x), 2.0])

    def test_nested_tuple_output(self):
        def f(x):
            ((a, b),) = wrap(lambda x: ((x.sin(), x.cos()),), x)
            return a + b

        x = torch.randn(2, 3)

        counters.clear()
        arg_count = ifdynstaticdefault(2, 4)
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 4, return_graph=True
        )
        self.assertEqual(len(counters["graph_break"]), 0)

        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        a: "f32[2, 3]" = wrap[0]
        b: "f32[2, 3]" = wrap[1];  wrap = None

        add: "f32[2, 3]" = a + b;  a = b = None
        return (add,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[2, 3]"):
            sin: "f32[2, 3]" = l_x_.sin()
            cos: "f32[2, 3]" = l_x_.cos();  l_x_ = None
            return (sin, cos)
""",
        )

    def test_output_with_dict(self):
        def f(x):
            return wrap(lambda x: [{"a": -x}], x)

        x = torch.randn(3)

        counters.clear()

        arg_count = ifdynstaticdefault(2, 3)
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 2, return_graph=True
        )
        self.assertEqual(len(counters["graph_break"]), 0)

        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3]"):
            neg: "f32[3]" = -l_x_;  l_x_ = None
            return (neg,)
""",
        )

    def test_access_module_attr(self):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)

        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_make_closure(self):
        def f(x, y):
            def g(x):
                return x + y

            return g(x)

        def h(x, y):
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(h, default_args_generator((x, y)), arg_count)

    def test_internal_nonlocal(self):
        def f(x, y):
            w = 1

            def g(x):
                nonlocal w
                w = x
                return x

            def h(x):
                nonlocal w
                w = w + 1
                return x

            g(x)
            h(x)
            return w + y

        def h(x, y):
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(h, default_args_generator((x, y)), arg_count)

    def test_capture_numpy_number(self):
        import numpy as np

        y = np.float32(1.0)

        def f(x):
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # np.number are lifted to graph inputs
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_freevars_as_inputs_to_wrap(self):
        y = torch.randn(3)

        def f(x):
            return wrap(lambda x, y: x + y, x, y)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_lift_tensor_constant(self):
        def f(x):
            y = torch.tensor(1.0)
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, expected_opcount=3
        )

    def test_nested_wrap(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        mod = MockModule()

        # Two levels of wrap ops
        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            return wrap(gn, x)

        arg_count = ifdynstaticdefault(4, 5)
        self._test_wrap_simple(
            fn, default_args_generator((torch.randn(10, 10),)), arg_count
        )

    def test_fn_with_kwargs_in_torch_ops(self):
        def fn(x):
            return wrap(lambda z: torch.cos(input=z), x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count)

    def test_hooks(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        forward_handles = {}
        activations = {}

        def save_activations(mod, inp, out):
            activations[name] = inp

        for name, module in model.named_children():
            forward_handles[name] = module.register_forward_hook(save_activations)

        @torch.compile(backend="eager")
        def fn(x):
            return wrap(lambda x: model(x), x)

        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((10, 10))
            pred = fn(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(activations.keys() == forward_handles.keys())

    def _get_source_fn_stack(self, gm, node_names):
        ret = {}
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in node_names:
                    actual_stack = [
                        name for name, _ in node.meta.get("source_fn_stack", [])
                    ]
                    ret[node.name] = actual_stack
        return ret

    def test_wrap_source_fn_stack(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        mod = MockModule()

        def gn(x):
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            return wrap(gn, x)

        backend = EagerAndRecordGraphs()
        inp = torch.randn((4, 4))
        torch.compile(fn, backend=backend, fullgraph=True)(inp)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "linear"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['wrap', 'add'],
 'cos': ['wrap', 'cos'],
 'linear': ['wrap', 'wrap', 'linear']}""",
        )

    def test_cond_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            def true_fn(pred2, x, y):
                return x + y

            def false_fn(pred2, x, y):
                def true_fn2(x, y):
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)
        cond_f(pred, pred2, xs, y)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin", "sub"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['cond', 'add'],
 'cos': ['cond', 'cond', 'cos'],
 'sin': ['cond', 'cond', 'sin'],
 'sub': ['cond', 'cond', 'sub']}""",
        )

    def test_map_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=backend, fullgraph=True)
        def map_f(xs, y):
            def inner(x, y):
                def inner2(x, y):
                    return x + y

                return control_flow.map(inner2, x, y) * y.cos()

            return control_flow.map(inner, xs, y).sin()

        map_f(xs, y)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['map_impl', 'map_impl', 'add'],
 'cos': ['map_impl', 'cos'],
 'sin': ['sin']}""",
        )

    def test_grad_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        def fn(x):
            return x.sin().sum()

        @torch.compile(backend=backend, fullgraph=False)
        def wrapper_fn(x):
            return torch.func.grad(torch.func.grad(fn))(x)

        x = torch.randn(())

        wrapper_fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"sum_1", "sin"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sin': ['sin']}""",
        )

    def test_vmap_multiply_scalar(self):
        @torch.compile(backend="inductor", fullgraph=True)
        def g(x):
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 3.14)

        x = torch.randn(3)
        y = g(x)
        self.assertEqual(y, x * 3.14)

        @torch.compile(backend="inductor", fullgraph=True)
        def f(x):
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 314)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x * 314)

    def test_vmap_source_fn_stack(self):
        backend = EagerAndRecordGraphs()

        def inner_fn(x):
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return torch.func.vmap(lambda x: inner_fn(x.cos()))(x)

        x = torch.randn(3, 3, 3, 3)
        fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(
            gm, {"sum_1", "sum_2", "batched_output"}
        )
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sum_1': ['sum_1'], 'sum_2': ['sum_2']}""",
        )

    # https://github.com/pytorch/pytorch/issues/137061
    def test_dynamic_shapes_over_vmap_batch_size(self):
        def gn(a, b, c, d):
            return a + b + c + d

        def fn(func, a, b, c, d):
            a = torch.arange(a)
            b = torch.arange(b)
            c = torch.arange(c)
            d = torch.arange(d)
            func = torch.vmap(func, in_dims=(0, None, None, None))
            func = torch.vmap(func, in_dims=(None, 0, None, None))
            func = torch.vmap(func, in_dims=(None, None, 0, None))
            func = torch.vmap(func, in_dims=(None, None, None, 0))
            return func(a, b, c, d)

        cnt = CompileCounterWithBackend("eager")
        # We generate corresponding dynamic shapes test case at
        # `test/dynamo/test_dynamic_shapes.py` automatically.
        compiled_fn = torch.compile(fn, backend=cnt)
        a, b, c, d = 2, 4, 8, 8
        self.assertEqual(fn(gn, a, b, c, d), compiled_fn(gn, a, b, c, d))
        self.assertEqual(cnt.frame_count, 1)

        a, b, c, d = 4, 8, 16, 16
        self.assertEqual(fn(gn, a, b, c, d), compiled_fn(gn, a, b, c, d))
        # Ensure no recompile if dynamic shapes enabled.
        self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))
        graph = cnt.graphs[0]

        # Check dynamic shapes generates correct graph.
        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                graph.code.strip(),
                """\
def forward(self, L_a_ : torch.SymInt, L_b_ : torch.SymInt, L_c_ : torch.SymInt, L_d_ : torch.SymInt):
    l_a_ = L_a_
    l_b_ = L_b_
    l_c_ = L_c_
    l_d_ = L_d_
    a = torch.arange(l_a_)
    b = torch.arange(l_b_)
    c = torch.arange(l_c_)
    d = torch.arange(l_d_)
    lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None
    _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(l_d_, 'error');  _vmap_increment_nesting = None
    child = torch._functorch.predispatch._add_batch_dim(d, 0, 1);  d = None
    lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None
    _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(l_c_, 'error');  _vmap_increment_nesting_1 = None
    child_1 = torch._functorch.predispatch._add_batch_dim(c, 0, 2);  c = None
    lazy_load_decompositions_2 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_2 = None
    _vmap_increment_nesting_2 = torch._functorch.predispatch._vmap_increment_nesting(l_b_, 'error');  _vmap_increment_nesting_2 = None
    child_2 = torch._functorch.predispatch._add_batch_dim(b, 0, 3);  b = None
    lazy_load_decompositions_3 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_3 = None
    _vmap_increment_nesting_3 = torch._functorch.predispatch._vmap_increment_nesting(l_a_, 'error');  _vmap_increment_nesting_3 = None
    _add_batch_dim_3 = torch._functorch.predispatch._add_batch_dim(a, 0, 4);  a = None
    add = _add_batch_dim_3 + child_2;  _add_batch_dim_3 = child_2 = None
    add_1 = add + child_1;  add = child_1 = None
    batched_outputs = add_1 + child;  add_1 = child = None
    batched_outputs_1 = torch._functorch.predispatch._remove_batch_dim(batched_outputs, 4, l_a_, 0);  batched_outputs = l_a_ = None
    _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
    batched_outputs_2 = torch._functorch.predispatch._remove_batch_dim(batched_outputs_1, 3, l_b_, 0);  batched_outputs_1 = l_b_ = None
    _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
    batched_outputs_3 = torch._functorch.predispatch._remove_batch_dim(batched_outputs_2, 2, l_c_, 0);  batched_outputs_2 = l_c_ = None
    _vmap_decrement_nesting_2 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_2 = None
    _remove_batch_dim_3 = torch._functorch.predispatch._remove_batch_dim(batched_outputs_3, 1, l_d_, 0);  batched_outputs_3 = l_d_ = None
    _vmap_decrement_nesting_3 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_3 = None
    return (_remove_batch_dim_3,)""",  # noqa: B950
            )

    def test_cond_pytree_operands(self):
        def _construct_pytree():
            a = torch.randn(3, 3)
            b = torch.randn(3, 3)
            c = torch.randn(3, 3)
            d = torch.randn(3, 3)
            e = torch.randn(3, 3)
            f = torch.randn(3, 3)
            g = torch.randn(3, 3)
            return (a, [[[b]]], c, (d, (e,), f), {"g": g})

        pred = torch.tensor(True)
        inp = _construct_pytree()

        def _reduce_sum(flattened):
            init = 0
            for val in flattened:
                init += val
            return init

        def _reduce_max(flattened):
            init = flattened[0]
            for val in flattened:
                init = max(val, init)
            return init

        def true_fn(pytree_in):
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_sum(flattened)

        def false_fn(pytree_in):
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_max(flattened)

        def fn(pred, pytree_in):
            return torch.cond(pred, true_fn, false_fn, [pytree_in])

        backend = EagerAndRecordGraphs()
        compiled_res = torch.compile(fn, backend=backend)(pred, inp)
        eager_res = fn(pred, inp)
        self.assertEqual(compiled_res, eager_res)
        graph = backend.graphs[0]

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, L_pred_ : torch.Tensor, L_pytree_in_0_ : torch.Tensor, L_pytree_in_1_0_0_0_ : torch.Tensor, L_pytree_in_2_ : torch.Tensor, L_pytree_in_3_0_ : torch.Tensor, L_pytree_in_3_1_0_ : torch.Tensor, L_pytree_in_3_2_ : torch.Tensor, L_pytree_in_4_g_ : torch.Tensor):
    l_pred_ = L_pred_
    l_pytree_in_0_ = L_pytree_in_0_
    l_pytree_in_1_0_0_0_ = L_pytree_in_1_0_0_0_
    l_pytree_in_2_ = L_pytree_in_2_
    l_pytree_in_3_0_ = L_pytree_in_3_0_
    l_pytree_in_3_1_0_ = L_pytree_in_3_1_0_
    l_pytree_in_3_2_ = L_pytree_in_3_2_
    l_pytree_in_4_g_ = L_pytree_in_4_g_
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, (l_pytree_in_0_, l_pytree_in_1_0_0_0_, l_pytree_in_2_, l_pytree_in_3_0_, l_pytree_in_3_1_0_, l_pytree_in_3_2_, l_pytree_in_4_g_));  l_pred_ = cond_true_0 = cond_false_0 = l_pytree_in_0_ = l_pytree_in_1_0_0_0_ = l_pytree_in_2_ = l_pytree_in_3_0_ = l_pytree_in_3_1_0_ = l_pytree_in_3_2_ = l_pytree_in_4_g_ = None
    getitem = cond[0];  cond = None
    return (getitem,)""",  # noqa: B950
        )

    def test_cond_pytree_operands_with_non_tensor_leaves(self):
        def fn(pred, pytree_in):
            return torch.cond(
                pred, lambda x: x[0] + 1, lambda x: x[0] * 2, (pytree_in,)
            )

        pred = torch.tensor(True)
        for pytree_in in [("string",), (1.0,)]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"Expect operands to be a tuple of possibly nested dict/list/tuple",
            ):
                fn(pred, pytree_in)

        for pytree_in in [("string",), (1.0,)]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn, backend="eager")(pred, pytree_in)

    def test_cond_with_empty_operands(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, y, z):
            def true_fn():
                return y + 2

            def false_fn():
                return z + 1

            return torch.cond(x, true_fn, false_fn)

        zeros = torch.zeros(1)
        ones = torch.ones(1)
        self.assertEqual(fn(zeros, ones, ones), torch.tensor([2.0]))
        self.assertEqual(fn(ones, ones, ones), torch.tensor([3.0]))

    def test_hopify_generic_wrap(self):
        from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper

        def my_hop_fn_impl(fn, *args, k=1, **kwargs):
            def wrapper(*args, **kwargs):
                out = fn(*args, **kwargs)
                if isinstance(out, tuple):
                    return (out[0] + k,)
                return out + k

            return wrapper

        def my_hop_fn(fn, *args, k=1, **kwargs):
            return dynamo_bypassing_wrapper(
                functools.partial(my_hop_fn_impl, k=k), fn, *args, **kwargs
            )

        def my_hop_fn_2_impl(fn, *args, g=None):
            def wrapper(*args, **kwargs):
                assert g is not None  # noqa: S101
                out = fn(*args)
                if isinstance(out, tuple):
                    return (g(out[0]),)
                return g(out)

            return wrapper

        def my_hop_fn_2(fn, *args, g=None, **kwargs):
            return dynamo_bypassing_wrapper(
                functools.partial(my_hop_fn_2_impl, g=g), fn, *args, **kwargs
            )

        def gn(x, h=1):
            return x.sin() + h

        def fn(x, b):
            out = my_hop_fn(gn, x, h=b, k=2)
            return out

        a = torch.rand((4, 4), requires_grad=True)
        b = torch.rand((4, 4))
        compiled_fn = torch.compile(
            fn, backend="aot_eager_decomp_partition", fullgraph=True
        )
        self.assertEqual(compiled_fn(a, b), fn(a, b))

        def g(x):
            return x.cos()

        def fn_2(x, b):
            out = my_hop_fn_2(fn, x, b, g=g)
            return out

        a = torch.rand((4, 4), requires_grad=True)
        compiled_fn_2 = torch.compile(
            fn_2, backend="aot_eager_decomp_partition", fullgraph=True
        )
        self.assertEqual(compiled_fn_2(a, b), fn_2(a, b))

    def test_hints_wrapper(self):
        def ref_fn(x, y):
            x = x + y
            x = torch.relu(x)
            x = x + y
            return torch.abs(x)

        def fn_with_hints(x, y):
            x = x + y

            def inner_body_fn(x, y):
                x = torch.relu(x)
                x = x + y
                return x

            def outer_body_fn(x, y):
                x = hints_wrapper(inner_body_fn, (x, y), {}, hints={"inner_body": True})
                x = torch.abs(x)
                return x

            res = hints_wrapper(outer_body_fn, (x, y), {}, hints={"outer_body": True})
            return res

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(2, 4)
        y = torch.ones(4)

        eager_res = fn_with_hints(x, y)
        compiled_res = torch.compile(fn_with_hints, backend=cnt)(x, y)
        ref_res = ref_fn(x, y)
        self.assertEqual(eager_res, ref_res)
        self.assertEqual(compiled_res, ref_res)
        self.assertEqual(len(cnt.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        graph = backend.graphs[0]
        self.assertExpectedInline(
            normalize_gm(graph.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4]", L_y_: "f32[4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        x: "f32[2, 4]" = l_x_ + l_y_;  l_x_ = None

        hints_wrapper_body_1 = self.hints_wrapper_body_1
        hints_wrapper = torch.ops.higher_order.hints_wrapper(hints_wrapper_body_1, (x, l_y_), {}, hints = {'outer_body': True});  hints_wrapper_body_1 = x = l_y_ = None
        getitem: "f32[2, 4]" = hints_wrapper[0];  hints_wrapper = None
        return (getitem,)

    class hints_wrapper_body_1(torch.nn.Module):
        def forward(self, x: "f32[2, 4]", l_y_: "f32[4]"):
            hints_wrapper_body_0 = self.hints_wrapper_body_0
            hints_wrapper = torch.ops.higher_order.hints_wrapper(hints_wrapper_body_0, (x, l_y_), {}, hints = {'inner_body': True});  hints_wrapper_body_0 = x = l_y_ = None
            getitem: "f32[2, 4]" = hints_wrapper[0];  hints_wrapper = None

            x_1: "f32[2, 4]" = torch.abs(getitem);  getitem = None
            return (x_1,)

        class hints_wrapper_body_0(torch.nn.Module):
            def forward(self, x: "f32[2, 4]", l_y_: "f32[4]"):
                x_1: "f32[2, 4]" = torch.relu(x);  x = None

                x_2: "f32[2, 4]" = x_1 + l_y_;  x_1 = l_y_ = None
                return (x_2,)
""",
        )

    def test_hints_wrapper_no_hints(self):
        def fn_with_hints(x, y):
            def outer_body_fn(x, y):
                x = torch.add(x, y)
                return x

            res = hints_wrapper(outer_body_fn, (x, y), {})
            return res

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(2, 4)
        y = torch.ones(4)

        msg = "hints_wrapper: improper args/kwargs"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn_with_hints, backend=cnt)(x, y)

    def test_hints_wrapper_incorrect_type(self):
        def fn_with_hints(x, y):
            def outer_body_fn(x, y):
                x = torch.add(x, y)
                return x

            res = hints_wrapper(outer_body_fn, (x, y), {}, hints={"test": (True,)})
            return res

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(2, 4)
        y = torch.ones(4)

        msg = r"hints must be a dict containing int, float, bool or str value,"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn_with_hints, backend=cnt)(x, y)

    def test_hints_wrapper_pytree_inputs(self):
        def fn_with_hints(x, y):
            def outer_body_fn(x):
                res = torch.add(x[0], x[1]["test"])
                return res

            res = hints_wrapper(
                outer_body_fn, ((x, {"test": y}),), {}, hints={"test": True}
            )
            return res

        x = torch.randn(2, 4)
        y = torch.ones(4)

        msg = r"args must be a tuple of tensors, ints, floats, or bools,"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn_with_hints(x, y)

    @requires_gpu_and_triton
    def test_wrap_inductor_compiled_regions_option(self):
        """
        Test that wrap_inductor_compiled_regions option wraps compiled regions
        in inductor_compiled_code HOP, making them visible to DebugMode.
        """
        from torch.utils._debug_mode import DebugMode

        # Test with wrapping enabled
        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn_wrapped(x, y):
            return torch.matmul(x, y)

        # Test with wrapping disabled (default)
        @torch.compile(backend="inductor", fullgraph=True)
        def fn_not_wrapped(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        # Test wrapped version - HOP should be visible in DebugMode
        with DebugMode() as debug_mode_wrapped:
            result_wrapped = fn_wrapped(x, y)

        debug_string_wrapped = debug_mode_wrapped.debug_string()
        self.assertIn("inductor_compiled_code", debug_string_wrapped)

        # Test non-wrapped version - HOP should NOT be visible
        with DebugMode() as debug_mode_not_wrapped:
            result_not_wrapped = fn_not_wrapped(x, y)

        debug_string_not_wrapped = debug_mode_not_wrapped.debug_string()
        self.assertNotIn("inductor_compiled_code", debug_string_not_wrapped)

        # Both should produce correct results
        expected = torch.matmul(x, y)
        self.assertEqual(result_wrapped, expected)
        self.assertEqual(result_not_wrapped, expected)

    @requires_gpu_and_triton
    def test_wrap_inductor_compiled_regions_with_backward(self):
        """
        Test that wrap_inductor_compiled_regions works correctly with autograd.
        """
        from torch.utils._debug_mode import DebugMode

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device=GPU_TYPE, requires_grad=True)
        y = torch.randn(4, 4, device=GPU_TYPE, requires_grad=True)

        # Clone for eager comparison
        x_eager = x.detach().clone().requires_grad_(True)
        y_eager = y.detach().clone().requires_grad_(True)

        # Compiled forward and backward
        with DebugMode() as debug_mode:
            result = fn(x, y)
            loss = result.sum()
            loss.backward()

        # HOP should be visible in forward pass
        self.assertIn("inductor_compiled_code", debug_mode.debug_string())

        # Eager forward and backward for comparison
        expected = torch.matmul(x_eager, y_eager)
        expected_loss = expected.sum()
        expected_loss.backward()

        # Check correctness
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, x_eager.grad)
        self.assertEqual(y.grad, y_eager.grad)
