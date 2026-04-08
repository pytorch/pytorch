# Owner(s): ["module: dynamo"]
# flake8: noqa: B006, B950, C405, C416, E731, F403, F405, F841
# ruff: noqa: B006,C405,C416,E731,F403,F405,F841,SIM113,UP032
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPart6:
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

    def test_grad_state_mutated(self):
        prior = torch.is_grad_enabled()
        value = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            value = torch.is_grad_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            check_state()
            torch.set_grad_enabled(False)
            return x + 1

        try:
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if torch.is_grad_enabled() is not False:
                raise AssertionError("Expected grad disabled after fn()")

            value = None
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if torch.is_grad_enabled() is not False:
                raise AssertionError("Expected grad disabled after fn()")

            if cnt.frame_count != 1:
                raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")
        finally:
            torch.set_grad_enabled(prior)

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

    def test_torch_guards_stack_frame_register_inlining(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me(x, y, z):
            r = torch.cat((x, x)) + y
            r2 = torch.cat((y, y)) + z
            return r, r2

        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 1)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[0].line, "r, r2 = uwu_inline_me(x, y, z)")

    def test_fullgraph_capture(self):
        from torch._dynamo.convert_frame import fullgraph_capture
        from torch._dynamo.utils import dynamo_timed, get_metrics_context

        def foo(x):
            if x.shape[1] >= 3:
                return x + x.shape[0]
            else:
                return x - x.shape[0]

        x = torch.randn(4, 3)
        with (
            get_metrics_context(),
            dynamo_timed(""),
        ):
            capture_output = fullgraph_capture(foo, (x,))
            graph_capture_output = capture_output.graph_capture_output
            fn = graph_capture_output.build_guards(foo.__code__)

            for guard in graph_capture_output.output_graph.guards:
                if guard.source == torch._guards.GuardSource.SHAPE_ENV:
                    dynamic = guard.code_list is not None
                    if dynamic:
                        self.assertEqual(
                            guard.code_list,
                            [
                                "L['x'].stride()[0] == L['x'].size()[1]",
                                "2 <= L['x'].size()[0]",
                                "3 <= L['x'].size()[1]",
                            ],
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    else:
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(4, 2)}))
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(1, 3)}))
                    break

            backend_input = capture_output.backend_input
            self.assertTrue(fn.guard_manager.check({"x": x}))
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in graph_capture_output.import_sources.items()
        }
        self.assertEqual(
            foo(x),
            types.FunctionType(
                graph_capture_output.bytecode,
                {
                    **import_sources,
                    backend_input.backend_id: backend_input.graph_module,
                },
            )(x),
        )

    def test_torch_guards_stack_frame_register_inlining_deep(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me_deep(x, y):
            return torch.cat((x, x)) + y

        def uwu_inline_me(x, y, z):
            r = uwu_inline_me_deep(x, y)
            r2 = uwu_inline_me_deep(y, z)
            return r, r2

        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 3)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[1].name, "uwu_inline_me")
        self.assertEqual(seen_frames[2].line, "r2 = uwu_inline_me_deep(y, z)")

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

    def test_error_on_recompile(self):
        @torch.compile(backend="eager")
        def fn(a, b):
            return a + b

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            with self.assertRaises(torch._dynamo.exc.RecompileError):
                fn(torch.rand(2, 3), torch.rand(2, 3))
                fn(torch.rand(2, 3), (1, 2, 3))

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

    def test_add_sizes(self):
        def func(x):
            y = x.size()
            return y + y

        eager_out = func(torch.ones(10, 10, 3))
        compile_out = torch.compile(func, backend="eager")(torch.ones(10, 10, 3))
        self.assertTrue(isinstance(compile_out, torch.Size))
        self.assertEqual(eager_out, compile_out)

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

    def test_recursion_depth_guards(self):
        @torch.compile(dynamic=True, backend="eager")
        def foo(*args, **kwargs):
            if sum(args) == 0:
                return 0
            return 1

        args = list(range(2000))
        foo(*args)
        # Previously would have crashed

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

    def test_scalar_tensor_is_equivalent_to_symint_argument(self):
        class GumbelTopKSampler(torch.nn.Module):
            def __init__(self, T, k):
                super().__init__()
                self.T = torch.nn.Parameter(
                    torch.tensor(T, dtype=torch.float32), requires_grad=False
                )
                self.k = torch.nn.Parameter(
                    torch.tensor(k, dtype=torch.int32), requires_grad=False
                )

            def sample_discrete(self, logits):
                threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
                samples = torch.ge(logits.squeeze(1), threshold).float()
                return samples

            def forward(self, logits):
                dsamples = self.sample_discrete(logits)
                return dsamples

        x = torch.rand([4, 4, 4, 4])
        m = GumbelTopKSampler(T=4, k=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_symint_list_argument(self):
        class Jitter(torch.nn.Module):
            def __init__(self, jitter_val):
                super().__init__()
                self.jitter_val = jitter_val

            def roll_tensor(self, input):
                h_shift = self.jitter_val - 1
                w_shift = self.jitter_val + 1
                return torch.roll(
                    torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3
                )

            def forward(self, input):
                return self.roll_tensor(input)

        x = torch.rand([4, 4, 4, 4])
        m = Jitter(jitter_val=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_int_list_argument(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                permute = torch.tensor([0, 2, 1])
                x = input.permute(*permute)
                return x

        x = torch.randn(2, 3, 4)
        m = MyModel()
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_torch_variable_hasattr(self):
        def fn(x):
            if hasattr(torch.nn, "Module"):
                return x * x
            return x + 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = torch.rand([4, 4])
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

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

    def test_fn_hasattr__name__1(self):
        def fn():
            foo = lambda x: x + 1
            return hasattr(foo, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__2(self):
        def bar(x):
            return torch.sin(x)

        def fn():
            return hasattr(bar, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertTrue(fn_out)

    def test_fn_hasattr__name__3(self):
        def bar(x, y):
            return torch.sin(x) + torch.cos(y)

        baz = functools.partial(bar, y=4)

        def fn():
            return hasattr(baz, "__name__")

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertFalse(fn_out)

    def test_constant_hasattr_returns_bool(self):
        """Test that hasattr on constant values properly returns boolean ConstantVariable."""

        # Test various constant types
        def fn():
            # String constant
            s = "hello"
            result1 = hasattr(s, "upper")  # True
            result2 = hasattr(s, "nonexistent")  # False

            # Integer constant
            i = 42
            result3 = hasattr(i, "bit_length")  # True
            result4 = hasattr(i, "fake_method")  # False

            # Float constant
            f = 3.14
            result5 = hasattr(f, "is_integer")  # True
            result6 = hasattr(f, "missing_attr")  # False

            # Use all results to ensure they're compiled
            return (result1, result2, result3, result4, result5, result6)

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        fn_out = fn()
        compiled_out = compiled_fn()
        self.assertEqual(fn_out, compiled_out)
        self.assertEqual(fn_out, (True, False, True, False, True, False))

    def test_class_hasattr_sourceless_descriptor(self):
        """Test that hasattr on sourceless UserDefinedClassVariable does not graph break."""

        class FlagDescriptor:
            def __get__(self, instance, owner):
                if hasattr(owner, "flag"):
                    return 1
                return 0

        class WithFlag:
            flag = True
            prop = FlagDescriptor()

        class WithoutFlag:
            prop = FlagDescriptor()

        def fn(x, obj):
            return x + obj.prop

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        x = torch.randn(3)
        self.assertEqual(fn(x, WithFlag()), compiled_fn(x, WithFlag()))
        self.assertEqual(fn(x, WithoutFlag()), compiled_fn(x, WithoutFlag()))

    def test_torch_objects_as_keys(self):
        remap = {torch.float16: torch.float32}

        def fn():
            return torch.randn(3, dtype=remap[torch.float16])

        opt = torch.compile(fn, backend="eager")
        opt()

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

    def test_jacfwd_one_hot_dynamic_compile(self):
        import torch.nn.functional as F

        MAX, BATCH = 3, 37

        def func(x, idxs):
            return x.square() * F.one_hot(idxs, MAX)

        def jacfunc(x, idxs):
            return torch.func.jacfwd(func, argnums=(0,))(x, idxs)

        idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
        x = torch.rand((BATCH, MAX), dtype=torch.float64)
        eager = jacfunc(x, idxs)

        compiled = torch.compile(jacfunc, backend="eager", dynamic=True)
        out_comp = compiled(x, idxs)
        self.assertEqual(eager[0], out_comp[0])

    def test_tracing_nested_py_tree_mixed_all(self):
        def fn(xs):
            flat_xs, spec = python_pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            return python_pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

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

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_out_variant_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
            )

            @torch.library.impl(lib, "split_with_sizes_copy", "Meta")
            @torch.library.impl(lib, "split_with_sizes_copy", "CPU")
            def split_with_sizes_copy(
                all_gather_output: torch.Tensor,
                all_gather_input_split_sizes: typing.List[int],
                dim: int,
                out: typing.List[torch.Tensor],
            ) -> None:
                torch.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
                )

            @torch.compile(backend="eager", fullgraph=True)
            def f1(all_gather_output, all_gather_input_split_sizes, dim, out):
                return torch.ops.mylib.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim, out=out
                )

            all_gather_output = torch.randn(2, 272)
            all_gather_input_split_sizes = [128, 8, 128, 8]
            dim = 1
            out = [
                torch.empty(2, 128),
                torch.empty(2, 8),
                torch.empty(2, 128),
                torch.empty(2, 8),
            ]
            f1(all_gather_output, all_gather_input_split_sizes, dim, out)

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
            )

            @torch.library.impl(lib, "chunk_cat", "Meta")
            @torch.library.impl(lib, "chunk_cat", "CPU")
            def chunk_cat(
                tensors: typing.List[torch.Tensor],
                dim: int,
                num_chunks: int,
                out: torch.Tensor,
            ) -> None:
                torch._chunk_cat(tensors, dim, num_chunks, out=out)

            @torch.compile(backend="eager", fullgraph=True)
            def f2(tensors, dim, num_chunks, out):
                return torch.ops.mylib.chunk_cat(tensors, dim, num_chunks, out=out)

            x = torch.zeros(100, dtype=torch.int64)
            tensors = [
                torch.randn(16, 16),
                torch.randn(16),
                torch.randn(16, 16),
                torch.randn(16),
            ]
            dim = 0
            num_chunks = 2
            out = torch.empty(2, 272)
            f2(tensors, dim, num_chunks, out)

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

    def test_reconstruct_set_across_graph_break(self):
        def foo(x, y):
            setty = set()
            for t in x:
                setty.add(t)
            print("Break!")
            return y * len(setty)

        x = torch.randn(10, 10)
        y = torch.randn(2, 2)

        counter = CompileCounter()
        foo = torch.compile(foo, backend=counter)
        result = foo([x, x, x, x, y], y)

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

    def test_sourceless_namedtuple(self):
        from collections import namedtuple

        CustomDtype = namedtuple("CustomDtype", ["dtype", "higher_dtype"])

        class CustomTensor(torch.Tensor):
            _data: torch.Tensor
            custom_dtype: CustomDtype
            __torch_function__ = torch._C._disabled_torch_function_impl
            __slots__ = [
                "_data",
                "custom_dtype",
            ]

            def __new__(
                cls,
                data: torch.Tensor,
                custom_dtype: CustomDtype,
            ):
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    data.size(),
                    strides=data.stride(),
                    storage_offset=data.storage_offset(),
                    dtype=custom_dtype.dtype,
                    layout=data.layout,
                    requires_grad=data.requires_grad,
                    device=data.device,
                )
                self._data = data
                self.custom_dtype = custom_dtype
                return self

            def __tensor_flatten__(self):
                meta = {
                    "custom_dtype": self.custom_dtype,
                }
                return ["_data"], meta

            @staticmethod
            def __tensor_unflatten__(
                inner_tensors: dict, metadata, outer_size, outer_stride
            ):
                return CustomTensor(
                    inner_tensors["_data"],
                    metadata["custom_dtype"],
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
                return func(*args, **kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = CustomTensor(x, CustomDtype(torch.float32, torch.bfloat16))
            return y, y.custom_dtype

        fn(torch.ones(2, 2, device="cpu"))

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

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_dim_order(self):
        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def f(x):
            x = x.permute(3, 0, 2, 1)
            return x, x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def g(x):
            return x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h0(xs, ambiguity_check=False):
            u0, u1, u2 = xs.tolist()
            torch._check(u2 >= u0)
            torch._check(u1 >= u0)
            # stride ordering still isn't unique here, should raise
            y = torch.empty_strided([4, 4, 4], [u0, u1, u2])
            return y.dim_order(ambiguity_check=ambiguity_check)

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h1(xs, ambiguity_check=False):
            u0, u1, u2 = xs.tolist()
            y = torch.empty_strided([4, 4, 4], [u0, u0, u0])  # no ordering
            return y.dim_order(ambiguity_check=ambiguity_check)

        # check that for functions permuting contiguous input, the original stride is recovered with dim_order.
        def test(x):
            stride_inp = tuple(x.stride())
            f_out, f_order = f(x)
            self.assertEqual(stride_inp, tuple(f_out.stride(i) for i in f_order))

        # shape: [4, u0, 5, u1]
        x0 = torch.randn(4, 1, 5, 2)
        torch._dynamo.decorators.mark_unbacked(x0, 1)
        torch._dynamo.decorators.mark_unbacked(x0, 3)
        test(x0)

        # shape: [u0, u1, u2, u3]
        x1 = torch.randn(4, 1, 5, 2)
        for i in range(x1.ndim):
            torch._dynamo.decorators.mark_unbacked(x1, i)
        test(x1)

        # custom strides (all integers)
        x2 = torch.randn(10000)
        x2 = x2.as_strided([4, 4, 4, 4], [1, 2, 4, 8])
        if g(x2) != (3, 2, 1, 0):
            raise AssertionError(f"Expected g(x2) == (3, 2, 1, 0), got {g(x2)}")

        # custom unbacked strides with no ordering: ambiguity check should raise
        xs = torch.tensor([2, 3, 4])
        h0(xs)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h0(xs, ambiguity_check=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h1(xs, ambiguity_check=True)

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

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_scalar(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([3])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_1d(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([2, 1])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_kd(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([[[2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1]]])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @patch.object(torch._dynamo.config, "specialize_int", True)
    def test_tolist_0d(self):
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list

        x = torch.tensor(42)
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    def test_tolist_kd_dynamic(self):
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list, x * 10

        x = torch.randint(3, 5, [5, 5])
        eager = fn(x)
        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=False)
        compiled = compiled_fn(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

        # Value change, no recompiles
        x = torch.randint(7, 9, [5, 5])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 1)

        # Size change, forced recompiles
        x = torch.randint(3, 5, [3, 3])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 2)

    def test_tolist_float(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor(
            [[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]]
        )
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)(x)
        self.assertEqual(eager, compiled)
        # Nothing to compile here
        self.assertEqual(counter.frame_count, 0)

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

    # just to be sure in case anyone tries to run this in older versions of Python
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

    def test_derpy_nn_module_usage(self):
        def ff1(x):
            self = mod1
            return torch.sigmoid(self.mod2(x) + self.param1)

        def ff2(x):
            self = mod2
            return torch.cos(torch.sin(x) * self.param2 + 10)

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()
        mod1.register_module("mod2", mod2)
        mod1.register_parameter("param1", torch.nn.Parameter(torch.randn(10)))
        mod1.forward = ff1
        mod2.register_parameter("param2", torch.nn.Parameter(torch.randn(10)))
        mod2.forward = ff2
        mod1.eval()

        x = torch.randn(10)
        expected = mod1(x)
        counter = CompileCounter()
        actual = torch.compile(mod1, backend=counter, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(counter.op_count, 6)

    def test_default_args_device_dtype(self):
        class Foo:
            def __init__(
                self,
                dtype: torch.dtype = torch.float16,
                device: torch.device = torch.device("cpu"),
            ) -> None:
                self.value = torch.tensor(10, dtype=dtype, device=device)

        def fn():
            return Foo().value + 1

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn()
        res = opt_func()
        self.assertEqual(ref, res)

    def test_torch_dtype_python_type(self):
        def fn(target):
            target_dtype = target.dtype
            a = torch.zeros(2, 3, dtype=target_dtype)
            # Constant assert at trace time
            assert isinstance(target_dtype, torch.dtype)  # noqa: S101
            b = torch.zeros(2, 3, dtype=target_dtype)
            c = torch.zeros(2, 3, dtype=target_dtype)
            return a + b + c

        from torch._dynamo.variables import ConstantVariable

        dtype = torch.float16
        expected_variable = ConstantVariable(dtype)
        self.assertEqual(expected_variable.python_type(), type(dtype))

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        a = torch.tensor([2, 3], dtype=dtype)
        res = opt_func(a)
        self.assertIsInstance(res, torch.Tensor)

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
