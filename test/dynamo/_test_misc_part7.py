# Owner(s): ["module: dynamo"]
# flake8: noqa: B950, F401, F403, F405, F841
# ruff: noqa: F401,F403,F405,F841,UP008
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPart7:
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

    def test_storage_return(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = torch.sin(x + 1)
            storage = x.untyped_storage()
            storage.resize_(0)
            y = torch.cos(y)
            return y, storage

        x = torch.randn(10)
        expected = torch.cos(torch.sin(x + 1))
        y, s = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)
        self.assertIs(s, x.untyped_storage())

    def test_flat_name_to_original_fqn(self):
        class FooBarModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("0", torch.nn.Parameter(torch.randn(3, 4)))
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )

            def forward(self, x):
                return ((x + self.test_buf) * getattr(self, "0")) / self.test_param

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo_bar = FooBarModule()
                self.register_parameter(
                    "test_param", torch.nn.Parameter(torch.randn(3, 4))
                )
                self.test_buf = torch.nn.Buffer(torch.randn(3, 4))

            def forward(self, x):
                return (self.foo_bar(x) + self.test_param) * self.test_buf

        gm, _ = torch._dynamo.export(TestModule(), torch.randn(3, 4))
        self.assertIn("dynamo_flat_name_to_original_fqn", gm.meta)
        expected_fqn = {
            "L__self___test_param": "test_param",
            "L__self___test_buf": "test_buf",
            "L__self___foo_bar_0": "foo_bar.0",
            "L__self___foo_bar_test_param": "foo_bar.test_param",
            "L__self___foo_bar_test_buf": "foo_bar.test_buf",
        }
        self.assertEqual(expected_fqn, gm.meta["dynamo_flat_name_to_original_fqn"])

    def test_proxy_frozen_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_reconstruct_frozen_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor

        def fn(x, y):
            dc = TestDataClass(x, y)
            torch._dynamo.graph_break()
            return dc.x + dc.y

        fn_opt = torch.compile(backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

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

    def test_frozen_dataclass_default_value(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default=5)
            a: int = 6

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.z + dc.a

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_default_factory(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(default_factory=list)
            a: int = dataclasses.field(default_factory=lambda: [5])

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.a[0]

        def fn(x, y):
            dc = TestDataClass(x, y)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_kw_only(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int = dataclasses.field(kw_only=True)
            a: int = dataclasses.field(kw_only=True)

        @allow_in_graph
        def inner_fn(dc):
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_attr_access(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: torch.Tensor
            y: torch.Tensor
            z: int
            a: int

        def inner_fn(dc):
            return dc.x + dc.y + dc.a + dc.z

        def fn(x, y):
            dc = TestDataClass(x, y, z=5, a=2)
            return inner_fn(dc)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)

        self.assertEqual(actual, expected)

    def test_frozen_dataclass_hashable(self):
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            x: float
            y: float
            z: int
            a: int

        def inner_fn(dc, x, y):
            d = {}
            d[dc] = 2
            return dc.x + dc.y + d[dc] + x + y

        def fn(x, y):
            dc = TestDataClass(x=3.2, y=2.5, z=5, a=2)
            return inner_fn(dc, x, y)

        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        inps = (torch.ones(2, 2), torch.ones(2, 2))
        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertEqual(actual, expected)

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

    def test_frozen_dataclass_in_compile(self):
        from torch.utils._pytree import MappingKey, SequenceKey

        def fn(x):
            path = (MappingKey("a"), SequenceKey(0))
            msg = f"path={path}"
            return x * 2, msg

        x = torch.randn(4, 4)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, fullgraph=True)(x)
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1], compiled_result[1])

    def test_frozen_dataclass_treespec_method_and_fields(self):
        from torch.utils._pytree import tree_flatten

        def fn(x):
            d = {"a": x, "b": [x * 2, x * 3]}
            flat, spec = tree_flatten(d)
            is_leaf = spec.is_leaf()
            return sum(flat), spec.num_leaves, spec.num_nodes, is_leaf

        x = torch.randn(4)
        eager_result = fn(x)
        compiled_result = torch.compile(fn, fullgraph=True)(x)
        for i in range(4):
            self.assertEqual(eager_result[i], compiled_result[i])

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

    def test_default_dtype_change(self):
        @torch.compile(backend="eager")
        def foo():
            def inner(a, b, res_dtype):
                print(a, b, res_dtype)
                self.assertEqual(torch.result_type(a, b), res_dtype)

            inner(torch.tensor(1, device="cpu"), 1.0, torch.get_default_dtype())

        with set_default_dtype(torch.float):
            foo()
        with set_default_dtype(torch.double):
            foo()

    def test_numpy_ufunc_out(self):
        @torch.compile(backend="eager")
        def foo():
            x = np.arange(5)
            out = np.empty((x.shape[0], x.shape[0]))
            res_out = np.sin(x, out=out)
            assert res_out is out  # noqa: S101

        foo()

    # Unfortunately, we don't currently preserve the ids of
    # res_out and out correctly across the graph break
    @unittest.expectedFailure
    def test_numpy_ufunc_out_graph_break(self):
        @torch.compile(backend="eager")
        def foo():
            x = np.arange(5)
            out = np.empty((x.shape[0], x.shape[0]))
            res_out = np.sin(x, out=out)
            torch._dynamo.graph_break()
            assert res_out is out  # noqa: S101

        foo()

    @wrapDeterministicFlagAPITest
    def test_backward_deterministic_mode_mismatch_warning(self):
        @torch.compile(backend="aot_eager")
        def func(a, b):
            return a + b

        for forward_deterministic, backward_deterministic in itertools.product(
            [True, False], [True, False]
        ):
            torch.use_deterministic_algorithms(forward_deterministic)
            a = torch.randn(10, requires_grad=True)
            res = func(a, 1)
            grad = torch.ones_like(res)
            torch.use_deterministic_algorithms(backward_deterministic)

            if not forward_deterministic and backward_deterministic:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^This compiled backward function is being run with torch\.use_deterministic_algorithms",
                ):
                    res.backward(grad)

            else:
                res.backward(grad)

    @skipIfWindows(
        msg="AssertionError: False is not true : Encountered an unexpected fallback to 'aten pow' in dynamo compiled code"
    )
    @unittest.skipIf(
        torch._inductor.config.cpu_backend != "cpp",
        "Skip for non cpp backend CPU as comments contain 'aten.pow' ",
    )
    def test_torch_dynamo_codegen_pow(self):
        def pow(x):
            return x**2

        x = np.arange(8)
        pow_opt = torch.compile(pow, backend="eager")

        actual, source_code = run_and_get_code(pow_opt, x)
        expect = pow(x)

        self.assertEqual(expect, actual)

        self.assertTrue(
            all("aten.pow" not in code for code in source_code),
            msg="Encountered an unexpected fallback to 'aten pow' in dynamo compiled code",
        )

    def test_graph_break_compilation_metrics(self):
        def fn(x):
            x.cos()
            torch._dynamo.graph_break()
            x.sin()
            torch._dynamo.graph_break()
            return x.cos()

        torch._dynamo.utils.clear_compilation_metrics()
        x = torch.rand((4, 4))
        f = torch.compile(fn, backend="eager")
        f(x)
        metrics = torch._dynamo.utils.get_compilation_metrics()
        # Should only be one restart per event
        (restart_reason,) = metrics[0].restart_reasons
        self.assertTrue(
            "User-inserted graph break" in restart_reason,
            "Should have logged graph break reason",
        )
        self.assertTrue(
            metrics[0].dynamo_time_before_restart_s
            <= metrics[0].entire_frame_compile_time_s
        )

        (restart_reason,) = metrics[1].restart_reasons
        self.assertTrue(
            "User-inserted graph break" in restart_reason,
            "Should have logged graph break reason",
        )
        self.assertTrue(
            metrics[1].dynamo_time_before_restart_s
            <= metrics[1].entire_frame_compile_time_s
        )

        # No restarts
        self.assertTrue(
            len(metrics[2].restart_reasons) == 0, "Last compile has no graph break"
        )
        self.assertTrue(metrics[2].dynamo_time_before_restart_s == 0)

    def test_graph_break_compilation_metrics_on_failure(self):
        def fn(x):
            return x.sin()

        def broken_backend(gm, example_inputs):
            raise RuntimeError("broken backend")

        x = torch.rand((4, 4))
        f = torch.compile(fn, backend=broken_backend)
        with unittest.mock.patch("torch._dynamo.config.suppress_errors", True):
            torch._dynamo.utils.clear_compilation_metrics()
            f(x)
            metrics = torch._dynamo.utils.get_compilation_metrics()
            for metric in metrics:
                self.assertTrue(metric.dynamo_time_before_restart_s > 0)
                self.assertTrue(
                    "RuntimeError: broken backend" in metric.fail_reason,
                    "Should have logged fail reason",
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

    def _test_compile_model_free(self, model_inp_ctr, weakref_watch):
        """
        Args:
        model_inp_ctr
            - constructor that returns a new model and inputs to that model
        weakref_watch
            - function that returns a layer of the model for weakref to
              finalize on, so we can check that the layer is freed after
              the model goes out of scope
        """
        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            mod, inp = model_inp_ctr()
            weakref.finalize(weakref_watch(mod), finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        gc.collect()
        self.assertTrue(cleared)

    def test_custom_module_free(self):
        """Test that a model is freed when it goes out of scope"""

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super(Mod, self).__init__()
                self.fc = torch.nn.Linear(100, 100)

            def forward(self, out):
                return self.fc(out)

        self._test_compile_model_free(
            lambda: (Mod(), torch.randn(100, 100)),
            lambda mod: mod.fc,
        )

    def test_sequential_module_free(self):
        self._test_compile_model_free(
            lambda: (
                torch.nn.Sequential(
                    torch.nn.Linear(100, 100),
                    torch.nn.ReLU(),
                ),
                torch.randn(100, 100),
            ),
            lambda mod: mod[0],
        )

    def test_linear_module_free(self):
        self._test_compile_model_free(
            lambda: (torch.nn.Linear(100, 100), torch.randn(100, 100)),
            lambda mod: mod,
        )

    def test_outside_linear_module_free(self):
        # Compared to test_linear_module_free, the linear
        # layer is not the code object that is directly compiled.

        # This test does not use _test_compile_model_free because of difficulty
        # in handling variable fc.

        cleared = False

        def finalize():
            nonlocal cleared
            cleared = True

        def run():
            fc = torch.nn.Linear(100, 100)

            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc_ref = fc

                def forward(self, x):
                    return self.fc_ref(x)

            mod = Mod()
            inp = torch.randn(100, 100)
            weakref.finalize(fc, finalize)
            torch.compile(mod, backend="eager")(inp)

        run()
        # del fc  # This should delete all the references
        gc.collect()
        self.assertTrue(cleared)

    def test_parameter_free(self):
        def model_inp_ctr():
            param = torch.nn.Parameter(torch.randn(100, 100))

            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = param

                def forward(self, x):
                    return self.param * x[0]

            # return param to keep it alive in _test_compile_model_free
            return Mod(), (torch.randn(100, 100), param)

        self._test_compile_model_free(model_inp_ctr, lambda mod: mod.param)

    def test_conditional_list_comp_in_context(self):
        def fn(inp):
            try:
                return [torch.sin(x) for x in inp if x is not None]
            except Exception:
                pass

        inp = [torch.randn(3, 3) for _ in range(3)] + [None]
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(inp)

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

    def test_super_after_graph_break(self):
        class Foo(torch.nn.Sequential):
            def __init__(self, layers):
                torch._dynamo.graph_break()
                super().__init__(*layers)

        def fn(x):
            layers = [torch.nn.Linear(3, 3) for _ in range(3)]
            mod = Foo(layers)
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(3, 3))

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

    def test_inspect_signature_bind(self):
        import inspect

        def inner(a, b, *ar, c=10, d=11, **kw):
            pass

        def fn(x, apply_defaults):
            sig = inspect.signature(inner)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            bound.arguments["d"] = 13
            if apply_defaults:
                bound.apply_defaults()
            return (
                sig,
                bound.signature,
                bound,
                bound.arguments,
                bound.args,
                bound.kwargs,
                x + 1,
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        for apply_defaults in (True, False):
            _, _, bound0, arguments0, args0, kwargs0, _ = fn(
                torch.ones(3, 3), apply_defaults
            )
            _, _, bound1, arguments1, args1, kwargs1, _ = opt_fn(
                torch.ones(3, 3), apply_defaults
            )

            self.assertEqual(bound0, bound1)
            self.assertEqual(arguments0, arguments1)
            self.assertEqual(args0, args1)
            self.assertEqual(kwargs0, kwargs1)
            self.assertTrue(args1)
            self.assertTrue(kwargs1)

    def test_inspect_signature_bind_non_user_function(self):
        import inspect

        class Foo:
            def __init__(self, a, b, *ar, c=10, d=11, **kw):
                pass

        def fn(x):
            sig = inspect.signature(Foo)
            bound = sig.bind(1, 2, 3, d=12, e=15)
            return bound, x + 1

        opt_fn = torch.compile(fn, backend="eager")
        bound0, _ = fn(torch.ones(3, 3))
        bound1, _ = opt_fn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

        import traceback

        # choose a function that is skipped but has defaults
        self.assertTrue(hasattr(traceback.print_exc, "__kwdefaults__"))
        self.assertIs(
            torch._dynamo.trace_rules.lookup(traceback.print_exc),
            torch._dynamo.variables.UserFunctionVariable,
        )

        def gn(x):
            sig = inspect.signature(traceback.print_exc)
            bound = sig.bind()
            return bound, x + 1

        opt_gn = torch.compile(gn, backend="eager", fullgraph=True)
        bound0, _ = gn(torch.ones(3, 3))
        bound1, _ = opt_gn(torch.ones(3, 3))

        self.assertEqual(bound0, bound1)

    def test_sourceless_mapping_proxy(self):
        # Test that Dynamo can handle a sourceless MappingProxyType.
        # This occurs when type.__dict__['__dict__'].__get__ is called
        # and returns a mappingproxy that was created during tracing.
        _get_dunder_dict = type.__dict__["__dict__"].__get__

        class MyClass:
            a = 1
            b = 2

        def fn(x):
            d = _get_dunder_dict(MyClass)
            return x + len(d)

        t = torch.randn(3)
        ref = fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(t)
        self.assertEqual(ref, res)

    def test_sourceless_inspect_parameter(self):
        import inspect

        class MyClass:
            param = inspect.Parameter(
                "x", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=42
            )

        _get_dunder_dict = type.__dict__["__dict__"].__get__

        def fn(x):
            d = _get_dunder_dict(MyClass)
            p = d["param"]
            return x + p.default

        t = torch.randn(3)
        ref = fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(t)
        self.assertEqual(ref, res)

    def test_inspect_signature_parameters(self):
        import inspect

        def fn(x, gn):
            d = inspect.signature(gn).parameters
            if d["a"].default is inspect.Parameter.empty:
                return torch.sin(x + 1)
            else:
                return torch.cos(x + 1)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            return a + b

        x = torch.randn(2, 3)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(fn(x, gn), opt_fn(x, gn))

    def test_inspect_signature_caching(self):
        """Test that inspect.signature results are cached for repeated calls."""
        import inspect
        from unittest.mock import patch

        def target_func(a, b, c=3):
            return a + b + c

        def other_func(x, y):
            return x * y

        def fn():
            results = []
            for _ in range(10):
                sig1 = inspect.signature(target_func)
                sig2 = inspect.signature(other_func)
                results.append(len(sig1.parameters))
                results.append(len(sig2.parameters))
            return sum(results)

        from torch._dynamo.output_graph import OutputGraph

        original_cleanup = OutputGraph.cleanup
        unique_calls = 0

        def capturing_cleanup(self):
            nonlocal unique_calls
            unique_calls = len(self.signature_cache)
            return original_cleanup(self)

        with patch.object(OutputGraph, "cleanup", capturing_cleanup):
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled_fn()

        # 10 iterations * (3 params from target_func + 2 params from other_func) = 50
        self.assertEqual(result, 50)
        # signature_cache should have exactly 2 entries (one per unique function)
        self.assertEqual(unique_calls, 2)

    def test_inspect_signature_caching_methods(self):
        """Test that inspect.signature caching works for methods."""
        import inspect
        from unittest.mock import patch

        class MyClass:
            def method_a(self, x, y, z):
                return x + y + z

            def method_b(self, a):
                return a * 2

        obj = MyClass()

        def fn():
            results = []
            for _ in range(10):
                sig1 = inspect.signature(obj.method_a)
                sig2 = inspect.signature(obj.method_b)
                # Note: bound methods don't include 'self' in signature
                results.append(len(sig1.parameters))
                results.append(len(sig2.parameters))
            return sum(results)

        from torch._dynamo.output_graph import OutputGraph

        original_cleanup = OutputGraph.cleanup
        unique_calls = 0

        def capturing_cleanup(self):
            nonlocal unique_calls
            unique_calls = len(self.signature_cache)
            return original_cleanup(self)

        with patch.object(OutputGraph, "cleanup", capturing_cleanup):
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled_fn()

        # 10 iterations * (3 params from method_a + 1 param from method_b) = 40
        self.assertEqual(result, 40)
        # signature_cache should have exactly 2 entries (one per unique method)
        self.assertEqual(unique_calls, 2)

    def test_inspect_variable_redirect(self):
        """Test that InspectVariable is used and redirects property accesses."""
        import inspect
        from unittest.mock import patch

        from torch._dynamo.variables.user_defined import InspectVariable

        redirected_attrs = []
        original_var_getattr = InspectVariable.var_getattr

        def tracking_var_getattr(self, tx, name):
            redirects = self._PROPERTY_REDIRECTS.get(type(self.value), {})
            if name in redirects:
                redirected_attrs.append(name)
            return original_var_getattr(self, tx, name)

        def fn(x, gn):
            sig = inspect.signature(gn)
            params = sig.parameters
            param = params["a"]
            return x + param.kind + len(param.name)

        def gn(a: torch.Tensor, b: int) -> torch.Tensor:
            return a + b

        x = torch.randn(2, 3)
        with patch.object(InspectVariable, "var_getattr", tracking_var_getattr):
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            result = opt_fn(x, gn)

        self.assertEqual(result, fn(x, gn))
        self.assertIn("parameters", redirected_attrs)
        self.assertIn("kind", redirected_attrs)
        self.assertIn("name", redirected_attrs)

    def test_grad_none(self):
        def fn(x, y):
            x.grad = torch.abs(y)
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        opt_fn = torch.compile(fn, backend="eager")
        z = opt_fn(x, y)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_grad_non_none(self):
        def fn(x, y):
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_fn = torch.compile(fn, backend=cnt)
        z = opt_fn(x, y)

        # Ensure that the generated graph returns only one output. We want the
        # add_ on the grad to be part of the graph itself, so that inductor can
        # theoretically move the add_ and resulting copy_ nodes at the right
        # place to free memory.
        self.assertEqual(len(list(cnt.graphs[0].graph.nodes)[-1].all_input_nodes), 1)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_new_with_int_list(self):
        # Make sure torch.Tensor.new(int argument list) behaves the same on dynamo.
        def fn(x):
            return x.new(*x.size()) + 5

        optfn = torch.compile(backend="eager")(fn)

        x = torch.arange(10).view(2, 5)

        expected = fn(x)
        actual = optfn(x)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    def test_dynamic_shapes_as_strided(self):
        def fn(t, new_size, new_stride):
            tmp = t.as_strided(new_size, new_stride)
            tmp = tmp.view(-1)
            return t * tmp.sum()

        optfn = torch.compile(backend="eager", dynamic=True)(fn)

        x = torch.randn(3)
        new_size = [0, 3]
        new_stride = [3, 1]

        expected = fn(x, new_size, new_stride)
        actual = optfn(x, new_size, new_stride)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

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

    def test_custom_dict(self):
        class MyDict(dict):
            pass

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = MyDict(d)

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

    def test_descriptor(self):
        class lazy_property:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __get__(self, instance, obj_type=None):
                value = self.wrapped(instance)
                setattr(instance, self.wrapped.__name__, value)
                return value

        class UserDefined:
            def __init__(self) -> None:
                self.a = 3

            @lazy_property
            def length(self):
                return 3

            def run(self, x):
                return x * self.length

        obj = UserDefined()

        def fn(x):
            return obj.run(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        # Opt_fn is deliberately called first to trigger the __get__ function.
        # Otherwise, the setattr removes the lazy property.
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)
        ref = opt_fn(x)
        res = fn(x)
        self.assertEqual(ref, res)

    def test_descriptor_side_effect(self):
        # This pattern (readonly descriptor but writable value in `__dict__`) is
        # from scipy `_make_tuple_bunch`:
        # https://github.com/scipy/scipy/blob/maintenance/1.9.x/scipy/_lib/_bunch.py#L32-L226
        def fget(obj):
            return obj.__dict__["field"]

        class MyClass:
            def __init__(self, n):
                self.__dict__["field"] = n

            field = property(fget)

        def fn(x):
            obj = MyClass(42)
            return x + obj.field, obj

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_t, ref_obj = fn(x)
        res_t, res_obj = opt_fn(x)
        self.assertEqual(ref_t, res_t)
        self.assertEqual(ref_obj.field, res_obj.field)

    def test_data_descriptor_priority_over_instance_dict(self):
        # CPython: data descriptors on the type take priority over instance
        # dict values. Verify Dynamo follows this ordering.
        class Foo:
            @property
            def x(self):
                return 10

        foo = Foo()
        # Manually put a different value in the instance dict.
        # The property (data descriptor) should still win.
        foo.__dict__["x"] = 999

        def fn(t):
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        self.assertEqual(ref, t + 10)

    def test_instance_dict_priority_over_non_data_descriptor(self):
        # CPython: instance dict values take priority over non-data
        # descriptors (those with only __get__, no __set__/__delete__).
        class Desc:
            def __init__(self, val):
                self.val = val

            def __get__(self, obj, cls):
                return self.val * 100

        class Foo:
            x = Desc(7)

        foo = Foo()
        # Instance dict value should shadow the non-data descriptor.
        foo.__dict__["x"] = 10

        def fn(t):
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        # CPython: instance dict wins → foo.x is 10, not 700
        self.assertEqual(ref, t + 10)

    def test_user_defined_data_descriptor(self):
        # A user-defined data descriptor (has __get__ + __set__) on the type
        # should be invoked even when the same name exists in the instance dict.
        class ValidatedAttr:
            def __set_name__(self, owner, name):
                self.storage_name = "_" + name

            def __get__(self, obj, cls):
                if obj is None:
                    return self
                return getattr(obj, self.storage_name)

            def __set__(self, obj, value):
                setattr(obj, self.storage_name, value)

        class Foo:
            x = ValidatedAttr()

            def __init__(self, x):
                self.x = x

        foo = Foo(10)

        def fn(t):
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        self.assertEqual(ref, t + 10)

    def test_property_setter(self):
        class Box:
            def __init__(self, value):
                self._value = torch.tensor([value], dtype=torch.float32)

            @property
            def value(self):
                return self._value + 1

            @value.setter
            def value(self, new_value):
                self._value = new_value * 2

        def fn(b):
            b.value = 5
            return b.value

        b = Box(0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(Box(0))
        res = opt_fn(b)
        self.assertEqual(ref, res)

    def test_property_setter_in_init(self):
        class Clamped:
            def __init__(self, val):
                self.val = val

            @property
            def val(self):
                return self._val

            @val.setter
            def val(self, v):
                self._val = torch.clamp(v, 0, 100)

        def fn(x):
            obj = Clamped(x)
            return obj.val

        x = torch.tensor([200.0])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_assert_size_stride(self):
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(
            AssertionError,
            "expected size 2==5, stride 12==9 at dim=0; expected size 3==6, stride 4==9 at dim=1; expected size 4==7, stride 1==10 at dim=2",
        ):
            torch._C._dynamo.guards.assert_size_stride(x, (5, 6, 7), (9, 9, 10))
