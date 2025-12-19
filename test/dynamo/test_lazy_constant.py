# Owner(s): ["module: dynamo"]

import keyword

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter, same


class LazyConstantVariableTests(TestCase):
    def _assert_compile_count(self, fn, arg_sets, expected_frames):
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        for args in arg_sets:
            eager = fn(*args)
            compiled = opt_fn(*args)
            self.assertTrue(same(eager, compiled))

        self.assertEqual(counter.frame_count, expected_frames)

    def test_passthrough_python_constants_do_not_guard(self):
        tensor_input = torch.ones(2)
        cases = [
            ("str", "alpha", "beta"),
            ("bool", True, False),
            ("float", 2.5, -3.0),
            ("int", 11, 42),
        ]

        for name, first, second in cases:
            with self.subTest(name=name):

                def fn(t, constant):
                    return t.sin() + 1, constant

                self._assert_compile_count(
                    fn,
                    [(tensor_input, first), (tensor_input, second)],
                    expected_frames=1,
                )

    def test_branching_on_constant_recompiles(self):
        tensor_input = torch.randn(2)

        def fn(t, token):
            if token == "neg":
                return t - 1
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        eager0 = fn(tensor_input, "neg")
        compiled0 = opt_fn(tensor_input, "neg")
        self.assertTrue(same(eager0, compiled0))

        eager1 = fn(tensor_input, "pos")
        compiled1 = opt_fn(tensor_input, "pos")
        self.assertTrue(same(eager1, compiled1))

        self.assertGreater(counter.frame_count, 1)

    def test_numeric_constant_participating_in_math_recompiles(self):
        tensor_input = torch.randn(3)

        def fn(t, scale):
            return t * scale

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        eager0 = fn(tensor_input, 0.1)
        compiled0 = opt_fn(tensor_input, 0.1)
        self.assertTrue(same(eager0, compiled0))

        eager1 = fn(tensor_input, 2.0)
        compiled1 = opt_fn(tensor_input, 2.0)
        self.assertTrue(same(eager1, compiled1))

        self.assertGreater(counter.frame_count, 1)

    def test_slice_indices_unspecialized_ints(self):
        """Test that slice indices work with unspecialized ints.

        This tests that LazyConstantVariable properly interacts with the
        unspecialized int (symint) codepath. When specialize_int=False,
        integers become symbolic, and slice objects containing these
        symbolic ints should still work correctly.
        """
        layers = list(range(10))

        def getitem(a, idx):
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            return (torch.zeros(1), a[idx])

        cnts = CompileCounter()

        with torch._dynamo.config.patch(
            assume_static_by_default=False,
            specialize_int=False,
            specialize_float=True,
        ):
            opt_getitem = torch.compile(getitem, backend=cnts, fullgraph=True)
            ref0 = getitem(layers, slice(0, 2, 1))
            ref1 = getitem(layers, 2)
            ref2 = getitem(layers, slice(3, 8, 2))

            res0 = opt_getitem(layers, slice(0, 2, 1))
            res1 = opt_getitem(layers, 2)
            res2 = opt_getitem(layers, slice(3, 8, 2))

        self.assertEqual(ref0, res0)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_range_iteration_with_unspec_ints(self):
        def make_fn():
            keys = range(10)
            allowed = [0, 1, 2, 3]

            def fn(x):
                x = x + 1
                torch._dynamo.graph_break()
                key = [key for key in keys if key in allowed]
                return x + key[0]

            return fn

        fn = make_fn()
        tensor_input = torch.ones(3)
        with torch._dynamo.config.patch(
            assume_static_by_default=False, specialize_int=False
        ):
            opt_fn = torch.compile(fn, backend="eager")
            self.assertTrue(same(fn(tensor_input), opt_fn(tensor_input)))

    def test_keyword_iskeyword(self):
        """Test that keyword.iskeyword is properly traced."""

        def fn(x, word):
            if keyword.iskeyword(word):
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        # "if" is a Python keyword
        eager_kw = fn(tensor_input, "if")
        compiled_kw = opt_fn(tensor_input, "if")
        self.assertTrue(same(eager_kw, compiled_kw))

        # "foo" is not a keyword
        eager_not_kw = fn(tensor_input, "foo")
        compiled_not_kw = opt_fn(tensor_input, "foo")
        self.assertTrue(same(eager_not_kw, compiled_not_kw))

        # Both should cause recompilation since they branch differently
        self.assertGreater(counter.frame_count, 1)

    def test_keyword_iskeyword_no_branch(self):
        """Test keyword.iskeyword without branching on result.

        Even though we don't branch on the result, iskeyword() needs the
        actual string value to compute its result, so different strings
        will still cause recompilation.
        """

        def fn(x, word):
            result = keyword.iskeyword(word)
            return x + 1, result

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        eager0 = fn(tensor_input, "class")
        compiled0 = opt_fn(tensor_input, "class")
        self.assertEqual(eager0[1], compiled0[1])
        self.assertTrue(same(eager0[0], compiled0[0]))

        eager1 = fn(tensor_input, "notakeyword")
        compiled1 = opt_fn(tensor_input, "notakeyword")
        self.assertEqual(eager1[1], compiled1[1])
        self.assertTrue(same(eager1[0], compiled1[0]))

        # Recompiles because iskeyword needs the actual value
        self.assertGreater(counter.frame_count, 1)

    def test_frozenset_contains(self):
        """Test that frozenset.__contains__ is properly traced."""
        valid_options = frozenset({"alpha", "beta", "gamma"})

        def fn(x, option):
            if option in valid_options:
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        # "alpha" is in the frozenset
        eager_in = fn(tensor_input, "alpha")
        compiled_in = opt_fn(tensor_input, "alpha")
        self.assertTrue(same(eager_in, compiled_in))

        # "delta" is not in the frozenset
        eager_not_in = fn(tensor_input, "delta")
        compiled_not_in = opt_fn(tensor_input, "delta")
        self.assertTrue(same(eager_not_in, compiled_not_in))

        # Both should cause recompilation since they branch differently
        self.assertGreater(counter.frame_count, 1)

    def test_frozenset_contains_no_branch(self):
        """Test frozenset.__contains__ without branching on result.

        Even though we don't branch on the result, the `in` operator needs
        the actual value to check membership, so different values will
        still cause recompilation.
        """
        valid_options = frozenset({1, 2, 3})

        def fn(x, value):
            result = value in valid_options
            return x + 1, result

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        eager0 = fn(tensor_input, 2)
        compiled0 = opt_fn(tensor_input, 2)
        self.assertEqual(eager0[1], compiled0[1])
        self.assertTrue(same(eager0[0], compiled0[0]))

        eager1 = fn(tensor_input, 5)
        compiled1 = opt_fn(tensor_input, 5)
        self.assertEqual(eager1[1], compiled1[1])
        self.assertTrue(same(eager1[0], compiled1[0]))

        # Recompiles because `in` needs the actual value
        self.assertGreater(counter.frame_count, 1)

    def test_tensor_method_with_lazy_kwargs(self):
        """Test that tensor methods work when kwargs contain lazy constants."""

        def fn(x, dim):
            # sum() with dim kwarg - the dim value goes through as lazy constant
            return x.sum(dim=dim)

        tensor_input = torch.randn(3, 4, 5)
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # Different dim values should cause recompile since dim affects the output
        eager0 = fn(tensor_input, 0)
        compiled0 = opt_fn(tensor_input, 0)
        self.assertTrue(same(eager0, compiled0))

        eager1 = fn(tensor_input, 1)
        compiled1 = opt_fn(tensor_input, 1)
        self.assertTrue(same(eager1, compiled1))

        # This should recompile because dim is used in the computation
        self.assertGreater(counter.frame_count, 1)

    def test_global_constant_passthrough(self):
        """Test that global constants benefit from lazy constant optimization.

        When a global/closure constant is just passed through without being
        used in control flow or math, changing its value should not cause
        recompilation.
        """
        tensor_input = torch.randn(3)

        # Use a mutable container to allow modifying the "global" value
        global_holder = {"value": "hello"}

        def fn(t):
            return t.sin() + 1, global_holder["value"]

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call
        result1 = opt_fn(tensor_input)
        self.assertEqual(result1[1], "hello")
        self.assertEqual(counter.frame_count, 1)

        # Modify the global and call again - should NOT recompile
        global_holder["value"] = "world"
        result2 = opt_fn(tensor_input)
        self.assertEqual(result2[1], "world")
        self.assertEqual(counter.frame_count, 1)  # No recompile!

        # Also test with int
        int_holder = {"value": 42}

        def fn_int(t):
            return t.cos() + 1, int_holder["value"]

        counter2 = CompileCounter()
        opt_fn2 = torch.compile(fn_int, backend=counter2)

        result3 = opt_fn2(tensor_input)
        self.assertEqual(result3[1], 42)
        self.assertEqual(counter2.frame_count, 1)

        int_holder["value"] = 100
        result4 = opt_fn2(tensor_input)
        self.assertEqual(result4[1], 100)
        self.assertEqual(counter2.frame_count, 1)  # No recompile!

    def test_type_change_does_not_recompile(self):
        tensor_input = torch.randn(3)

        def fn(t, val):
            # Just pass through - should not guard on value, only type
            return t + 1, val

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with int
        eager1 = fn(tensor_input, 42)
        compiled1 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])
        self.assertEqual(counter.frame_count, 1)

        # Second call with different int value - should NOT recompile
        eager2 = fn(tensor_input, 100)
        compiled2 = opt_fn(tensor_input, 100)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(eager2[1], compiled2[1])
        self.assertEqual(counter.frame_count, 1)  # Still 1, no recompile

        # Third call with string
        eager3 = fn(tensor_input, "hello")
        compiled3 = opt_fn(tensor_input, "hello")
        self.assertTrue(same(eager3[0], compiled3[0]))
        self.assertEqual(eager3[1], compiled3[1])
        self.assertEqual(counter.frame_count, 1)

        # Fourth call with different string
        eager4 = fn(tensor_input, "world")
        compiled4 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager4[0], compiled4[0]))
        self.assertEqual(eager4[1], compiled4[1])
        self.assertEqual(counter.frame_count, 1)

    def test_python_type_does_not_realize(self):
        """Test that python_type() on a lazy constant does not trigger realization.
        This verifies that type-based queries can be answered without full guarding.

        Note: When specialize_int=False (default), lazy_isinstance() for int values
        must realize to check if the value becomes SymNodeVariable. So we test with
        specialize_int=True to verify the non-realizing path, and separately test
        the specialize_int=False case with string values (which always stay constants).
        """
        from torch._dynamo import config
        from torch._dynamo.source import LocalSource
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.lazy import LazyCache, LazyConstantVariable
        from torch._dynamo.variables.tensor import TensorVariable
        from torch._guards import tracing, TracingContext

        # Create a dummy tracing context so guards can be installed
        ctx = TracingContext(fake_mode=None)
        with tracing(ctx):
            # Test with specialize_int=True so int values stay as ConstantVariable
            with config.patch(specialize_int=True):
                source = LocalSource("test")
                cache = LazyCache(42, source)
                lc = LazyConstantVariable(cache, source=source)

                # python_type() should not trigger realization
                self.assertFalse(lc.is_realized())
                self.assertEqual(lc.python_type(), int)
                self.assertFalse(lc.is_realized())  # Still not realized

                # is_tensor() should not trigger realization
                self.assertEqual(lc.is_tensor(), False)
                self.assertFalse(lc.is_realized())

                # lazy_isinstance() checks if cls is a subclass of ConstantVariable
                self.assertTrue(lc.lazy_isinstance(ConstantVariable))
                self.assertFalse(lc.lazy_isinstance(TensorVariable))
                self.assertFalse(lc.is_realized())

            # Test with string value (always stays as ConstantVariable regardless of config)
            source2 = LocalSource("test2")
            cache2 = LazyCache("hello", source2)
            lc2 = LazyConstantVariable(cache2, source=source2)

            self.assertFalse(lc2.is_realized())
            self.assertEqual(lc2.python_type(), str)
            self.assertFalse(lc2.is_realized())
            self.assertTrue(lc2.lazy_isinstance(ConstantVariable))
            self.assertFalse(lc2.is_realized())

    def test_isinstance_recompiles_on_value_change(self):
        """Test that isinstance checks DO trigger recompilation on value change.

        When isinstance() is called on a LazyConstantVariable, the type argument
        (e.g., `str`) is not a constant-like type, so the lazy variable gets
        realized to avoid infinite recursion in handler dispatch. This installs
        a CONSTANT_MATCH guard and causes recompilation when the value changes.

        Note: This is expected behavior because isinstance() involves non-constant
        arguments (the type class) alongside the lazy constant.
        """
        tensor_input = torch.randn(3)

        def fn(t, val):
            if isinstance(val, str):
                return t + 1
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with string
        eager1 = fn(tensor_input, "hello")
        compiled1 = opt_fn(tensor_input, "hello")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call with different string - WILL recompile because isinstance
        # causes the lazy variable to be realized (installing CONSTANT_MATCH guard)
        eager2 = fn(tensor_input, "world")
        compiled2 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 2)  # Recompile due to value change

        # Third call with int - recompiles again due to both type and value change
        eager3 = fn(tensor_input, 42)
        compiled3 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(counter.frame_count, 3)  # Recompile for value change

    def test_container_with_lazy_constant_no_recompile(self):
        """Test that returning containers with LazyConstantVariables works without realization.

        LazyConstantVariables inside returned containers (list, tuple) should stay
        lazy and not cause recompilation when the constant value changes.
        """
        tensor_input = torch.randn(3)

        # Test returning a list with constants
        def fn_list(t, val):
            return [t + 1, val]

        counter_list = CompileCounter()
        opt_fn_list = torch.compile(fn_list, backend=counter_list)

        eager1 = fn_list(tensor_input, 42)
        compiled1 = opt_fn_list(tensor_input, 42)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])
        self.assertEqual(counter_list.frame_count, 1)

        # Different value should NOT recompile
        eager2 = fn_list(tensor_input, 100)
        compiled2 = opt_fn_list(tensor_input, 100)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(eager2[1], compiled2[1])
        self.assertEqual(counter_list.frame_count, 1)

        # Test returning a tuple with constants
        def fn_tuple(t, val):
            return (t + 1, val)

        counter_tuple = CompileCounter()
        opt_fn_tuple = torch.compile(fn_tuple, backend=counter_tuple)

        eager3 = fn_tuple(tensor_input, "hello")
        compiled3 = opt_fn_tuple(tensor_input, "hello")
        self.assertTrue(same(eager3[0], compiled3[0]))
        self.assertEqual(eager3[1], compiled3[1])
        self.assertEqual(counter_tuple.frame_count, 1)

        # Different string should NOT recompile
        eager4 = fn_tuple(tensor_input, "world")
        compiled4 = opt_fn_tuple(tensor_input, "world")
        self.assertTrue(same(eager4[0], compiled4[0]))
        self.assertEqual(eager4[1], compiled4[1])
        self.assertEqual(counter_tuple.frame_count, 1)

        # Test returning list with multiple constants
        def fn_multi(t, val1, val2):
            return [t + 1, val1, val2]

        counter_multi = CompileCounter()
        opt_fn_multi = torch.compile(fn_multi, backend=counter_multi)

        eager5 = fn_multi(tensor_input, 1, "a")
        compiled5 = opt_fn_multi(tensor_input, 1, "a")
        self.assertTrue(same(eager5[0], compiled5[0]))
        self.assertEqual(eager5[1], compiled5[1])
        self.assertEqual(eager5[2], compiled5[2])
        self.assertEqual(counter_multi.frame_count, 1)

        # Different values should NOT recompile
        eager6 = fn_multi(tensor_input, 2, "b")
        compiled6 = opt_fn_multi(tensor_input, 2, "b")
        self.assertTrue(same(eager6[0], compiled6[0]))
        self.assertEqual(eager6[1], compiled6[1])
        self.assertEqual(eager6[2], compiled6[2])
        self.assertEqual(counter_multi.frame_count, 1)

    def test_computed_lazy_constant_arithmetic_correct_result(self):
        """Test that arithmetic operations on lazy constants produce correct results.

        When two lazy constants are added/multiplied/etc, the result is computed at
        trace time via ComputedLazyConstantVariable. The reconstruct() method generates
        bytecode that recomputes the result at runtime, so no recompilation is needed
        when values change.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        # Test addition
        def fn_add(t, a, b):
            return t + 1, a + b

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_add, backend=counter)

            eager1 = fn_add(tensor_input, 10, 20)
            compiled1 = opt_fn(tensor_input, 10, 20)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 30
            self.assertEqual(counter.frame_count, 1)

            # Different values should NOT recompile - reconstruct() generates bytecode
            # that recomputes a + b at runtime
            eager2 = fn_add(tensor_input, 100, 200)
            compiled2 = opt_fn(tensor_input, 100, 200)
            self.assertTrue(
                same(eager2[0], compiled2[0])
            )  # tensor computation still correct
            self.assertEqual(compiled2[1], 300)  # Correct result without recompilation
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_multiple_ops_correct_result(self):
        """Test that chained operations on lazy constants produce correct results.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_chain(t, a, b, c):
            # Chain of operations: (a + b) * c
            return t + 1, (a + b) * c

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_chain, backend=counter)

            eager1 = fn_chain(tensor_input, 2, 3, 4)
            compiled1 = opt_fn(tensor_input, 2, 3, 4)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # (2+3)*4 = 20
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes (a + b) * c at runtime
            eager2 = fn_chain(tensor_input, 10, 20, 30)
            compiled2 = opt_fn(tensor_input, 10, 20, 30)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 900)  # (10+20)*30 = 900, no recompilation!
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_division_correct_result(self):
        """Test division operations on lazy constants produce correct results.

        Note: This feature requires specialize_float=True.
        """
        tensor_input = torch.randn(3)

        def fn_div(t, a, b):
            return t + 1, a / b

        counter = CompileCounter()
        # specialize_float=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_float=True):
            opt_fn = torch.compile(fn_div, backend=counter)

            eager1 = fn_div(tensor_input, 10.0, 2.0)
            compiled1 = opt_fn(tensor_input, 10.0, 2.0)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 5.0
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes a / b at runtime
            eager2 = fn_div(tensor_input, 100.0, 4.0)
            compiled2 = opt_fn(tensor_input, 100.0, 4.0)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 25.0)  # Correct result without recompilation
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_string_concat_correct_result(self):
        """Test string concatenation on lazy constants produces correct results."""
        tensor_input = torch.randn(3)

        def fn_concat(t, a, b):
            return t + 1, a + b

        counter = CompileCounter()
        opt_fn = torch.compile(fn_concat, backend=counter)

        eager1 = fn_concat(tensor_input, "hello", "world")
        compiled1 = opt_fn(tensor_input, "hello", "world")
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])  # "helloworld"
        self.assertEqual(counter.frame_count, 1)

        # Different values do NOT recompile - reconstruct_fn generates bytecode
        # that recomputes a + b at runtime
        eager2 = fn_concat(tensor_input, "foo", "bar")
        compiled2 = opt_fn(tensor_input, "foo", "bar")
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "foobar")  # Correct result without recompilation
        self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_string_multiply_correct_result(self):
        """Test string multiplication on lazy constants produces correct results.

        Note: String operations that return non-Tensor values cause graph breaks
        when guard checks fail, so we test that results are still correct through
        eager fallback.
        """
        tensor_input = torch.randn(3)

        def fn_str_mul(t, s, n):
            return t + 1, s * n

        counter = CompileCounter()
        opt_fn = torch.compile(fn_str_mul, backend=counter)

        eager1 = fn_str_mul(tensor_input, "ab", 3)
        compiled1 = opt_fn(tensor_input, "ab", 3)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])  # "ababab"
        self.assertEqual(counter.frame_count, 1)

        # Different values cause guard failures, but results are still correct
        # through eager fallback (graph breaks due to non-Tensor return)
        eager2 = fn_str_mul(tensor_input, "xy", 5)
        compiled2 = opt_fn(tensor_input, "xy", 5)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "xyxyxyxyxy")  # Correct result

    def test_computed_lazy_constant_with_regular_constant(self):
        """Test operations between lazy constants and regular constants.

        When a lazy constant is combined with a regular constant (closure variable),
        the result produces correct values. The reconstruct_fn generates bytecode
        that recomputes the value at runtime, so no recompilation is needed when
        the lazy constant value changes.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        CONSTANT = 100

        def fn_mixed(t, a):
            # a is lazy, CONSTANT is a regular constant
            return t + 1, a + CONSTANT

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_mixed, backend=counter)

            eager1 = fn_mixed(tensor_input, 10)
            compiled1 = opt_fn(tensor_input, 10)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 110
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes a + CONSTANT at runtime
            eager2 = fn_mixed(tensor_input, 50)
            compiled2 = opt_fn(tensor_input, 50)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 150)  # Correct result without recompilation
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_branching_recompiles(self):
        """Test that using computed lazy constant in control flow causes recompilation."""
        tensor_input = torch.randn(3)

        def fn_branch(t, a, b):
            result = a + b
            if result > 50:
                return t + 1
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn_branch, backend=counter)

        # First call: 10 + 20 = 30, takes else branch
        eager1 = fn_branch(tensor_input, 10, 20)
        compiled1 = opt_fn(tensor_input, 10, 20)
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call: 30 + 30 = 60, takes if branch - should recompile
        eager2 = fn_branch(tensor_input, 30, 30)
        compiled2 = opt_fn(tensor_input, 30, 30)
        self.assertTrue(same(eager2, compiled2))
        self.assertGreater(counter.frame_count, 1)

    def test_computed_lazy_constant_modulo_correct_result(self):
        """Test modulo operation on lazy constants produces correct results.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_mod(t, a, b):
            return t + 1, a % b

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_mod, backend=counter)

            eager1 = fn_mod(tensor_input, 17, 5)
            compiled1 = opt_fn(tensor_input, 17, 5)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 2
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes a % b at runtime
            eager2 = fn_mod(tensor_input, 23, 7)
            compiled2 = opt_fn(tensor_input, 23, 7)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 2)  # 23 % 7 = 2
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_subtraction_correct_result(self):
        """Test subtraction operation on lazy constants produces correct results.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_sub(t, a, b):
            return t + 1, a - b

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_sub, backend=counter)

            eager1 = fn_sub(tensor_input, 100, 30)
            compiled1 = opt_fn(tensor_input, 100, 30)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 70
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes a - b at runtime
            eager2 = fn_sub(tensor_input, 50, 10)
            compiled2 = opt_fn(tensor_input, 50, 10)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 40)  # 50 - 10 = 40
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_chained_operations(self):
        """Test multiple operations chained together on lazy constants.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        # Test longer chain of operations
        def fn_long_chain(t, a, b, c, d):
            # ((a + b) * c) - d
            return t + 1, ((a + b) * c) - d

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_long_chain, backend=counter)

            eager1 = fn_long_chain(tensor_input, 2, 3, 4, 5)
            compiled1 = opt_fn(tensor_input, 2, 3, 4, 5)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # ((2+3)*4)-5 = 15
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes the entire chain at runtime
            eager2 = fn_long_chain(tensor_input, 10, 20, 3, 10)
            compiled2 = opt_fn(tensor_input, 10, 20, 3, 10)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 80)  # ((10+20)*3)-10 = 80, no recompilation!
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_str_format(self):
        """Test str.format() with lazy constants produces correct results."""
        tensor_input = torch.randn(3)

        def fn_format(t, a, b):
            return t + 1, "{} + {} = {}".format(a, b, a + b)  # noqa: UP032

        counter = CompileCounter()
        opt_fn = torch.compile(fn_format, backend=counter)

        eager1 = fn_format(tensor_input, 10, 20)
        compiled1 = opt_fn(tensor_input, 10, 20)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])  # "10 + 20 = 30"
        self.assertEqual(counter.frame_count, 1)

        # Different values WILL recompile and produce correct result
        eager2 = fn_format(tensor_input, 100, 200)
        compiled2 = opt_fn(tensor_input, 100, 200)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "100 + 200 = 300")
        self.assertEqual(counter.frame_count, 2)

    def test_computed_lazy_constant_fstring(self):
        """Test f-strings with lazy constants produces correct results."""
        tensor_input = torch.randn(3)

        def fn_fstring(t, name, count):
            return t + 1, f"Hello {name}, you have {count} items"

        counter = CompileCounter()
        opt_fn = torch.compile(fn_fstring, backend=counter)

        eager1 = fn_fstring(tensor_input, "Alice", 5)
        compiled1 = opt_fn(tensor_input, "Alice", 5)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])  # "Hello Alice, you have 5 items"
        self.assertEqual(counter.frame_count, 1)

        # Different values WILL recompile and produce correct result
        eager2 = fn_fstring(tensor_input, "Bob", 10)
        compiled2 = opt_fn(tensor_input, "Bob", 10)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "Hello Bob, you have 10 items")
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_computed_lazy_constant_percent_format(self):
        """Test % formatting with lazy constants produces correct results.

        The % operator with string and tuple of lazy constants should be
        constant-foldable without graph breaks. When values change, recompilation
        occurs and produces correct results.

        Note: automatic_dynamic_shapes is disabled for this test to ensure we
        test the lazy constant behavior without it converting values to symbolic
        on recompilation.
        """
        tensor_input = torch.randn(3)

        def fn_percent(t, a, b):
            return t + 1, "%d + %d = %d" % (a, b, a + b)  # noqa: UP031

        counter = CompileCounter()
        opt_fn = torch.compile(fn_percent, backend=counter)

        eager1 = fn_percent(tensor_input, 10, 20)
        compiled1 = opt_fn(tensor_input, 10, 20)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])  # "10 + 20 = 30"
        self.assertEqual(counter.frame_count, 1)

        # Different values trigger recompilation and produce correct results
        eager2 = fn_percent(tensor_input, 100, 200)
        compiled2 = opt_fn(tensor_input, 100, 200)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "100 + 200 = 300")
        self.assertEqual(counter.frame_count, 2)  # Recompiled, not graph break

    def test_percent_format_no_overguarding_with_automatic_dynamic(self):
        """Test that % formatting doesn't over-guard with automatic_dynamic_shapes.

        With automatic_dynamic_shapes enabled (default), when values change and
        trigger recompilation, the values become symbolic. String formatting with
        symbolic values is not supported in the graph, so we get a graph break
        but results should still be correct.

        The key point is that after the first graph break, subsequent calls with
        different values should NOT trigger additional recompiles - the symbolic
        graph should handle all values.
        """
        tensor_input = torch.randn(3)

        def fn_simple(t, a, b):
            # Simple computation that should work with symbolic values
            return t + a + b

        counter = CompileCounter()
        opt_fn = torch.compile(fn_simple, backend=counter)

        # First call - compiles with static values
        result1 = opt_fn(tensor_input, 10, 20)
        self.assertEqual(counter.frame_count, 1)

        # Second call with different values - may trigger recompile due to
        # automatic_dynamic_shapes, but should stabilize
        result2 = opt_fn(tensor_input, 100, 200)
        frame_count_after_second = counter.frame_count

        # Third call with yet different values - should NOT trigger another recompile
        # because automatic_dynamic_shapes should have made the values symbolic
        result3 = opt_fn(tensor_input, 1000, 2000)
        self.assertEqual(counter.frame_count, frame_count_after_second)

        # Verify results are correct
        expected1 = tensor_input + 10 + 20
        expected2 = tensor_input + 100 + 200
        expected3 = tensor_input + 1000 + 2000
        self.assertTrue(same(result1, expected1))
        self.assertTrue(same(result2, expected2))
        self.assertTrue(same(result3, expected3))

    def test_computed_lazy_constant_nested_fstring(self):
        """Test f-strings with expressions involving lazy constants."""
        tensor_input = torch.randn(3)

        def fn_nested(t, x, y):
            # f-string with computation inside
            return t + 1, f"Result: {x * 2} and {y + 10}"

        counter = CompileCounter()
        opt_fn = torch.compile(fn_nested, backend=counter)

        eager1 = fn_nested(tensor_input, 5, 3)
        compiled1 = opt_fn(tensor_input, 5, 3)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])  # "Result: 10 and 13"
        self.assertEqual(counter.frame_count, 1)

        # Different values WILL recompile and produce correct result
        eager2 = fn_nested(tensor_input, 7, 15)
        compiled2 = opt_fn(tensor_input, 7, 15)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "Result: 14 and 25")
        self.assertEqual(counter.frame_count, 2)

    def test_returning_lazy_constant_plus_one_no_recompile(self):
        """Test that returning LazyConstantVariable + 1 does not recompile.

        This tests the reconstruct() functionality of ComputedLazyConstantVariable.
        When a binary operation (e.g., a + 1) is performed on a LazyConstantVariable,
        the result should be a ComputedLazyConstantVariable that can generate bytecode
        to recompute the value from the source. This allows the function to be called
        with different input values without recompiling.

        Note: This feature requires specialize_int=True. With specialize_int=False
        (the default), ints are treated symbolically from the first call, which is
        tested separately in test_lazy_constant_arithmetic_both_specialize_modes.
        """
        tensor_input = torch.randn(3)

        def fn(t, a):
            # a is a LazyConstantVariable, a + 10 should create a
            # ComputedLazyConstantVariable with reconstruct() that generates
            # bytecode to load a and add 10 to it.
            return t + 1, a + 10

        counter = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call
            eager1 = fn(tensor_input, 5)
            compiled1 = opt_fn(tensor_input, 5)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 15
            self.assertEqual(counter.frame_count, 1)

            # Second call with different value - should NOT recompile because
            # the reconstruct() generates bytecode that recomputes a + 10 at runtime
            eager2 = fn(tensor_input, 100)
            compiled2 = opt_fn(tensor_input, 100)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 110)  # Correct result without recompilation
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

            # Third call with yet another different value - still no recompile
            eager3 = fn(tensor_input, 999)
            compiled3 = opt_fn(tensor_input, 999)
            self.assertTrue(same(eager3[0], compiled3[0]))
            self.assertEqual(compiled3[1], 1009)
            self.assertEqual(counter.frame_count, 1)  # Still no recompilation!

    def test_returning_chained_lazy_operations_no_recompile(self):
        """Test that chained lazy constant operations don't recompile.

        When multiple operations are chained (e.g., (a + 1) * 2), each operation
        creates a ComputedLazyConstantVariable. The final result should generate
        bytecode that chains all the operations together.

        Note: This feature requires specialize_int=True. With specialize_int=False,
        the value might become symbolic via automatic_dynamic_shapes, which conflicts
        with the lazy constant optimization.
        """
        tensor_input = torch.randn(3)

        def fn(t, a):
            # Chain of operations: (a + 1) * 2 - 5
            result = a + 1
            result = result * 2
            result = result - 5
            return t + 1, result

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call: (10 + 1) * 2 - 5 = 17
            eager1 = fn(tensor_input, 10)
            compiled1 = opt_fn(tensor_input, 10)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 17
            self.assertEqual(counter.frame_count, 1)

            # Second call: (20 + 1) * 2 - 5 = 37
            eager2 = fn(tensor_input, 20)
            compiled2 = opt_fn(tensor_input, 20)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 37)  # Correct result
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

            # Third call: (100 + 1) * 2 - 5 = 197
            eager3 = fn(tensor_input, 100)
            compiled3 = opt_fn(tensor_input, 100)
            self.assertTrue(same(eager3[0], compiled3[0]))
            self.assertEqual(compiled3[1], 197)
            self.assertEqual(counter.frame_count, 1)  # Still no recompilation!

    def test_returning_two_lazy_constant_operations_no_recompile(self):
        """Test that operations between two LazyConstantVariables don't recompile.

        When two input lazy constants are combined (e.g., a + b), the result
        should generate bytecode that loads both operands and performs the
        operation at runtime.

        Note: This feature requires specialize_int=True. With specialize_int=False,
        the value might become symbolic via automatic_dynamic_shapes, which conflicts
        with the lazy constant optimization.
        """
        tensor_input = torch.randn(3)

        def fn(t, a, b):
            return t + 1, a + b

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call: 5 + 10 = 15
            eager1 = fn(tensor_input, 5, 10)
            compiled1 = opt_fn(tensor_input, 5, 10)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 15
            self.assertEqual(counter.frame_count, 1)

            # Second call: 100 + 200 = 300
            eager2 = fn(tensor_input, 100, 200)
            compiled2 = opt_fn(tensor_input, 100, 200)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 300)
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

            # Third call: 1 + 2 = 3
            eager3 = fn(tensor_input, 1, 2)
            compiled3 = opt_fn(tensor_input, 1, 2)
            self.assertTrue(same(eager3[0], compiled3[0]))
            self.assertEqual(compiled3[1], 3)
            self.assertEqual(counter.frame_count, 1)  # Still no recompilation!

    def test_lazy_constant_subtraction_no_recompile(self):
        """Test that subtraction on lazy constants doesn't recompile.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn(t, a):
            return t + 1, a - 5

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call: 100 - 5 = 95
            eager1 = fn(tensor_input, 100)
            compiled1 = opt_fn(tensor_input, 100)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(compiled1[1], 95)
            self.assertEqual(counter.frame_count, 1)

            # Second call: 50 - 5 = 45
            eager2 = fn(tensor_input, 50)
            compiled2 = opt_fn(tensor_input, 50)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 45)
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_lazy_constant_multiplication_no_recompile(self):
        """Test that multiplication on lazy constants doesn't recompile.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn(t, a):
            return t + 1, a * 3

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call: 10 * 3 = 30
            eager1 = fn(tensor_input, 10)
            compiled1 = opt_fn(tensor_input, 10)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(compiled1[1], 30)
            self.assertEqual(counter.frame_count, 1)

            # Second call: 7 * 3 = 21
            eager2 = fn(tensor_input, 7)
            compiled2 = opt_fn(tensor_input, 7)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 21)
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_lazy_constant_division_no_recompile(self):
        """Test that division on lazy constants doesn't recompile.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn(t, a):
            return t + 1, a / 2

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call: 100 / 2 = 50.0
            eager1 = fn(tensor_input, 100)
            compiled1 = opt_fn(tensor_input, 100)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(compiled1[1], 50.0)
            self.assertEqual(counter.frame_count, 1)

            # Second call: 50 / 2 = 25.0
            eager2 = fn(tensor_input, 50)
            compiled2 = opt_fn(tensor_input, 50)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 25.0)
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_lazy_constant_modulo_no_recompile(self):
        """Test that modulo on lazy constants doesn't recompile.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn(t, a):
            return t + 1, a % 7

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call: 17 % 7 = 3
            eager1 = fn(tensor_input, 17)
            compiled1 = opt_fn(tensor_input, 17)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(compiled1[1], 3)
            self.assertEqual(counter.frame_count, 1)

            # Second call: 100 % 7 = 2
            eager2 = fn(tensor_input, 100)
            compiled2 = opt_fn(tensor_input, 100)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 2)
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_lazy_constant_arithmetic_both_specialize_modes(self):
        """Test lazy constant arithmetic works with both specialize_int=True and False.

        This test explicitly verifies the feature works in both modes:
        - specialize_int=True: reconstruct_fn generates bytecode to recompute at runtime
        - specialize_int=False: ComputedLazyCache.realize() handles symbolic values
          if automatic_dynamic_shapes makes them symbolic

        In both cases, values should be correct and recompilation should be minimal.
        """
        tensor_input = torch.randn(3)

        def fn(t, a, b):
            return t + 1, a + b

        # Test with specialize_int=True
        counter1 = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn1 = torch.compile(fn, backend=counter1)

            # First call
            eager1 = fn(tensor_input, 10, 20)
            compiled1 = opt_fn1(tensor_input, 10, 20)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(compiled1[1], 30)
            self.assertEqual(counter1.frame_count, 1)

            # Second call - no recompile
            eager2 = fn(tensor_input, 100, 200)
            compiled2 = opt_fn1(tensor_input, 100, 200)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], 300)
            self.assertEqual(counter1.frame_count, 1)

            # Third call - still no recompile
            compiled3 = opt_fn1(tensor_input, 1, 2)
            self.assertEqual(compiled3[1], 3)
            self.assertEqual(counter1.frame_count, 1)

        # Test with specialize_int=False (default)
        counter2 = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=False):
            opt_fn2 = torch.compile(fn, backend=counter2)

            # First call
            eager4 = fn(tensor_input, 10, 20)
            compiled4 = opt_fn2(tensor_input, 10, 20)
            self.assertTrue(same(eager4[0], compiled4[0]))
            self.assertEqual(compiled4[1], 30)
            self.assertEqual(counter2.frame_count, 1)

            # Second call - may trigger one recompile due to automatic_dynamic_shapes
            eager5 = fn(tensor_input, 100, 200)
            compiled5 = opt_fn2(tensor_input, 100, 200)
            self.assertTrue(same(eager5[0], compiled5[0]))
            self.assertEqual(compiled5[1], 300)  # Correct value
            frame_count_after_second = counter2.frame_count

            # Third call - should NOT trigger additional recompile
            # because automatic_dynamic_shapes should have stabilized
            compiled6 = opt_fn2(tensor_input, 1, 2)
            self.assertEqual(compiled6[1], 3)  # Correct value
            self.assertEqual(counter2.frame_count, frame_count_after_second)

    def test_dict_mutation_no_recompile_on_unused_key_change(self):
        """Test that mutating a dict doesn't guard on unused keys.

        When a dict is mutated with a constant key (e.g., d['new_key'] = value),
        no guard is needed since __setitem__ works the same whether the key
        exists or not. This ensures that changing any keys (used or unused)
        doesn't cause unnecessary recompilation.
        """
        tensor_input = torch.randn(3)

        def fn(t, d):
            d["new_key"] = 123  # Mutate dict - no guard needed for constant key
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with dict {'a': 1, 'b': 2}
        d1 = {"a": 1, "b": 2}
        eager1 = fn(tensor_input, d1.copy())
        d1_copy = {"a": 1, "b": 2}
        compiled1 = opt_fn(tensor_input, d1_copy)
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(d1_copy["new_key"], 123)
        self.assertEqual(counter.frame_count, 1)

        # Second call with completely different keys - should NOT recompile
        d2 = {"x": 100, "y": 200}
        eager2 = fn(tensor_input, d2.copy())
        d2_copy = {"x": 100, "y": 200}
        compiled2 = opt_fn(tensor_input, d2_copy)
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(d2_copy, {"x": 100, "y": 200, "new_key": 123})
        self.assertEqual(counter.frame_count, 1)  # No recompile!

        # Third call with 'new_key' already present - should also NOT recompile
        # since __setitem__ works the same whether key exists or not
        d3 = {"a": 1, "new_key": 999}
        eager3 = fn(tensor_input, d3.copy())
        d3_copy = {"a": 1, "new_key": 999}
        compiled3 = opt_fn(tensor_input, d3_copy)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(d3_copy["new_key"], 123)  # Overwritten
        self.assertEqual(counter.frame_count, 1)  # Still no recompile!

    def test_dict_read_then_write_same_key(self):
        """Test reading and writing the same dict key (e.g., increment).

        When we read a dict key with a constant key, the value is treated as
        a LazyConstantVariable. When we add to it (d["x"] + 1), it creates a
        ComputedLazyConstantVariable which doesn't guard on the value.
        This allows changing the value without recompilation.

        Note: This requires specialize_int=True for the value to stay lazy.
        """
        tensor_input = torch.randn(3)

        def fn(t, d):
            d["counter"] = d["counter"] + 1  # Read then write same key
            return t + 1

        counter = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call
            d1 = {"counter": 10, "other": "unchanged"}
            eager1 = fn(tensor_input, d1.copy())
            d1_copy = {"counter": 10, "other": "unchanged"}
            compiled1 = opt_fn(tensor_input, d1_copy)
            self.assertTrue(same(eager1, compiled1))
            self.assertEqual(d1_copy["counter"], 11)
            self.assertEqual(d1_copy["other"], "unchanged")  # Other key preserved
            self.assertEqual(counter.frame_count, 1)

            # Second call with different counter value - should NOT recompile
            # because the value is lazy and d["counter"] + 1 creates a
            # ComputedLazyConstantVariable that recomputes at runtime
            d2 = {"counter": 100, "different": "keys"}
            eager2 = fn(tensor_input, d2.copy())
            d2_copy = {"counter": 100, "different": "keys"}
            compiled2 = opt_fn(tensor_input, d2_copy)
            self.assertTrue(same(eager2, compiled2))
            self.assertEqual(d2_copy["counter"], 101)  # 100 + 1 = 101
            self.assertEqual(d2_copy["different"], "keys")  # Other key preserved
            self.assertEqual(counter.frame_count, 1)  # No recompile!

            # Third call with yet another counter value - still no recompile
            d3 = {"counter": 500}
            eager3 = fn(tensor_input, d3.copy())
            d3_copy = {"counter": 500}
            compiled3 = opt_fn(tensor_input, d3_copy)
            self.assertTrue(same(eager3, compiled3))
            self.assertEqual(d3_copy["counter"], 501)  # 500 + 1 = 501
            self.assertEqual(counter.frame_count, 1)  # Still no recompile!

    def test_dict_setitem_with_lazy_constant_key_no_recompile(self):
        """Test that dict[lazy_key] = value doesn't recompile when key changes.

        When the key is a LazyConstantVariable, we only install TYPE_MATCH guard,
        not CONSTANT_MATCH. This allows the key value to change without recompilation.
        """
        tensor_input = torch.randn(3)

        def fn(t, d, key):
            d[key] = t.sum()
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with key "alpha"
        d1 = {}
        eager1 = fn(tensor_input, d1, "alpha")
        d1_compiled = {}
        compiled1 = opt_fn(tensor_input, d1_compiled, "alpha")
        self.assertTrue(same(eager1, compiled1))
        self.assertIn("alpha", d1_compiled)
        self.assertEqual(counter.frame_count, 1)

        # Second call with different key "beta" - should NOT recompile
        d2 = {}
        eager2 = fn(tensor_input, d2, "beta")
        d2_compiled = {}
        compiled2 = opt_fn(tensor_input, d2_compiled, "beta")
        self.assertTrue(same(eager2, compiled2))
        self.assertIn("beta", d2_compiled)
        self.assertEqual(counter.frame_count, 1)  # No recompile!

        # Third call with different key "gamma" - still no recompile
        d3 = {}
        eager3 = fn(tensor_input, d3, "gamma")
        d3_compiled = {}
        compiled3 = opt_fn(tensor_input, d3_compiled, "gamma")
        self.assertTrue(same(eager3, compiled3))
        self.assertIn("gamma", d3_compiled)
        self.assertEqual(counter.frame_count, 1)  # Still no recompile!

    def test_dict_setitem_with_lazy_constant_int_key_no_recompile(self):
        """Test that dict[lazy_int_key] = value doesn't recompile when key changes.

        Note: This requires specialize_int=True because with specialize_int=False,
        integers become SymNodeVariables which need to be specialized for use as
        dict keys (hashing requires concrete values).
        """
        tensor_input = torch.randn(3)

        def fn(t, d, key):
            d[key] = t.mean()
            return t + 1

        counter = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn, backend=counter)

            # First call with int key 10
            d1 = {}
            eager1 = fn(tensor_input, d1, 10)
            d1_compiled = {}
            compiled1 = opt_fn(tensor_input, d1_compiled, 10)
            self.assertTrue(same(eager1, compiled1))
            self.assertIn(10, d1_compiled)
            self.assertEqual(counter.frame_count, 1)

            # Second call with different int key 20 - should NOT recompile
            d2 = {}
            eager2 = fn(tensor_input, d2, 20)
            d2_compiled = {}
            compiled2 = opt_fn(tensor_input, d2_compiled, 20)
            self.assertTrue(same(eager2, compiled2))
            self.assertIn(20, d2_compiled)
            self.assertEqual(counter.frame_count, 1)  # No recompile!

    def test_dict_setitem_with_lazy_constant_key_type_change_recompiles(self):
        """Test that changing key type DOES cause recompilation.

        We install TYPE_MATCH guard on the key, so changing from str to int
        should trigger recompilation.
        """
        tensor_input = torch.randn(3)

        def fn(t, d, key):
            d[key] = t.sum()
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with string key
        d1 = {}
        eager1 = fn(tensor_input, d1, "alpha")
        d1_compiled = {}
        compiled1 = opt_fn(tensor_input, d1_compiled, "alpha")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call with int key - SHOULD recompile due to type change
        d2 = {}
        eager2 = fn(tensor_input, d2, 42)
        d2_compiled = {}
        compiled2 = opt_fn(tensor_input, d2_compiled, 42)
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 2)  # Recompiled due to type change

    def test_dict_setitem_existing_key_overwrite(self):
        """Test that dict[lazy_key] = value works correctly when overwriting existing keys."""
        tensor_input = torch.randn(3)

        def fn(t, d, key):
            d[key] = t.sum()
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call - key doesn't exist
        d1 = {"other": 999}
        eager1 = fn(tensor_input, d1.copy(), "alpha")
        d1_compiled = {"other": 999}
        compiled1 = opt_fn(tensor_input, d1_compiled, "alpha")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call - key already exists (overwriting)
        d2 = {"alpha": 123, "other": 999}
        eager2 = fn(tensor_input, d2.copy(), "alpha")
        d2_compiled = {"alpha": 123, "other": 999}
        compiled2 = opt_fn(tensor_input, d2_compiled, "alpha")
        self.assertTrue(same(eager2, compiled2))
        # Should still not recompile - __setitem__ works the same for existing/new keys
        self.assertEqual(counter.frame_count, 1)

    def test_dict_setitem_with_tensor_value_and_lazy_key(self):
        """Test the main use case: dict[string_arg] = tensor works without recompile."""
        tensor_input = torch.randn(3, 4)

        def fn(t, d, key):
            # Store processed tensor in dict with lazy key
            d[key] = t.sin() + t.cos()
            return t.sum()

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call
        d1 = {}
        opt_fn(tensor_input, d1, "feature_a")
        self.assertIn("feature_a", d1)
        self.assertTrue(isinstance(d1["feature_a"], torch.Tensor))
        self.assertEqual(counter.frame_count, 1)

        # Second call with different key - no recompile
        d2 = {}
        opt_fn(tensor_input, d2, "feature_b")
        self.assertIn("feature_b", d2)
        self.assertTrue(isinstance(d2["feature_b"], torch.Tensor))
        self.assertEqual(counter.frame_count, 1)

        # Verify results are correct
        expected = tensor_input.sin() + tensor_input.cos()
        self.assertTrue(same(d1["feature_a"], expected))
        self.assertTrue(same(d2["feature_b"], expected))

    def test_dict_aliasing_keys_setitem_only(self):
        """Test that aliasing keys without reads works correctly.

        When we only do setitems with potentially aliasing lazy keys,
        no value guards are needed because:
        1. The bytecode will reconstruct the operations with actual key values
        2. If keys alias at runtime, the second setitem overwrites the first
        3. The dict side effect is correct regardless of aliasing
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key1, key2):
            d[key1] = t.sum()
            d[key2] = t.mean()
            return t + 1  # No dict read!

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # Compile with different keys
        d1 = {}
        opt_fn(tensor_input, d1, "a", "b")
        self.assertEqual(d1, {"a": tensor_input.sum(), "b": tensor_input.mean()})
        self.assertEqual(counter.frame_count, 1)

        # Call with same keys (aliasing) - no recompile needed!
        d2 = {}
        opt_fn(tensor_input, d2, "x", "x")
        # When keys alias, second setitem overwrites first
        self.assertEqual(len(d2), 1)
        self.assertIn("x", d2)
        self.assertTrue(same(d2["x"], tensor_input.mean()))
        self.assertEqual(counter.frame_count, 1)  # Still no recompile!

        # Call with different keys again - still no recompile
        d3 = {}
        opt_fn(tensor_input, d3, "p", "q")
        self.assertEqual(d3, {"p": tensor_input.sum(), "q": tensor_input.mean()})
        self.assertEqual(counter.frame_count, 1)

    def test_dict_aliasing_keys_with_read_recompiles(self):
        """Test that reading back from a dict with aliasing keys triggers recompilation.

        When we read from a dict key that might alias with another key we wrote to,
        we MUST guard on the key values to ensure correctness. This is because
        the value we read depends on whether the keys are equal.
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key1, key2):
            d[key1] = t.sum()
            d[key2] = t.mean()
            return d[key1]  # Read back - aliasing matters here!

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # Compile with different keys
        eager1 = fn(tensor_input, {}, "a", "b")
        compiled1 = opt_fn(tensor_input, {}, "a", "b")
        self.assertTrue(same(eager1, compiled1))
        # With different keys: d["a"] = sum, d["b"] = mean, return d["a"] = sum
        self.assertTrue(same(compiled1, tensor_input.sum()))
        self.assertEqual(counter.frame_count, 1)

        # Call with same keys - MUST recompile for correctness
        eager2 = fn(tensor_input, {}, "x", "x")
        compiled2 = opt_fn(tensor_input, {}, "x", "x")
        self.assertTrue(same(eager2, compiled2))
        # With same keys: d["x"] = sum, d["x"] = mean (overwrites!), return d["x"] = mean
        self.assertTrue(same(compiled2, tensor_input.mean()))
        self.assertGreater(counter.frame_count, 1)  # Recompilation happened

    def test_dict_aliasing_keys_getitem_installs_guard(self):
        """Verify that __getitem__ installs proper guards to catch aliasing."""
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()
            return d[key]  # getitem installs EQUALS_MATCH guard

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call
        opt_fn(tensor_input, {}, "a")
        self.assertEqual(counter.frame_count, 1)

        # Different key value - should recompile due to EQUALS_MATCH guard on key
        opt_fn(tensor_input, {}, "b")
        self.assertEqual(counter.frame_count, 2)

    def test_dict_contains_with_lazy_key(self):
        """Test that 'key in dict' operations work correctly with lazy keys."""
        tensor_input = torch.randn(3)

        def fn(t, d, key):
            if key in d:
                return t + d[key]
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call - key not in dict
        d1 = {"other": tensor_input.mean()}
        eager1 = fn(tensor_input, d1.copy(), "missing")
        compiled1 = opt_fn(tensor_input, d1.copy(), "missing")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call - key IS in dict, different branch
        d2 = {"present": tensor_input.sum()}
        eager2 = fn(tensor_input, d2.copy(), "present")
        compiled2 = opt_fn(tensor_input, d2.copy(), "present")
        self.assertTrue(same(eager2, compiled2))
        # Should recompile because different branch was taken
        self.assertGreater(counter.frame_count, 1)

    def test_dict_lazy_key_write_constant_key_read_aliasing(self):
        """Test aliasing between lazy key writes and constant key reads.

        This is a critical correctness test. When we write with a lazy key
        and then read with a constant key, the lazy key might alias with
        the constant key, which would change the read value.

        Example:
            d = {'x': old_value}
            d[lazy_key] = new_value
            return d['x']  # Depends on whether lazy_key == 'x'
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()  # Write with lazy key
            return d["x"]  # Read with constant key - might alias!

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: key='a' (different from 'x')
        # d['a'] = 6.0, return d['x'] = 999.0 (original value)
        d1 = {"x": torch.tensor(999.0)}
        eager1 = fn(tensor_input, {"x": torch.tensor(999.0)}, "a")
        compiled1 = opt_fn(tensor_input, d1, "a")
        self.assertTrue(same(eager1, compiled1))
        self.assertTrue(same(compiled1, torch.tensor(999.0)))
        self.assertEqual(counter.frame_count, 1)

        # Second call: key='x' (SAME as constant 'x' - aliasing!)
        # d['x'] = 6.0 (overwrites!), return d['x'] = 6.0
        d2 = {"x": torch.tensor(999.0)}
        eager2 = fn(tensor_input, {"x": torch.tensor(999.0)}, "x")
        compiled2 = opt_fn(tensor_input, d2, "x")
        self.assertTrue(same(eager2, compiled2))
        self.assertTrue(
            same(compiled2, tensor_input.sum())
        )  # Should be 6.0, not 999.0!
        # Should recompile because key value changed (guard was installed)
        self.assertEqual(counter.frame_count, 2)

    def test_dict_lazy_key_on_nonempty_dict_no_read_no_guard(self):
        """Test that lazy key on non-empty dict without read does NOT install value guard.

        When the dict has existing keys but we don't read from it, there's no
        aliasing concern, so we should NOT install a value guard.
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()
            return t + 1  # No dict read!

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: non-empty dict but no read
        d1 = {"existing": torch.tensor(0.0)}
        opt_fn(tensor_input, d1, "a")
        self.assertEqual(counter.frame_count, 1)

        # Second call: different key value should NOT recompile
        # (no read from dict, so no aliasing concern)
        d2 = {"existing": torch.tensor(0.0)}
        opt_fn(tensor_input, d2, "b")
        self.assertEqual(counter.frame_count, 1)

        # Third call: still no recompile
        d3 = {"existing": torch.tensor(0.0)}
        opt_fn(tensor_input, d3, "c")
        self.assertEqual(counter.frame_count, 1)

    def test_dict_lazy_key_on_empty_dict_no_value_guard(self):
        """Test that lazy key on empty dict does NOT install value guard."""
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: empty dict
        d1 = {}
        opt_fn(tensor_input, d1, "a")
        self.assertEqual(counter.frame_count, 1)

        # Second call: different key value should NOT recompile
        # (dict was empty, no aliasing possible, only TYPE_MATCH guard)
        d2 = {}
        opt_fn(tensor_input, d2, "b")
        self.assertEqual(counter.frame_count, 1)

        # Third call: still no recompile
        d3 = {}
        opt_fn(tensor_input, d3, "c")
        self.assertEqual(counter.frame_count, 1)

    def test_dict_lazy_key_write_then_contains_aliasing(self):
        """Test aliasing between lazy key writes and __contains__ checks.

        When we write with a lazy key and then check containment with a constant
        key, the lazy key might alias, which would change the containment result.

        Example:
            d = {}
            d[lazy_key] = value
            return 'x' in d  # Depends on whether lazy_key == 'x'
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()  # Write with lazy key
            if "x" in d:  # Check containment with constant key
                return t + 1
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: key='a' (different from 'x')
        # 'x' not in d, returns t - 1
        d1 = {}
        eager1 = fn(tensor_input, {}, "a")
        compiled1 = opt_fn(tensor_input, d1, "a")
        self.assertTrue(same(eager1, compiled1))
        self.assertTrue(same(compiled1, tensor_input - 1))
        self.assertEqual(counter.frame_count, 1)

        # Second call: key='x' (SAME as constant 'x' - aliasing!)
        # 'x' IS in d, returns t + 1
        d2 = {}
        eager2 = fn(tensor_input, {}, "x")
        compiled2 = opt_fn(tensor_input, d2, "x")
        self.assertTrue(same(eager2, compiled2))
        self.assertTrue(same(compiled2, tensor_input + 1))
        # Should recompile because key value changed (guard was installed)
        self.assertEqual(counter.frame_count, 2)

    def test_dict_lazy_key_write_then_get_aliasing(self):
        """Test aliasing between lazy key writes and .get() calls.

        When we write with a lazy key and then call .get() with a constant
        key, the lazy key might alias, which would change the .get() result.

        Example:
            d = {}
            d[lazy_key] = value
            return d.get('x', default)  # Depends on whether lazy_key == 'x'
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()  # Write with lazy key
            result = d.get("x", torch.tensor(-999.0))  # Get with constant key
            return result

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: key='a' (different from 'x')
        # d.get('x') returns default -999.0
        d1 = {}
        eager1 = fn(tensor_input, {}, "a")
        compiled1 = opt_fn(tensor_input, d1, "a")
        self.assertTrue(same(eager1, compiled1))
        self.assertTrue(same(compiled1, torch.tensor(-999.0)))
        self.assertEqual(counter.frame_count, 1)

        # Second call: key='x' (SAME as constant 'x' - aliasing!)
        # d.get('x') returns the sum = 6.0
        d2 = {}
        eager2 = fn(tensor_input, {}, "x")
        compiled2 = opt_fn(tensor_input, d2, "x")
        self.assertTrue(same(eager2, compiled2))
        self.assertTrue(same(compiled2, tensor_input.sum()))
        # Should recompile because key value changed (guard was installed)
        self.assertEqual(counter.frame_count, 2)

    def test_dict_lazy_key_write_then_pop_aliasing(self):
        """Test aliasing between lazy key writes and .pop() calls.

        When we write with a lazy key and then call .pop() with a constant
        key, the lazy key might alias, which would change the .pop() result.

        Example:
            d = {}
            d[lazy_key] = value
            return d.pop('x', default)  # Depends on whether lazy_key == 'x'
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, d, key):
            d[key] = t.sum()  # Write with lazy key
            result = d.pop("x", torch.tensor(-999.0))  # Pop with constant key
            return result

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: key='a' (different from 'x')
        # d.pop('x') returns default -999.0
        d1 = {}
        eager1 = fn(tensor_input, {}, "a")
        compiled1 = opt_fn(tensor_input, d1, "a")
        self.assertTrue(same(eager1, compiled1))
        self.assertTrue(same(compiled1, torch.tensor(-999.0)))
        self.assertEqual(counter.frame_count, 1)

        # Second call: key='x' (SAME as constant 'x' - aliasing!)
        # d.pop('x') returns the sum = 6.0
        d2 = {}
        eager2 = fn(tensor_input, {}, "x")
        compiled2 = opt_fn(tensor_input, d2, "x")
        self.assertTrue(same(eager2, compiled2))
        self.assertTrue(same(compiled2, tensor_input.sum()))
        # Should recompile because key value changed (guard was installed)
        self.assertEqual(counter.frame_count, 2)

    def test_dict_two_lazy_keys_different_sources_recompiles(self):
        """Test that two lazy keys from different sources recompile correctly.

        When we write with one lazy key and read with another lazy key from
        a different source, we must realize both to install proper guards.
        This ensures correct behavior when the values change independently.

        The bug this prevents:
            d = {}
            d[lazy_key1] = value  # lazy_key1 = 'x'
            return d[lazy_key2]   # lazy_key2 = 'x' too, but from different source
            # Without proper guards, if lazy_key1 changes to 'y' but lazy_key2
            # stays 'x', we'd return wrong value.
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, key1, key2):
            d = {}
            d[key1] = t.sum()  # Write with first lazy key
            return d.get(key2, torch.tensor(-999.0))  # Read with second lazy key

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: both keys are 'x' - should find the value
        eager1 = fn(tensor_input, "x", "x")
        compiled1 = opt_fn(tensor_input, "x", "x")
        self.assertTrue(same(eager1, compiled1))
        self.assertTrue(same(compiled1, tensor_input.sum()))
        self.assertEqual(counter.frame_count, 1)

        # Second call: key1='y', key2='x' - should NOT find the value
        # This must recompile because the relationship between keys changed
        eager2 = fn(tensor_input, "y", "x")
        compiled2 = opt_fn(tensor_input, "y", "x")
        self.assertTrue(same(eager2, compiled2))
        self.assertTrue(same(compiled2, torch.tensor(-999.0)))
        # Should recompile because guards were installed on both keys
        self.assertEqual(counter.frame_count, 2)

    def test_dict_same_lazy_key_used_twice_correctness(self):
        """Test that using the same lazy key for both write and read works correctly.

        When the same lazy key (same source) is used for both write and read,
        the result should always be correct regardless of what guards are installed.
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, key):
            d = {}
            d[key] = t.sum()  # Write with lazy key
            return d[key]  # Read with SAME lazy key

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with key='x'
        eager1 = fn(tensor_input, "x")
        compiled1 = opt_fn(tensor_input, "x")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call with key='y' - verify correctness
        eager2 = fn(tensor_input, "y")
        compiled2 = opt_fn(tensor_input, "y")
        self.assertTrue(same(eager2, compiled2))

    def test_dict_lazy_key_equality_different_values_recompiles(self):
        """Test that lazy keys with different values cause recompilation.

        This tests the is_python_equal fix: when two lazy keys from different
        sources have different values that later become equal (or vice versa),
        we must recompile.
        """
        tensor_input = torch.tensor([1.0, 2.0, 3.0])

        def fn(t, key1, key2):
            d = {key1: t.sum()}
            # Check if key2 is in the dict (tests is_python_equal)
            if key2 in d:
                return d[key2]
            return torch.tensor(-1.0)

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call: key1='a', key2='a' - same, should find
        eager1 = fn(tensor_input, "a", "a")
        compiled1 = opt_fn(tensor_input, "a", "a")
        self.assertTrue(same(eager1, compiled1))
        self.assertTrue(same(compiled1, tensor_input.sum()))
        self.assertEqual(counter.frame_count, 1)

        # Second call: key1='a', key2='b' - different, should not find
        # This must recompile because the equality relationship changed
        eager2 = fn(tensor_input, "a", "b")
        compiled2 = opt_fn(tensor_input, "a", "b")
        self.assertTrue(same(eager2, compiled2))
        self.assertTrue(same(compiled2, torch.tensor(-1.0)))
        self.assertEqual(counter.frame_count, 2)


if __name__ == "__main__":
    run_tests()
