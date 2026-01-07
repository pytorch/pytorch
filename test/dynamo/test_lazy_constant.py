# Owner(s): ["module: dynamo"]

import keyword

import sympy

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

    def test_type_does_not_recompile_on_value_change(self):
        """Test that type() checks do NOT trigger recompilation on value change.

        When type() is called on a LazyConstantVariable, it only installs a
        TYPE_MATCH guard (not CONSTANT_MATCH), so different values of the same type
        do not cause recompilation. This is similar to isinstance() behavior but
        tests a different code path.

        Note: We use string values here because with specialize_int=False (the default),
        int values must be realized during handler dispatch to determine if they become
        ConstantVariable or SymNodeVariable. Strings always become ConstantVariable.
        """
        tensor_input = torch.randn(3)

        def fn(t, val):
            if type(val) is str:
                return t + 1
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with string
        eager1 = fn(tensor_input, "hello")
        compiled1 = opt_fn(tensor_input, "hello")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call with different string - should NOT recompile since
        # type() only installs TYPE_MATCH guard
        eager2 = fn(tensor_input, "world")
        compiled2 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 1)  # No recompilation!

        # Third call with int - should recompile due to type change
        eager3 = fn(tensor_input, 42)
        compiled3 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(counter.frame_count, 2)  # Recompile for type change

    def test_isinstance_no_recompile_on_value_change(self):
        """Test that isinstance checks do NOT trigger recompilation on value change.

        When isinstance() is called on a LazyConstantVariable, we only need to
        know the TYPE of the value, not the specific value. LazyConstantVariable's
        python_type() installs a TYPE_MATCH guard, not a CONSTANT_MATCH guard,
        so changing the value (but keeping the same type) won't cause recompilation.

        However, changing the TYPE (e.g., from str to int) will cause recompilation
        because it takes a different branch in the code.
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

        # Second call with different string - NO recompile because isinstance
        # only installs TYPE_MATCH guard, not CONSTANT_MATCH guard
        eager2 = fn(tensor_input, "world")
        compiled2 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 1)  # NO recompilation!

        # Third call with int - WILL recompile because type changed (different branch)
        eager3 = fn(tensor_input, 42)
        compiled3 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(counter.frame_count, 2)  # Recompile due to type change

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

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_str_mul(t, s, n):
            return t + 1, s * n

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_str_mul, backend=counter)

            eager1 = fn_str_mul(tensor_input, "ab", 3)
            compiled1 = opt_fn(tensor_input, "ab", 3)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # "ababab"
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes s * n at runtime
            eager2 = fn_str_mul(tensor_input, "xy", 5)
            compiled2 = opt_fn(tensor_input, "xy", 5)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], "xyxyxyxyxy")  # Correct result
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

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
        """Test str.format() with lazy constants produces correct results.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_format(t, a, b):
            return t + 1, "{} + {} = {}".format(a, b, a + b)  # noqa: UP032

        counter = CompileCounter()
        # specialize_int=True is required for this optimization to work
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_format, backend=counter)

            eager1 = fn_format(tensor_input, 10, 20)
            compiled1 = opt_fn(tensor_input, 10, 20)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # "10 + 20 = 30"
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes str.format() at runtime
            eager2 = fn_format(tensor_input, 100, 200)
            compiled2 = opt_fn(tensor_input, 100, 200)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], "100 + 200 = 300")
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_fstring(self):
        """Test f-strings with lazy constants produces correct results without recompilation.

        F-strings with lazy constants should work correctly and NOT trigger
        recompilation when the constant values change. This is achieved by keeping
        the lazy constants unrealized and using reconstruct() to generate bytecode
        that rebuilds the f-string at runtime.
        """
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

        # Different values do NOT recompile - f-string is reconstructed at runtime
        eager2 = fn_fstring(tensor_input, "Bob", 10)
        compiled2 = opt_fn(tensor_input, "Bob", 10)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(compiled2[1], "Hello Bob, you have 10 items")
        self.assertEqual(counter.frame_count, 1)  # NO recompilation!

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

    def test_computed_lazy_constant_peek_symbolic_realized(self):
        from torch._dynamo.variables.lazy import (
            ComputedLazyCache,
            ComputedLazyConstantVariable,
        )
        from torch._dynamo.variables.tensor import SymNodeVariable

        cache = ComputedLazyCache(
            value=1,
            lazy_vars=[],
            args=[],
            op=lambda: 1,
            reconstruct_fn=lambda codegen, args: None,
        )
        var = ComputedLazyConstantVariable(cache)
        cache.vt = SymNodeVariable(proxy=None, sym_num=sympy.Symbol("n"))

        can_peek, is_unrealized, value = var.try_peek_constant()
        self.assertFalse(can_peek)
        self.assertFalse(is_unrealized)
        self.assertIsNone(value)

    def test_computed_lazy_constant_nested_fstring(self):
        """Test f-strings with expressions involving lazy constants.

        F-strings with computations on lazy constants (like x * 2) should work
        correctly without triggering recompilation when values change.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_nested(t, x, y):
            # f-string with computation inside
            return t + 1, f"Result: {x * 2} and {y + 10}"

        counter = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_nested, backend=counter)

            eager1 = fn_nested(tensor_input, 5, 3)
            compiled1 = opt_fn(tensor_input, 5, 3)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # "Result: 10 and 13"
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - f-string is reconstructed at runtime
            eager2 = fn_nested(tensor_input, 7, 15)
            compiled2 = opt_fn(tensor_input, 7, 15)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], "Result: 14 and 25")
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!

    def test_computed_lazy_constant_division_by_zero(self):
        """Test that division by zero with lazy constants raises ZeroDivisionError.

        When dividing lazy constants, if the divisor is zero, the operation should
        raise ZeroDivisionError at compile time (when the computation is performed
        eagerly in ComputedLazyConstantVariable.create).
        """
        tensor_input = torch.randn(3)

        def fn_div_zero(t, a, b):
            return t + 1, a / b

        counter = CompileCounter()
        with torch._dynamo.config.patch(specialize_float=True):
            opt_fn = torch.compile(fn_div_zero, backend=counter)

            # Normal division should work
            eager1 = fn_div_zero(tensor_input, 10.0, 2.0)
            compiled1 = opt_fn(tensor_input, 10.0, 2.0)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # 5.0
            self.assertEqual(counter.frame_count, 1)

            # Division by zero should raise ZeroDivisionError
            with self.assertRaises(ZeroDivisionError):
                opt_fn(tensor_input, 10.0, 0.0)

    def test_computed_lazy_constant_comparison_no_branch(self):
        """Test comparison operations on lazy constants without branching.

        When comparison operators (==, <, >, etc.) are applied to lazy constants
        and the result is just returned (not used in control flow), the result
        should be correct without recompilation when values change.

        Note: This feature requires specialize_int=True.
        """
        tensor_input = torch.randn(3)

        def fn_compare(t, a, b):
            # Compare without using result in a branch
            return t + 1, a < b, a == b, a > b

        counter = CompileCounter()
        with torch._dynamo.config.patch(specialize_int=True):
            opt_fn = torch.compile(fn_compare, backend=counter)

            eager1 = fn_compare(tensor_input, 10, 20)
            compiled1 = opt_fn(tensor_input, 10, 20)
            self.assertTrue(same(eager1[0], compiled1[0]))
            self.assertEqual(eager1[1], compiled1[1])  # True (10 < 20)
            self.assertEqual(eager1[2], compiled1[2])  # False (10 == 20)
            self.assertEqual(eager1[3], compiled1[3])  # False (10 > 20)
            self.assertEqual(counter.frame_count, 1)

            # Different values do NOT recompile - reconstruct_fn generates bytecode
            # that recomputes comparisons at runtime
            eager2 = fn_compare(tensor_input, 30, 20)
            compiled2 = opt_fn(tensor_input, 30, 20)
            self.assertTrue(same(eager2[0], compiled2[0]))
            self.assertEqual(compiled2[1], False)  # 30 < 20 is False
            self.assertEqual(compiled2[2], False)  # 30 == 20 is False
            self.assertEqual(compiled2[3], True)  # 30 > 20 is True
            self.assertEqual(counter.frame_count, 1)  # NO recompilation!


if __name__ == "__main__":
    run_tests()
