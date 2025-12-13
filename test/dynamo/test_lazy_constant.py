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
        """Test keyword.iskeyword without branching on result."""

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
        """Test frozenset.__contains__ without branching on result."""
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

    def test_global_constant_passthrough(self):
        """Test that global constants also benefit from lazy constant optimization."""
        GLOBAL_STR_1 = "hello"
        GLOBAL_STR_2 = "world"
        GLOBAL_INT_1 = 42

        def fn_str(t):
            return t.sin() + 1, GLOBAL_STR_1

        def fn_int(t):
            return t.cos() + 1, GLOBAL_INT_1

        tensor_input = torch.randn(3)

        # Test string global - changing the global shouldn't cause recompile
        # if the value is just passed through
        counter = CompileCounter()
        opt_fn = torch.compile(fn_str, backend=counter)

        result1 = opt_fn(tensor_input)
        self.assertEqual(result1[1], GLOBAL_STR_1)

        # Modify the closure to use different global
        def fn_str2(t):
            return t.sin() + 1, GLOBAL_STR_2

        counter2 = CompileCounter()
        opt_fn2 = torch.compile(fn_str2, backend=counter2)
        result2 = opt_fn2(tensor_input)
        self.assertEqual(result2[1], GLOBAL_STR_2)

        # Test int global
        counter3 = CompileCounter()
        opt_fn3 = torch.compile(fn_int, backend=counter3)
        result3 = opt_fn3(tensor_input)
        self.assertEqual(result3[1], GLOBAL_INT_1)

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

    @torch._dynamo.config.patch(specialize_int=True)
    def test_python_type_does_not_realize(self):
        """Test that python_type() on a lazy constant does not trigger realization.
        This verifies that type-based queries can be answered without full guarding.

        Note: This test requires specialize_int=True because with specialize_int=False,
        lazy_isinstance must realize to determine if the int becomes SymNodeVariable.
        """
        from torch._dynamo.source import LocalSource
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.lazy import LazyCache, LazyConstantVariable
        from torch._dynamo.variables.tensor import TensorVariable
        from torch._guards import tracing, TracingContext

        # Create a dummy tracing context so guards can be installed
        ctx = TracingContext(fake_mode=None)
        with tracing(ctx):
            # Create a LazyConstantVariable directly
            # to test that python_type() doesn't trigger realization
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
            # (i.e., whether this would realize to a ConstantVariable-like type)
            self.assertTrue(lc.lazy_isinstance(ConstantVariable))
            self.assertFalse(lc.lazy_isinstance(TensorVariable))
            self.assertFalse(lc.is_realized())

    def test_isinstance_does_not_recompile_on_value_change(self):
        """Test that isinstance checks do NOT trigger recompilation on value change.

        When isinstance() is called on a LazyConstantVariable, it only installs a
        TYPE_MATCH guard (not CONSTANT_MATCH), so different values of the same type
        do not cause recompilation.

        Note: We use string values here because with specialize_int=False (the default),
        int values must be realized during handler dispatch to determine if they become
        ConstantVariable or SymNodeVariable. Strings always become ConstantVariable.
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

        # Second call with different string - should NOT recompile since
        # isinstance only installs TYPE_MATCH guard
        eager2 = fn(tensor_input, "world")
        compiled2 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 1)  # No recompilation!

        # Third call with int - should recompile due to type change
        eager3 = fn(tensor_input, 42)
        compiled3 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(counter.frame_count, 2)  # Recompile for type change

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


if __name__ == "__main__":
    run_tests()
