"""
Tests for the updated size_hint API in inductor sizevars.

This tests the new behavior where:
- size_hint() requires explicit fallback parameter
- size_hint_or_throw() explicitly throws on unbacked shapes
"""

import unittest
import sympy
import torch
from torch._inductor.sizevars import SizeVars
from torch.fx.experimental.symbolic_shapes import ShapeEnv


class TestSizeVarsSizeHint(unittest.TestCase):
    def setUp(self):
        """Set up test environment with SizeVars instance."""
        self.shape_env = ShapeEnv()
        self.sizevars = SizeVars(self.shape_env)
    
    def test_size_hint_requires_fallback_parameter(self):
        """Test that size_hint() requires explicit fallback parameter."""
        # Create a simple expression 
        expr = sympy.Integer(5)
        
        # Test that size_hint with fallback works
        result = self.sizevars.size_hint(expr, fallback=10)
        self.assertEqual(result, 5)
        
        # Test that size_hint without fallback would fail (if we tried to call it)
        # Note: This tests the new function signature requiring fallback
        with self.assertRaises(TypeError):
            # This should fail due to missing required fallback parameter
            self.sizevars.size_hint(expr)  # type: ignore

    def test_size_hint_with_backed_shapes(self):
        """Test size_hint works correctly with backed (concrete) shapes."""
        # Test with integer
        result = self.sizevars.size_hint(42, fallback=10)
        self.assertEqual(result, 42)
        
        # Test with symbolic integer
        expr = sympy.Integer(100)
        result = self.sizevars.size_hint(expr, fallback=10)
        self.assertEqual(result, 100)
        
        # Test with simple arithmetic
        expr = sympy.Integer(3) * sympy.Integer(4)
        result = self.sizevars.size_hint(expr, fallback=10)
        self.assertEqual(result, 12)

    def test_size_hint_fallback_for_unbacked_shapes(self):
        """Test that size_hint returns fallback value for unbacked shapes."""
        # Create an unbacked symbol (this simulates what happens with dynamic shapes)
        x = sympy.Symbol('x')
        
        # When the symbol has no backing value, should return fallback
        result = self.sizevars.size_hint(x, fallback=99)
        self.assertEqual(result, 99)

    def test_size_hint_or_throw_with_backed_shapes(self):
        """Test size_hint_or_throw works correctly with backed shapes."""
        # Test with integer
        result = self.sizevars.size_hint_or_throw(42)
        self.assertEqual(result, 42)
        
        # Test with symbolic integer  
        expr = sympy.Integer(100)
        result = self.sizevars.size_hint_or_throw(expr)
        self.assertEqual(result, 100)

    def test_size_hint_or_throw_throws_on_unbacked(self):
        """Test that size_hint_or_throw throws on unbacked shapes."""
        # Create an unbacked symbol
        x = sympy.Symbol('x')
        
        # Should raise an exception when trying to convert unbacked symbol to int
        with self.assertRaises((TypeError, ValueError)):
            self.sizevars.size_hint_or_throw(x)

    def test_size_hints_batch_operation(self):
        """Test size_hints works correctly with multiple expressions."""
        exprs = [sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)]
        
        result = self.sizevars.size_hints(exprs, fallback=10)
        self.assertEqual(result, (1, 2, 3))

    def test_size_hints_with_fallback(self):
        """Test size_hints uses fallback for unbacked shapes."""
        # Mix of backed and unbacked expressions
        x = sympy.Symbol('x')
        exprs = [sympy.Integer(1), x, sympy.Integer(3)]
        
        result = self.sizevars.size_hints(exprs, fallback=99)
        self.assertEqual(result, (1, 99, 3))

    def test_size_hints_or_throw_batch_operation(self):
        """Test size_hints_or_throw works with backed shapes."""
        exprs = [sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)]
        
        result = self.sizevars.size_hints_or_throw(exprs)
        self.assertEqual(result, (1, 2, 3))

    def test_size_hints_or_throw_throws_on_any_unbacked(self):
        """Test that size_hints_or_throw throws if any shape is unbacked."""
        # Mix of backed and unbacked expressions
        x = sympy.Symbol('x')
        exprs = [sympy.Integer(1), x, sympy.Integer(3)]
        
        with self.assertRaises((TypeError, ValueError)):
            self.sizevars.size_hints_or_throw(exprs)

    def test_backwards_compatibility_with_explicit_fallback(self):
        """Test that code migrated to explicit fallback still works."""
        # This simulates migrated code that now provides explicit fallback
        expr = sympy.Integer(50)
        
        # Old pattern that would have worked: size_hint(expr)
        # New pattern that should work: size_hint(expr, fallback=default)
        result = self.sizevars.size_hint(expr, fallback=100)
        self.assertEqual(result, 50)
        
        # Test that fallback is ignored when not needed
        self.assertEqual(result, 50)  # Should return actual value, not fallback

    def test_different_fallback_values(self):
        """Test that different fallback values work correctly."""
        x = sympy.Symbol('x')
        
        # Test various fallback values
        self.assertEqual(self.sizevars.size_hint(x, fallback=1), 1)
        self.assertEqual(self.sizevars.size_hint(x, fallback=1024), 1024)
        self.assertEqual(self.sizevars.size_hint(x, fallback=0), 0)
        self.assertEqual(self.sizevars.size_hint(x, fallback=-1), -1)


if __name__ == '__main__':
    unittest.main()