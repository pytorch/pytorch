# Owner(s): ["module: dynamo"]

"""
Unit tests for ModelSpec DSL.

Tests cover:
- Core data structures (Arg, KwArg, ModelSpec)
- Constraint types and generation
- Builder pattern functionality
"""

import torch
from torch._dynamo.model_spec.model_spec import (
    Arg,
    COMPILE,
    Context,
    KwArg,
    ModelSpec,
    RAISE_ERROR,
)
from torch._dynamo.model_spec.types import (
    DtypeConstraint,
    NoneConstraint,
    RankConstraint,
    ShapeConstraint,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestConstraint(TestCase):
    """Test constraint classes."""

    def test_constraint_guard_and_check_generation(self):
        """Test that all constraint types can generate guards and checks."""
        # Shape constraint (static)
        shape_constraint = ShapeConstraint(shape=(3, 4), dynamic_dims=None)
        self.assertIn("x.shape[0] == 3", shape_constraint.to_guard_expression("x"))
        self.assertIn("torch._check", shape_constraint.to_check("x"))

        # Shape constraint (dynamic)
        dyn_shape_constraint = ShapeConstraint(shape=(1, 4), dynamic_dims=[0])
        self.assertIn("dynamic: s0", dyn_shape_constraint.to_guard_expression("x"))

        # Dtype constraint
        dtype_constraint = DtypeConstraint(dtype=torch.float32)
        self.assertIn("x.dtype", dtype_constraint.to_guard_expression("x"))
        self.assertIn("torch._check", dtype_constraint.to_check("x"))

        # Rank constraint
        rank_constraint = RankConstraint(rank=4)
        self.assertEqual("x.dim() == 4", rank_constraint.to_guard_expression("x"))
        self.assertIn("torch._check", rank_constraint.to_check("x"))

        # None constraint
        none_constraint = NoneConstraint(is_none=False)
        self.assertEqual("x is not None", none_constraint.to_guard_expression("x"))


class TestArg(TestCase):
    """Test Arg class and builder pattern."""

    def test_arg_basic_creation(self):
        """Test basic Arg creation and validation."""
        arg = Arg(0)
        self.assertEqual(arg.position, 0)
        self.assertEqual(len(arg._constraints), 0)

        with self.assertRaises(ValueError):
            Arg(-1)

    def test_arg_shape_constraints(self):
        """Test static and dynamic shape methods."""
        # Static with explicit shape
        arg = Arg(0).static((3, 4))
        constraints = arg.constraints
        shape_constraint = next(
            (c for c in constraints if isinstance(c, ShapeConstraint)), None
        )
        self.assertIsNotNone(shape_constraint)
        self.assertEqual(shape_constraint.shape, (3, 4))

        # Static from example
        example = torch.randn(5, 6)
        arg = Arg(0, example_input=example).static()
        constraints = arg.constraints
        shape_constraint = next(
            (c for c in constraints if isinstance(c, ShapeConstraint)), None
        )
        self.assertIsNotNone(shape_constraint)
        self.assertEqual(shape_constraint.shape, (5, 6))

        # Dynamic specific dimension
        arg = Arg(0, example_input=torch.randn(3, 4, 5)).dynamic(idx=1)
        constraints = arg.constraints
        shape_constraint = next(
            (c for c in constraints if isinstance(c, ShapeConstraint)), None
        )
        self.assertIsNotNone(shape_constraint)
        self.assertIn(1, shape_constraint.dynamic_dims)

        # Dynamic all dimensions
        arg = Arg(0, example_input=torch.randn(3, 4)).dynamic()
        constraints = arg.constraints
        shape_constraint = next(
            (c for c in constraints if isinstance(c, ShapeConstraint)), None
        )
        self.assertIsNotNone(shape_constraint)
        self.assertEqual(shape_constraint.dynamic_dims, [0, 1])

    def test_arg_dimension_constraints_without_example(self):
        """Test that dimension constraints work without example_input."""
        # Can use dynamic(idx) without example_input when rank is specified
        arg = Arg(0).rank(4).dynamic(idx=2)
        constraints = arg.constraints
        rank_constraint = next(
            (c for c in constraints if isinstance(c, RankConstraint)), None
        )
        self.assertIsNotNone(rank_constraint)
        self.assertEqual(rank_constraint.rank, 4)
        # Note: ShapeConstraint won't be generated without metadata

        # Can use static(idx) without size specified
        arg = Arg(0).rank(2).static(idx=1)
        constraints = arg.constraints
        self.assertIsNotNone(constraints)

        # Conflict detection: static then dynamic on same dimension
        with self.assertRaises(ValueError) as cm:
            Arg(0).rank(2).static(idx=1).dynamic(idx=1)
        self.assertIn("already marked as static", str(cm.exception))

        # Conflict detection: dynamic then static on same dimension
        with self.assertRaises(ValueError) as cm:
            Arg(0).rank(2).dynamic(idx=1).static(idx=1)
        self.assertIn("already marked as dynamic", str(cm.exception))

        # Can stack multiple dimension constraints on different dimensions
        arg = Arg(0).rank(4).static(idx=0).dynamic(idx=1).static(idx=2).dynamic(idx=3)
        constraints = arg.constraints
        rank_constraint = next(
            (c for c in constraints if isinstance(c, RankConstraint)), None
        )
        self.assertIsNotNone(rank_constraint)
        self.assertEqual(rank_constraint.rank, 4)

        # When example_input is provided, ShapeConstraint is generated
        example = torch.randn(2, 3, 4, 5)
        arg = Arg(0, example_input=example).rank(4).dynamic(idx=1).dynamic(idx=3)
        constraints = arg.constraints
        shape_constraint = next(
            (c for c in constraints if isinstance(c, ShapeConstraint)), None
        )
        self.assertIsNotNone(shape_constraint)
        self.assertEqual(shape_constraint.shape, (2, 3, 4, 5))
        self.assertEqual(shape_constraint.dynamic_dims, [1, 3])

    def test_arg_type_constraints(self):
        """Test dtype, rank, and other type constraints."""
        arg = Arg(0).dtype("float32").rank(4).notNone()

        constraints = arg.constraints
        self.assertEqual(len(constraints), 3)
        self.assertIsInstance(constraints[0], DtypeConstraint)
        self.assertIsInstance(constraints[1], NoneConstraint)
        self.assertIsInstance(constraints[2], RankConstraint)

        # Invalid dtype
        with self.assertRaises(ValueError):
            Arg(0).dtype("invalid")

    def test_arg_builder_chaining(self):
        """Test builder pattern chaining."""
        arg = Arg(0).rank(4).dtype("float32").notNone()
        self.assertEqual(len(arg.constraints), 3)

    def test_arg_properties(self):
        """Test Arg position and constraints properties."""
        arg = Arg(0).rank(4).dtype("float32")

        self.assertEqual(arg.position, 0)
        self.assertEqual(len(arg.constraints), 2)


class TestKwArg(TestCase):
    """Test KwArg class."""

    def test_kwarg_basic(self):
        """Test KwArg works like Arg but for keyword arguments."""
        kwarg = KwArg("x").rank(2).dtype("float32")

        self.assertEqual(len(kwarg.constraints), 2)
        self.assertEqual(kwarg.name, "x")

        with self.assertRaises(ValueError):
            KwArg("")


class TestModelSpec(TestCase):
    """Test ModelSpec class."""

    def test_modelspec_creation(self):
        """Test creating ModelSpec with function and module."""
        # With function
        spec = ModelSpec(lambda x: x * 2)
        self.assertEqual(len(spec._rules), 0)

        # With module
        model = torch.nn.Linear(10, 10)
        spec = ModelSpec(model)
        self.assertEqual(spec.model, model)

    def test_modelspec_add_rules(self):
        """Test adding compilation rules."""
        spec = ModelSpec(lambda x, y: x + y)

        # Single condition
        spec.add(Arg(0).rank(2))
        self.assertEqual(len(spec._rules), 1)

        # Multiple conditions
        spec.add(Arg(0).rank(2), Arg(1).rank(2))
        self.assertEqual(len(spec._rules), 2)

        # With contexts
        spec.add(Arg(0).rank(3), ctxs=[Context.GRAD])
        self.assertIn(Context.GRAD, spec._rules[2].contexts)

        # Requires at least one condition
        with self.assertRaises(ValueError):
            spec.add()

    def test_modelspec_custom_dispatcher(self):
        """Test custom dispatcher decorator."""
        spec = ModelSpec(lambda x: x)

        @ModelSpec.custom_dispatcher
        def my_dispatcher(x):
            if x.shape[0] > 10:
                return COMPILE
            return RAISE_ERROR

        spec.add(Arg(0).notNone(), dispatcher=my_dispatcher)
        self.assertEqual(spec._rules[0].dispatcher, my_dispatcher)
        self.assertTrue(hasattr(spec._rules[0].dispatcher, "_is_model_spec_dispatcher"))

        # Test decorator syntax
        @spec.addWithDispatch(Arg(0).notNone())
        def another_dispatcher(x):
            return COMPILE

        self.assertEqual(len(spec._rules), 2)
        self.assertTrue(hasattr(another_dispatcher, "_is_model_spec_dispatcher"))

    def test_modelspec_default_rule(self):
        """Test default fallback rule."""
        spec = ModelSpec(lambda x: x)
        spec.default()
        self.assertIsNotNone(spec._default_rule)

        # Can only have one default
        with self.assertRaises(ValueError):
            spec.default()


class TestIntegration(TestCase):
    """Integration tests for the full DSL."""

    def test_complex_spec(self):
        """Test creating a complex spec with multiple features."""

        def my_model(x, y, z=None):
            return x + y if z is None else x + y + z

        spec = ModelSpec(my_model)

        # Static shapes
        spec.add(Arg(0).static((3, 4)), Arg(1).static((3, 4)), KwArg("z").isNone())

        # Dynamic shapes with context
        spec.add(
            Arg(0).rank(2).dynamic(idx=0).dtype("float32"),
            Arg(1).rank(2).dynamic(idx=0).dtype("float32"),
            ctxs=[Context.GRAD],
        )

        # Custom dispatcher
        @spec.addWithDispatch(Arg(0).notNone(), Arg(1).notNone())
        def batch_dispatcher(x, y, z):
            if x.shape[0] % 8 == 0:
                return COMPILE
            return RAISE_ERROR

        spec.default(ctxs=[Context.NO_GRAD])

        # Verify structure
        self.assertEqual(len(spec._rules), 3)
        self.assertIsNotNone(spec._default_rule)

    def test_constraints_generation(self):
        """Test that Arg constraints can generate guards and checks."""
        arg = Arg(0).rank(4).dtype("float32").notNone()

        # All constraints preserved
        self.assertEqual(len(arg.constraints), 3)

        # All constraints can generate guards and checks
        for constraint in arg.constraints:
            guard_expr = constraint.to_guard_expression("x")
            self.assertIsInstance(guard_expr, str)
            self.assertGreater(len(guard_expr), 0)

            check_code = constraint.to_check("x")
            self.assertIsInstance(check_code, str)


if __name__ == "__main__":
    run_tests()
