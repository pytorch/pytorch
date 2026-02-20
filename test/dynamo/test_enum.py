# Owner(s): ["module: dynamo"]

import enum
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same, skipIfNotPy312


class EnumTests(torch._dynamo.test_case.TestCase):
    """Tests for enum support in torch.compile."""

    def test_enum_as_dict_key(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

        def fn(x):
            y = x + 2
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            torch._dynamo.graph_break()
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            b = y * 2
            return a, b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_enum_as_dict_key_with_overloaded_str(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

            def __str__(self):
                return self.value

        def fn(x):
            y = x + 2
            z = {
                MyEnum.FOO: torch.tensor(1),
                MyEnum.BAR: 10,
                "MyEnum.BAR": torch.tensor(8),
                5: torch.rand(3),
            }
            torch._dynamo.graph_break()
            a = z[MyEnum.FOO] + z["MyEnum.BAR"]
            b = y * 2
            return a, b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.rand(3)
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_enum_no_graphbreaks(self):
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        def fn(x, foo):
            if foo is Foo.FOO:
                x = torch.add(x, 1.0)
            x = torch.mul(x, 1.0)
            return x

        x = torch.randn(1)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, Foo.FOO)
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, Foo.BAR)
        self.assertEqual(cnts.op_count, 1)

    def test_enum_membership_check(self):
        class Reduction(enum.Enum):
            SUM = "sum"
            MAX = "max"
            MIN = "min"

        def fn_enum_in(x, reduction):
            if reduction in Reduction:
                x = torch.add(x, 1.0)
            return x

        def fn_enum_not_in(x, reduction):
            if reduction not in Reduction:
                raise ValueError("Unknown reduction")
            x = torch.mul(x, 2.0)
            return x

        x = torch.randn(4)
        reduction = Reduction.SUM

        # Test `in` operator
        opt_fn = torch.compile(fn_enum_in, backend="eager", fullgraph=True)
        ref = fn_enum_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

        # Test `not in` operator
        opt_fn = torch.compile(fn_enum_not_in, backend="eager", fullgraph=True)
        ref = fn_enum_not_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

    @skipIfNotPy312
    def test_enum_membership_check_constant(self):
        class Reduction(enum.Enum):
            SUM = "sum"
            MAX = "max"
            MIN = "min"

        def fn_enum_in(x, reduction):
            if reduction in Reduction:
                x = torch.add(x, 1.0)
            return x

        def fn_enum_not_in(x, reduction):
            if reduction not in Reduction:
                raise ValueError("Unknown reduction")
            x = torch.mul(x, 2.0)
            return x

        x = torch.randn(4)
        reduction = "sum"

        # Test `in` operator for constants
        opt_fn = torch.compile(fn_enum_in, backend="eager", fullgraph=True)
        ref = fn_enum_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

        # Test `not in` operator for constants
        opt_fn = torch.compile(fn_enum_not_in, backend="eager", fullgraph=True)
        ref = fn_enum_not_in(x, reduction)
        res = opt_fn(x, reduction)
        self.assertEqual(ref, res)

    def test_enum_guards(self):
        class MyEnum(enum.Enum):
            FOO = 10
            BAR = 20

        def fn(x, y):
            if y == MyEnum.FOO:
                return x + 1
            else:
                return x - 1

        x = torch.rand(3)
        y = MyEnum.BAR
        ref = fn(x, y)
        opt_fn = torch.compile(backend="eager")(fn)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_enum_method(self):
        class Bool(enum.IntEnum):
            TRUE = enum.auto()
            FALSE = enum.auto()

            def is_true(self, x):
                # Return `x + 1` to make sure Dynamo actually traced into this,
                # rather than invoking it.
                return self == Bool.TRUE, x + 1

        def f(x, e):
            cond, y = e.is_true(x)
            if cond:
                return y + 2
            else:
                return y - 2

        opt_f = torch.compile(fullgraph=True)(f)
        args = [torch.zeros(1), Bool.TRUE]
        ref_out = f(*args)
        opt_out = opt_f(*args)
        self.assertTrue(same(ref_out, opt_out))

    def test_enum_subclass(self):
        # Copied from inspect.py

        class _ParameterKind(enum.IntEnum):
            POSITIONAL_ONLY = "positional-only"

            def __new__(cls, description):
                value = len(cls.__members__)
                member = int.__new__(cls, value)
                member._value_ = value
                member.description = description
                return member

            def __str__(self):
                return self.name

        _POSITIONAL_ONLY = _ParameterKind.POSITIONAL_ONLY

        def fn(x):
            _ParameterKind(_POSITIONAL_ONLY)
            return torch.cos(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_user_function_variable_supports_enum_argument(self):
        class Foo(enum.Enum):
            FOO = 0
            BAR = 1

        def gn(x, y=Foo.FOO):
            if y is Foo.FOO:
                return x
            else:
                return x + 1

        def fn(x):
            return gn(x)

        x = torch.randn(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_enum_membership_custom_metaclass(self):
        """Test that custom metaclass __contains__ override is respected."""

        class CustomEnumMeta(enum.EnumMeta):
            def __contains__(self, item):
                # Custom behavior: always return False
                return False

        class MyEnum(enum.Enum, metaclass=CustomEnumMeta):
            A = 1
            B = 2

        # Verify eager behavior
        self.assertFalse(MyEnum.A in MyEnum)
        self.assertFalse(1 in MyEnum)

        def fn(x, member):
            if member in MyEnum:
                return x + 1
            return x - 1

        x = torch.randn(4)
        # With custom metaclass, membership should always be False
        ref = fn(x, MyEnum.A)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, MyEnum.A)
        self.assertEqual(ref, res)
        # Should return x - 1 since custom __contains__ returns False
        self.assertEqual(res, x - 1)

    def test_enum_len(self):
        """Test len() on Enum class."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            return x + len(Color)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @unittest.expectedFailure  # TODO: Support Enum.__iter__
    def test_enum_iter(self):
        """Test iterating over Enum class."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            total = 0
            for color in Color:
                total += color.value
            return x + total

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @unittest.expectedFailure  # TODO: Support Enum.__iter__ (list uses iter)
    def test_enum_list(self):
        """Test list(Enum) to get all members."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            colors = list(Color)
            return x + len(colors)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_getitem(self):
        """Test Enum.__getitem__ (access by name)."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            color = Color["RED"]
            if color == Color.RED:
                return x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @unittest.expectedFailure  # TODO: Support Enum.__reversed__
    def test_enum_reversed(self):
        """Test reversed() on Enum class."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            colors = list(reversed(Color))
            # Check order is reversed: BLUE, GREEN, RED
            if colors[0] == Color.BLUE and colors[2] == Color.RED:
                return x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_call(self):
        """Test Enum(value) to get member by value."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            color = Color(1)
            if color == Color.RED:
                return x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_enum_name_value_properties(self):
        """Test accessing .name and .value properties of enum members."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x, color):
            if color.name == "RED":
                return x + color.value
            return x

        x = torch.randn(4)
        ref = fn(x, Color.RED)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Color.RED)
        self.assertEqual(ref, res)

    def test_int_enum_arithmetic(self):
        """Test IntEnum arithmetic operations."""

        class Priority(enum.IntEnum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        def fn(x, priority):
            return x + priority

        x = torch.randn(4)
        ref = fn(x, Priority.HIGH)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Priority.HIGH)
        self.assertEqual(ref, res)

    @unittest.expectedFailure  # TODO: Support Flag enum membership check
    def test_flag_enum(self):
        """Test Flag enum operations."""

        # It checks if a flag is set in a combined value, not if it's a member of the Flag class.
        # This requires different handling in dynamo.

        class Permission(enum.Flag):
            READ = 1
            WRITE = 2
            EXECUTE = 4

        def fn(x, perm):
            if Permission.READ in perm:
                return x + 1
            return x

        x = torch.randn(4)
        combined = Permission.READ | Permission.WRITE
        ref = fn(x, combined)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, combined)
        self.assertEqual(ref, res)

    def test_enum_comparison(self):
        """Test enum comparison operations."""

        class Priority(enum.Enum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        def fn(x, a, b):
            if a == b:
                return x + 1
            elif a is b:
                return x + 2
            else:
                return x - 1

        x = torch.randn(4)

        # Same member
        ref = fn(x, Priority.LOW, Priority.LOW)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Priority.LOW, Priority.LOW)
        self.assertEqual(ref, res)

        # Different members
        torch._dynamo.reset()
        ref = fn(x, Priority.LOW, Priority.HIGH)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, Priority.LOW, Priority.HIGH)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
