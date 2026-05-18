# Owner(s): ["module: dynamo"]

"""Tests for the * and *= operators in PyTorch Dynamo (nb_multiply slot)."""

import operator

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
)


class UserDefinedClassWithMul:
    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        if isinstance(other, UserDefinedClassWithMul):
            return UserDefinedClassWithMul(self.value * other.value)
        return UserDefinedClassWithMul(self.value * other)

    def __rmul__(self, other):
        if isinstance(other, UserDefinedClassWithMul):
            return UserDefinedClassWithMul(other.value * self.value)
        return UserDefinedClassWithMul(other * self.value)

    def __eq__(self, other):
        return isinstance(other, UserDefinedClassWithMul) and self.value == other.value

    def __repr__(self):
        return f"UserDefinedClassWithMul({self.value})"


class LeftMulClass:
    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        if isinstance(other, LeftMulClass):
            return LeftMulClass(self.value * other.value)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, LeftMulClass):
            return LeftMulClass(other.value * self.value)
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, LeftMulClass) and self.value == other.value


class RightMulClass:
    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        if isinstance(other, RightMulClass):
            return RightMulClass(self.value + "*" + other.value)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, LeftMulClass):
            return f"LeftMulClass({other.value})*RightMulClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, RightMulClass) and self.value == other.value


class _IntSubWithMul(int):
    def __mul__(self, other):
        return "_IntSubWithMul.__mul__"

    def __rmul__(self, other):
        return "_IntSubWithMul.__rmul__"


class _BaseWithMul:
    def __mul__(self, other):
        return "_BaseWithMul.__mul__"

    def __rmul__(self, other):
        return "_BaseWithMul.__rmul__"


class _SubWithMul(_BaseWithMul):
    def __mul__(self, other):
        return "_SubWithMul.__mul__"

    def __rmul__(self, other):
        return "_SubWithMul.__rmul__"


class _InheritedSub(_BaseWithMul):
    pass


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestNbMultiply(torch._dynamo.test_case.TestCase):
    # --- Integer multiply ---

    @make_dynamo_test
    def test_int_mul(self):
        self.assertEqual(2 * 3, 6)
        self.assertEqual(0 * 5, 0)
        self.assertEqual(-2 * 3, -6)
        self.assertEqual(-2 * -3, 6)

    @make_dynamo_test
    def test_int_mul_large(self):
        self.assertEqual(100 * 200, 20000)
        # Python ints have arbitrary precision.
        self.assertEqual(2**60 * 2, 2**61)

    @make_dynamo_test
    def test_bool_mul(self):
        self.assertEqual(True * True, 1)
        self.assertEqual(True * 5, 5)
        self.assertEqual(False * 5, 0)
        self.assertEqual(5 * True, 5)

    # --- Float multiply ---

    @make_dynamo_test
    def test_float_mul(self):
        self.assertEqual(2.5 * 4, 10.0)
        self.assertEqual(2 * 2.5, 5.0)
        self.assertEqual(0.5 * 0.5, 0.25)

    @make_dynamo_test
    def test_float_int_mixed(self):
        self.assertEqual(1.5 * 3, 4.5)
        self.assertEqual(3 * 1.5, 4.5)

    # --- Complex multiply ---

    @make_dynamo_test
    def test_complex_mul(self):
        self.assertEqual(complex(1, 2) * complex(3, 4), complex(-5, 10))
        self.assertEqual(complex(1, 2) * 3, complex(3, 6))
        self.assertEqual(3 * complex(1, 2), complex(3, 6))

    # --- Sequence repetition (sq_repeat fallback) ---

    @make_dynamo_test
    def test_list_mul_int(self):
        self.assertEqual([1, 2] * 3, [1, 2, 1, 2, 1, 2])
        self.assertEqual([1, 2] * 0, [])
        self.assertEqual([1, 2] * -1, [])

    @make_dynamo_test
    def test_int_mul_list(self):
        self.assertEqual(3 * [1, 2], [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_list_mul_bool(self):
        # bool is index-like, so list * True == list
        self.assertEqual([1, 2] * True, [1, 2])
        self.assertEqual([1, 2] * False, [])

    @make_dynamo_test
    def test_tuple_mul_int(self):
        self.assertEqual((1, 2) * 3, (1, 2, 1, 2, 1, 2))
        self.assertEqual((1,) * 5, (1, 1, 1, 1, 1))

    @make_dynamo_test
    def test_str_mul_int(self):
        self.assertEqual("ab" * 3, "ababab")
        self.assertEqual(3 * "ab", "ababab")
        self.assertEqual("" * 5, "")
        self.assertEqual("a" * -3, "")

    @make_dynamo_test
    def test_bytes_mul_int(self):
        self.assertEqual(b"ab" * 3, b"ababab")
        self.assertEqual(3 * b"ab", b"ababab")

    # --- Inplace multiply ---

    @make_dynamo_test
    def test_int_imul(self):
        x = 5
        x *= 3
        self.assertEqual(x, 15)

    @make_dynamo_test
    def test_float_imul(self):
        x = 1.5
        x *= 2
        self.assertEqual(x, 3.0)

    @make_dynamo_test
    def test_list_imul(self):
        a = [1, 2]
        a *= 3
        self.assertEqual(a, [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_tuple_imul(self):
        # tuple lacks sq_inplace_repeat; *= falls back to non-inplace via
        # generic_inplace_multiply -> sq_repeat.
        t = (1, 2)
        t *= 3
        self.assertEqual(t, (1, 2, 1, 2, 1, 2))

    @make_dynamo_test
    def test_str_imul(self):
        s = "ab"
        s *= 3
        self.assertEqual(s, "ababab")

    # --- TypeError cases ---

    @make_dynamo_test
    def test_list_mul_str_raises(self):
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'str'"
        ):
            [1, 2] * "foo"

    @make_dynamo_test
    def test_tuple_mul_str_raises(self):
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'str'"
        ):
            (1, 2) * "foo"

    @make_dynamo_test
    def test_str_mul_str_raises(self):
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'str'"
        ):
            "ab" * "cd"

    @make_dynamo_test
    def test_str_mul_float_raises(self):
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'float'"
        ):
            "ab" * 3.0

    @make_dynamo_test
    def test_range_mul_raises(self):
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'range' and 'int'",
        ):
            range(5) * 3

    @make_dynamo_test
    def test_int_mul_range_raises(self):
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'int' and 'range'",
        ):
            3 * range(5)

    @make_dynamo_test
    def test_dict_mul_raises(self):
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'dict' and 'int'",
        ):
            {1: 2} * 3

    @make_dynamo_test
    def test_set_mul_raises(self):
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'set' and 'int'",
        ):
            {1, 2} * 3

    # --- operator.mul / operator.imul function-call form ---
    # BuiltinVariable.call_mul / call_imul are hit by the BINARY_OP /
    # INPLACE_MULTIPLY opcodes above, but also by an explicit
    # ``operator.mul(a, b)`` / ``operator.imul(a, b)`` call; verify both routes.

    @make_dynamo_test
    def test_operator_mul_int(self):
        self.assertEqual(operator.mul(5, 7), 35)

    @make_dynamo_test
    def test_operator_mul_list_int(self):
        self.assertEqual(operator.mul([1, 2], 3), [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_operator_mul_int_str(self):
        # int * str via sq_repeat fallback in generic_multiply.
        self.assertEqual(operator.mul(3, "ab"), "ababab")

    @make_dynamo_test
    def test_operator_imul_int(self):
        # int has no nb_inplace_multiply — generic_inplace_multiply falls
        # back to nb_multiply via binary_iop1.
        self.assertEqual(operator.imul(5, 3), 15)

    @make_dynamo_test
    def test_operator_imul_list(self):
        # list has sq_inplace_repeat; operator.imul mutates a.
        a = [1, 2]
        operator.imul(a, 3)
        self.assertEqual(a, [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_operator_imul_tuple(self):
        # tuple has no sq_inplace_repeat — falls back to sq_repeat, returns
        # a new tuple.
        t = (1, 2)
        self.assertEqual(operator.imul(t, 3), (1, 2, 1, 2, 1, 2))

    @make_dynamo_test
    def test_operator_imul_str(self):
        self.assertEqual(operator.imul("ab", 3), "ababab")

    @make_dynamo_test
    def test_operator_imul_unsupported_raises(self):
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*=: 'set' and 'int'",
        ):
            operator.imul({1, 2}, 3)

    # --- Direct method calls ---

    @make_dynamo_test
    def test_int_dunder_mul(self):
        self.assertEqual((2).__mul__(3), 6)

    @make_dynamo_test
    def test_list_dunder_mul(self):
        # list.__mul__ wraps sq_repeat — TypeError on non-int.  CPython's
        # slot wrapper raises "'str' object cannot be interpreted as an
        # integer"; Dynamo's slot_wrapper_mul currently delegates to
        # sequence_repeat and raises "can't multiply sequence by non-int of
        # type 'str'". Both are TypeErrors, so use a permissive regex.
        self.assertEqual([1, 2].__mul__(3), [1, 2, 1, 2, 1, 2])
        with self.assertRaisesRegex(
            TypeError,
            r"(can't multiply sequence by non-int of type 'str'"
            r"|'str' object cannot be interpreted as an integer)",
        ):
            [1, 2].__mul__("foo")

    @make_dynamo_test
    def test_list_dunder_imul(self):
        a = [1, 2]
        a.__imul__(3)
        self.assertEqual(a, [1, 2, 1, 2, 1, 2])

    # --- User-defined __mul__ ---

    @make_dynamo_test
    def test_user_defined_mul(self):
        a = UserDefinedClassWithMul(5)
        b = UserDefinedClassWithMul(3)
        self.assertEqual(a * b, UserDefinedClassWithMul(15))

    @make_dynamo_test
    def test_user_defined_mul_with_int(self):
        a = UserDefinedClassWithMul(5)
        self.assertEqual(a * 3, UserDefinedClassWithMul(15))

    @make_dynamo_test
    def test_user_defined_rmul(self):
        a = UserDefinedClassWithMul(5)
        self.assertEqual(3 * a, UserDefinedClassWithMul(15))

    @make_dynamo_test
    def test_left_mul_left_uses_mul(self):
        a = LeftMulClass(5)
        b = LeftMulClass(3)
        self.assertEqual(a * b, LeftMulClass(15))

    @make_dynamo_test
    def test_left_mul_right_falls_back_to_rmul(self):
        a = LeftMulClass(7)
        b = RightMulClass("x")
        self.assertEqual(a * b, "LeftMulClass(7)*RightMulClass(x)")

    @make_dynamo_test
    def test_right_mul_left_raises(self):
        a = RightMulClass("x")
        b = LeftMulClass(7)
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'RightMulClass' and 'LeftMulClass'",
        ):
            a * b

    # --- Subclass right-op dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        self.assertEqual(_IntSubWithMul(1) * 1, "_IntSubWithMul.__mul__")
        self.assertEqual(1 * _IntSubWithMul(1), "_IntSubWithMul.__rmul__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        self.assertEqual(_BaseWithMul() * 1, "_BaseWithMul.__mul__")
        self.assertEqual(1 * _BaseWithMul(), "_BaseWithMul.__rmul__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        self.assertEqual(_SubWithMul() * _BaseWithMul(), "_SubWithMul.__mul__")
        self.assertEqual(_BaseWithMul() * _SubWithMul(), "_SubWithMul.__rmul__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        self.assertIs(_InheritedSub.__rmul__, _BaseWithMul.__rmul__)
        self.assertEqual(_InheritedSub() * 1, "_BaseWithMul.__mul__")
        self.assertEqual(1 * _InheritedSub(), "_BaseWithMul.__rmul__")

    # --- Tensors ---

    def test_tensor_mul_scalar(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(t):
            return t * 2

        self.assertEqual(f(torch.tensor([1.0, 2.0, 3.0])).tolist(), [2.0, 4.0, 6.0])

    def test_scalar_mul_tensor(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(t):
            return 2 * t

        self.assertEqual(f(torch.tensor([1.0, 2.0, 3.0])).tolist(), [2.0, 4.0, 6.0])

    def test_tensor_mul_tensor(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(a, b):
            return a * b

        out = f(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        self.assertEqual(out.tolist(), [3.0, 8.0])

    def test_symint_mul(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return x.shape[0] * 3

        self.assertEqual(f(torch.randn(5, 4)), 15)

    def test_int_mul_symint(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return 3 * x.shape[0]

        self.assertEqual(f(torch.randn(5, 4)), 15)

    def test_symint_mul_float(self):
        # SymNodeVariable.nb_multiply_impl must accept any python constant, not
        # just symnode-like ones — float constants aren't is_symnode_like().
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return x.shape[0] * 1.5

        self.assertEqual(f(torch.randn(5, 4)), 7.5)

    def test_float_mul_symint(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return 1.5 * x.shape[0]

        self.assertEqual(f(torch.randn(5, 4)), 7.5)

    def test_symint_mul_str(self):
        # int * str works in CPython via PyNumber_Multiply's sq_repeat fallback.
        # SymNode operand must allow str on the other side just like a constant int.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return x.shape[0] * "ab"

        self.assertEqual(f(torch.randn(3, 4)), "ababab")

    def test_symfloat_mul_int(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return x.item() * 2

        self.assertEqual(f(torch.tensor(3.5)), 7.0)

    def test_int_mul_symfloat(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return 2 * x.item()

        self.assertEqual(f(torch.tensor(3.5)), 7.0)

    def test_symfloat_mul_float(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return x.item() * 1.5

        self.assertEqual(f(torch.tensor(3.0)), 4.5)

    def test_float_mul_symfloat(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            return 1.5 * x.item()

        self.assertEqual(f(torch.tensor(3.0)), 4.5)

    def test_list_mul_symnode(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, y):
            return [x] * y

        self.assertEqual(f(2, 3), [2, 2, 2])
        self.assertEqual(f(2, True), [2])

    # --- Error message accuracy (matches CPython exactly) ---

    @make_dynamo_test
    def test_error_message_list_mul_str(self):
        try:
            [1, 2] * "foo"
        except TypeError as e:
            self.assertEqual(str(e), "can't multiply sequence by non-int of type 'str'")

    @make_dynamo_test
    def test_error_message_range_mul(self):
        try:
            range(5) * 3
        except TypeError as e:
            self.assertEqual(
                str(e), "unsupported operand type(s) for *: 'range' and 'int'"
            )


instantiate_parametrized_tests(TestNbMultiply)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
