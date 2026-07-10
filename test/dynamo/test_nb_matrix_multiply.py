# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class UserDefinedClassWithMatMul:
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, UserDefinedClassWithMatMul):
            return UserDefinedClassWithMatMul(self.value * other.value)
        return UserDefinedClassWithMatMul(self.value * other)

    def __imatmul__(self, other):
        if isinstance(other, UserDefinedClassWithMatMul):
            self.value *= other.value
        else:
            self.value *= other
        return self

    def __rmatmul__(self, other):
        if isinstance(other, UserDefinedClassWithMatMul):
            return UserDefinedClassWithMatMul(other.value * self.value)
        return UserDefinedClassWithMatMul(other * self.value)

    def __eq__(self, other):
        return (
            isinstance(other, UserDefinedClassWithMatMul) and self.value == other.value
        )

    def __repr__(self):
        return f"UserDefinedClassWithMatMul({self.value})"


class LeftMatMulClass:
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, LeftMatMulClass):
            return LeftMatMulClass(self.value * other.value)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, LeftMatMulClass):
            return LeftMatMulClass(other.value * self.value)
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, LeftMatMulClass) and self.value == other.value


class RightMatMulClass:
    def __init__(self, value):
        self.value = value

    def __matmul__(self, other):
        if isinstance(other, RightMatMulClass):
            return RightMatMulClass(self.value + "*" + other.value)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, LeftMatMulClass):
            return f"LeftMatMulClass({other.value})@RightMatMulClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, RightMatMulClass) and self.value == other.value


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestNbMatrixMultiply(torch._dynamo.test_case.TestCase):
    # --- user defined matrix multiply ---

    @make_dynamo_test
    def test_user_defined_matmul(self):
        self.assertEqual(
            UserDefinedClassWithMatMul(2) @ UserDefinedClassWithMatMul(3),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_user_defined_matmul_with_int(self):
        self.assertEqual(
            UserDefinedClassWithMatMul(2) @ 3, UserDefinedClassWithMatMul(6)
        )

    @make_dynamo_test
    def test_user_defined_rmatmul_with_int(self):
        self.assertEqual(
            3 @ UserDefinedClassWithMatMul(2), UserDefinedClassWithMatMul(6)
        )

    # --- using dunder ---

    @make_dynamo_test
    def test_user_defined_matmul_dunder(self):
        self.assertEqual(
            UserDefinedClassWithMatMul(2).__matmul__(UserDefinedClassWithMatMul(3)),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_user_defined_rmatmul_dunder(self):
        self.assertEqual(
            UserDefinedClassWithMatMul(2).__rmatmul__(3),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_user_defined_imatmul_dunder(self):
        x = UserDefinedClassWithMatMul(2)
        out = x.__imatmul__(UserDefinedClassWithMatMul(3))
        self.assertIs(out, x)
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    # --- Inplace matrix multiply ---

    @make_dynamo_test
    def test_user_defined_imatmul(self):
        x = UserDefinedClassWithMatMul(2)
        x @= UserDefinedClassWithMatMul(3)
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    @make_dynamo_test
    def test_user_defined_imatmul_with_int(self):
        x = UserDefinedClassWithMatMul(2)
        x @= 3
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    # --- using torch.compile ---

    def test_compile_matmul_tensor(self):
        def fn(x, y):
            return x @ y

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_compile_imatmul_tensor(self):
        def fn(x, y):
            x = x.clone()
            x @= y
            return x

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_compile_matmul_dunder(self):
        def fn(x, y):
            return x.__matmul__(y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 5)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_compile_int_tensor_matmul(self):
        def fn(a, b):
            return a @ b

        a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(a, b), fn(a, b))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
