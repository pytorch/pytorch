import unittest

from torchgen import local
from torchgen.api.types import (
    BaseCType,
    boolT,
    ConstRefCType,
    CType,
    longT,
    MutRefCType,
    NamedCType,
    OptionalCType,
    TupleCType,
    VectorCType,
    voidT,
)
from torchgen.executorch.api.et_cpp import argument_type, return_type, returns_type
from torchgen.executorch.api.types import ArrayRefCType, scalarT, tensorListT, tensorT
from torchgen.model import Argument, FunctionSchema, Return


class ExecutorchCppTest(unittest.TestCase):
    """
    Test torchgen.executorch.api.cpp
    """

    def _test_argumenttype_type(self, arg_str: str, expected: NamedCType) -> None:
        arg = Argument.parse(arg_str)
        self.assertEqual(str(argument_type(arg, binds=arg.name)), str(expected))

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_argumenttype_type(self) -> None:
        data = [
            ("Tensor self", NamedCType("self", ConstRefCType(BaseCType(tensorT)))),
            ("Tensor(a!) out", NamedCType("out", MutRefCType(BaseCType(tensorT)))),
            (
                "Tensor? opt",
                NamedCType("opt", ConstRefCType(OptionalCType(BaseCType(tensorT)))),
            ),
            ("Scalar scalar", NamedCType("scalar", ConstRefCType(BaseCType(scalarT)))),
            (
                "Scalar? scalar",
                NamedCType("scalar", ConstRefCType(OptionalCType(BaseCType(scalarT)))),
            ),
            ("int[] size", NamedCType("size", ArrayRefCType(BaseCType(longT)))),
            ("int? dim", NamedCType("dim", OptionalCType(BaseCType(longT)))),
            ("Tensor[] weight", NamedCType("weight", BaseCType(tensorListT))),
            (
                "Scalar[] spacing",
                NamedCType("spacing", ArrayRefCType(ConstRefCType(BaseCType(scalarT)))),
            ),
            (
                "Tensor?[] weight",
                NamedCType("weight", ArrayRefCType(OptionalCType(BaseCType(tensorT)))),
            ),
            (
                "SymInt[]? output_size",
                NamedCType(
                    "output_size", OptionalCType(ArrayRefCType(BaseCType(longT)))
                ),
            ),
            (
                "int[]? dims",
                NamedCType("dims", OptionalCType(ArrayRefCType(BaseCType(longT)))),
            ),
            (
                "bool[3] output_mask",
                NamedCType("output_mask", ArrayRefCType(BaseCType(boolT))),
            ),
        ]
        for d in data:
            self._test_argumenttype_type(*d)

    def _test_returntype_type(self, ret_str: str, expected: CType) -> None:
        ret = Return.parse(ret_str)
        self.assertEqual(str(return_type(ret)), str(expected))

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_returntype_type(self) -> None:
        data = [
            ("Tensor", BaseCType(tensorT)),
            ("Tensor(a!)", MutRefCType(BaseCType(tensorT))),
            ("Tensor[]", VectorCType(BaseCType(tensorT))),
        ]
        for d in data:
            self._test_returntype_type(*d)

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_returns_type(self) -> None:
        func = FunctionSchema.parse(
            "min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"
        )
        expected = TupleCType([BaseCType(tensorT), BaseCType(tensorT)])
        self.assertEqual(str(returns_type(func.returns)), str(expected))

    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    def test_void_return_type(self) -> None:
        func = FunctionSchema.parse(
            "_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()"
        )
        expected = BaseCType(voidT)
        self.assertEqual(str(returns_type(func.returns)), str(expected))


if __name__ == "__main__":
    unittest.main()
