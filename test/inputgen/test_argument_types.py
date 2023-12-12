# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argument.type import ArgType


class TestArgType(TestCase):
    def test_methods(self):
        argtype = ArgType.Tensor
        self.assertTrue(argtype.is_tensor())

        argtype = ArgType.TensorList
        self.assertTrue(argtype.is_tensor_list())

        argtype = ArgType.Scalar
        self.assertTrue(argtype.is_scalar())

        argtype = ArgType.DimList
        self.assertTrue(argtype.is_dim_list())


if __name__ == "__main__":
    run_tests()
