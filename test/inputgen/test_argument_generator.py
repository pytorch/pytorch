# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argument.engine import MetaArg
from torch.testing._internal.inputgen.argument.gen import (
    ArgumentGenerator,
    TensorGenerator,
)
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.variable.solve import SolvableVariable


class TestTensorGenerator(TestCase):
    def test_gen(self):
        v = SolvableVariable(float)
        v.Ge(13)
        v.Le(51)
        tg = TensorGenerator(dtype=torch.float64, structure=(2, 3), space=v.space)
        tensor = tg.gen()

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, torch.float64)
        self.assertGreaterEqual(tensor.min(), 13)
        self.assertLessEqual(tensor.max(), 51)


class TestArgumentGenerator(TestCase):
    def test_gen_optional(self):
        m = MetaArg(argtype=ArgType.TensorOpt, optional=True)
        tensor = ArgumentGenerator(m).gen()
        self.assertEqual(tensor, None)

    def test_gen_scalar(self):
        m = MetaArg(argtype=ArgType.Scalar, value=True)
        scalar = ArgumentGenerator(m).gen()
        self.assertIs(scalar, True)

    def test_gen_dim_list(self):
        m = MetaArg(argtype=ArgType.DimList, structure=(2, 3))
        dim_list = ArgumentGenerator(m).gen()
        self.assertEqual(dim_list, [2, 3])

    def test_gen_tensor(self):
        v = SolvableVariable(float)
        v.Ge(13)
        v.Le(51)
        m = MetaArg(
            argtype=ArgType.Tensor, dtype=torch.float64, structure=(2, 3), value=v.space
        )
        tensor = ArgumentGenerator(m).gen()

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, torch.float64)
        self.assertGreaterEqual(tensor.min(), 13)
        self.assertLessEqual(tensor.max(), 51)

    def test_gen_tensor_list(self):
        v = SolvableVariable(float)
        v.Ge(13)
        v.Le(51)
        m = MetaArg(
            argtype=ArgType.TensorOptList,
            dtype=[torch.float64, torch.int32, None],
            structure=((2, 3), (3,), None),
            value=v.space,
        )
        tensors = ArgumentGenerator(m).gen()

        self.assertEqual(len(tensors), 3)
        self.assertEqual(tensors[0].shape, (2, 3))
        self.assertEqual(tensors[0].dtype, torch.float64)
        self.assertGreaterEqual(tensors[0].min(), 13)
        self.assertLessEqual(tensors[0].max(), 51)

        self.assertEqual(tensors[1].shape, (3,))
        self.assertEqual(tensors[1].dtype, torch.int32)
        self.assertGreaterEqual(tensors[1].min(), 13)
        self.assertLessEqual(tensors[1].max(), 51)

        self.assertEqual(tensors[2], None)


if __name__ == "__main__":
    run_tests()
