# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.variable.type import ScalarDtype


class TestAttribute(TestCase):
    def test_hierarchy(self):
        self.assertEqual(
            Attribute.hierarchy(ArgType.ScalarOpt),
            [Attribute.OPTIONAL, Attribute.DTYPE, Attribute.VALUE],
        )
        self.assertEqual(
            Attribute.hierarchy(ArgType.TensorOpt),
            [
                Attribute.OPTIONAL,
                Attribute.DTYPE,
                Attribute.RANK,
                Attribute.SIZE,
                Attribute.VALUE,
            ],
        )

    def test_vtype(self):
        attr = Attribute.OPTIONAL
        self.assertEqual(attr.get_vtype(), bool)

        attr = Attribute.LENGTH
        self.assertEqual(attr.get_vtype(), int)

        attr = Attribute.RANK
        self.assertEqual(attr.get_vtype(), int)

        attr = Attribute.SIZE
        self.assertEqual(attr.get_vtype(), int)

        attr = Attribute.DTYPE
        self.assertEqual(attr.get_vtype(ArgType.Tensor), torch.dtype)

        attr = Attribute.DTYPE
        self.assertEqual(attr.get_vtype(ArgType.Scalar), ScalarDtype)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.Dim), int)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.Float), float)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.String), str)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.ScalarType), torch.dtype)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.Scalar, ScalarDtype.bool), bool)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.Scalar, ScalarDtype.int), int)

        attr = Attribute.VALUE
        self.assertEqual(attr.get_vtype(ArgType.Scalar, ScalarDtype.float), float)

    def test_custom_limits(self):
        attr = Attribute.OPTIONAL
        self.assertEqual(attr.get_custom_limits(), None)


if __name__ == "__main__":
    run_tests()
