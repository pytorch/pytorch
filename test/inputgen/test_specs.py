# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import (
    Constraint,
    ConstraintProducer as cp,
    ConstraintSuffix,
)


class TestArgSpecs(TestCase):
    def test_inpos(self):
        arg = InPosArg(ArgType.Tensor, name="self")
        self.assertEqual(arg.name, "self")
        self.assertEqual(arg.type, ArgType.Tensor)
        self.assertFalse(arg.kw())
        self.assertFalse(arg.out())
        self.assertFalse(arg.ret())

    def test_inkw(self):
        arg = InKwArg(ArgType.Scalar, name="alpha")
        self.assertEqual(arg.name, "alpha")
        self.assertEqual(arg.type, ArgType.Scalar)
        self.assertTrue(arg.kw())
        self.assertFalse(arg.out())
        self.assertFalse(arg.ret())

    def test_out(self):
        arg = OutArg(ArgType.TensorList)
        self.assertEqual(arg.name, "out")
        self.assertEqual(arg.type, ArgType.TensorList)
        self.assertTrue(arg.kw())
        self.assertTrue(arg.out())
        self.assertFalse(arg.ret())

    def test_ret(self):
        arg = RetArg(ArgType.Tensor)
        self.assertEqual(arg.name, "__ret")
        self.assertEqual(arg.type, ArgType.Tensor)
        self.assertFalse(arg.kw())
        self.assertFalse(arg.out())
        self.assertTrue(arg.ret())


if __name__ == "__main__":
    run_tests()
