# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.engine import AttributeEngine
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import ConstraintProducer as cp
from torch.testing._internal.inputgen.variable.type import ScalarDtype


class TestAttributeEngine(TestCase):
    def test_engine(self):
        constraints = [
            cp.Value.Ge(lambda x, y: 1),
            cp.Value.Le(lambda x, y: x + 3),
            cp.Value.Ne(lambda x, y: y + 1),
        ]
        x = 2
        y = 1

        engine = AttributeEngine(
            Attribute.VALUE, constraints, True, ArgType.Scalar, ScalarDtype.float
        )
        values = engine.gen(Attribute.VALUE, x, y)
        self.assertEqual(len(values), 6)
        self.assertTrue(all(v >= 1 for v in values))
        self.assertTrue(all(v <= 5 for v in values))
        self.assertTrue(all(v != 2 for v in values))

        values = engine.gen(Attribute.DTYPE, x, y)
        self.assertEqual(len(values), 1)

        engine = AttributeEngine(
            Attribute.VALUE, constraints, False, ArgType.Scalar, ScalarDtype.float
        )
        values = engine.gen(Attribute.VALUE, x, y)
        self.assertEqual(len(values), 9)
        self.assertTrue(float("-inf") in values)
        self.assertTrue(0.9999999999999999 in values)
        self.assertTrue(2.0 in values)
        self.assertTrue(5.000000000000001 in values)
        self.assertTrue(float("inf") in values)


if __name__ == "__main__":
    run_tests()
