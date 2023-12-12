# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import (
    Constraint,
    ConstraintProducer as cp,
    ConstraintSuffix,
)


class TestConstraint(TestCase):
    def test_constraint(self):
        constraint = cp.Optional.Ne(lambda deps: False)
        self.assertTrue(isinstance(constraint, Constraint))
        self.assertEqual(constraint.attribute, Attribute.OPTIONAL)
        self.assertEqual(constraint.suffix, ConstraintSuffix.NE)
        self.assertEqual(constraint.fn(None), False)


if __name__ == "__main__":
    run_tests()
