# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argument.engine import StructuralEngine
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import ConstraintProducer as cp


class TestStructuralEngine(TestCase):
    def test_engine(self):
        constraints = [
            cp.Rank.Le(lambda deps: deps[0] + 2),
            cp.Size.NotIn(lambda deps, length, ix: [1, 3]),
            cp.Size.Le(lambda deps, length, ix: 5),
            cp.Value.Ne(lambda deps: 0),
        ]
        deps = [2]

        engine = StructuralEngine(ArgType.Tensor, constraints, deps, True)
        for s in engine.gen(Attribute.VALUE):
            self.assertTrue(1 <= len(s) <= 4)
            self.assertTrue(all(v in [2, 4, 5] for v in s))


if __name__ == "__main__":
    run_tests()
