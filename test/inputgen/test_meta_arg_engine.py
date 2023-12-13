# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argument.engine import MetaArgEngine
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import ConstraintProducer as cp
from torch.testing._internal.inputgen.variable.type import SUPPORTED_TENSOR_DTYPES


class TestMetaArgEngine(TestCase):
    def test_tensor(self):
        constraints = [
            cp.Rank.Le(lambda deps: deps[0] + 2),
            cp.Size.NotIn(lambda deps, length, ix: [1, 3]),
            cp.Size.Le(lambda deps, length, ix: 5),
            cp.Value.Ne(lambda deps: 0),
        ]
        deps = [2]

        engine = MetaArgEngine(ArgType.Tensor, constraints, deps, True)
        ms = list(engine.gen(Attribute.DTYPE))
        self.assertEqual(len(ms), len(SUPPORTED_TENSOR_DTYPES))
        self.assertEqual({m.dtype for m in ms}, set(SUPPORTED_TENSOR_DTYPES))
        self.assertTrue(all(0 <= m.rank() <= 4 for m in ms))
        for m in ms:
            self.assertTrue(
                all(0 <= size <= 5 and size not in [1, 3] for size in m.structure)
            )
        for m in ms:
            self.assertEqual(str(m.value), "[-inf, 0.0) (0.0, inf]")

        ms = list(engine.gen(Attribute.RANK))
        self.assertEqual(len(ms), 4)
        ranks = {len(m.structure) for m in ms}
        self.assertTrue(0 in ranks)
        self.assertTrue(4 in ranks)
        self.assertTrue(all(0 <= r <= 4 for r in ranks))

    # def test_tensor_list(self):
    #     constraints = [
    #       cp.Length.Eq(lambda deps: deps[0]),
    #       cp.Rank.Le(lambda deps, length, ix: ix + 2),
    #       cp.Size.NotIn(lambda deps, length, ix: [1, 3]),
    #       cp.Size.Le(lambda deps, length, ix: 5),
    #       cp.Value.Ne(lambda deps: 0),
    #     ]
    #     deps = [2]

    #     engine = MetaArgEngine(ArgType.TensorList, constraints, deps, True)
    #     for m in engine.gen(None):
    #       print(m)

    def test_dim_list(self):
        constraints = [
            cp.Length.Le(lambda deps: deps[0] + deps[1]),
            cp.Value.Gen(
                lambda deps, length: ({(deps[0],) * length}, {(deps[1],) * length})
            ),
        ]
        deps = [2, 3]

        engine = MetaArgEngine(ArgType.DimList, constraints, deps, True)
        ms = list(engine.gen(Attribute.VALUE))
        self.assertEqual(len(ms), 1)
        self.assertTrue(1 <= len(ms[0].value) <= 5)
        self.assertTrue(all(v == 2 for v in ms[0].value))

        engine = MetaArgEngine(ArgType.DimList, constraints, deps, False)
        ms = list(engine.gen(Attribute.VALUE))
        self.assertEqual(len(ms), 1)
        self.assertTrue(1 <= len(ms[0].value) <= 5)
        self.assertTrue(all(v == 3 for v in ms[0].value))


if __name__ == "__main__":
    run_tests()
