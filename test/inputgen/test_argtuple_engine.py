# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argtuple.engine import MetaArgTupleEngine
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.specs.model import (
    ConstraintProducer as cp,
    InPosArg,
    Spec,
)


class TestMetaArgTupleEngine(TestCase):
    def test_size(self):
        spec = Spec(
            op="test_size",  # (Tensor self, int dim) -> int
            inspec=[
                InPosArg(ArgType.Tensor, name="self"),
                InPosArg(
                    ArgType.Dim,
                    name="dim",
                    deps=[0],
                    constraints=[
                        cp.Value.Ge(
                            lambda deps: -deps[0].dim() if deps[0].dim() > 0 else None
                        ),
                        cp.Value.Ge(lambda deps: -1 if deps[0].dim() == 0 else None),
                        cp.Value.Le(
                            lambda deps: deps[0].dim() - 1
                            if deps[0].dim() > 0
                            else None
                        ),
                        cp.Value.Le(lambda deps: 0 if deps[0].dim() == 0 else None),
                    ],
                ),
            ],
            outspec=[],
        )

        for meta_tuple in MetaArgTupleEngine(spec).gen(True):
            t, dim = meta_tuple
            self.assertEqual(t.argtype, ArgType.Tensor)
            self.assertEqual(dim.argtype, ArgType.Dim)
            if t.rank() == 0:
                self.assertTrue(dim.value in [-1, 0])
            else:
                self.assertTrue(dim.value >= -t.rank())
                self.assertTrue(dim.value < t.rank())


if __name__ == "__main__":
    run_tests()
