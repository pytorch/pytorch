# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.argtuple.gen import ArgumentTupleGenerator
from torch.testing._internal.inputgen.argument.type import ArgType
from torch.testing._internal.inputgen.specs.model import (
    ConstraintProducer as cp,
    InPosArg,
    Spec,
)


class TestArgumentTupleGenerator(TestCase):
    def test_gen(self):
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

        for args, kwargs in ArgumentTupleGenerator(spec).gen():
            self.assertEqual(len(args), 2)
            self.assertEqual(kwargs, {})
            t = args[0]
            dim = args[1]
            self.assertTrue(isinstance(t, torch.Tensor))
            self.assertTrue(isinstance(dim, int))
            if t.dim() == 0:
                self.assertTrue(dim in [-1, 0])
            else:
                self.assertTrue(dim >= -t.dim())
                self.assertTrue(dim < t.dim())


if __name__ == "__main__":
    run_tests()
