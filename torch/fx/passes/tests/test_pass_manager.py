import unittest

from caffe2.torch.fx.passes.pass_manager import (
    inplace_wrapper,
    PassManagerBuilder,
    these_before_those_pass_constraint,
    this_before_that_pass_constraint,
)


class TestPassManager(unittest.TestCase):
    def test_pass_manager_builder(self):
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManagerBuilder.build_from_passlist(passes)
        pm.validate()

    def test_this_before_that_pass_constraint(self):
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManagerBuilder.build_from_passlist(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        self.assertRaises(RuntimeError, pm.validate)

    def test_these_before_those_pass_constraint(self):
        passes = [lambda x: 2 * x for _ in range(10)]
        constraint = these_before_those_pass_constraint(passes[-1], passes[0])
        pm = PassManagerBuilder.build_from_passlist(
            [inplace_wrapper(p) for p in passes]
        )

        # add unfulfillable constraint
        pm.add_constraint(constraint)

        self.assertRaises(RuntimeError, pm.validate)
