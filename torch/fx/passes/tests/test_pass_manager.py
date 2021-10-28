import unittest

from caffe2.torch.fx.passes.pass_manager import PassManagerBuilder, ThisBeforeThatConstraint

class TestPassManager(unittest.TestCase):
    def test_pass_manager_builder(self):
        passes = [
            lambda x: 2 * x for _ in range(10)
        ]
        pm = PassManagerBuilder.build_from_passlist(passes)
        pm.validate()

        # add unfulfillable constraint
        pm.add_constraint(ThisBeforeThatConstraint(passes[-1], passes[0]))

        self.assertRaises(RuntimeError, pm.validate)
