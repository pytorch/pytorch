import unittest

from ..pass_manager import (
    PassManager,
    this_before_that_pass_constraint,
)


class TestPassManager(unittest.TestCase):
    def test_pass_manager_builder(self) -> None:
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)
        pm.validate()

    def test_this_before_that_pass_constraint(self) -> None:
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        self.assertRaises(RuntimeError, pm.validate)
