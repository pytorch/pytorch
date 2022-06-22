# Owner(s): ["oncall: fx"]

import torch
import torch.fx as fx

from torch.testing._internal.common_utils import TestCase
from torch.fx.passes.infra.pass_manager import (
    PassManager,
    this_before_that_pass_constraint,
)

def replace_add_with_mul_pass(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.add:
            node.target = torch.mul

def replace_mul_with_sub_pass(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.mul:
            node.target = torch.div

class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TestPassManager(TestCase):
    def test_pass_manager(self):
        """
        Tests that the pass manager runs the passes correctly.
        """

        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_sub_pass], steps=5)

        pm.validate_constraints()
        self.assertEqual(len(pm.passes), 2)

        pm(traced_m)

        # Check that all call_function nodes are divs
        for node in traced_m.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, torch.div)

    def test_this_before_that_pass_constraint(self):
        """
        Tests the construction of constraints
        """
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        self.assertRaises(RuntimeError, pm.validate_constraints)


    def test_pass_manager_checks(self):
        """
        Tests that users can add in check functions correctly
        """
        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[replace_add_with_mul_pass, replace_mul_with_sub_pass])

        def check_div_target(graph_module):
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target != torch.div:
                    raise ValueError("Target should be div!")
        pm.add_checks(check_div_target)

        self.assertRaises(ValueError, pm, traced_m)

    def test_pass_manager_bad_checks(self):
        """
        Checks that we error if we pass in a check function with the wrong parameters
        """
        def check_bad_args(graph_module, i):
            pass

        pm = PassManager()
        self.assertRaises(TypeError, pm.add_checks, check_bad_args)
