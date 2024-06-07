# Owner(s): ["module: fx"]

import torch
import torch.fx as fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.infra.pass_manager import (
    _topological_sort_passes,
    pass_result_wrapper,
    PassManager,
    this_before_that_pass_constraint,
)

from torch.testing._internal.common_utils import TestCase


# Pass that uses PassBase and returns a PassResult (best scenario)
class ReplaceAddWithMulPass(PassBase):
    def call(self, gm) -> PassResult:
        modified = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                node.target = torch.mul
                modified = True
        return PassResult(gm, modified)


# Pass that is a callable and returns a PassResult
def replace_mul_with_div_pass(gm) -> PassResult:
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.mul:
            node.target = torch.div
            modified = True
    return PassResult(gm, modified)


# Pass that is a PassBase and does not return a PassResult
# Need to wrap with pass_result_wrapper or else it will fail
class ReplaceDivWithSubPass(PassBase):
    def call(self, gm) -> PassResult:
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.div:
                node.target = torch.sub


# Pass that is a callable and does not return a PassResult
# Need to wrap with pass_result_wrapper or else it will fail
def replace_sub_with_add_pass(gm) -> PassResult:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.sub:
            node.target = torch.add


class AddModule(torch.nn.Module):
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
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                pass_result_wrapper(ReplaceDivWithSubPass()),
                pass_result_wrapper(replace_sub_with_add_pass),
            ],
            steps=5,
        )

        pm.validate_constraints()
        self.assertEqual(len(pm.passes), 4)

        res = pm(traced_m)
        modified_m = res.graph_module
        assert isinstance(modified_m, fx.GraphModule)

        # Check that all call_function nodes are divs
        for node in modified_m.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(node.target, torch.add)

    def test_this_before_that_pass_constraint(self):
        """
        Tests the construction of constraints
        """
        passes = [lambda x: 2 * x for _ in range(10)]
        pm = PassManager(passes)

        # add unfulfillable constraint
        pm.add_constraint(this_before_that_pass_constraint(passes[-1], passes[0]))

        with self.assertRaises(RuntimeError):
            pm.validate_constraints()

    def test_pass_manager_checks(self):
        """
        Tests that users can add in check functions correctly
        """
        m = AddModule()
        traced_m = fx.symbolic_trace(m)
        pm = PassManager(passes=[ReplaceAddWithMulPass(), replace_mul_with_div_pass])

        def check_div_target(graph_module):
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target != torch.div:
                    raise ValueError("Target should be div!")

        pm.add_checks(check_div_target)

        with self.assertRaises(ValueError):
            pm(traced_m)

    def test_pass_manager_bad_checks(self):
        """
        Checks that we error if we pass in a check function with the wrong parameters
        """

        def check_bad_args(graph_module, i):
            pass

        pm = PassManager()
        self.assertRaises(TypeError, pm.add_checks, check_bad_args)

    def test_topological_sort(self):
        """
        Tests that passes are correctly ordered based on contraints.
        """

        def pass0(x):
            return x

        def pass1(x):
            return x + 1

        def pass2(x):
            return x + 2

        def pass3(x):
            return x + 3

        def pass4(x):
            return x + 4

        def pass5(x):
            return x + 5

        # Not passing any constraints should keep the original order
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        sorted = _topological_sort_passes(passes, [])
        self.assertEqual(sorted, passes)

        # Graph that we are constructing:
        #     5 ---->  0  <---- 4
        #     |                 |
        #     +-> 2 -> 3 -> 1 <-+
        # Which has a possible topological order of: [4, 5, 0, 2, 3, 1]
        passes = [pass0, pass1, pass2, pass3, pass4, pass5]
        constraints = [
            this_before_that_pass_constraint(pass5, pass0),
            this_before_that_pass_constraint(pass5, pass2),
            this_before_that_pass_constraint(pass4, pass0),
            this_before_that_pass_constraint(pass4, pass1),
            this_before_that_pass_constraint(pass2, pass3),
            this_before_that_pass_constraint(pass3, pass1),
        ]
        sorted = _topological_sort_passes(passes, constraints)
        self.assertEqual(sorted, [pass4, pass5, pass0, pass2, pass3, pass1])

        # Circular dependency should result in the circular_dep flag being set
        passes = [pass0, pass1, pass2]
        constraints = [
            this_before_that_pass_constraint(passes[0], passes[1]),
            this_before_that_pass_constraint(passes[1], passes[2]),
            this_before_that_pass_constraint(passes[2], passes[0]),
        ]
        with self.assertRaises(RuntimeError) as e:
            _topological_sort_passes(passes, constraints)
        expected_error_msg = (
            f"Circular dependency detected within the following passes: {passes}"
        )
        self.assertEqual(e.exception.args[0], expected_error_msg)

    def test_pass_manager_error(self):
        """
        Tests error catching + debug
        """

        def pass_fail(graph_module):
            raise RuntimeError("bad")

        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                ReplaceDivWithSubPass(),
                pass_result_wrapper(replace_sub_with_add_pass),
            ],
        )

        # Comment out this line to see the actual error message
        error_msg = (
            "ReplaceDivWithSubPass.*ReplaceAddWithMulPass.*replace_mul_with_div_pass"
        )
        with self.assertRaisesRegex(Exception, error_msg):
            pm(traced_m)

        pm = PassManager(
            passes=[
                ReplaceAddWithMulPass(),
                replace_mul_with_div_pass,
                pass_result_wrapper(ReplaceDivWithSubPass()),
                pass_result_wrapper(replace_sub_with_add_pass),
                pass_fail,
            ],
        )

        # Comment out this line to see the actual error message
        error_msg = "pass_fail.*ReplaceAddWithMulPass.*replace_mul_with_div_pass.*ReplaceDivWithSubPass.*replace_sub_with_add_pass"
        with self.assertRaisesRegex(Exception, error_msg):
            pm(traced_m)
