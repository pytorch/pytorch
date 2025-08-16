# Owner(s): ["module: fx"]

# pyre-strict
import torch
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.splitter_base import (
    FxNetAccNodesFinder,
    NodeEventTracker,
    ShapeProp,
)
from torch.testing._internal.common_utils import TestCase


# Wrappepr function to make it supported
@torch.fx.wrap
def sup_f(x):
    return x


class TestFxSplitNodeFinder(TestCase):
    def testFinder(self):
        class IsNodeSupported(OperatorSupportBase):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return "sup_" in node.name

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x = sup_f(x)
                y = sup_f(y)
                b = x + y  # non-supported to break graph
                return sup_f(b)

        gm = torch.fx.symbolic_trace(TestModule())
        ShapeProp(gm).propagate(*(torch.rand((2, 2)), 3))
        finder = FxNetAccNodesFinder(gm, IsNodeSupported(), False)

        # override tracker without having to run with env var
        tracker = NodeEventTracker(
            1,  # mode: just enable the tracker without dumping
            "",  # dump_path. We don't need it.
        )
        finder.tracker = tracker

        acc_nodes = finder()

        def getEvents(tracker, node):
            return [tracker.events[idx] for idx in tracker.node_events[node.name]]

        # check that acc nodes events are as expected
        for node in gm.graph.nodes:
            if node.name == "sup_f_1":
                # this node should be removed from acc nodes.
                self.assertFalse(node in acc_nodes)
                events = getEvents(tracker, node)
                # 2 events.
                self.assertEqual(len(events), 2)
                # 1st event is init_acc as supported operator
                self.assertTrue(
                    events[0].desc.startswith(
                        "init_acc|callable_and_operator_supported"
                    )
                )
                # 2nd event is del_acc as non-tensor output with cpu user
                self.assertTrue(
                    events[1].desc.startswith("acc_del|non_tensor_output_with_cpu_user")
                )
            elif node.name.startswith("sup_f"):
                # other supported nodes should remain in acc nodes.
                self.assertTrue(node in acc_nodes)
                events = getEvents(tracker, node)
                self.assertEqual(len(events), 1)
                self.assertTrue(
                    events[0].desc.startswith(
                        "init_acc|callable_and_operator_supported"
                    )
                )
            else:
                # other nodes are on cpu.
                self.assertFalse(node in acc_nodes)
