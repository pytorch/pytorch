# Owner(s): ["module: fx"]

import unittest

import torch
import torch.fx


class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return x + self.linear(x)


class M2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = M1()

    def forward(self, x):
        return x + self.m1(x)


class TestRecordStackTraces(unittest.TestCase):
    def _trace(self):
        m = M2()
        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(m)
        return m, graph

    def test_readable_shows_user_forward_frames(self):
        # Regression test for gh#130861: with record_stack_traces enabled,
        # print_readable() should point at the user's forward frames, not at
        # torch's internal tracer/dispatch frames.
        m, graph = self._trace()
        readable = torch.fx.GraphModule(m, graph).print_readable()
        self.assertIn("code: return x + self.linear(x)", readable)
        self.assertIn("code: return x + self.m1(x)", readable)

    def test_internal_frames_are_not_recorded(self):
        m, graph = self._trace()
        for node in graph.nodes:
            stack_trace = getattr(node, "stack_trace", None)
            if stack_trace:
                self.assertNotIn(
                    "fx/proxy.py",
                    stack_trace,
                    f"tracer internals leaked into {node.name} stack trace",
                )
                self.assertNotIn(
                    "nn/modules/module.py",
                    stack_trace,
                    f"module dispatch internals leaked into {node.name}",
                )


if __name__ == "__main__":
    unittest.main()
