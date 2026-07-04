"""Verify torch.fx.Tracer.record_stack_traces captures user code stack traces."""

import sys

import torch
import torch.fx
import torch.nn as nn


class M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return x + self.linear(x)


class M2(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = M1()

    def forward(self, x):
        return x + self.m1(x)


def main() -> None:
    m = M2()
    tracer = torch.fx.Tracer()
    tracer.record_stack_traces = True
    graph = tracer.trace(m)
    gm = torch.fx.GraphModule(m, graph)

    readable = gm.print_readable(print_output=False)
    print(readable)

    # Issue #130861: stack traces must reference user forward code, not FX internals.
    assert "code: return x + self.linear(x)" in readable, (
        "Missing stack trace for M1.forward; got:\n" + readable
    )
    assert "code: return x + self.m1(x)" in readable, (
        "Missing stack trace for M2.forward; got:\n" + readable
    )

    internal_suffixes = (
        "torch/fx/proxy.py",
        "torch/fx/_symbolic_trace.py",
        "torch/_ops.py",
        "torch/_tensor.py",
    )
    for node in graph.nodes:
        stack_trace = node.stack_trace
        if node.op in {"placeholder", "output"}:
            continue
        assert stack_trace is not None, f"node {node.name} has no stack_trace"
        for suffix in internal_suffixes:
            assert suffix not in stack_trace, (
                f"node {node.name} stack_trace leaks internal frame {suffix}:\n"
                f"{stack_trace}"
            )

    print("verify_fix.py: PASSED", file=sys.stderr)


if __name__ == "__main__":
    main()
