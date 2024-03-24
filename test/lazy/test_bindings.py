# Owner(s): ["oncall: jit"]

import torch._lazy.metrics

def test_metrics():
    names = torch._lazy.metrics.counter_names()
    assert len(names) == 0, f"Expected no counter names, but got {names}"
