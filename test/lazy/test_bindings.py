# Owner(s): ["oncall: jit"]

import torch._lazy.metrics
from torch.testing._internal.common_utils import run_tests


def test_metrics():
    names = torch._lazy.metrics.counter_names()
    assert len(names) == 0, f"Expected no counter names, but got {names}"


if __name__ == "__main__":
    run_tests()
