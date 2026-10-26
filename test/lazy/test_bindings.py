# Owner(s): ["oncall: jit"]

import torch._lazy.metrics
from torch.testing._internal import tci
from torch.testing._internal.common_utils import run_tests


@tci.test(platform=(tci.platform.linux_x86_64, tci.platform.linux_x86_64_cuda_l4))
def test_metrics():
    names = torch._lazy.metrics.counter_names()
    if len(names) != 0:
        raise AssertionError(f"Expected no counter names, but got {names}")


if __name__ == "__main__":
    run_tests()
