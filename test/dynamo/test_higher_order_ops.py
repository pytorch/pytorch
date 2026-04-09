# Owner(s): ["module: dynamo"]
# ruff: noqa: F401
# flake8: noqa: F401

import sys
from pathlib import Path

from torch.testing._internal.common_device_type import instantiate_device_type_tests

try:
    from ._higher_order_ops_test_activation_checkpointing import ActivationCheckpointingTests
    from ._higher_order_ops_test_core import HigherOrderOpTests
    from ._higher_order_ops_test_functorch import FuncTorchHigherOrderOpTests
    from ._higher_order_ops_test_opinfo import TestHigherOrderOpsOpInfo
    from ._higher_order_ops_test_vmap_guards import HigherOrderOpVmapGuardTests
except ImportError:
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    from _higher_order_ops_test_activation_checkpointing import ActivationCheckpointingTests
    from _higher_order_ops_test_core import HigherOrderOpTests
    from _higher_order_ops_test_functorch import FuncTorchHigherOrderOpTests
    from _higher_order_ops_test_opinfo import TestHigherOrderOpsOpInfo
    from _higher_order_ops_test_vmap_guards import HigherOrderOpVmapGuardTests


instantiate_device_type_tests(TestHigherOrderOpsOpInfo, globals(), only_for=("cuda",))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
