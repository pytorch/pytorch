from cli.lib.pytorch.base import BasePytorchTestPlan
from cli.lib.pytorch.plans.core_tests import CORE_TEST_PLANS


# Central registry — add new plan dicts here as they are created
PYTORCH_TEST_LIBRARY: dict[str, BasePytorchTestPlan] = {
    **CORE_TEST_PLANS,
}
