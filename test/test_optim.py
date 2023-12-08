# Owner(s): ["module: optimizer"]

from optim.test_optim import TestOptim, TestDifferentiableOptimizer  # noqa: F401
from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_swa_utils import TestSWAUtils  # noqa: F401
from torch.testing._internal.common_optimizers import optim_db, optims, OptimizerErrorEnum
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU
from torch.testing._internal.common_utils import run_tests, TestCase

class TestOptimRenewed(TestCase):

    @onlyCPU
    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None])
    def test_errors(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        for error_input in error_inputs:
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                optim = optim_cls(params, **kwargs)
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    optim.step()
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")


instantiate_device_type_tests(TestOptimRenewed, globals(), allow_mps=True)


if __name__ == '__main__':
    run_tests()
