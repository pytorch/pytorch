# Owner(s): ["module: optimizer"]
from copy import deepcopy

import torch
from optim.test_optim import TestOptim, TestDifferentiableOptimizer  # noqa: F401
from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_swa_utils import TestSWAUtils  # noqa: F401
from torch.testing._internal.common_optimizers import optim_db, optims, OptimizerErrorEnum
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU, skipMPS
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


    def _test_derived_optimizers(self, device, dtype, optim_info, flag):
        assert flag in ("foreach", "fused")

        # why 7? iteration 7 is where we start to see differences for RAdam
        # params interacting with the small eps value, because that's right
        # after rho_t becomes greater than 5 in step 6.
        kIterations = 7

        optim_inputs = optim_info.optim_inputs_func()
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            updated_params, state = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if (kwargs.get("capturable", False) and
               (str(device) == "cpu" or optim_cls.__name__ == "ASGD")):
                # capturable is not supported on CPU nor in single tensor ASGD
                continue
            for flag_value in (False, True):
                kwargs[flag] = flag_value
                input = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=device
                ).reshape(3, 2)

                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=dtype, device=device)

                # foreach/fused optimizers should be tested with a
                # zero_size tensor as its last param.
                # ref: https://github.com/pytorch/pytorch/issues/100701
                empty_params = [torch.empty((), device=device, dtype=dtype)]
                params = list(model.parameters()) + empty_params

                optimizer = optim_cls(params, **kwargs)

                for i in range(kIterations):
                    optimizer.zero_grad()
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                    # Test that step behaves as expected (a no-op) when grads are set to None
                    if i == 3:
                        optimizer.zero_grad(set_to_none=True)

                    optimizer.step()

                state.append(optimizer.state)
                updated_params.append(model.parameters())

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                self.assertEqual(og_p, new_p)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]

                for k in og_p_state:
                    self.assertEqual(og_p_state[k], new_p_state[k])


    @skipMPS  # MPS doesn't support torch.float64, see https://github.com/pytorch/pytorch/issues/115350
    @optims([optim for optim in optim_db if "foreach" in optim.supported_impls], dtypes=[torch.float64])
    def test_foreach_matches_forloop(self, device, dtype, optim_info):
        self._test_derived_optimizers(device, dtype, optim_info, "foreach")


    @onlyCPU
    @optims(optim_db)
    def test_optim_infos_do_not_specify_global_cliquey_kwargs(self, device, dtype, optim_info):
        global_cliquey_flags = ["foreach", "fused", "differentiable"]
        for optim_input in optim_info.optim_inputs_func():
            self.assertFalse(any(f for f in global_cliquey_flags if f in optim_input.kwargs))


instantiate_device_type_tests(TestOptimRenewed, globals(), allow_mps=True)


if __name__ == '__main__':
    run_tests()
