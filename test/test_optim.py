# Owner(s): ["module: optimizer"]
import functools
import unittest
from copy import deepcopy

import torch
from optim.test_optim import TestOptim, TestDifferentiableOptimizer  # noqa: F401
from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_swa_utils import TestSWAUtils  # noqa: F401
from torch.nn import Parameter
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_optimizers import (
    optim_db, optims, OptimizerErrorEnum, _get_optim_inputs_including_global_cliquey_kwargs)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, largeTensorTest, onlyCPU, onlyCUDA, skipMPS)
from torch.testing._internal.common_utils import markDynamoStrictTest, parametrize, run_tests, TestCase


FP16_REDUCED_PRECISION = {'atol': 1e-5, 'rtol': 1e-4}


def _make_radam_single_tensor_non_capturable(optim_cls, kwargs):
    # Remove this function once https://github.com/pytorch/pytorch/issues/118230 is completed
    if optim_cls == torch.optim.RAdam and not kwargs.get("foreach", False) and kwargs.get("capturable", False):
        # Radam does not support capturable single tensor
        kwargs["capturable"] = False

@markDynamoStrictTest
class TestOptimRenewed(TestCase):

    @onlyCPU
    @optims(optim_db)
    def test_optim_infos_do_not_specify_global_cliquey_kwargs(self, device, dtype, optim_info):
        global_cliquey_flags = ["foreach", "fused", "differentiable"]
        for optim_input in optim_info.optim_inputs_func(device=device):
            self.assertFalse(any(f for f in global_cliquey_flags if f in optim_input.kwargs))


    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None])
    def test_errors(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        for error_input in error_inputs:
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(error_input.error_type, error_input.error_regex):
                        optim_cls(params, **kwargs)
                else:
                    with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                        optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                optim = optim_cls(params, **kwargs)
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(error_input.error_type, error_input.error_regex):
                        optim.step()
                else:
                    with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                        optim.step()
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")


    @parametrize("contiguous", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction(self, device, dtype, optim_info, contiguous):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            if "foreach" in optim_info.supported_impls:
                optim_input.kwargs["foreach"] = False  # force forloop
            if contiguous:
                weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
                bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            else:
                weight = Parameter(torch.randn((10, 5, 2), device=device, dtype=dtype)[..., 0])
                bias = Parameter(torch.randn((10, 2), device=device, dtype=dtype)[..., 0])
            input = torch.randn(5, device=device, dtype=dtype)

            # https://github.com/pytorch/pytorch/issues/118230
            _make_radam_single_tensor_non_capturable(optim_cls, optim_input.kwargs)
            optimizer = optim_cls([weight, bias], **optim_input.kwargs)

            def closure():
                optimizer.zero_grad()
                loss = (weight.mv(input) + bias).pow(2).sum()
                loss.backward()
                if optim_cls.__name__ == "SparseAdam":
                    # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
                    # which we know does NOT represent the expected use case!
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                return loss

            initial_value = closure().item()
            for _ in range(20):
                optimizer.step(closure)

            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(closure().item(), initial_value)
            else:
                self.assertLess(closure().item(), initial_value)


    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction_multigpu(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device="cuda")
        for optim_input in optim_inputs:
            if "foreach" in optim_info.supported_impls:
                optim_input.kwargs["foreach"] = False  # force forloop

            # https://github.com/pytorch/pytorch/issues/118230
            _make_radam_single_tensor_non_capturable(optim_cls, optim_input.kwargs)

            weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
            bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
            input = torch.randn(5, device="cuda:0", dtype=dtype)
            optimizer = optim_cls([weight, bias], **optim_input.kwargs)

            def closure():
                optimizer.zero_grad()
                loss = (weight.mv(input).cuda(1) + bias).pow(2).sum()
                loss.backward()
                if optim_cls.__name__ == "SparseAdam":
                    # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
                    # which we know does NOT represent the expected use case!
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                return loss

            initial_value = closure().item()
            for _ in range(20):
                optimizer.step(closure)

            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(closure().item(), initial_value)
            else:
                self.assertLess(closure().item(), initial_value)


    @skipMPS
    @optims(optim_db, dtypes=[torch.complex64])
    def test_complex(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))
        for optim_input in all_optim_inputs:
            # Last param is intentionally real to test that we can mix real and complex
            complex_params = [
                torch.randn(10, 5, dtype=dtype, requires_grad=True),
                torch.randn(10, dtype=dtype, requires_grad=True),
                torch.randn(10, 5, dtype=torch.float32, requires_grad=True),
            ]
            real_params = [
                (
                    torch.view_as_real(param).detach().clone().requires_grad_()
                    if param.is_complex()
                    else param.detach().clone().requires_grad_()
                )
                for param in complex_params
            ]
            # https://github.com/pytorch/pytorch/issues/118230
            _make_radam_single_tensor_non_capturable(optim_cls, optim_input.kwargs)
            complex_optimizer = optim_cls(complex_params, **optim_input.kwargs)
            real_optimizer = optim_cls(real_params, **optim_input.kwargs)
            real_steps = []
            complex_steps = []
            grads_losses = []

            def real_closure():
                for p in real_params:
                    p.grad = torch.randn_like(p)
                    grads_losses.append(p.grad.clone())
                    real_steps.append(p.clone())
                    if optim_info.only_supports_sparse_grads:
                        # SparseAdam, for example, requires sparse gradients. We convert the Tensor layout,
                        # which we know does NOT represent the expected use case!
                        p.grad = p.grad.to_sparse()
                grads_losses.append(torch.randn(1))

                return grads_losses[-1].clone()

            def complex_closure():
                for p in complex_params:
                    if torch.is_complex(p):
                        p.grad = torch.view_as_complex(grads_losses.pop(0))
                        complex_steps.append(torch.view_as_real_copy(p))
                    else:
                        p.grad = grads_losses.pop(0)
                        complex_steps.append(p.clone())
                    if optim_info.only_supports_sparse_grads:
                        # SparseAdam, for example, requires sparse gradients. We convert the Tensor layout,
                        # which we know does NOT represent the expected use case!
                        p.grad = p.grad.to_sparse()
                return grads_losses.pop(0)

            for _ in range(3):
                if optim_info.step_requires_closure:
                    # LBFGS, for example, requires closure and calls it internally
                    real_optimizer.step(real_closure)
                    complex_optimizer.step(complex_closure)
                else:
                    # For other optimizers, we call closure explicitly to set the gradients
                    real_closure()
                    complex_closure()
                    real_optimizer.step()
                    complex_optimizer.step()

            # All intermediate steps should be the same
            # also checks steps taken within for example a line search
            self.assertEqual(complex_steps, real_steps)


    def _test_derived_optimizers(self, device, dtype, optim_info, flag, reduced_precision=False, assert_step_dtype=None):
        """
        Given a flag 'fused' or 'foreach', test for parity of optimizer state
        and updated parameters between when the flag is set to True and False
        for provided optimizer configurations.
        """
        assert flag in ("foreach", "fused")

        # why 7? iteration 7 is where we start to see differences for RAdam
        # params interacting with the small eps value, because that's right
        # after rho_t becomes greater than 5 in step 6.
        kIterations = 7

        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            updated_params, state = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if (kwargs.get("capturable", False) and str(device) == "cpu"):
                # capturable is not supported on CPU
                continue
            for flag_value in (False, True):
                kwargs[flag] = flag_value

                # https://github.com/pytorch/pytorch/issues/118230
                _make_radam_single_tensor_non_capturable(optim_cls, kwargs)

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
                empty_param = torch.empty((), device=device, dtype=dtype, requires_grad=True)
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                optimizer = optim_cls(params, **kwargs)

                for i in range(kIterations):
                    optimizer.zero_grad()

                    # Test that step behaves as expected (a no-op) when grads are set to None
                    if i != 3:
                        output = model(input)
                        loss = output.sum()
                        loss.backward()

                    optimizer.step()

                if assert_step_dtype is not None:
                    p_state = optimizer.state[params[0]]
                    if torch.is_tensor(p_state.get("step", None)):
                        self.assertEqual(p_state["step"].dtype, assert_step_dtype)

                state.append(optimizer.state)
                updated_params.append(model.parameters())

            assert_eq_kwargs = {} if not reduced_precision else FP16_REDUCED_PRECISION

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                self.assertEqual(og_p, new_p, **assert_eq_kwargs)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]

                for k in og_p_state:
                    self.assertEqual(og_p_state[k], new_p_state[k], **assert_eq_kwargs)


    @skipMPS  # MPS doesn't support torch.float64, see https://github.com/pytorch/pytorch/issues/115350
    @optims([optim for optim in optim_db if "foreach" in optim.supported_impls], dtypes=[torch.float64])
    def test_foreach_matches_forloop(self, device, dtype, optim_info):
        self._test_derived_optimizers(device, dtype, optim_info, "foreach")


    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("impl", ["foreach", "fused"])
    @optims([optim for optim in optim_db if "foreach" in optim.supported_impls or "fused" in optim.supported_impls])
    def test_mixed_device_dtype(self, device, dtype, optim_info, impl):
        """
        Similar in essence to _test_derived_optimizers above. The main difference is that
        _test_derived_optimizers uses model parameters whereas we randomly pass in
        parameters of different dtypes and devices here. We need multiple GPUs (vs just a
        CPU and GPU) because fused adam only works on GPUs. (Thus we only run the tests
        that call into this helper when TEST_MULTIGPU.)
        """
        assert impl in ("foreach", "fused")
        if impl == "foreach" and "foreach" not in optim_info.supported_impls:
            return unittest.skip(f"foreach not supported for {optim_info.optim_cls.__name__}")
        elif impl == "fused" and "fused" not in optim_info.supported_impls:
            return unittest.skip(f"fused not supported for {optim_info.optim_cls.__name__}")

        params = [
            torch.rand(2, 3, dtype=torch.float64, device='cuda:0', requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device='cuda:0', requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device='cuda:0', requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device='cuda:0', requires_grad=True),
            torch.rand(2, 3, dtype=torch.float64, device='cuda:1', requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device='cuda:1', requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device='cuda:1', requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device='cuda:1', requires_grad=True),
            torch.randint(1024, (2, 3), dtype=torch.int64, device='cuda:1', requires_grad=False),
        ]

        for p in params:
            if p.requires_grad:
                p.grad = torch.rand_like(p, device=p.device, dtype=p.dtype)

        kIterations = 7 if impl == "foreach" else 1
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            updated_params, state = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and str(device) == "cpu":
                # capturable is not supported on CPU
                continue
            for use_impl in (False, True):
                kwargs[impl] = use_impl
                params_clone = []
                for p in params:
                    p_clone = p.clone().detach()
                    if p.requires_grad:
                        p_clone.requires_grad = True
                        p_clone.grad = p.grad.clone().detach()
                        params_clone.append(p_clone)

                # https://github.com/pytorch/pytorch/issues/118230
                _make_radam_single_tensor_non_capturable(optim_cls, kwargs)
                optimizer = optim_cls(params_clone, **kwargs)
                for _ in range(kIterations):
                    optimizer.step()

                state.append(optimizer.state)
                updated_params.append(params_clone)

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                # Increasing the tolerance as we are collating lots of ops together for optimizers and
                # the designated tolerances are for single op only.
                single_rtol, single_atol = torch.testing._comparison.get_tolerances(new_p.dtype, rtol=None, atol=None)
                rtol = 5 * single_rtol
                atol = 5 * single_atol

                self.assertEqual(og_p, new_p, rtol=rtol, atol=atol)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]

                for k in og_p_state:
                    actual = new_p_state[k]
                    self.assertEqual(og_p_state[k], actual, rtol=rtol, atol=atol)


    @onlyCUDA
    @optims([optim for optim in optim_db if "foreach" in optim.supported_impls], dtypes=[torch.float64])
    def test_set_default_dtype_works_with_foreach(self, device, dtype, optim_info):
        # https://github.com/pytorch/pytorch/issues/110940
        # We coerce step to always be float32 unless the
        # default dtype is higher prec float64
        old_default_dtype = torch.get_default_dtype()
        for default_dtype in [torch.float64, torch.float16]:
            torch.set_default_dtype(default_dtype)
            self._test_derived_optimizers(
                device,
                dtype,
                optim_info,
                "foreach",
                reduced_precision=default_dtype == torch.float16,
                assert_step_dtype=torch.float64 if default_dtype == torch.float64 else torch.float32,
            )
            torch.set_default_dtype(old_default_dtype)



    @onlyCUDA
    @largeTensorTest("72GB", "cuda")
    @optims([optim for optim in optim_db if "foreach" in optim.supported_impls], dtypes=[torch.float16])
    def test_foreach_large_tensor(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2 ** 32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, foreach=True, **optim_input.kwargs)
            optimizer.step()


    @onlyCUDA
    @optims([optim for optim in optim_db if "foreach" in optim.supported_impls], dtypes=[torch.float32])
    def test_peak_memory_foreach(self, device, dtype, optim_info):
        nparams = 10
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            kwargs = deepcopy(optim_input.kwargs)
            max_mems = []
            for flag_value in (False, True):
                kwargs["foreach"] = flag_value

                # https://github.com/pytorch/pytorch/issues/118230
                _make_radam_single_tensor_non_capturable(optim_cls, kwargs)

                # The 128 is critical here! Our CUDACachingAllocator allocates in blocks of 512,
                # meaning any tensor that occupies <512 bytes of memory will allocate a whole
                # 512 bytes anyway. We use 128 (since datasize would be 4 bytes) so that param
                # is size 512 exactly, making our later calculations for intermediate_size easy.
                param = torch.rand(128, device=device, dtype=dtype)
                params = [torch.rand_like(param) for _ in range(nparams)]

                optimizer = optim_cls(params, **kwargs)

                for p in params:
                    p.grad = torch.rand_like(p)

                optimizer.step()
                import gc
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                optimizer.step()
                gc.collect()
                max_mems.append(torch.cuda.max_memory_allocated())

            st_max_mem, mt_max_mem = max_mems
            intermediate_size = nparams * param.nelement() * param.element_size()
            nintermediates = 1  # we expect a budget of 1 intermediate most of the time
            if kwargs.get('capturable') or optim_cls.__name__ in ["Adadelta", "ASGD", "RAdam"]:
                # with capturable in Adam(W), we have 2 extra intermediates for the bias_corrections
                # with Adadelta, we have 2 extra for (acc_delta + eps) and (square_avg + eps)
                # ASGD allocates axs, 2x mus, 2x etas, and grads at the same time
                nintermediates = 3
                if optim_cls.__name__ == "NAdam":
                    # with capturable in NAdam, we have 3 extra intermediates for the
                    # bias_correction, mus, and mu_nexts
                    nintermediates = 5

                if optim_cls.__name__ == "RAdam":
                    # RAdam has four intermediates with capturable
                    # num, unrect_step_size, buffer, grouped_grads
                    nintermediates = 4

            elif optim_cls.__name__ in ["NAdam", "Adagrad", "RMSprop"]:
                # NAdam uses two intermediates at the same time (grads & exp_avg_sq_sqrt)
                # Adagrad uses std and grads at the same time
                # RMSprop uses avg and grads
                nintermediates = 2

            self.assertLessEqual(mt_max_mem, st_max_mem + intermediate_size * nintermediates)


    @onlyCUDA
    @optims([optim for optim in optim_db if "fused" in optim.supported_impls], dtypes=[torch.float64])
    def test_fused_matches_forloop(self, device, dtype, optim_info):
        self._test_derived_optimizers(device, dtype, optim_info, "fused")


    @onlyCUDA
    @largeTensorTest("64GB", "cuda")
    @optims([optim for optim in optim_db if "fused" in optim.supported_impls], dtypes=[torch.float16])
    def test_fused_large_tensor(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2 ** 32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
            optimizer.step()


    @onlyCUDA
    @parametrize("impl", ["fused", "capturable"])
    @optims([optim for optim in optim_db if "fused" in optim.supported_impls], dtypes=[torch.float32])
    def test_cpu_load_state_dict(self, device, dtype, impl, optim_info):
        # NOTE: This SIMULATES a fused/capturable optimizer with state moved to CPU, issue 103256
        # How do we get there? Users typically create CUDA models on fused optimizers and then
        # store checkpoints on CPU as CUDA memory is limited with torch.load(...map_location="cpu").
        # Since this is a unit test, it is more expedient to simulate what the state_dict
        # would look like, which is basically CPU tensors with fused/capturable flag = True.
        optim_cls = optim_info.optim_cls
        if optim_cls.__name__ == "SGD" and impl == "capturable":
            # Capturable SGD does not exist
            self.skipTest("SGD does not currently support capturable")

        cpu_optim_inputs = optim_info.optim_inputs_func(device="cpu")
        for optim_input in cpu_optim_inputs:
            param = torch.tensor([0.1, 0.2], dtype=dtype, device="cpu")
            optimizer = optim_cls([param], **optim_input.kwargs)
            param.grad = torch.rand_like(param)
            optimizer.step()
            optim_state_dict_cpu = deepcopy(optimizer.state_dict())
            optim_state_dict_cpu["param_groups"][0][impl] = True

            # load
            optim_input.kwargs[impl] = True
            param_cuda = param.clone().detach().to(device="cuda")
            optimizer_cuda = optim_cls([param_cuda], **optim_input.kwargs)
            optimizer_cuda.load_state_dict(optim_state_dict_cpu)
            optimizer_cuda.zero_grad()
            param_cuda.grad = torch.rand_like(param_cuda)
            optimizer_cuda.step()


    @optims(optim_db, dtypes=[torch.float32])
    def test_param_groups_weight_decay(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))
        for optim_input in all_optim_inputs:
            weight_kwargs = optim_input.kwargs
            bias_kwargs = deepcopy(optim_input.kwargs)
            bias_kwargs["weight_decay"] = 0.0

            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            input = torch.randn(5, device=device, dtype=dtype)

            optimizer = optim_cls([dict(params=[weight], **weight_kwargs), dict(params=[bias], **bias_kwargs)])

            loss = (weight.mv(input) + bias).pow(2).sum()
            initial_value = loss.item()
            for _ in range(20):
                optimizer.zero_grad()
                loss = (weight.mv(input) + bias).pow(2).sum()
                loss.backward()
                if optim_cls.__name__ == "SparseAdam":
                    # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
                    # which we know does NOT represent the expected use case!
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                optimizer.step()

            # Test that the direction of loss moved appropriately
            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(loss.item(), initial_value)
            else:
                self.assertLess(loss.item(), initial_value)


    @optims(optim_db, dtypes=[torch.float32])
    def test_param_groups_lr(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))
        for optim_input in all_optim_inputs:
            # optim_input.kwargs will be the param group kwargs, which should have >0 lr
            if "lr" not in optim_input.kwargs or optim_input.kwargs["lr"] == 0:
                optim_input.kwargs["lr"] = 1e-3
            outer_kwargs = {"lr": 1e-28}
            if optim_cls.__name__ == "Rprop":
                # Allow min step size to be 0
                outer_kwargs["step_sizes"] = (0, 50)

            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            irrelevant = Parameter(torch.randn(2, device=device, dtype=dtype))
            irrelevant_clone = irrelevant.clone()
            input = torch.randn(5, device=device, dtype=dtype)
            optimizer = optim_cls(
                [dict(params=[weight, bias], **optim_input.kwargs), dict(params=[irrelevant])],
                **outer_kwargs)

            loss = (weight.mv(input) + bias).pow(2).sum()
            initial_value = loss.item()
            for _ in range(20):
                optimizer.zero_grad()
                loss = (weight.mv(input) + bias).pow(2).sum()
                loss.backward()
                irrelevant.grad = torch.rand_like(irrelevant)
                if optim_cls.__name__ == "SparseAdam":
                    # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
                    # which we know does NOT represent the expected use case!
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                    irrelevant.grad = irrelevant.grad.to_sparse()
                optimizer.step()

            # Test that the direction of loss moved appropriately
            if optim_input.kwargs.get("maximize", False):
                self.assertGreater(loss.item(), initial_value)
            else:
                self.assertLess(loss.item(), initial_value)

            # Test that irrelevant parameters were not updated since lr was almost 0
            self.assertEqual(irrelevant, irrelevant_clone)


    @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_when_params_have_no_grad(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info)
        params = [
            torch.randn(2, 3, requires_grad=False, device=device, dtype=dtype)
            for _ in range(2)]
        old_params = [p.clone().detach() for p in params]

        def closure():
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            _make_radam_single_tensor_non_capturable(optim_cls, optim_input.kwargs)
            optimizer = optim_cls(params, **optim_input.kwargs)
            optimizer.step(closure)
            self.assertEqual(old_params, params)


    @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_for_zero_grads(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info)
        param = torch.randn((5, 1), device=device, dtype=dtype, requires_grad=True)
        old_param = param.clone().detach()

        def closure():
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs
            _make_radam_single_tensor_non_capturable(optim_cls, optim_input.kwargs)

            # params will decay even if grads are empty if weight_decay != 0,
            # and capturable doesn't work for CPU tensors
            if kwargs.get("weight_decay", 0) != 0:
                continue

            # AdamW params will be updated regardless of grads due to lr, so make lr smaller
            if optim_cls.__name__ == "AdamW":
                kwargs["lr"] = torch.tensor(1e-4) if isinstance(kwargs.get("lr", 1e-4), torch.Tensor) else 1e-4

            if kwargs.get("differentiable", False):
                params = [param.clone()]
            else:
                params = [param]

            optimizer = optim_cls(params, **kwargs)
            if optim_cls.__name__ == "SparseAdam":
                # Intentionally construct a multidimensional empty v for the sparse grad
                # Single dim v passes the test while multidim correctly repros the issue
                # https://github.com/pytorch/pytorch/issues/82486
                i = torch.empty((1, 0), device=device, dtype=dtype)
                v = torch.empty((0, 1), device=device, dtype=dtype)
                params[0].grad = torch.sparse_coo_tensor(i, v, (5, 1), device=device, dtype=dtype)
            else:
                params[0].grad = torch.zeros_like(params[0])
            optimizer.step(closure)
            self.assertEqual(old_param, params[0])


    @optims(optim_db, dtypes=[torch.float32])
    def test_optimizer_can_be_printed(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info)
        params = [Parameter(torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)) for _ in range(2)]
        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            optimizer.__repr__()


    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_deterministic(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))
        weight = Parameter(torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype))
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        params = [weight, bias]

        def fwd_bwd(optim, w, b, i):
            optim.zero_grad()
            loss = (w.mv(i) + b).pow(2).sum()
            loss.backward()
            return loss

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # Prime the optimizer
            for _ in range(10):
                optimizer.step(closure)

            # Clone the weights and construct a new optimizer for them
            with torch.no_grad():
                weight_c = Parameter(weight.clone())
                bias_c = Parameter(bias.clone())

            optimizer_c = optim_cls([weight_c, bias_c], **optim_input.kwargs)
            closure_c = functools.partial(fwd_bwd, optimizer_c, weight_c, bias_c, input)

            # Load the state dict from the original optimizer into the new one
            optimizer_c.load_state_dict(deepcopy(optimizer.state_dict()))

            # Run both optimizers in parallel
            for _ in range(10):
                optimizer.step(closure)
                optimizer_c.step(closure_c)
                self.assertEqual(weight, weight_c)
                self.assertEqual(bias, bias_c)

            # Make sure state dict is deterministic with equal (not identical) parameters
            self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())

            # Make sure repeated parameters have identical representation (see #36831)
            optimizer_c.param_groups.extend(optimizer_c.param_groups)
            self.assertEqual(
                optimizer.state_dict()["param_groups"][-1],
                optimizer_c.state_dict()["param_groups"][-1]
            )

    @optims(optim_db, dtypes=[torch.float32])
    def test_can_load_older_state_dict(self, device, dtype, optim_info):
        new_flags = ["maximize", "foreach", "fused", "differentiable", "capturable"]
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))
        for optim_input in all_optim_inputs:
            torch.manual_seed(1)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(4, 2, 1, stride=2),
                torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
            )
            model.to(dtype=dtype, device=device)
            input = torch.rand(1, 4, 16, 16, device=device, dtype=dtype)
            optimizer = optim_cls(model.parameters(), **optim_input.kwargs)

            def fwd_bwd(optim, mod, i):
                optim.zero_grad()
                loss = mod(i).sum()
                loss.backward()
                return loss

            for _ in range(3):
                if optim_cls.__name__ == "LBFGS":
                    optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
                else:
                    fwd_bwd(optimizer, model, input)
                    optimizer.step()

            # old_state_dict has all new flags del'd
            old_state_dict = deepcopy(optimizer.state_dict())
            old_state_dict_pg = old_state_dict["param_groups"]
            for group in old_state_dict_pg:
                for flag in new_flags:
                    if flag in group:
                        del group[flag]

            optimizer.load_state_dict(old_state_dict)

            # Make sure we can still step
            if optim_cls.__name__ == "LBFGS":
                optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
            else:
                fwd_bwd(optimizer, model, input)
                optimizer.step()


    @optims(optim_db, dtypes=[torch.float32])
    def test_load_nontensor_step(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))
        params = [Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)]
        for p in params:
            p.grad = torch.rand_like(p)
            # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
            # which we know does NOT represent the expected use case!
            if optim_cls.__name__ == "SparseAdam":
                p.grad = p.grad.to_sparse()

        # Needed for LBFGS
        lbfgs_loss = torch.rand(1, device=device, dtype=dtype)

        def closure():
            return lbfgs_loss if optim_cls.__name__ == "LBFGS" else None

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs
            # See https://github.com/pytorch/pytorch/issues/117836 for Adamax
            # See https://github.com/pytorch/pytorch/issues/118230 for RAdam
            if optim_cls.__name__ in ["Adamax", "RAdam"] and kwargs.get("capturable", False) and not kwargs.get("foreach", False):
                continue

            optimizer = optim_cls(params, **optim_input.kwargs)
            for _ in range(3):
                optimizer.step(closure)
            state_dict = deepcopy(optimizer.state_dict())
            for p_state in state_dict["state"].values():
                if "step" in p_state and torch.is_tensor(p_state["step"]):
                    p_state["step"] = p_state["step"].item()
            optimizer.load_state_dict(state_dict)
            optimizer.step(closure)


    @onlyCUDA
    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_with_cuda_params(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # We limit our configs to CPU only, because we will be moving them to CUDA later
        cpu_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs("cpu", dtype, optim_info, skip=("differentiable",))

        # Needed for LBFGS
        lbfgs_loss = torch.rand(1, device=device, dtype=dtype)

        def closure():
            return lbfgs_loss if optim_cls.__name__ == "LBFGS" else None

        for optim_input in cpu_optim_inputs:
            params = [Parameter(torch.randn(2, 3, device="cpu", dtype=dtype)) for _ in range(2)]
            for p in params:
                p.grad = torch.randn_like(p)
                # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
                # which we know does NOT represent the expected use case!
                if optim_cls.__name__ == "SparseAdam":
                    p.grad = p.grad.to_sparse()

            optimizer = optim_cls(params, **optim_input.kwargs)

            for _ in range(3):
                optimizer.step(closure)

            with torch.no_grad():
                params_cuda = [p.to(device="cuda") for p in params]
                for (i, p) in enumerate(params_cuda):
                    p.grad = params[i].grad.to(device="cuda")
            optimizer_cuda = optim_cls(params_cuda, **optim_input.kwargs)

            state_dict_cpu = deepcopy(optimizer.state_dict())
            state_dict_cuda = deepcopy(optimizer.state_dict())
            optimizer_cuda.load_state_dict(state_dict_cuda)

            # Make sure state_dict_cuda isn't modified by merely calling load_state_dict
            self.assertEqual(state_dict_cpu, state_dict_cuda)

            # Make sure that device of state['step'] is still CPU _unless_ torch.compile() added a capturable!
            capturable = state_dict_cpu["param_groups"][0].get("capturable", False)
            new_state_dict = optimizer_cuda.state_dict()
            for state_cpu, state_cuda in zip(state_dict_cpu["state"].values(), new_state_dict["state"].values()):
                if "step" in state_cpu and torch.is_tensor(state_cpu["step"]):
                    self.assertEqual(state_cuda["step"].device.type, "cuda" if capturable else "cpu")

            for _ in range(5):
                optimizer.step(closure)
                optimizer_cuda.step(closure)
                self.assertEqual(params, params_cuda)
                self.assertEqual(optimizer.state_dict(), optimizer_cuda.state_dict())


    @optims(optim_db, dtypes=[torch.float32])
    def test_deepcopy_copies_all_public_attrs(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(device, dtype, optim_info, skip=("differentiable",))

        params = [Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)]
        for p in params:
            p.grad = torch.rand_like(p)
            if optim_cls.__name__ == "SparseAdam":
                # SparseAdam requires sparse gradients. For this test, we convert the Tensor layout,
                # which we know does NOT represent the expected use case!
                p.grad = p.grad.to_sparse()

        # Needed for LBFGS
        def closure():
            return 1 if optim_cls.__name__ == "LBFGS" else None

        def getPublicAttrs(obj):
            return {k for k in obj.__dict__ if not k.startswith("_")}

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)

            # Make some state
            for _ in range(3):
                optimizer.step(closure)

            self.assertEqual(getPublicAttrs(optimizer), getPublicAttrs(deepcopy(optimizer)))


instantiate_device_type_tests(TestOptimRenewed, globals(), allow_mps=True)


if __name__ == '__main__':
    run_tests()
