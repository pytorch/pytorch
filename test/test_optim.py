# Owner(s): ["module: optimizer"]
import functools
import math
import tempfile
import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from optim.test_lrscheduler import TestLRScheduler  # noqa: F401
from optim.test_optim import TestDifferentiableOptimizer  # noqa: F401
from optim.test_swa_utils import TestSWAUtils  # noqa: F401

import torch
from torch.nn import Parameter
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipMPS,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_optimizers import (
    _get_device_type,
    _get_optim_inputs_including_global_cliquey_kwargs,
    optim_db,
    OptimizerErrorEnum,
    optims,
    TensorTracker,
)
from torch.testing._internal.common_utils import (
    markDynamoStrictTest,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfS390X,
)


FP16_REDUCED_PRECISION = {"atol": 1e-5, "rtol": 1e-4}


def rosenbrock(tensor):
    assert tensor.size() == torch.Size(
        [2]
    ), f"Requires tensor with 2 scalars but got {tensor.size()}"
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def drosenbrock(tensor):
    assert tensor.size() == torch.Size(
        [2]
    ), f"Requires tensor with 2 scalars but got {tensor.size()}"
    x, y = tensor
    return torch.stack((-400 * x * (y - x**2) - 2 * (1 - x), 200 * (y - x**2)))


@markDynamoStrictTest
class TestOptimRenewed(TestCase):
    """
    This test class validates the core optimizers and is structured as the correctness of:
    - The update algorithms (forloop implementation)
        * Every optimizer's algorithm is most readably implemented through a big for-loop
          over all the parameters, which is what we refer to as the forloop or single tensor
          implementation. These algorithms are manually validated by comparing to the paper
          and systematically validated by assuring that the loss goes the right direction
          when the optimizer has been applied.
        * This implementation should compose with optimizer hyperparameters well, such as
          supporting Tensor LRs, the capturable API, and sparse and complex parameters.
    - Each varying implementation
        * We then have implementations that improve upon the performance of the forloop
          implementation by leveraging fusion, namely our foreach (mult_tensor) and fused
          implementations.
        * These variations are validated numerically by comparing with the forloop version
          of the optimizer. In fact, we test most variations this way--we see the forloop
          implementation as the ground truth and expect that improvements to it in any way
          should be just as correct.
        * Both params and optimizer states should be validated numerically.
    - state_dict APIs
        * The optimizer instance should be serializable
        * Calling save and load should be deterministic
        * Moving between devices should be seamless
        * BC - load_state_dict should be able to handle older optimizer states
    - Hook APIs (everything should fire in the right order)
    - LR Scheduler integration (composing should not error + should go the right direction)
    - Parameter groups (should be equivalent to having multiple optimizers)
    - Erroring (what should error should error)

    We also cover different ways of generating parameters and grads:
    - With parameters, we either generate them randomly given specific shapes or we take
      them from a sample NN module.
        * Variety is important here because NN modules have type Parameter and randomly
          generated tensors have type Tensor.
        * Parameters can be sparse for a subset of the optimizers (check out OptimizerInfo)
        * Complex parameters should be handled using view_as_real
        * Parameters can be spread across different devices and different dtypes for any
          given optimizer
        * Parameters can be contiguous and noncontiguous
    - With grads, we follow suit from the parameters.
        * Grads can also be None, empty, or zero-valued, and this should not disrupt training.
    """

    @onlyCPU
    @optims(optim_db)
    def test_optim_infos_do_not_specify_global_cliquey_kwargs(
        self, device, dtype, optim_info
    ):
        global_cliquey_flags = ["foreach", "fused", "differentiable"]
        for optim_input in optim_info.optim_inputs_func(device=device):
            self.assertFalse(
                any(f for f in global_cliquey_flags if f in optim_input.kwargs)
            )

    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None])
    def test_errors(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        for error_input in error_inputs:
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim_cls(params, **kwargs)
                else:
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                optim = optim_cls(params, **kwargs)
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim.step()
                else:
                    with self.assertRaisesRegex(
                        error_input.error_type, error_input.error_regex
                    ):
                        optim.step()
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")

    @parametrize("contiguous", [True, False])
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction(
        self, device, dtype, optim_info, contiguous, with_lrsched
    ):
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )

        for schedulers_constructor in schedulers_constructors:
            # with tensor LR we need fresh inputs for each scheduler
            # or mutating it will carry across iters
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop
                if contiguous:
                    weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
                    bias = Parameter(torch.randn((10), device=device, dtype=dtype))
                else:
                    weight = Parameter(
                        torch.randn((10, 5, 2), device=device, dtype=dtype)[..., 0]
                    )
                    bias = Parameter(
                        torch.randn((10, 2), device=device, dtype=dtype)[..., 0]
                    )
                input = torch.randn(5, device=device, dtype=dtype)

                optimizer = optim_cls([weight, bias], **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                def closure():
                    optimizer.zero_grad()
                    loss = (weight.mv(input) + bias).pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    if optim_info.step_requires_closure:
                        loss = optimizer.step(closure)
                    else:
                        loss = closure()
                        optimizer.step()

                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction_multigpu(
        self, device, dtype, optim_info, with_lrsched
    ):
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (
            optim_info.scheduler_inputs if with_lrsched else [None]
        )
        for schedulers_constructor in schedulers_constructors:
            # We need a fresh set of inputs if we have a tensor LR
            # to not carry mutations across iterations.
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop

                weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
                bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
                inpt = torch.randn(5, device="cuda:0", dtype=dtype)

                optimizer = optim_cls([weight, bias], **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])
                ]

                def closure():
                    optimizer.zero_grad()
                    loss = (weight.mv(inpt).cuda(1) + bias).pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    loss = optimizer.step(closure)
                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    def test_param_group_with_lrscheduler_goes_right_direction(
        self, device, dtype, optim_info
    ):
        optim_cls = optim_info.optim_cls

        for schedulers_c in optim_info.scheduler_inputs:
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            inpt = torch.randn(5, device=device, dtype=dtype)

            # avoid endless recompiles by wrapping LR in a tensor if we're compiling
            lr = torch.tensor(0.01) if torch.compiler.is_compiling() else 0.01
            optimizer = optim_cls([{"params": [weight]}, {"params": [bias], "lr": lr}])
            schedulers = [scheduler_c(optimizer) for scheduler_c in schedulers_c]

            def closure():
                optimizer.zero_grad()
                loss = (weight.mv(inpt) + bias).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    weight.grad = weight.grad.to_sparse()
                    bias.grad = bias.grad.to_sparse()
                return loss

            initial_value = closure().item()
            for _ in range(20):
                loss = optimizer.step(closure)
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss)
                    else:
                        scheduler.step()

            self.assertLess(closure().item(), initial_value)

    @optims(optim_db, dtypes=[torch.float32])
    def test_tensor_lr(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input in all_optim_inputs:
            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            weight_c = weight.detach().clone().requires_grad_(True)
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            bias_c = bias.detach().clone().requires_grad_(True)
            inpt = torch.randn(5, device=device, dtype=dtype)

            kwargs = optim_input.kwargs
            if "lr" in kwargs:
                del kwargs["lr"]

            kwargs["lr"] = 1.0 if optim_info.step_requires_closure else 1e-3
            optimizer_r = optim_cls([weight, bias], **kwargs)

            try:
                kwargs["lr"] = torch.tensor(kwargs["lr"])
                optimizer = optim_cls([weight_c, bias_c], **kwargs)
            except ValueError as e:
                self.assertRegex(str(e), ".*lr as a Tensor is not supported.*")
                continue

            def closure(optim, w, b, i):
                optim.zero_grad()
                loss = (w.mv(i) + b).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    w.grad = w.grad.to_sparse()
                    b.grad = b.grad.to_sparse()
                return loss

            for _ in range(5):
                if optim_info.step_requires_closure:
                    optimizer_r.step(
                        functools.partial(closure, optimizer_r, weight, bias, inpt)
                    )
                    optimizer.step(
                        functools.partial(closure, optimizer, weight_c, bias_c, inpt)
                    )
                else:
                    closure(optimizer_r, weight, bias, inpt)
                    closure(optimizer, weight_c, bias_c, inpt)

                self.assertEqual(weight, weight_c)
                self.assertEqual(bias, bias_c)

    @parametrize("with_lrsched", [True, False])
    @optims(
        [o for o in optim_db if o.supports_sparse or o.only_supports_sparse_grads],
        dtypes=[torch.float64],
    )
    def test_rosenbrock_sparse(self, device, dtype, optim_info, with_lrsched):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Fused impls do not support sparse gradients
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        kwarg_updates, schedulers_constructors = optim_info.metadata_for_sparse

        if with_lrsched and len(schedulers_constructors) == 0:
            return

        supported_inputs = []
        if len(kwarg_updates) != 0:
            seen = set()
            for i in all_optim_inputs:
                for k in kwarg_updates:
                    if k in i.kwargs:
                        del i.kwargs[k]
                hashable_kwargs = tuple(sorted(i.kwargs.items()))
                if len(i.kwargs) > 0 and hashable_kwargs not in seen:
                    supported_inputs.append(i)
                    seen.add(hashable_kwargs)
                    if "lr" in kwarg_updates:
                        i.kwargs["lr"] = kwarg_updates["lr"]
        else:
            supported_inputs = all_optim_inputs

        for optim_input in supported_inputs:
            kwargs = optim_input.kwargs
            multi_tensor = kwargs.get("foreach", False)

            # For rosenbrock tests, it is mandated that the param is a tensor with 2 numbers
            if multi_tensor:
                params_t = [
                    torch.tensor([1.5, 1.5]),
                    torch.tensor([1.5, 1.5], dtype=dtype),
                ]
            else:
                params_t = [torch.tensor([1.5, 1.5])]

            params = [Parameter(param_t) for param_t in params_t]
            optimizer = optim_cls(params, **kwargs)
            schedulers = [
                s(optimizer) for s in (schedulers_constructors if with_lrsched else [])
            ]

            if not optim_info.only_supports_sparse_grads:
                params_c = [Parameter(param_t.clone()) for param_t in params_t]
                optimizer_c = optim_cls(params_c, **kwargs)
                schedulers_c = [
                    s(optimizer_c)
                    for s in (schedulers_constructors if with_lrsched else [])
                ]

            solution = torch.tensor([1, 1])
            with torch.no_grad():
                initial_dist = sum(param.dist(solution) for param in params)

            def get_grad(param, sparse_grad, w):
                grad = drosenbrock(param)
                # NB: We torture test the optimizer by returning an
                # uncoalesced sparse tensor

                # Depending on w, provide only the x or y gradient
                if sparse_grad:
                    if w:
                        i = torch.tensor([[0, 0]], dtype=torch.int64)
                        x = grad[0]
                        v = torch.tensor([x / 4.0, x - x / 4.0])
                    else:
                        i = torch.tensor([[1, 1]], dtype=torch.int64)
                        y = grad[1]
                        v = torch.tensor([y - y / 4.0, y / 4.0])
                    grad_out = torch.sparse_coo_tensor(i, v, (2,), dtype=v.dtype)
                else:
                    if w:
                        grad_out = torch.tensor([grad[0], 0], dtype=param.dtype)
                    else:
                        grad_out = torch.tensor([0, grad[1]], dtype=param.dtype)
                return grad_out

            def eval(params, sparse_grad, w):
                optimizer.zero_grad()
                if multi_tensor:
                    loss = sum(rosenbrock(param) for param in params)
                else:
                    loss = rosenbrock(params[0])
                loss.backward()

                grads_out = [get_grad(param, sparse_grad, w) for param in params]
                with torch.no_grad():
                    params[0].grad = grads_out[0]
                    if multi_tensor:
                        params[1].grad = grads_out[1].to(dtype=dtype)
                return loss

            for i in range(1800):
                # Do cyclic coordinate descent
                w = i % 2
                optimizer.step(functools.partial(eval, params, True, w))
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(rosenbrock(params[0]))
                    else:
                        scheduler.step()
                if not optim_info.only_supports_sparse_grads:
                    optimizer_c.step(functools.partial(eval, params_c, False, w))
                    for scheduler in schedulers_c:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(rosenbrock(params_c[0]))
                        else:
                            scheduler.step()
                    # Tolerance is increased due to floating point error from different
                    # code path for dense case: x v.s. x - x / 4.0 + x / 4.0
                    self.assertEqual(params, params_c, atol=5e-6, rtol=5e-6)

            if not kwargs.get("maximize", False):
                self.assertLessEqual(
                    sum(param.dist(solution) for param in params), initial_dist
                )
            else:
                self.assertGreaterEqual(
                    sum(rosenbrock(param) for param in params),
                    sum(rosenbrock(param_t) for param_t in params_t),
                )

    @skipMPS
    @optims([o for o in optim_db if o.supports_complex], dtypes=[torch.complex64])
    def test_complex(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Also skip fused, since our fused kernels do not support complex
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        for optim_input in all_optim_inputs:
            # Last param is intentionally real to test that we can mix real and complex
            complex_params = [
                torch.randn(10, 5, device=device, dtype=dtype, requires_grad=True),
                torch.randn(10, device=device, dtype=dtype, requires_grad=True),
                torch.randn(
                    10, 5, device=device, dtype=torch.float32, requires_grad=True
                ),
            ]
            real_params = [
                (
                    torch.view_as_real(param).detach().clone().requires_grad_()
                    if param.is_complex()
                    else param.detach().clone().requires_grad_()
                )
                for param in complex_params
            ]

            complex_optimizer = optim_cls(complex_params, **optim_input.kwargs)
            real_optimizer = optim_cls(real_params, **optim_input.kwargs)
            real_steps = []
            complex_steps = []
            grads_losses = []

            def real_closure():
                for param in real_params:
                    grad = torch.randn_like(param)
                    param.grad = grad
                    real_steps.append(param.detach().clone())
                    grads_losses.append(grad.clone())
                loss = torch.randn(1)
                grads_losses.append(loss.clone())
                return loss

            def complex_closure():
                for param in complex_params:
                    if torch.is_complex(param):
                        grad = torch.view_as_complex(grads_losses.pop(0))
                        complex_steps.append(torch.view_as_real_copy(param.detach()))
                    else:
                        grad = grads_losses.pop(0)
                        complex_steps.append(param.detach().clone())
                    param.grad = grad
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

            # Final Parameters should be the same
            complex_params_asreal = [
                torch.view_as_real(param) if param.is_complex() else param
                for param in complex_params
            ]
            self.assertEqual(real_params, complex_params_asreal)

            # All intermediate steps should also be the same
            # also checks steps taken within for example a line search
            self.assertEqual(complex_steps, real_steps)

    @skipMPS
    @xfailIfS390X
    @optims([o for o in optim_db if o.supports_complex], dtypes=[torch.complex64])
    def test_complex_2d(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        # Also skip fused, since our fused kernels do not support complex
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable", "fused")
        )
        for optim_input in all_optim_inputs:
            if optim_info.step_requires_closure:
                # Why? The way we implement complex is by turning complex params into view_as_real
                # alternatives. For example, an size (M,N) tensor will become (M,N,2). In this test,
                # we break apart a tensor into its real and imaginary parts, which would be 2x(M,N).
                # For other pointwise optimizers, this distinction is trivial, but for LBFGS where
                # there are reductions across all parameters (and all the grads get flattened into
                # one long Tensor), this ordering matters. Why? Reductions are not deterministic
                # because addition between floating point numbers is not associative, i.e.,
                # a + b + c != a + c + b. Thus, we add a seed here to control the discrepancy that
                # will happen with LBFGS. Note that in test_complex above, there is no need for a seed
                # nor for increased tolerance, because results should be bitwise equivalent.
                torch.manual_seed(2024)

            a1 = torch.randn(2, device=device, dtype=dtype, requires_grad=True)
            a1_real = a1.real.detach().clone()
            a1_imag = a1.imag.detach().clone()
            a1_real.requires_grad_()
            a1_imag.requires_grad_()
            optim1 = optim_cls([a1], **optim_input.kwargs)
            optim2 = optim_cls([a1_real, a1_imag], **optim_input.kwargs)

            a1_reals = TensorTracker()
            a1_imags = TensorTracker()
            a1_grad_reals = TensorTracker()
            a1_grad_imags = TensorTracker()
            losses = TensorTracker()

            def closure1():
                optim1.zero_grad()
                loss = rosenbrock(a1).abs()
                loss.backward()

                # Track clones to best test accuracy
                a1_reals.add(a1.real)
                a1_imags.add(a1.imag)
                a1_grad_reals.add(a1.grad.real)
                a1_grad_imags.add(a1.grad.imag)

                losses.add(loss)

                return loss

            def closure2():
                optim2.zero_grad()
                a1_reals.pop_check_set(a1_real, self)
                a1_imags.pop_check_set(a1_imag, self)
                a2 = torch.complex(a1_real, a1_imag)
                loss = rosenbrock(a2).abs()
                losses.pop_check_set(loss, self)
                loss.backward()
                a1_grad_reals.pop_check_set(a1_real.grad, self)
                a1_grad_imags.pop_check_set(a1_imag.grad, self)
                return loss

            for _ in range(3):
                if optim_info.step_requires_closure:
                    # LBFGS, for example, requires closure and calls it internally
                    optim1.step(closure1)
                    optim2.step(closure2)
                else:
                    closure1()
                    closure2()
                    optim1.step()
                    optim2.step()

                self.assertEqual(a1.real, a1_real)
                self.assertEqual(a1.imag, a1_imag)

            self.assertTrue(a1_reals.all_popped())
            self.assertTrue(a1_imags.all_popped())
            self.assertTrue(a1_grad_reals.all_popped())
            self.assertTrue(a1_grad_imags.all_popped())
            self.assertTrue(losses.all_popped())

    def _compare_between(
        self, inputs, models, optimizers, assert_eq_kwargs=None, assert_step_dtype=None
    ):
        # why 7? iteration 7 is where we start to see differences for RAdam
        # params interacting with the small eps value, because that's right
        # after rho_t becomes greater than 5 in step 6.
        if assert_eq_kwargs is None:
            assert_eq_kwargs = {}
        kIterations = 7
        tracker = TensorTracker(assert_eq_kwargs)
        for i in range(kIterations):
            state, updated_params = [], []
            if not isinstance(inputs, list):
                inputs = [inputs, inputs]
            for input, model, optimizer in zip(inputs, models, optimizers):
                optimizer.zero_grad()

                if i == 3:
                    # Freeze a layer to test if the step of this layer in 'fused' or 'foreach'
                    # is same as the step in 'forloop'.
                    model[2].requires_grad_(False)
                if i == 5:
                    # Unfreeze the layer after 2 iters.
                    model[2].requires_grad_(True)

                # Test that step behaves as expected (a no-op) when grads are set to None
                if i != 2:
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                optimizer.step()
                state.append(optimizer.state)
                updated_params.append(model.parameters())

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                tracker.add(og_p)
                tracker.pop_check_set(new_p, self)

                # check that optimizer states are the same
                og_p_state = og_state[og_p]
                new_p_state = new_state[new_p]
                if assert_step_dtype is not None:
                    if torch.is_tensor(og_p_state.get("step", None)):
                        self.assertEqual(og_p_state["step"].dtype, assert_step_dtype)
                    if torch.is_tensor(new_p_state.get("step", None)):
                        self.assertEqual(new_p_state["step"].dtype, assert_step_dtype)
                for k in og_p_state:
                    tracker.add(og_p_state[k])
                    tracker.pop_check_set(new_p_state[k], self)

            self.assertTrue(tracker.all_popped())

    def _test_derived_optimizers(
        self,
        device,
        dtype,
        optim_info,
        flag,
        reduced_precision=False,
        assert_step_dtype=None,
    ):
        """
        Given a flag 'fused' or 'foreach', test for parity of optimizer state
        and updated parameters between when the flag is set to True and False
        for provided optimizer configurations.
        """
        assert flag in ("foreach", "fused")
        assert_eq_kwargs = {} if not reduced_precision else FP16_REDUCED_PRECISION

        optim_inputs = optim_info.optim_inputs_func(device=device, dtype=dtype)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            models, optimizers = [], []
            kwargs = deepcopy(optim_input.kwargs)
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
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
                empty_param = torch.empty(
                    (), device=device, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                optimizer = optim_cls(params, **kwargs)
                models.append(model)
                optimizers.append(optimizer)

            self._compare_between(
                input, models, optimizers, assert_eq_kwargs, assert_step_dtype
            )

    @skipMPS  # MPS doesn't support torch.float64, see https://github.com/pytorch/pytorch/issues/115350
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_foreach_matches_forloop(self, device, dtype, optim_info):
        self._test_derived_optimizers(device, dtype, optim_info, "foreach")

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("impl", ["foreach", "fused"])
    @optims(
        [
            optim
            for optim in optim_db
            if "foreach" in optim.supported_impls or "fused" in optim.supported_impls
        ]
    )
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
            return unittest.skip(
                f"foreach not supported for {optim_info.optim_cls.__name__}"
            )
        elif impl == "fused" and "cuda" not in optim_info.supports_fused_on:
            return unittest.skip(
                f"fused not supported for {optim_info.optim_cls.__name__} on cuda"
            )

        params = [
            torch.rand(2, 3, dtype=torch.float64, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device="cuda:0", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float64, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float32, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.float16, device="cuda:1", requires_grad=True),
            torch.rand(2, 3, dtype=torch.bfloat16, device="cuda:1", requires_grad=True),
            torch.randint(
                1024, (2, 3), dtype=torch.int64, device="cuda:1", requires_grad=False
            ),
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
            if kwargs.get("capturable", False) and _get_device_type(device) == "cpu":
                # capturable is not supported on CPU
                continue
            for use_impl in (False, True):
                kwargs[impl] = use_impl
                params_clone = []
                for p in params:
                    p_clone = p.detach().clone()
                    if p.requires_grad:
                        p_clone.requires_grad = True
                        p_clone.grad = p.grad.detach().clone()
                        params_clone.append(p_clone)

                optimizer = optim_cls(params_clone, **kwargs)
                for _ in range(kIterations):
                    optimizer.step()

                state.append(optimizer.state)
                updated_params.append(params_clone)

            og_state, new_state = state
            for og_p, new_p in zip(updated_params[0], updated_params[1]):
                # Increasing the tolerance as we are collating lots of ops together for optimizers and
                # the designated tolerances are for single op only.
                single_rtol, single_atol = torch.testing._comparison.get_tolerances(
                    new_p.dtype, rtol=None, atol=None
                )
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
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float64],
    )
    def test_set_default_dtype_works_with_foreach(self, device, dtype, optim_info):
        # https://github.com/pytorch/pytorch/issues/110940
        # We coerce step to always be float32 unless the
        # default dtype is higher prec float64
        old_default_dtype = torch.get_default_dtype()
        for default_dtype in [torch.float64, torch.float16]:
            try:
                torch.set_default_dtype(default_dtype)
                self._test_derived_optimizers(
                    device,
                    dtype,
                    optim_info,
                    "foreach",
                    reduced_precision=default_dtype == torch.float16,
                    assert_step_dtype=(
                        torch.float64
                        if default_dtype == torch.float64
                        else torch.float32
                    ),
                )
            finally:
                torch.set_default_dtype(old_default_dtype)

    @onlyCUDA
    @largeTensorTest("72GB", "cuda")
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    def test_foreach_large_tensor(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, foreach=True, **optim_input.kwargs)
            optimizer.step()

    @onlyCUDA
    @optims(
        [optim for optim in optim_db if "foreach" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_peak_memory_foreach(self, device, dtype, optim_info):
        nparams = 10
        optim_inputs = optim_info.optim_inputs_func(device=device)
        optim_cls = optim_info.optim_cls
        for optim_input in optim_inputs:
            kwargs = deepcopy(optim_input.kwargs)
            max_mems = []
            for flag_value in (False, True):
                kwargs["foreach"] = flag_value
                # The 16 * 8 = 128 is critical here! Our CUDACachingAllocator allocates in blocks
                # of 512, meaning any tensor that occupies <512 bytes of memory will allocate a
                # whole 512 bytes anyway. We use 128 (cuz datasize would be 4 bytes) so that param
                # is size 512 exactly, making our later calculations for intermediate_size easy.
                param = torch.rand(16, 8, device=device, dtype=dtype)
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

            # Check the param group directly to handle if the compiler set capturable
            if optimizer.param_groups[0].get(
                "capturable", False
            ) or optim_cls.__name__ in ["Adadelta", "ASGD", "RAdam"]:
                # with capturable in Adam(W), we have 2 extra intermediates for the bias_corrections
                # with Adadelta, we have 2 extra for (acc_delta + eps) and (square_avg + eps)
                # ASGD allocates axs, 2x mus, 2x etas, and grads at the same time
                nintermediates = 3
                if optim_cls.__name__ == "NAdam":
                    # with capturable in NAdam, we have 3 extra intermediates for the
                    # bias_correction, mus, and mu_nexts
                    if TEST_WITH_TORCHDYNAMO:
                        # With dynamo, the eager/FX backend appears to hold memory longer than
                        # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                        nintermediates = 8
                    else:
                        nintermediates = 5

                if optim_cls.__name__ == "RAdam":
                    # RAdam has four intermediates with capturable
                    # num, unrect_step_size, buffer, grouped_grads
                    if TEST_WITH_TORCHDYNAMO:
                        # With dynamo, the eager/FX backend appears to hold memory than
                        # vanilla eager: https://github.com/pytorch/pytorch/issues/125511
                        nintermediates = 6
                    else:
                        nintermediates = 4

            elif optim_cls.__name__ in ["NAdam", "Adagrad", "RMSprop", "Adafactor"]:
                # NAdam uses two intermediates at the same time (grads & exp_avg_sq_sqrt)
                # Adagrad uses std and grads at the same time
                # RMSprop uses avg and grads
                # Adafactor uses row/col var and its mean
                nintermediates = 2

                if optim_cls.__name__ == "Adafactor" and kwargs.get("maximize", False):
                    # When maximize is True, Adafactor also tracks device_grad
                    nintermediates = 3

            # Dynamo ST uses less mem than eager in the case of Adam/Adagrad/Nadam/RAdam
            # which makes the foreach memory check fail
            if TEST_WITH_TORCHDYNAMO:
                st_max_mem += 6000

            expected_max_mem = st_max_mem + intermediate_size * nintermediates
            # hipcc currently can't generate efficient code for the small buffer optimization
            # code path (see Note [small buffer optimization] for details), thus we always
            # dynamically allocate the tensor metadata for ROCM. Adjusting the expected max
            # memory usage to account for this.
            if TEST_WITH_ROCM:
                expected_max_mem *= 1.02

            self.assertLessEqual(mt_max_mem, expected_max_mem)

    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_matches_forloop(self, device, dtype, optim_info):
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        if _get_device_type(device) == "mps" and dtype not in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ):
            self.skipTest(
                "MPS supports only torch.float16, torch.float32 and torch.bfloat16"
            )
        self._test_derived_optimizers(device, dtype, optim_info, "fused")

    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=(torch.float32,),
    )
    def test_fused_error_on_params_on_meta(self, device, dtype, optim_info):
        if _get_device_type(device) not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )

        with torch.device("meta"):
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 3),
                torch.nn.Sigmoid(),
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid(),
            ).to(dtype)

        optimizer = optim_info.optim_cls(model.parameters(), fused=True)
        with torch.device("meta"):
            for p in model.parameters():
                p.grad = torch.rand_like(p)

        with self.assertRaisesRegex(
            RuntimeError,
            "`fused=True` requires all the params to be floating point Tensors",
        ):
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        model.to_empty(device=device)
        for p in model.parameters():
            p.grad = torch.rand_like(p)
        optimizer.step()

    @onlyNativeDeviceTypes
    @largeTensorTest("64GB")
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float16],
    )
    def test_fused_large_tensor(self, device, dtype, optim_info):
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        for optim_input in optim_inputs:
            params = [torch.ones(2**32, device=device, dtype=dtype)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
            optimizer.step()

    @onlyCUDA
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_fused_does_not_step_if_foundinf(self, device, dtype, optim_info):
        if device not in optim_info.supports_fused_on:
            self.skipTest(
                f"{device} is not supported for fused on {optim_info.optim_cls.__name__}"
            )
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device=device)
        num_params = 5
        for optim_input in optim_inputs:
            for no_grad_scale in (False, True):
                params = [
                    torch.ones((1,), device=device, dtype=dtype)
                    for _ in range(num_params)
                ]
                params_c = [param.detach().clone() for param in params]
                for p in params:
                    p.grad = torch.ones_like(p)
                optimizer = optim_cls(params, fused=True, **optim_input.kwargs)
                optimizer.grad_scale = (
                    None
                    if no_grad_scale
                    else torch.ones((1,), dtype=dtype, device=device)
                )
                optimizer.found_inf = torch.ones((), dtype=dtype, device=device)
                optimizer.step()
                for p in params:
                    if "step" in optimizer.state[p]:
                        self.assertEqual(
                            torch.zeros((), dtype=dtype, device=device),
                            optimizer.state[p]["step"],
                        )
                self.assertEqual(params, params_c)

    @parametrize("impl", ["fused", "capturable"])
    @optims(
        [optim for optim in optim_db if "fused" in optim.supported_impls],
        dtypes=[torch.float32],
    )
    def test_cpu_load_state_dict(self, device, dtype, impl, optim_info):
        # NOTE: This SIMULATES a fused/capturable optimizer with state moved to CPU, issue 103256
        # How do we get there? Users typically create CUDA models on fused optimizers and then
        # store checkpoints on CPU as CUDA memory is limited with torch.load(...map_location="cpu").
        # Since this is a unit test, it is more expedient to simulate what the state_dict
        # would look like, which is basically CPU tensors with fused/capturable flag = True.
        optim_cls = optim_info.optim_cls
        opt_name = optim_cls.__name__
        if opt_name in ("SGD", "Adagrad") and impl == "capturable":
            # Capturable SGD/Adagrad does not exist
            self.skipTest("SGD does not currently support capturable")
        if _get_device_type(device) == "cpu":
            self.skipTest("Test is only for non-cpu devices")
        elif (
            impl == "fused"
            and _get_device_type(device) not in optim_info.supports_fused_on
        ):
            self.skipTest(f"{device} is not supported for fused on {opt_name}")
        elif impl == "capturable" and _get_device_type(device) == "mps":
            self.skipTest("MPS does not support capturable")

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
            param_device = param.detach().clone().to(device=device)
            optimizer_device = optim_cls([param_device], **optim_input.kwargs)
            optimizer_device.load_state_dict(optim_state_dict_cpu)
            optimizer_device.zero_grad()
            param_device.grad = torch.rand_like(param_device)
            optimizer_device.step()

    @optims(optim_db, dtypes=[torch.float32])
    def test_param_groups_weight_decay(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input in all_optim_inputs:
            weight_kwargs = optim_input.kwargs
            bias_kwargs = deepcopy(optim_input.kwargs)
            bias_kwargs["weight_decay"] = 0.0

            weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
            bias = Parameter(torch.randn((10), device=device, dtype=dtype))
            input = torch.randn(5, device=device, dtype=dtype)

            optimizer = optim_cls(
                [
                    dict(params=[weight], **weight_kwargs),
                    dict(params=[bias], **bias_kwargs),
                ]
            )

            loss = (weight.mv(input) + bias).pow(2).sum()
            initial_value = loss.item()
            for _ in range(20):
                optimizer.zero_grad()
                loss = (weight.mv(input) + bias).pow(2).sum()
                loss.backward()
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
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
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
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
                [
                    dict(params=[weight, bias], **optim_input.kwargs),
                    dict(params=[irrelevant]),
                ],
                **outer_kwargs,
            )

            loss = (weight.mv(input) + bias).pow(2).sum()
            initial_value = loss.item()
            for _ in range(20):
                optimizer.zero_grad()
                loss = (weight.mv(input) + bias).pow(2).sum()
                loss.backward()
                irrelevant.grad = torch.rand_like(irrelevant)
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
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
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        params = [
            torch.randn(2, 3, requires_grad=False, device=device, dtype=dtype)
            for _ in range(2)
        ]
        old_params = [p.detach().clone() for p in params]

        def closure():
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            optimizer.step(closure)

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_is_noop_for_zero_grads(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        param = torch.randn((5, 1), device=device, dtype=dtype, requires_grad=True)
        old_param = param.detach().clone()

        def closure():
            return torch.tensor([1], device=device, dtype=dtype)

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs

            # params will decay even if grads are empty if weight_decay != 0,
            # and capturable doesn't work for CPU tensors
            if kwargs.get("weight_decay", 0) != 0:
                continue

            # AdamW params will be updated regardless of grads due to lr, so make lr smaller
            if optim_cls.__name__ == "AdamW":
                kwargs["lr"] = (
                    torch.tensor(1e-5)
                    if isinstance(kwargs.get("lr", 1e-5), torch.Tensor)
                    else 1e-5
                )

            if kwargs.get("differentiable", False):
                params = [param.clone()]
            else:
                params = [param]

            optimizer = optim_cls(params, **kwargs)
            if optim_info.only_supports_sparse_grads:
                # Intentionally construct a multidimensional empty v for the sparse grad
                # Single dim v passes the test while multidim correctly repros the issue
                # https://github.com/pytorch/pytorch/issues/82486
                i = torch.empty((1, 0), device=device, dtype=dtype)
                v = torch.empty((0, 1), device=device, dtype=dtype)
                params[0].grad = torch.sparse_coo_tensor(
                    i, v, (5, 1), device=device, dtype=dtype
                )
            else:
                params[0].grad = torch.zeros_like(params[0])
            optimizer.step(closure)
            self.assertEqual(old_param, params[0])

    @optims(optim_db, dtypes=[torch.float32])
    def test_optimizer_can_be_printed(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        params = [
            Parameter(torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype))
            for _ in range(2)
        ]
        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)
            optimizer.__repr__()

    @parametrize("is_named_optim0", [True, False])
    @parametrize("is_named_optim1", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_deterministic(
        self, device, dtype, optim_info, is_named_optim0, is_named_optim1
    ):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        weight = Parameter(
            torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)
        )
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        params = [weight, bias]

        def make_named_param(param, is_named):
            if not is_named:
                return param
            return [(f"name{i}", p) for i, p in enumerate(param)]

        def without_param_names(state_dict):
            new_state_dict = deepcopy(state_dict)
            for pg in new_state_dict["param_groups"]:
                pg.pop("param_names", None)
            return new_state_dict

        def fwd_bwd(optim, w, b, i):
            optim.zero_grad()
            loss = (w.mv(i) + b).pow(2).sum()
            loss.backward()
            if optim_info.only_supports_sparse_grads:
                if w.grad is not None:
                    w.grad = w.grad.to_sparse()
                if b.grad is not None:
                    b.grad = b.grad.to_sparse()
            return loss

        for optim_input in all_optim_inputs:
            params_in = make_named_param(params, is_named=is_named_optim0)
            optimizer = optim_cls(params_in, **optim_input.kwargs)
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # Prime the optimizer
            for _ in range(10):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

            # Clone the weights and construct a new optimizer for them
            with torch.no_grad():
                weight_c = Parameter(weight.clone())
                bias_c = Parameter(bias.clone())
            params_c = make_named_param([weight_c, bias_c], is_named=is_named_optim1)
            optimizer_c = optim_cls(params_c, **optim_input.kwargs)
            closure_c = functools.partial(fwd_bwd, optimizer_c, weight_c, bias_c, input)

            # Load the state dict from the original optimizer into the new one
            optimizer_c.load_state_dict(deepcopy(optimizer.state_dict()))

            # Run both optimizers in parallel
            for _ in range(10):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                    optimizer_c.step(closure_c)
                else:
                    closure()
                    closure_c()
                    optimizer.step()
                    optimizer_c.step()

                self.assertEqual(weight, weight_c)
                self.assertEqual(bias, bias_c)

            # Make sure state dict is deterministic with equal (not identical) parameters
            # Param names are optional and not needed to be the consistent.
            self.assertEqual(
                without_param_names(optimizer.state_dict()),
                without_param_names(optimizer_c.state_dict()),
            )

            # Make sure repeated parameters have identical representation (see #36831)
            optimizer_c.param_groups.extend(optimizer_c.param_groups)
            self.assertEqual(
                without_param_names(optimizer.state_dict())["param_groups"][-1],
                without_param_names(optimizer_c.state_dict())["param_groups"][-1],
            )

    @optims(optim_db, dtypes=[torch.float32])
    def test_can_load_older_state_dict(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
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
                if optim_info.step_requires_closure:
                    optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
                else:
                    fwd_bwd(optimizer, model, input)
                    optimizer.step()

            # old_state_dict has all new flags del'd
            old_state_dict = deepcopy(optimizer.state_dict())
            old_state_dict_pg = old_state_dict["param_groups"]
            for group in old_state_dict_pg:
                for flag in optim_info.not_og_supported_flags:
                    if flag in group:
                        del group[flag]

            optimizer.load_state_dict(old_state_dict)

            # Make sure we can still step
            if optim_info.step_requires_closure:
                optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
            else:
                fwd_bwd(optimizer, model, input)
                optimizer.step()

    @parametrize("is_named_optim0", [True, False])
    @parametrize("is_named_optim1", [True, False])
    @optims(
        [o for o in optim_db if not o.only_supports_sparse_grads],
        dtypes=[torch.float32],
    )
    def test_can_load_from_to_named_state_dict(
        self, device, dtype, optim_info, is_named_optim0, is_named_optim1
    ):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        for optim_input in all_optim_inputs:
            torch.manual_seed(1)
            model = torch.nn.Sequential(
                torch.nn.Conv2d(4, 2, 1, stride=2),
                torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1),
            )
            model.to(dtype=dtype, device=device)
            input = torch.rand(1, 4, 16, 16, device=device, dtype=dtype)

            def fwd_bwd(optim, mod, i):
                optim.zero_grad()
                loss = mod(i).sum()
                loss.backward()
                return loss

            # test for parameters, named_parameters, and 2 groups:
            params_to_optimizer = (
                model.named_parameters() if is_named_optim0 else model.parameters()
            )
            optimizer = optim_cls(params_to_optimizer, **optim_input.kwargs)

            for _ in range(3):
                if optim_info.step_requires_closure:
                    optimizer.step(functools.partial(fwd_bwd, optimizer, model, input))
                else:
                    fwd_bwd(optimizer, model, input)
                    optimizer.step()

            # old_state_dict has all new flags del'd
            old_state_dict = deepcopy(optimizer.state_dict())

            params_to_optimizer2 = (
                model.named_parameters() if is_named_optim1 else model.parameters()
            )
            optimizer2 = optim_cls(params_to_optimizer2, **optim_input.kwargs)
            optimizer2.load_state_dict(old_state_dict)

            # Make sure we can still step
            if optim_info.step_requires_closure:
                optimizer2.step(functools.partial(fwd_bwd, optimizer2, model, input))
            else:
                fwd_bwd(optimizer2, model, input)
                optimizer2.step()

            # Make sure that param_names are preserved when provided to at least one of the optimizers
            if is_named_optim0 or is_named_optim1:
                self.assertEqual(
                    optimizer2.state_dict()["param_groups"][0]["param_names"],
                    ["0.weight", "0.bias", "1.weight", "1.bias"],
                )

    @parametrize("is_named_optim", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_save_load_equality_with_weights_only(
        self, device, dtype, optim_info, is_named_optim
    ):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        weight = Parameter(
            torch.randn(2, 3, requires_grad=True, device=device, dtype=dtype)
        )
        bias = Parameter(torch.randn(2, requires_grad=True, device=device, dtype=dtype))
        input = torch.randn(3, requires_grad=True, device=device, dtype=dtype)
        params = [weight, bias]

        def make_named_param(param, is_named):
            if not is_named:
                return param
            return [(f"name{i}", p) for i, p in enumerate(param)]

        def fwd_bwd(optim, w, b, i):
            optim.zero_grad()
            loss = (w.mv(i) + b).pow(2).sum()
            loss.backward()
            if optim_info.only_supports_sparse_grads:
                weight.grad = weight.grad.to_sparse()
                bias.grad = bias.grad.to_sparse()
            return loss

        for optim_input in all_optim_inputs:
            params_in = make_named_param(params, is_named=is_named_optim)
            optimizer = optim_cls(params_in, **optim_input.kwargs)
            closure = functools.partial(fwd_bwd, optimizer, weight, bias, input)

            # Prime the optimizer
            for _ in range(3):
                optimizer.step(closure)

            sd = optimizer.state_dict()

            # === Check saved/loaded state_dict are the same (including weights_only load). ===
            with tempfile.TemporaryFile() as f:
                torch.save(sd, f)
                f.seek(0)
                sd_copy = torch.load(f)
                self.assertEqual(sd_copy, sd)
                del sd_copy
                f.seek(0)
                sd_copy_wo = torch.load(f, weights_only=True)
                self.assertEqual(sd_copy_wo, sd)

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_nontensor_step(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )
        params = [
            Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)
        ]
        for p in params:
            p.grad = torch.rand_like(p)
            if optim_info.only_supports_sparse_grads:
                # For this test, we naively convert the Tensor layout, which we know does
                # NOT represent the expected use case for optims like SparseAdam!
                p.grad = p.grad.to_sparse()

        # Needed for second order optims like LBFGS
        closure_loss = torch.rand(1, device=device, dtype=dtype)

        def closure():
            return closure_loss if optim_info.step_requires_closure else None

        for optim_input in all_optim_inputs:
            kwargs = optim_input.kwargs
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
        cpu_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            "cpu", dtype, optim_info, skip=("differentiable",)
        )

        # Needed for second order optims like LBFGS
        closure_loss = torch.rand(1, device=device, dtype=dtype)

        def closure():
            return closure_loss if optim_info.step_requires_closure else None

        for optim_input in cpu_optim_inputs:
            if (
                "fused" in optim_input.kwargs
                and "cuda" not in optim_info.supports_fused_on
            ):
                self.skipTest(
                    f"cuda is not supported for fused on {optim_cls.__name__}"
                )
            params = [
                Parameter(torch.randn(2, 3, device="cpu", dtype=dtype))
                for _ in range(2)
            ]
            for p in params:
                p.grad = torch.randn_like(p)
                if optim_info.only_supports_sparse_grads:
                    # For this test, we naively convert the Tensor layout, which we know does
                    # NOT represent the expected use case for optims like SparseAdam!
                    p.grad = p.grad.to_sparse()

            optimizer = optim_cls(params, **optim_input.kwargs)

            for _ in range(3):
                optimizer.step(closure)

            with torch.no_grad():
                params_cuda = [p.to(device="cuda") for p in params]
                for i, p in enumerate(params_cuda):
                    p.grad = params[i].grad.to(device="cuda")
            optimizer_cuda = optim_cls(params_cuda, **optim_input.kwargs)

            state_dict_cpu = deepcopy(optimizer.state_dict())
            state_dict_cuda = deepcopy(optimizer.state_dict())
            optimizer_cuda.load_state_dict(state_dict_cuda)

            # Make sure state_dict_cuda isn't modified by merely calling load_state_dict
            self.assertEqual(state_dict_cpu, state_dict_cuda)

            # Make sure that device of state['step'] is still CPU _unless_ torch.compile() added a capturable!
            capturable = state_dict_cpu["param_groups"][0].get("capturable", False)
            fused = state_dict_cpu["param_groups"][0].get("fused", False)
            new_state_dict = optimizer_cuda.state_dict()
            for state_cpu, state_cuda in zip(
                state_dict_cpu["state"].values(), new_state_dict["state"].values()
            ):
                if "step" in state_cpu and torch.is_tensor(state_cpu["step"]):
                    self.assertEqual(
                        state_cuda["step"].device.type,
                        "cuda" if capturable or fused else "cpu",
                    )

            for _ in range(5):
                optimizer.step(closure)
                optimizer_cuda.step(closure)
                self.assertEqual(params, params_cuda)
                self.assertEqual(optimizer.state_dict(), optimizer_cuda.state_dict())

    @staticmethod
    def _state_dict_pre_hook(optimizer: Optimizer) -> None:
        optimizer.state["test"] = 1

    @staticmethod
    def _state_dict_post_hook(
        optimizer: Optimizer, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "test" in state_dict["state"]:
            state_dict["state"].pop("test")
            state_dict["ran_state_dict_pre_hook"] = True
        else:
            state_dict["ran_state_dict_pre_hook"] = False
        return state_dict

    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_pre_hook(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            optim.register_state_dict_pre_hook(self.__class__._state_dict_pre_hook)
            state_dict = optim.state_dict()
            self.assertEqual(state_dict["state"]["test"], 1)

    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_post_hook(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            optim.register_state_dict_post_hook(self.__class__._state_dict_post_hook)
            state_dict = optim.state_dict()
            self.assertFalse(state_dict["ran_state_dict_pre_hook"])

    @optims(optim_db, dtypes=[torch.float32])
    def test_state_dict_pre_post_hook(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            optim.register_state_dict_pre_hook(self.__class__._state_dict_pre_hook)
            optim.register_state_dict_post_hook(self.__class__._state_dict_post_hook)
            state_dict = optim.state_dict()
            self.assertFalse("test" in state_dict["state"])
            self.assertTrue(state_dict["ran_state_dict_pre_hook"])

    @staticmethod
    def _load_state_dict_pre_hook1(
        optimizer: Optimizer, state_dict: Dict[str, Any]
    ) -> None:
        state_dict["param_groups"][0]["lr"] = 0.002

    @staticmethod
    def _load_state_dict_pre_hook2(
        optimizer: Optimizer, state_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        # The typical use case for returning a state dict is to drastically modify the state dict.
        # I will simulate by simply making a deep copy and ensuring that my_state_dict still gets used
        my_state_dict = deepcopy(state_dict)
        my_state_dict["param_groups"][0]["lr"] = 0.003
        return my_state_dict

    @staticmethod
    def _load_state_dict_post_hook(optimizer: Optimizer) -> None:
        optimizer.state["ran_load_state_dict_pre_hook2"] = (
            optimizer.param_groups[0]["lr"] == 0.003
        )
        optimizer.state["ran_load_state_dict_post_hook"] = True

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_pre_hook_and_prepend(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)
            state_dict = optim.state_dict()

            # usually one would have a new optim instance here, but it's all the same here
            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook1
            )
            optim.load_state_dict(state_dict)
            self.assertEqual(optim.param_groups[0]["lr"], 0.002)

            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook2, prepend=True
            )
            optim.load_state_dict(state_dict)
            # If prepend were False would be 0.003 but since prepend is True, the other hook overrides
            self.assertEqual(optim.param_groups[0]["lr"], 0.002)

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_post_hook(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)

            optim.register_load_state_dict_post_hook(
                self.__class__._load_state_dict_post_hook
            )
            optim.load_state_dict(optim.state_dict())
            self.assertFalse(optim.state["ran_load_state_dict_pre_hook2"])
            self.assertTrue(optim.state["ran_load_state_dict_post_hook"])

    @optims(optim_db, dtypes=[torch.float32])
    def test_load_state_dict_pre_post_hook(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            param = torch.rand(2, 3, device=device, dtype=dtype, requires_grad=True)
            optim = optim_cls([param], **optim_input.kwargs)

            optim.register_load_state_dict_pre_hook(
                self.__class__._load_state_dict_pre_hook2
            )
            optim.register_load_state_dict_post_hook(
                self.__class__._load_state_dict_post_hook
            )
            optim.load_state_dict(optim.state_dict())
            self.assertTrue(optim.state["ran_load_state_dict_pre_hook2"])
            self.assertTrue(optim.state["ran_load_state_dict_post_hook"])

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_post_hook(self, device, dtype, optim_info):
        def post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data += 2

        params = [torch.tensor([1, 1], device=device, dtype=dtype)]

        def dummy_closure():
            return 1

        closure = dummy_closure if optim_info.step_requires_closure else None

        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            data = 2
            hook_handle = optim.register_step_post_hook(post_hook)

            optim.step(closure)
            optim.step(closure)
            # check if post hooks were registered
            self.assertEqual(data, 6)

            # remove handles, take step and verify that hook is no longer registered
            hook_handle.remove()

            optim.step(closure)
            self.assertEqual(data, 6)

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_pre_hook(self, device, dtype, optim_info):
        def pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data += 2

        params = [torch.tensor([1, 1], device=device, dtype=dtype)]

        def dummy_closure():
            return 1

        closure = dummy_closure if optim_info.step_requires_closure else None

        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            data = 5
            hook_handle = optim.register_step_pre_hook(pre_hook)

            optim.step(closure)
            optim.step(closure)
            # check if pre hooks were registered
            self.assertEqual(data, 9)

            # remove handles, take step and verify that hook is no longer registered
            hook_handle.remove()

            optim.step(closure)
            self.assertEqual(data, 9)

    @optims(optim_db, dtypes=[torch.float32])
    def test_step_all_hooks(self, device, dtype, optim_info):
        def global_pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(0)

        def global_post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(5)

        def local_pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(1)

        def local_post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data.append(2)

        params = [torch.tensor([1, 1], device=device, dtype=dtype)]

        def dummy_closure():
            return 1

        closure = dummy_closure if optim_info.step_requires_closure else None

        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info
        )
        for optim_input in all_optim_inputs:
            optim = optim_info.optim_cls(params, **optim_input.kwargs)
            optim2 = SGD(params)
            data = []

            # register global hooks to both optimizers
            global_pre_handle = register_optimizer_step_pre_hook(global_pre_hook)
            global_post_handle = register_optimizer_step_post_hook(global_post_hook)

            # register local hooks
            first_pre_handle = optim.register_step_pre_hook(local_pre_hook)
            first_post_handle = optim.register_step_post_hook(local_post_hook)
            second_pre_handle = optim2.register_step_pre_hook(local_pre_hook)
            second_post_handle = optim2.register_step_post_hook(local_post_hook)

            optim.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5])
            optim2.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5])
            optim.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

            # remove all hooks
            global_pre_handle.remove()
            global_post_handle.remove()
            first_pre_handle.remove()
            first_post_handle.remove()
            second_pre_handle.remove()
            second_post_handle.remove()

            optim.step(closure)
            optim2.step(closure)
            self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

    @optims(optim_db, dtypes=[torch.float32])
    def test_deepcopy_copies_all_public_attrs(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls

        # Skip differentiable testing for now, see https://github.com/pytorch/pytorch/issues/116490
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",)
        )

        params = [
            Parameter(torch.randn(2, 3, device=device, dtype=dtype)) for _ in range(2)
        ]
        for p in params:
            p.grad = torch.rand_like(p)
            if optim_info.only_supports_sparse_grads:
                # For this test, we naively convert the Tensor layout, which we know does
                # NOT represent the expected use case for optims like SparseAdam!
                p.grad = p.grad.to_sparse()

        # Needed for second order optims like LBFGS
        def closure():
            return 1 if optim_info.step_requires_closure else None

        def getPublicAttrs(obj):
            return {k for k in obj.__dict__ if not k.startswith("_")}

        for optim_input in all_optim_inputs:
            optimizer = optim_cls(params, **optim_input.kwargs)

            # Make some state
            for _ in range(3):
                if optim_info.step_requires_closure:
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()

            self.assertEqual(
                getPublicAttrs(optimizer), getPublicAttrs(deepcopy(optimizer))
            )

    @optims(
        [optim for optim in optim_db if optim.step_requires_closure],
        dtypes=[torch.float32],
    )
    def test_second_order_optims_return_consistent_types(
        self, device, dtype, optim_info
    ):
        # Motivated by #7586
        optim_cls = optim_info.optim_cls
        params = [
            torch.randn(10, 5, device=device, dtype=dtype),
            torch.randn(10, device=device, dtype=dtype),
        ]

        def closure():
            return torch.tensor([10], device=device, dtype=dtype)

        for optim_input in optim_info.optim_inputs_func(device=device):
            # Currently, the only second order optim is LBFGS, so we just go ahead and modify
            # "tolerance_grad", but this may not scale if we add second order optims in the future
            kwargs = optim_input.kwargs
            kwargs["tolerance_grad"] = math.inf
            optim_inf = optim_cls(params, **kwargs)
            kwargs["tolerance_grad"] = -math.inf
            optim_neg_inf = optim_cls(params, **kwargs)

            res1 = optim_inf.step(closure)
            res2 = optim_neg_inf.step(closure)
            self.assertEqual(type(res1), type(res2))

    @onlyCUDA
    @optims(
        [
            optim
            for optim in optim_db
            if "cpu" in optim.supports_fused_on and "cuda" in optim.supports_fused_on
        ],
        dtypes=floating_types_and(
            torch.bfloat16,
            torch.float16,
        ),
    )
    def test_fused_cpu_matches_cuda(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        optim_inputs = optim_info.optim_inputs_func(device="cpu")
        for optim_input in optim_inputs:
            inpts, models, optimizers = [], [], []
            for dev in ("cpu", "cuda"):
                kwargs = optim_input.kwargs
                kwargs["fused"] = True
                inpt = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype, device=dev
                ).reshape(3, 2)

                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=dtype, device=dev)

                # foreach/fused optimizers should be tested with a
                # zero_size tensor as its last param.
                # ref: https://github.com/pytorch/pytorch/issues/100701
                empty_param = torch.empty(
                    (), device=dev, dtype=dtype, requires_grad=True
                )
                empty_param.grad = torch.rand_like(empty_param)
                params = list(model.parameters()) + [empty_param]

                optimizer = optim_cls(params, **kwargs)
                inpts.append(inpt)
                models.append(model)
                optimizers.append(optimizer)
        self._compare_between(inpts, models, optimizers)

    @onlyCUDA
    @optims(
        [
            o
            for o in optim_db
            if ("foreach" in o.supported_impls and o.optim_cls.__name__ != "Adafactor")
        ],
        dtypes=[torch.float32],
    )
    def test_defaults_changed_to_foreach(self, device, dtype, optim_info):
        # Test that the default implementations for optimizers are changed to foreach
        # except Adafactor, which defaults to the single tensor impl for memory efficiency.
        optim_cls = optim_info.optim_cls
        model = torch.nn.Linear(5, 5)
        model.to(dtype=dtype, device=device)
        inpt = torch.rand(2, 5, dtype=dtype, device=device)

        import inspect

        module = inspect.getmodule(optim_cls)

        for optim_input in optim_info.optim_inputs_func(device=device):
            optim = optim_cls(model.parameters(), **optim_input.kwargs)
            optim.zero_grad()
            output = model(inpt)
            loss = output.sum()
            loss.backward()
            with patch.object(
                module, f"_multi_tensor_{optim_cls.__name__.lower()}"
            ) as mocked_foreach_impl:
                optim.step()
                self.assertTrue(mocked_foreach_impl.called)

    @optims(optim_db, dtypes=[torch.float32])
    def test_non_empty_state(self, device, dtype, optim_info):
        # There are internal tests that check that the state is not empty
        optim_cls = optim_info.optim_cls
        model = torch.nn.Linear(5, 5)
        model.to(dtype=dtype, device=device)
        inpt = torch.rand(2, 5, dtype=dtype, device=device)

        for optim_input in optim_info.optim_inputs_func(device=device):
            optim = optim_cls(model.parameters(), **optim_input.kwargs)
            optim.zero_grad()
            output = model(inpt)
            loss = output.sum()
            loss.backward()

            if optim_info.only_supports_sparse_grads:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad.to_sparse()

            if optim_info.step_requires_closure:
                optim.step(lambda: 1.0)
            else:
                optim.step()

            for state in optim.state.values():
                self.assertGreater(len(state), 0)


instantiate_device_type_tests(TestOptimRenewed, globals(), allow_mps=True)


if __name__ == "__main__":
    run_tests()
