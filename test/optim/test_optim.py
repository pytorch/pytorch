# Owner(s): ["module: optimizer"]

import math
import unittest
import functools
import itertools
from copy import deepcopy

import torch
from torch.nn import Parameter
from torch.optim import (
    Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD, SparseAdam, Optimizer
)
from torch.optim.lr_scheduler import (
    StepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    ReduceLROnPlateau,
    PolynomialLR,
)
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    gradcheck,
    skipIfRocm,
    skipIfTorchDynamo
)

from torch._dynamo import disable as disable_dynamo

from torch.testing._internal.common_cuda import TEST_CUDA
from typing import Dict, Any, Tuple
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from unittest.mock import patch

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


def rosenbrock(tensor):
    assert tensor.size() == torch.Size([2]), f"Requires tensor with 2 scalars but got {tensor.size()}"
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def drosenbrock(tensor):
    assert tensor.size() == torch.Size([2]), f"Requires tensor with 2 scalars but got {tensor.size()}"
    x, y = tensor
    return torch.tensor((-400 * x * (y - x**2) - 2 * (1 - x), 200 * (y - x**2)))

@skipIfTorchDynamo("This is a TEMPORARY stopgap, see https://github.com/pytorch/pytorch/issues/103322")
class TestOptim(TestCase):
    exact_dtype = True

    def _test_rosenbrock_sparse(
        self,
        constructor,
        scheduler_constructors=None,
        sparse_only=False,
        maximize=False,
        multi_tensor=False
    ):
        if scheduler_constructors is None:
            scheduler_constructors = []
        # For rosenbrock tests, it is mandated that the param is a tensor with 2 numbers
        if multi_tensor:
            params_t = [torch.tensor([1.5, 1.5]), torch.tensor([1.5, 1.5], dtype=torch.float64)]
        else:
            params_t = [torch.tensor([1.5, 1.5])]

        params = [Parameter(param_t) for param_t in params_t]
        optimizer = constructor(params)
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        if not sparse_only:
            params_c = [Parameter(param_t.clone()) for param_t in params_t]
            optimizer_c = constructor(params_c)

        solution = torch.tensor([1, 1])
        with torch.no_grad():
            initial_dist = sum([param.dist(solution) for param in params])

        def get_grad(param, sparse_grad):
            grad = drosenbrock(param)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor

            # Depending on w, provide only the x or y gradient
            if sparse_grad:
                if w:
                    i = torch.LongTensor([[0, 0]])
                    x = grad[0]
                    v = torch.tensor([x / 4.0, x - x / 4.0])
                else:
                    i = torch.LongTensor([[1, 1]])
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

            grads_out = [get_grad(param, sparse_grad) for param in params]
            with torch.no_grad():
                params[0].grad = grads_out[0]
                if multi_tensor:
                    params[1].grad = grads_out[1].to(dtype=torch.float64)
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(rosenbrock(params[0]))
                else:
                    scheduler.step()
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                # Tolerance is increased due to floating point error from different
                # code path for dense case: x v.s. x - x / 4.0 + x / 4.0
                self.assertEqual(params, params_c, atol=5e-6, rtol=5e-6)

        if not maximize:
            self.assertLessEqual(
                sum([param.dist(solution) for param in params]),
                initial_dist
            )
        else:
            self.assertGreaterEqual(
                sum([rosenbrock(param) for param in params]),
                sum([rosenbrock(param_t) for param_t in params_t]),
            )

    def _test_basic_cases_template(
        self,
        weight_tensor,
        bias_tensor,
        input_tensor,
        constructor,
        scheduler_constructors,
        constructor_accepts_maximize=True,
        constructor_accepts_foreach=False,
    ):
        maximize_options = {False, constructor_accepts_maximize}
        foreach_options = {False, constructor_accepts_foreach}

        four_arg_constructor = constructor
        if constructor_accepts_maximize and constructor_accepts_foreach:
            pass
        elif constructor_accepts_maximize:

            def four_arg_constructor(weight, bias, maximize, foreach):
                self.assertFalse(foreach)
                return constructor(weight, bias, maximize)

        elif constructor_accepts_foreach:

            def four_arg_constructor(weight, bias, maximize, foreach):
                self.assertFalse(maximize)
                return constructor(weight, bias, foreach)

        else:

            def four_arg_constructor(weight, bias, maximize, foreach):
                self.assertFalse(maximize or foreach)
                return constructor(weight, bias)

        for maximize, foreach in itertools.product(maximize_options, foreach_options):
            with torch.no_grad():
                weight = Parameter(weight_tensor.clone().detach())
                bias = Parameter(bias_tensor.clone().detach())
                input = input_tensor.clone().detach().requires_grad_()
            optimizer = four_arg_constructor(weight, bias, maximize, foreach)
            schedulers = []
            for scheduler_constructor in scheduler_constructors:
                schedulers.append(scheduler_constructor(optimizer))

            # to check if the optimizer can be printed as a string
            optimizer.__repr__()

            def fn():
                optimizer.zero_grad()
                y = weight.mv(input)
                if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                    y = y.cuda(bias.get_device())
                loss = (y + bias).pow(2).sum()
                loss.backward()
                return loss

            initial_value = fn().item()
            for _ in range(200):
                optimizer.step(fn)
                for scheduler in schedulers:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        val_loss = fn()
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
            if maximize:
                self.assertGreater(fn().item(), initial_value)
            else:
                self.assertLess(fn().item(), initial_value)

    # Note: disable dynamo on this function
    # This allows us to continue running actual logic of the optimizer
    # tests in dynamo without tracing this test code which has a lot of unsupported
    # behavior
    @disable_dynamo(recursive=False)
    def _test_state_dict(self, weight, bias, input, constructor, atol=None, rtol=None):
        weight = Parameter(weight)
        bias = Parameter(bias)
        with torch.no_grad():
            input = input.clone().detach().requires_grad_()

        # Note: Disable dynamo on this function
        # This avoids a bug where input_cuda is not detected in the environment
        # because it currently is not defined in the local environmet. Unable to repro
        # anywhere else however and this is test code that we don't need to spend
        # time getting dynamo to trace unless the issue repros in real models.
        @disable_dynamo(recursive=False)
        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for _i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        with torch.no_grad():
            weight_c = Parameter(weight.clone().detach())
            bias_c = Parameter(bias.clone().detach())
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizers in parallel
        for _ in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict is deterministic with equal but not identical parameters
        self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())
        # Make sure repeated parameters have identical representation in state dict
        optimizer_c.param_groups.extend(optimizer_c.param_groups)
        self.assertEqual(
            optimizer.state_dict()["param_groups"][-1],
            optimizer_c.state_dict()["param_groups"][-1],
        )

        # Make sure that optimizers that support maximize can load older models
        old_state_dict = deepcopy(optimizer.state_dict())
        state_dict_no_maximize = deepcopy(optimizer.state_dict())
        if "maximize" in state_dict_no_maximize["param_groups"][0]:
            for group in state_dict_no_maximize["param_groups"]:
                del group["maximize"]
            optimizer.load_state_dict(state_dict_no_maximize)
            # Make sure we can still step
            optimizer.step()
            # Undo these changes before proceeding!
            optimizer.load_state_dict(old_state_dict)
        # Make sure that optimizers that support foreach can load older models
        state_dict_no_foreach = deepcopy(optimizer.state_dict())
        if "foreach" in state_dict_no_foreach["param_groups"][0]:
            for group in state_dict_no_foreach["param_groups"]:
                del group["foreach"]
            optimizer.load_state_dict(state_dict_no_foreach)
            # Make sure we can still step
            optimizer.step()
            # Undo these changes before proceeding!
            optimizer.load_state_dict(old_state_dict)

        # Make sure that loading optimizers with step not wrapped in tensor can work
        state_dict = optimizer.state_dict()
        if "step" in state_dict["state"][0] and torch.is_tensor(
            state_dict["state"][0]["step"]
        ):
            for state in state_dict["state"].values():
                state["step"] = state["step"].item()
            optimizer.load_state_dict(state_dict)
            optimizer.step()

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        with torch.no_grad():
            input_cuda = input.clone().detach().to(dtype=torch.float32, device="cuda")
            weight_cuda = Parameter(
                weight.clone().detach().to(dtype=torch.float32, device="cuda")
            )
            bias_cuda = Parameter(
                bias.clone().detach().to(dtype=torch.float32, device="cuda")
            )
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state_dict_c isn't modified by merely calling load_state_dict
        self.assertEqual(state_dict, state_dict_c)

        # Make sure that device of state['step'] is still CPU
        new_state_dict = optimizer_cuda.state_dict()
        if "step" in state_dict["state"][0] and torch.is_tensor(
            state_dict["state"][0]["step"]
        ):
            for state in new_state_dict["state"].values():
                self.assertEqual(state["step"].device.type, "cpu")

        for _ in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda, atol=atol, rtol=rtol)

        # validate deepcopy() copies all public attributes
        def getPublicAttr(obj):
            return {k for k in obj.__dict__ if not k.startswith("_")}

        self.assertEqual(getPublicAttr(optimizer), getPublicAttr(deepcopy(optimizer)))

    def _test_basic_cases(
        self,
        constructor,
        scheduler_constructors=None,
        ignore_multidevice=False,
        constructor_accepts_maximize=False,
        constructor_accepts_foreach=False,
        atol=None,
        rtol=None,
    ):
        if scheduler_constructors is None:
            scheduler_constructors = []

        def make_two_arg_constructor(
            constructor, maximize: bool, foreach: bool
        ):
            if constructor_accepts_maximize and constructor_accepts_foreach:
                return lambda weight, bias: constructor(weight, bias, maximize, foreach)
            if constructor_accepts_maximize:
                return lambda weight, bias: constructor(weight, bias, maximize)
            if constructor_accepts_foreach:
                return lambda weight, bias: constructor(weight, bias, foreach)
            return constructor

        for maximize, foreach in itertools.product(
            {False, constructor_accepts_maximize},
            {False, constructor_accepts_foreach},
        ):
            self._test_state_dict(
                torch.randn(10, 5),
                torch.randn(10),
                torch.randn(5),
                make_two_arg_constructor(constructor, maximize, foreach),
                atol=atol,
                rtol=rtol,
            )
        self._test_basic_cases_template(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor,
            scheduler_constructors,
            constructor_accepts_maximize,
            constructor_accepts_foreach,
        )

    def _test_complex_optimizer(self, optimizer_constructor):
        complex_param = torch.randn(5, 5, dtype=torch.complex64, requires_grad=True)
        real_param = torch.view_as_real(complex_param).detach().clone().requires_grad_()
        complex_opt = optimizer_constructor(complex_param)
        real_opt = optimizer_constructor(real_param)

        for _ in range(3):
            complex_param.grad = torch.randn_like(complex_param)
            real_param.grad = torch.view_as_real(complex_param.grad)
            complex_opt.step()
            real_opt.step()

            self.assertEqual(torch.view_as_real(complex_param), real_param)

    def _test_complex_2d(self, optimizer_constructor):
        a1 = torch.randn(2, dtype=torch.complex64, requires_grad=True)
        a1_real = a1.real.clone().detach()
        a1_imag = a1.imag.clone().detach()
        a1_real.requires_grad_()
        a1_imag.requires_grad_()
        optim1 = optimizer_constructor([a1])
        optim2 = optimizer_constructor([a1_real, a1_imag])

        for _ in range(10):
            optim1.zero_grad()
            optim2.zero_grad()
            a2 = torch.complex(a1_real, a1_imag)
            rosenbrock(a1).abs().backward()
            rosenbrock(a2).abs().backward()

            self.assertEqual(a1.grad.real, a1_real.grad)
            self.assertEqual(a1.grad.imag, a1_imag.grad)

            optim1.step()
            optim2.step()
            self.assertEqual(a1.real, a1_real)
            self.assertEqual(a1.imag, a1_imag)

    def _build_params_dict(self, weight, bias, **kwargs):
        return [{"params": [weight]}, dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def test_sgd(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                self._build_params_dict_single(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                self._build_params_dict_single(weight, bias, lr=1e-2),
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[lambda opt: StepLR(opt, gamma=0.9, step_size=10)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[
                lambda opt: LinearLR(
                    opt, start_factor=0.4, end_factor=0.8, total_iters=4
                )
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[lambda opt: ConstantLR(opt, factor=0.4, total_iters=4)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[lambda opt: PolynomialLR(opt, power=0.9, total_iters=4)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: LinearLR(
                    opt, start_factor=0.4, end_factor=0.6, total_iters=4
                ),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            [
                lambda opt: StepLR(opt, gamma=0.99, step_size=10),
                lambda opt: ExponentialLR(opt, gamma=0.99),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias],
                lr=1e-3,
                momentum=0.5,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias],
                lr=1e-3,
                momentum=0.5,
                weight_decay=1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: SGD(
                [weight, bias],
                nesterov=True,
                lr=1e-3,
                momentum=0.5,
                weight_decay=1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )


    def test_sgd_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: SGD(params, lr=4.8e-3, foreach=foreach),
                multi_tensor=foreach,
            )
            self._test_rosenbrock_sparse(
                lambda params: SGD(params, lr=0.0048, foreach=foreach),
                scheduler_constructors=[lambda opt: StepLR(opt, gamma=0.99999, step_size=300)],
                multi_tensor=foreach,
            )

    def test_sgd_complex(self):
        for foreach in (False, True):
            self._test_complex_optimizer(
                lambda param: SGD([param], lr=0.001, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: SGD([param], lr=0.001, momentum=1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: SGD(
                    [param], lr=0.001, momentum=1, weight_decay=1, foreach=foreach
                )
            )
            self._test_complex_optimizer(
                lambda param: SGD(
                    [param],
                    lr=0.001,
                    nesterov=True,
                    momentum=1,
                    weight_decay=1,
                    foreach=foreach,
                )
            )
            self._test_complex_optimizer(
                lambda param: SGD(
                    [param],
                    lr=0.001,
                    momentum=1,
                    dampening=0.5,
                    weight_decay=1,
                    foreach=foreach,
                )
            )


    def test_adam(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                [weight, bias],
                lr=1e-3,
                amsgrad=True,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                amsgrad=True,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            [lambda opt: ExponentialLR(opt, gamma=0.9)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            [lambda opt: LinearLR(opt, start_factor=0.4, total_iters=4)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            [lambda opt: ConstantLR(opt, factor=0.4, total_iters=4)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                [weight, bias],
                lr=1e-3,
                amsgrad=True,
                maximize=maximize,
                foreach=foreach,
            ),
            [
                lambda opt: ConstantLR(opt, factor=0.4, total_iters=4),
                lambda opt: ExponentialLR(opt, gamma=0.9),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                [weight, bias],
                lr=1e-3,
                amsgrad=True,
                maximize=maximize,
                foreach=foreach,
            ),
            [
                lambda opt: ExponentialLR(opt, gamma=0.9),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                amsgrad=True,
                maximize=maximize,
                foreach=foreach,
            ),
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )

        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            [lambda opt: PolynomialLR(opt, total_iters=4, power=0.9)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=torch.tensor(1e-3),
                maximize=maximize,
                foreach=False,  # foreach for lr tensors tested in multi configs
            ),
            [lambda opt: PolynomialLR(opt, total_iters=4, power=0.9)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )

    def test_adam_complex(self):
        for foreach in (False, True):
            self._test_complex_2d(functools.partial(Adam, foreach=foreach))
            self._test_complex_2d(functools.partial(Adam, foreach=foreach, amsgrad=True))
            self._test_complex_2d(functools.partial(Adam, foreach=foreach, weight_decay=0.2))
            self._test_complex_2d(functools.partial(Adam, foreach=foreach, weight_decay=0.2, amsgrad=True))
        self._test_complex_2d(Adam)
        self._test_complex_2d(functools.partial(
            Adam, lr=torch.tensor(.001), weight_decay=0.2, amsgrad=True,
        ))

    def test_adamw(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: AdamW(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: AdamW(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: AdamW(
                [weight, bias],
                lr=1e-3,
                weight_decay=1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: AdamW(
                [weight, bias],
                lr=1e-3,
                weight_decay=1,
                amsgrad=True,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: AdamW(
                [weight, bias],
                lr=torch.tensor(1e-3),
                weight_decay=1,
                amsgrad=True,
                maximize=maximize,
                foreach=False,  # foreach for lr tensors tested in multi configs
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )


    def test_adamw_complex(self):
        self._test_complex_2d(AdamW)
        self._test_complex_2d(functools.partial(
            AdamW, lr=torch.tensor(.001), weight_decay=0.2, amsgrad=True,
        ))
        for foreach in (False, True):
            self._test_complex_2d(functools.partial(AdamW, foreach=foreach))
            self._test_complex_2d(functools.partial(AdamW, foreach=foreach, amsgrad=True))
            self._test_complex_2d(functools.partial(AdamW, foreach=foreach, weight_decay=0.2))
            self._test_complex_2d(functools.partial(AdamW, foreach=foreach, weight_decay=0.2, amsgrad=True))

    def test_sparse_adam(self):
        self._test_rosenbrock_sparse(
            lambda params: SparseAdam(params, lr=4e-2), [], True
        )
        self._test_rosenbrock_sparse(
            lambda params: SparseAdam(params, lr=4e-2, maximize=True),
            scheduler_constructors=[],
            sparse_only=True,
            maximize=True,
        )

    # ROCm precision is too low to pass this test
    def test_adadelta(self):
        # Handles https://github.com/pytorch/pytorch/issues/69698
        self.rel_tol = 4e-3
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adadelta(
                [weight, bias], maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adadelta(
                self._build_params_dict(weight, bias, rho=0.95),
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adadelta(
                self._build_params_dict(weight, bias, rho=0.95),
                maximize=maximize,
                foreach=foreach,
            ),
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adadelta(
                [weight, bias], weight_decay=1, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )

    def test_adadelta_complex(self):
        # Handles https://github.com/pytorch/pytorch/issues/110606
        self.rel_tol = 2e-2
        for foreach in (False, True):
            self._test_complex_optimizer(lambda weight: Adadelta([weight], foreach=foreach))
            self._test_complex_optimizer(lambda weight: Adadelta([weight], rho=0.95, foreach=foreach))
            self._test_complex_optimizer(
                lambda weight: Adadelta([weight], rho=0.95, weight_decay=1, foreach=foreach)
            )

    def test_nadam(self):
        self._test_basic_cases(
            lambda weight, bias, foreach: NAdam(
                self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: NAdam(
                [weight, bias], lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: NAdam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                momentum_decay=6e-3,
                foreach=foreach,
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: NAdam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                momentum_decay=6e-3,
                foreach=foreach,
            ),
            [lambda opt: ExponentialLR(opt, gamma=0.9)],
            constructor_accepts_foreach=True,
        )
        # NAdamW tests
        self._test_basic_cases(
            lambda weight, bias, foreach: NAdam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                momentum_decay=6e-3,
                decoupled_weight_decay=True,
                foreach=foreach,
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: NAdam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                momentum_decay=6e-3,
                decoupled_weight_decay=True,
                foreach=foreach,
            ),
            [lambda opt: ExponentialLR(opt, gamma=0.9)],
            constructor_accepts_foreach=True,
        )


    def test_nadam_complex(self):
        for foreach in (False, True):
            self._test_complex_optimizer(
                lambda param: NAdam([param], lr=1e-1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: NAdam(
                    [param],
                    lr=1e-1,
                    weight_decay=0.01,
                    foreach=foreach,
                )
            )
            self._test_complex_optimizer(
                lambda param: NAdam(
                    [param],
                    lr=1e-1,
                    momentum_decay=0.01,
                    foreach=foreach,
                )
            )

    def test_adagrad(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adagrad(
                [weight, bias], lr=1e-1, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adagrad(
                [weight, bias],
                lr=1e-1,
                initial_accumulator_value=0.1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1,
                maximize=maximize,
                foreach=foreach,
            ),
            [lambda opt: ReduceLROnPlateau(opt)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1,
                maximize=maximize,
                foreach=foreach,
            ),
            [
                lambda opt: ReduceLROnPlateau(opt),
                lambda opt: ExponentialLR(opt, gamma=0.99),
            ],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )


    def test_adagrad_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: Adagrad(params, lr=1e-1, foreach=foreach),
                multi_tensor=foreach,
            )
            self._test_rosenbrock_sparse(
                lambda params: Adagrad(params, lr=0.1, foreach=foreach),
                scheduler_constructors=[
                    lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),
                    lambda opt: ReduceLROnPlateau(opt, threshold=1e-4),
                ],
                multi_tensor=foreach,
            )

    def test_adagrad_complex(self):
        for foreach in (False, True):
            self._test_complex_optimizer(
                lambda param: Adagrad([param], lr=1e-1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: Adagrad(
                    [param],
                    lr=1e-1,
                    initial_accumulator_value=0.1,
                    foreach=foreach,
                )
            )

    def test_adamax(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adamax(
                [weight, bias], lr=1e-1, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adamax(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: Adamax(
                [weight, bias],
                lr=1e-1,
                weight_decay=1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_complex_2d(Adamax)
        self._test_complex_2d(functools.partial(Adamax, foreach=True))


    def test_radam(self):
        self._test_basic_cases(
            lambda weight, bias, foreach: RAdam(
                [weight, bias], lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: RAdam(
                self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: RAdam(
                [weight, bias], lr=1e-3, weight_decay=0.1, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: RAdam(
                [weight, bias], lr=1e-3, foreach=foreach
            ),
            [
                lambda opt: ExponentialLR(opt, gamma=0.9),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_foreach=True,
        )
        # RAdamW tests
        self._test_basic_cases(
            lambda weight, bias, foreach: RAdam(
                [weight, bias], lr=1e-3, weight_decay=0.1, decoupled_weight_decay=True, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: RAdam(
                [weight, bias], lr=1e-3, weight_decay=0.1, decoupled_weight_decay=True, foreach=foreach
            ),
            [
                lambda opt: ExponentialLR(opt, gamma=0.9),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_foreach=True,
        )


    def test_radam_complex(self):
        for foreach in (False, True):
            self._test_complex_optimizer(
                lambda param: RAdam([param], lr=1e-1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: RAdam(
                    [param],
                    lr=1e-1,
                    weight_decay=0.01,
                    foreach=foreach,
                )
            )
            self._test_complex_optimizer(
                lambda param: RAdam(
                    [param],
                    lr=1e-1,
                    weight_decay=0.01,
                    decoupled_weight_decay=True,
                    foreach=foreach,
                )
            )

    def test_rmsprop(self):
        for foreach in (False, True):
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: RMSprop(
                    [weight, bias], lr=1e-2, maximize=maximize, foreach=foreach
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: RMSprop(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: RMSprop(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2,
                    centered=True,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: RMSprop(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2,
                    centered=True,
                    momentum=0.1,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: RMSprop(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2,
                    momentum=0.1,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: RMSprop(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2,
                    momentum=0.1,
                    weight_decay=1,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_complex_2d(lambda param: RMSprop(param, foreach=foreach))
            self._test_complex_2d(
                lambda param: RMSprop(param, centered=True, foreach=foreach)
            )
            self._test_complex_2d(
                lambda param: RMSprop(param, momentum=0.1, foreach=foreach)
            )
            self._test_complex_2d(
                lambda param: RMSprop(param, maximize=True, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: RMSprop([param], foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: RMSprop([param], centered=True, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: RMSprop([param], momentum=0.1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: RMSprop([param], maximize=True, foreach=foreach)
            )


    def test_asgd(self):
        for foreach in (False, True):
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: ASGD(
                    [weight, bias], lr=1e-3, t0=100, maximize=maximize, foreach=foreach
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: ASGD(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3,
                    t0=100,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: ASGD(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3,
                    weight_decay=1,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            # Ref: https://github.com/pytorch/pytorch/issues/84560
            # self._test_complex_2d(optimizer)
            self._test_complex_optimizer(
                lambda params: ASGD([params], foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda params: ASGD([params], maximize=True, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda params: ASGD(
                    [params], maximize=True, weight_decay=0.9, foreach=foreach
                )
            )
            self._test_complex_optimizer(
                lambda params: ASGD(
                    [params], maximize=False, weight_decay=0.9, foreach=foreach
                )
            )


    @skipIfRocm
    @skipIfTorchDynamo()
    def test_rprop(self):
        is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(
            0
        ) == (8, 6)
        for foreach in (False, True):
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: Rprop(
                    [weight, bias], lr=2e-4, maximize=maximize, foreach=foreach
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: Rprop(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=2e-4,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
                atol=4e-5 if is_cuda_sm86 else None,
                rtol=3e-5 if is_cuda_sm86 else None,
            )
            self._test_complex_2d(lambda param: Rprop(param, foreach=foreach))
            self._test_complex_optimizer(
                lambda param: Rprop([param], lr=0.001, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: Rprop(
                    [param], lr=0.001, maximize=True, foreach=foreach
                )
            )


    def test_lbfgs(self):
        self._test_basic_cases(
            lambda weight, bias: LBFGS([weight, bias]), ignore_multidevice=True
        )
        self._test_basic_cases(
            lambda weight, bias: LBFGS(
                [weight, bias], line_search_fn="strong_wolfe"
            ),
            ignore_multidevice=True,
        )

    def test_lbfgs_returns_consistent_type(self):
        params = [torch.randn(10, 5), torch.randn(10)]
        opt1 = LBFGS(params, 0.01, tolerance_grad=math.inf)
        opt2 = LBFGS(params, 0.01, tolerance_grad=-math.inf)

        def closure():
            return torch.tensor([10])

        res1 = opt1.step(closure)
        res2 = opt2.step(closure)
        self.assertEqual(type(res1), type(res2))


    def test_fused_optimizer_does_not_step_if_foundinf(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required.")

        from torch.optim import adam, adamw

        num_tensors = 5
        for functional_optim, amsgrad, no_grad_scale in itertools.product((adam.adam, adamw.adamw), (False, True), (False, True)):
            params, grads, exp_avgs, exp_avg_sqs = (
                [torch.ones((1,), device="cuda") for _ in range(num_tensors)] for _ in range(4))
            prev_params = [t.clone().detach() for t in params]
            max_exp_avg_sqs = [torch.ones((1,), device="cuda") for _ in range(num_tensors)] if amsgrad else []
            state_steps = [torch.ones((), dtype=torch.float32, device="cuda") for _ in range(num_tensors)]
            grad_scale = None if no_grad_scale else torch.ones((1,), dtype=torch.float32, device="cuda")
            found_inf = torch.ones((), dtype=torch.float32, device="cuda")

            functional_optim(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                foreach=False,
                capturable=False,
                fused=True,
                amsgrad=amsgrad,
                beta1=0.9,
                beta2=0.99,
                lr=1e-2,
                weight_decay=0.0,
                eps=1e-8,
                maximize=False,
                grad_scale=grad_scale,
                found_inf=found_inf,
            )

            self.assertEqual(
                state_steps,
                [
                    torch.ones((), dtype=torch.float32, device="cuda")
                    for _ in range(num_tensors)
                ],
            )
            self.assertEqual(params, prev_params)


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is required.")
    def test_fused_optimizer_load_state_dict(self):
        # NOTE: This SIMULATES a fused/capturable optimizer with state moved to CPU, issue 103256
        # How do we get there? Users typically create CUDA models on fused optimizers and then
        # store checkpoints on CPU as CUDA memory is limited with torch.load(...map_location="cpu").
        # Since this is a unit test, it is more expedient to simulate what the state_dict
        # would look like, which is basically CPU tensors with fused/capturable flag = True.
        for optimC, kwarg in itertools.product((Adam, AdamW), ("fused", "capturable")):
            input = torch.tensor([0.1, 0.2], dtype=torch.float32, device="cpu")
            optimizer = optimC([input])
            optimizer.zero_grad()
            input.grad = torch.rand_like(input)
            optimizer.step()
            optim_state_dict_cpu = deepcopy(optimizer.state_dict())
            optim_state_dict_cpu["param_groups"][0][kwarg] = True

            # load
            input_cuda = input.clone().detach().to(device="cuda")
            defaults = {kwarg: True}
            optimizer_cuda = optimC([input_cuda], **defaults)
            optimizer_cuda.load_state_dict(optim_state_dict_cpu)
            optimizer_cuda.zero_grad()
            input_cuda.grad = torch.rand_like(input_cuda)
            optimizer_cuda.step()


    @skipIfTorchDynamo()
    def test_post_hook(self):
        def post_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data += 2

        params = [torch.Tensor([1, 1])]
        opt = SGD(params, lr=0.001)
        data = 2
        hook_handle = opt.register_step_post_hook(post_hook)

        opt.step()
        opt.step()
        # check if pre hooks were registered
        self.assertEqual(data, 6)

        # remove handles, take step and verify that hook is no longer registered
        hook_handle.remove()

        opt.step()
        self.assertEqual(data, 6)

    @skipIfTorchDynamo()
    def test_pre_hook(self):
        def pre_hook(opt: Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
            nonlocal data
            data += 2

        params = [torch.Tensor([1, 1])]
        opt = SGD(params, lr=0.001)
        data = 5
        hook_handle = opt.register_step_pre_hook(pre_hook)

        opt.step()
        opt.step()
        # check if pre hooks were registered
        self.assertEqual(data, 9)

        # remove handles, take step and verify that hook is no longer registered
        hook_handle.remove()

        opt.step()
        self.assertEqual(data, 9)

    @skipIfTorchDynamo()
    def test_pre_and_post_hook(self):
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

        params = [torch.Tensor([1, 1])]
        opt1 = SGD(params, lr=0.001)
        opt2 = Adam(params, lr=0.01)
        data = []

        # register global hooks to both optimizers
        global_pre_handle = register_optimizer_step_pre_hook(global_pre_hook)
        global_post_handle = register_optimizer_step_post_hook(global_post_hook)

        # register local hooks
        first_pre_handle = opt1.register_step_pre_hook(local_pre_hook)
        first_post_handle = opt1.register_step_post_hook(local_post_hook)
        second_pre_handle = opt2.register_step_pre_hook(local_pre_hook)
        second_post_handle = opt2.register_step_post_hook(local_post_hook)

        opt1.step()
        self.assertListEqual(data, [0, 1, 2, 5])
        opt2.step()
        self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5])
        opt1.step()
        self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])

        # remove all hooks
        global_pre_handle.remove()
        global_post_handle.remove()
        first_pre_handle.remove()
        first_post_handle.remove()
        second_pre_handle.remove()
        second_post_handle.remove()

        opt1.step()
        opt2.step()
        self.assertListEqual(data, [0, 1, 2, 5, 0, 1, 2, 5, 0, 1, 2, 5])


    @staticmethod
    def _state_dict_pre_hook(optimizer: Optimizer) -> None:
        optimizer.state["test"] = 1

    @staticmethod
    def _state_dict_post_hook(optimizer: Optimizer, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "test" in state_dict["state"]:
            state_dict["state"].pop("test")
            state_dict["ran_state_dict_pre_hook"] = True
        else:
            state_dict["ran_state_dict_pre_hook"] = False
        return state_dict

    @staticmethod
    def _load_state_dict_pre_hook1(optimizer: Optimizer, state_dict: Dict[str, Any]) -> None:
        state_dict["param_groups"][0]["lr"] = 0.002

    @staticmethod
    def _load_state_dict_pre_hook2(optimizer: Optimizer, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        # The typical use case for returning a state dict is to drastically modify the state dict.
        # I will simulate by simply making a deep copy and ensuring that my_state_dict still gets used
        my_state_dict = deepcopy(state_dict)
        my_state_dict["param_groups"][0]["lr"] = 0.003
        return my_state_dict

    @staticmethod
    def _load_state_dict_post_hook(optimizer: Optimizer) -> None:
        optimizer.state["ran_load_state_dict_pre_hook2"] = optimizer.param_groups[0]["lr"] == 0.003
        optimizer.state["ran_load_state_dict_post_hook"] = True

    def test_state_dict_pre_hook(self):
        param = torch.rand(2, 3, requires_grad=True)
        param.grad = torch.rand(2, 3, requires_grad=True)
        opt = SGD([param], lr=0.001)
        opt.register_state_dict_pre_hook(self._state_dict_pre_hook)
        state_dict = opt.state_dict()
        self.assertEqual(state_dict["state"]["test"], 1)

    def test_state_dict_post_hook(self):
        param = torch.rand(2, 3, requires_grad=True)
        param.grad = torch.rand(2, 3, requires_grad=True)
        opt = SGD([param], lr=0.001)
        opt.register_state_dict_post_hook(self._state_dict_post_hook)
        state_dict = opt.state_dict()
        self.assertEqual(state_dict["ran_state_dict_pre_hook"], False)

    def test_state_dict_pre_post_hook(self):
        param = torch.rand(2, 3, requires_grad=True)
        param.grad = torch.rand(2, 3, requires_grad=True)
        opt = SGD([param], lr=0.001)
        opt.register_state_dict_pre_hook(self._state_dict_pre_hook)
        opt.register_state_dict_post_hook(self._state_dict_post_hook)
        state_dict = opt.state_dict()
        self.assertFalse("test" in state_dict["state"])
        self.assertEqual(state_dict["ran_state_dict_pre_hook"], True)

    def test_load_state_dict_pre_hook_and_prepend(self):
        param = torch.rand(2, 3, requires_grad=True)
        param.grad = torch.rand(2, 3, requires_grad=True)
        opt = SGD([param], lr=0.001)
        state_dict = opt.state_dict()

        # usually one would have a new opt instance here, but it's all the same here
        opt.register_load_state_dict_pre_hook(self._load_state_dict_pre_hook1)
        opt.load_state_dict(state_dict)
        self.assertEqual(opt.param_groups[0]["lr"], 0.002)

        opt.register_load_state_dict_pre_hook(self._load_state_dict_pre_hook2, prepend=True)
        opt.load_state_dict(state_dict)
        # If prepend were False would be 0.003 but since prepend is True, the other hook overrides
        self.assertEqual(opt.param_groups[0]["lr"], 0.002)

    def test_load_state_dict_post_hook(self):
        param = torch.rand(2, 3, requires_grad=True)
        param.grad = torch.rand(2, 3, requires_grad=True)
        opt = SGD([param], lr=0.001)

        opt.register_load_state_dict_post_hook(self._load_state_dict_post_hook)
        opt.load_state_dict(opt.state_dict())
        self.assertFalse(opt.state["ran_load_state_dict_pre_hook2"])
        self.assertTrue(opt.state["ran_load_state_dict_post_hook"])

    def test_load_state_dict_pre_post_hook(self):
        param = torch.rand(2, 3, requires_grad=True)
        param.grad = torch.rand(2, 3, requires_grad=True)
        opt = SGD([param], lr=0.001)

        opt.register_load_state_dict_pre_hook(self._load_state_dict_pre_hook2)
        opt.register_load_state_dict_post_hook(self._load_state_dict_post_hook)
        opt.load_state_dict(opt.state_dict())
        self.assertTrue(opt.state["ran_load_state_dict_pre_hook2"])
        self.assertTrue(opt.state["ran_load_state_dict_post_hook"])


def _diff_fn(p, grad, opt_differentiable_state, opt_class, kwargs, *ignored):
    # Ignored is the list of values in `opt_differentiable_state`, we do this
    # for `gradcheck` to correctly track the state tensors as function inputs
    # because otherwise it can't unpack the values in the `opt_differentiable_state`
    # dict
    p = p.clone()
    p.grad = grad
    opt_differentiable_state = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in opt_differentiable_state.items()
    }
    opt = opt_class([p], **kwargs)
    opt.state[p].update(opt_differentiable_state)
    opt.step()
    return (p,) + tuple(
        v
        for v in opt.state[p].values()
        if isinstance(v, torch.Tensor) and v.requires_grad
    )


@skipIfTorchDynamo("Differentiable optimizers not supported")
class TestDifferentiableOptimizer(TestCase):

    def test_sgd(self):
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        mbuff = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state = {"momentum_buffer": mbuff}
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                SGD,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )


    def test_adam(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adam,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )


    def test_rmsprop(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["step"] = 0
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["momentum_buffer"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        # This can cause issues with large values and nan due to sqrt ops
        state["grad_avg"] = 1e-2 * torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RMSprop,
                {
                    "lr": 0.9,
                    "maximize": True,
                    "momentum": 0.9,
                    "differentiable": True,
                    "centered": True,
                    "weight_decay": 0.1,
                },
                *state.values(),
            ),
        )


    def test_adadelta(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["square_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["acc_delta"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adadelta,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )


    def test_adagrad(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["sum"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adagrad,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )


    def test_adamax(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_inf"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Adamax,
                {"lr": 0.9, "weight_decay": 0.1, "differentiable": True},
                *state.values(),
            ),
        )


    @skipIfTorchDynamo("The inplace mu update fails with dynamo, "
                       "since this is only happening when differentiable is enabled, skipping for now")
    def test_asgd(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` `eta` & `mu` are not continuous variables (even though we define them as floats)
        # and so they shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["eta"] = torch.tensor(0.9, requires_grad=False, dtype=torch.float64)
        state["mu"] = torch.tensor(1.0, requires_grad=False, dtype=torch.float64)
        state["ax"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                ASGD,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_rprop(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["prev"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["step_size"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                Rprop,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

    def test_adamw(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["max_exp_avg_sq"] = torch.rand(
            10, requires_grad=True, dtype=torch.float64
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                AdamW,
                {"lr": 0.9, "differentiable": True, "amsgrad": True},
                *state.values(),
            ),
        )

    def test_nadam(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["mu_product"] = torch.tensor(1.0, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                NAdam,
                {"lr": 0.9, "decoupled_weight_decay": True, "differentiable": True},
                *state.values(),
            ),
        )

    def test_radam(self):
        state = {}
        p = torch.rand(10, requires_grad=True, dtype=torch.float64)
        grad = torch.rand(10, requires_grad=True, dtype=torch.float64)
        # `step` is not a continuous variable (even though we define it as a float)
        # and so it shouldn't require gradients.
        state["step"] = torch.tensor(10.0, requires_grad=False, dtype=torch.float64)
        state["exp_avg"] = torch.rand(10, requires_grad=True, dtype=torch.float64)
        state["exp_avg_sq"] = torch.rand(10, requires_grad=True, dtype=torch.float64)

        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )
        gradcheck(
            _diff_fn,
            (
                p,
                grad,
                state,
                RAdam,
                {"lr": 0.9, "weight_decay": 0.1, "decoupled_weight_decay": True, "differentiable": True},
                *state.values(),
            ),
        )

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_defaults_changed_to_foreach(self):
        from torch.optim import (adam, adamw, nadam, sgd, radam, rmsprop, rprop,
                                 asgd, adamax, adadelta, adagrad)
        multi_optims = ((Adam, adam, "_multi_tensor_adam"),
                        (AdamW, adamw, "_multi_tensor_adamw"),
                        (NAdam, nadam, "_multi_tensor_nadam"),
                        (SGD, sgd, "_multi_tensor_sgd"),
                        (RAdam, radam, "_multi_tensor_radam"),
                        (RMSprop, rmsprop, "_multi_tensor_rmsprop"),
                        (Rprop, rprop, "_multi_tensor_rprop"),
                        (ASGD, asgd, "_multi_tensor_asgd"),
                        (Adamax, adamax, "_multi_tensor_adamax"),
                        (Adadelta, adadelta, "_multi_tensor_adadelta"),
                        (Adagrad, adagrad, "_multi_tensor_adagrad"),)

        model = torch.nn.Linear(5, 5)
        model.to(dtype=torch.float64, device="cuda")
        input = torch.rand(2, 5, dtype=torch.float64, device="cuda")

        for opt, mod, func in multi_optims:
            defaults = {}
            if opt == SGD:
                defaults["lr"] = 1e-2
            optimizer = opt(model.parameters(), **defaults)
            optimizer.zero_grad()
            output = model(input)
            loss = output.sum()
            loss.backward()
            with patch.object(mod, func) as mocked_foreach_impl:
                optimizer.step()
                self.assertTrue(mocked_foreach_impl.called)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
