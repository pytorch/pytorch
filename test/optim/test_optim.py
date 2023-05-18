# Owner(s): ["module: optimizer"]

import math
import unittest
import functools
import itertools
from copy import deepcopy

import torch
import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adam, SGD, Optimizer
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
    run_tests,
    TEST_WITH_UBSAN,
    load_tests,
    gradcheck,
    skipIfRocm,
    skipIfTorchDynamo
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_device_type import largeTensorTest
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


class TestOptim(TestCase):
    exact_dtype = True

    def _test_rosenbrock_sparse(
        self,
        constructor,
        scheduler_constructors=None,
        sparse_only=False,
        maximize=False,
    ):
        if scheduler_constructors is None:
            scheduler_constructors = []
        # For rosenbrock tests, it is mandated that the param is a tensor with 2 numbers
        param_t = torch.tensor([1.5, 1.5])

        param = Parameter(param_t)
        optimizer = constructor([param])
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        if not sparse_only:
            param_c = Parameter(param_t.clone())
            optimizer_c = constructor([param_c])

        solution = torch.tensor([1, 1])
        with torch.no_grad():
            initial_dist = param.dist(solution)

        def eval(param, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(param)
            loss.backward()
            grad = drosenbrock(param)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.tensor([x / 4.0, x - x / 4.0])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.tensor([y - y / 4.0, y / 4.0])
            x = torch.sparse_coo_tensor(i, v, (2,), dtype=v.dtype)
            with torch.no_grad():
                if sparse_grad:
                    param.grad = x
                else:
                    param.grad = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, param, True, w))
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(rosenbrock(param))
                else:
                    scheduler.step()
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, param_c, False, w))
                self.assertEqual(param, param_c)

        if not maximize:
            self.assertLessEqual(param.dist(solution), initial_dist)
        else:
            self.assertGreaterEqual(rosenbrock(param), rosenbrock(param_t))

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

    def _test_state_dict(self, weight, bias, input, constructor, atol=None, rtol=None):
        weight = Parameter(weight)
        bias = Parameter(bias)
        with torch.no_grad():
            input = input.clone().detach().requires_grad_()

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
        # Run both optimizations in parallel
        for _ in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)
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

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        # Make sure that device of state['step'] is still CPU
        new_state_dict = optimizer_cuda.state_dict()
        if "step" in state_dict["state"][0] and torch.is_tensor(
            state_dict["state"][0]["step"]
        ):
            for state in new_state_dict["state"].values():
                self.assertEqual(state["step"].device.type, "cpu")

        for _i in range(20):
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
            lambda weight, bias, maximize, foreach: optim.SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
                self._build_params_dict_single(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
                self._build_params_dict_single(weight, bias, lr=1e-2),
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[lambda opt: StepLR(opt, gamma=0.9, step_size=10)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
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
            lambda weight, bias, maximize, foreach: optim.SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[lambda opt: ConstantLR(opt, factor=0.4, total_iters=4)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            scheduler_constructors=[lambda opt: PolynomialLR(opt, power=0.9, total_iters=4)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.SGD(
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
            lambda weight, bias, maximize, foreach: optim.SGD(
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
            lambda weight, bias, maximize, foreach: optim.SGD(
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
            lambda weight, bias, maximize, foreach: optim.SGD(
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
            lambda weight, bias, maximize, foreach: optim.SGD(
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
            lambda weight, bias, maximize, foreach: optim.SGD(
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
        with self.assertRaisesRegex(ValueError, "Invalid momentum value: -0.5"):
            optim.SGD(None, lr=1e-2, momentum=-0.5)

    def test_sgd_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: optim.SGD(params, lr=4.8e-3, foreach=foreach)
            )
            self._test_rosenbrock_sparse(
                lambda params: optim.SGD(params, lr=0.0048, foreach=foreach),
                scheduler_constructors=[lambda opt: StepLR(opt, gamma=0.99999, step_size=300)],
            )

    def test_sgd_complex(self):
        for foreach in (False, True):
            self._test_complex_optimizer(
                lambda param: optim.SGD([param], lr=0.001, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.SGD([param], lr=0.001, momentum=1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.SGD(
                    [param], lr=0.001, momentum=1, weight_decay=1, foreach=foreach
                )
            )
            self._test_complex_optimizer(
                lambda param: optim.SGD(
                    [param],
                    lr=0.001,
                    nesterov=True,
                    momentum=1,
                    weight_decay=1,
                    foreach=foreach,
                )
            )
            self._test_complex_optimizer(
                lambda param: optim.SGD(
                    [param],
                    lr=0.001,
                    momentum=1,
                    dampening=0.5,
                    weight_decay=1,
                    foreach=foreach,
                )
            )

    def _test_derived_optimizers_varying_tensors(self, optimizer_with_kwargs, kwarg):
        if not torch.cuda.is_available():
            return
        assert kwarg in ("foreach", "fused")

        # Specifically test that inputting params of different dtypes and devices
        # is handled equivalently on the foreach and fused implementations as the
        # single tensor implementations. We need multiple GPUs (vs just a CPU and
        # GPU) because fused adam only works on GPUs. (Thus we only run the tests
        # that call into this helper when TEST_MULTIGPU.)
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

        kIterations = 7 if kwarg == "foreach" else 1
        for optimizer_constructor, kwargs in optimizer_with_kwargs:
            res, state = [], []
            for enabled in (False, True):
                kwargs_clone = deepcopy(kwargs)
                kwargs_clone[kwarg] = enabled

                params_clone = []
                for p in params:
                    p_clone = p.clone().detach()
                    if p.requires_grad:
                        p_clone.requires_grad = True
                        p_clone.grad = p.grad.clone().detach()
                        params_clone.append(p_clone)

                optimizer = optimizer_constructor(params_clone, **kwargs_clone)
                for _ in range(kIterations):
                    optimizer.step()

                state.append(optimizer.state)
                res.append(params_clone)

            st_state = state[0]
            mt_state = state[1]
            for st_p, mt_p in zip(res[0], res[1]):
                self.assertEqual(st_p, mt_p)

                # check that optimizer states are the same
                st_p_state = st_state[st_p]
                mt_p_state = mt_state[mt_p]

                for k in st_p_state:
                    actual = mt_p_state[k]
                    # If `torch.optim.Adam` is `__init__`ed with either `fused=True` or `capturable=True`,
                    # `step` Tensor is 1D while usually it's 0D.
                    if (
                        k == "step"
                        and isinstance(actual, torch.Tensor)
                        and actual.ndim == 1
                    ):
                        actual = actual[0]
                    self.assertEqual(st_p_state[k], actual)

    def _test_derived_optimizers(self, optimizer_pairs_with_flags, flag):
        if not torch.cuda.is_available():
            return
        assert flag in ("foreach", "fused")

        # why 7? iteration 7 is where we start to see differences for RAdam
        # params interacting with the small eps value, because that's right
        # after rho_t becomes greater than 5 in step 6.
        kIterations = 7
        device = "cuda"
        for optimizer_constructor, params in optimizer_pairs_with_flags:
            res, state = [], []
            for flag_value in (False, True):
                input = torch.tensor(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float64, device=device
                ).reshape(3, 2)

                torch.manual_seed(1)
                model = torch.nn.Sequential(
                    torch.nn.Linear(2, 3),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(3, 1),
                    torch.nn.Sigmoid(),
                )
                model.to(dtype=torch.float64, device=device)
                params_with_flags = deepcopy(params)
                params_with_flags[flag] = flag_value

                # foreach/fused optimizers should be tested with a param_groups['params'] with
                # zero_size tensor as its last param.
                # ref: https://github.com/pytorch/pytorch/issues/100701
                empty_params = [torch.empty((), device=device, dtype=torch.float64)]

                optimizer = optimizer_constructor(
                    list(model.parameters()) + empty_params, **params_with_flags
                )

                for i in range(kIterations):
                    optimizer.zero_grad()
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                    # Test that step behaves as expected (a no-op) when grads are set to None
                    if i == 0:
                        optimizer.zero_grad(set_to_none=True)

                    optimizer.step()

                state.append(optimizer.state)
                res.append(model.parameters())

            st_state = state[0]
            mt_state = state[1]
            for st_p, mt_p in zip(res[0], res[1]):
                self.assertEqual(st_p, mt_p)

                # check that optimizer states are the same
                st_p_state = st_state[st_p]
                mt_p_state = mt_state[mt_p]

                for k in st_p_state:
                    self.assertEqual(st_p_state[k], mt_p_state[k])

    @property
    def _multi_tensor_optimizer_configs(self):
        return [
            (optim.Adam, dict(weight_decay=1.0, amsgrad=True, fused=False)),
            (optim.Adam, dict(weight_decay=1.0, amsgrad=False, fused=False)),
            (optim.Adam, dict(weight_decay=0.0, amsgrad=True, fused=False)),
            (optim.Adam, dict(weight_decay=0.0, amsgrad=False, fused=False)),
            (optim.AdamW, dict(weight_decay=1.0, amsgrad=True)),
            (optim.AdamW, dict(weight_decay=1.0, amsgrad=False)),
            (optim.AdamW, dict(weight_decay=0.0, amsgrad=True)),
            (optim.AdamW, dict(weight_decay=0.0, amsgrad=False)),
            (optim.NAdam, dict(weight_decay=0.0, momentum_decay=6e-3)),
            (optim.NAdam, dict(weight_decay=1.0, momentum_decay=6e-3)),
            (optim.NAdam, dict(weight_decay=0.0, momentum_decay=4e-3)),
            (optim.NAdam, dict(weight_decay=0.01, momentum_decay=4e-3)),
            (
                optim.SGD,
                dict(lr=0.2, momentum=1, dampening=0, weight_decay=1, nesterov=True),
            ),
            (
                optim.SGD,
                dict(lr=0.2, momentum=1, dampening=0.5, weight_decay=1, nesterov=False),
            ),
            (optim.RAdam, dict(weight_decay=0, eps=1e-6)),
            (optim.RAdam, dict(weight_decay=0)),
            (optim.RAdam, dict(weight_decay=1, eps=1e-6)),
            (optim.RAdam, dict(weight_decay=1)),
            (optim.RMSprop, dict(weight_decay=1, momentum=1, centered=True)),
            (optim.RMSprop, dict(weight_decay=1, momentum=0, centered=True)),
            (optim.RMSprop, dict(weight_decay=1, momentum=1, centered=False)),
            (optim.RMSprop, dict(weight_decay=0, momentum=1, centered=False)),
            (optim.Rprop, dict(lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50))),
            (optim.ASGD, dict(weight_decay=0)),
            (optim.ASGD, dict(weight_decay=1)),
            (optim.Adamax, dict(weight_decay=0)),
            (optim.Adamax, dict(weight_decay=1)),
            (optim.Adadelta, dict(weight_decay=0)),
            (optim.Adadelta, dict(weight_decay=1)),
            (optim.Adagrad, dict(weight_decay=0)),
            (optim.Adagrad, dict(weight_decay=1)),
        ]

    def test_multi_tensor_optimizers(self):
        self._test_derived_optimizers(self._multi_tensor_optimizer_configs, "foreach")

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_multi_tensor_optimizers_with_varying_tensors(self):
        self._test_derived_optimizers_varying_tensors(self._multi_tensor_optimizer_configs, "foreach")

    @unittest.skipIf(not torch.cuda.is_available(), "Requires a GPU")
    @largeTensorTest("72GB", "cuda")
    def test_multi_tensor_optimizers_with_large_tensors(self):
        for optimizer_ctor, optimizer_params in self._multi_tensor_optimizer_configs:
            # note(crcrpar): H100 wasn't sufficient for Adamax, surprisingly
            if optimizer_ctor == optim.Adamax:
                continue
            params = [torch.ones(2 ** 32, device="cuda", dtype=torch.float16)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optimizer_ctor(params, foreach=True, **optimizer_params)
            optimizer.step()

    @property
    def _fused_optimizer_configs(self):
        return tuple(itertools.product(
            (optim.Adam, optim.AdamW),
            (
                dict(weight_decay=1., amsgrad=False),
                dict(weight_decay=1., amsgrad=True),
                dict(weight_decay=0., amsgrad=False),
                dict(weight_decay=0., amsgrad=True),
            ),
        ))

    def test_fused_optimizers(self):
        self._test_derived_optimizers(self._fused_optimizer_configs, "fused")

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_fused_optimizers_with_varying_tensors(self):
        self._test_derived_optimizers_varying_tensors(self._fused_optimizer_configs, "fused")

    @unittest.skipIf(not torch.cuda.is_available(), "Requires a GPU")
    @largeTensorTest("64GB", "cuda")
    def test_fused_optimizers_with_large_tensors(self):
        for optimizer_ctor, optimizer_params in self._fused_optimizer_configs:
            params = [torch.ones(2 ** 32, device="cuda", dtype=torch.float16)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optimizer_ctor(params, fused=True, **optimizer_params)
            optimizer.step()

    def test_adam(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adam(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
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
            lambda weight, bias, maximize, foreach: optim.Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            [lambda opt: PolynomialLR(opt, total_iters=4, power=0.9)],
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_complex_2d(optim.Adam)
        self._test_complex_2d(functools.partial(optim.Adam, foreach=True))
        self._test_complex_2d(functools.partial(optim.Adam, foreach=True, weight_decay=0.2))

        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            optim.Adam(None, lr=1e-2, betas=(1.0, 0.0))

        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            optim.Adam(None, lr=1e-2, weight_decay=-1)

    def test_adamw(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.AdamW(
                [weight, bias], lr=1e-3, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.AdamW(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.AdamW(
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
            lambda weight, bias, maximize, foreach: optim.AdamW(
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
        self._test_complex_2d(optim.AdamW)
        self._test_complex_2d(functools.partial(optim.AdamW, foreach=True))
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            optim.AdamW(None, lr=1e-2, weight_decay=-1)

    def test_sparse_adam(self):
        self._test_rosenbrock_sparse(
            lambda params: optim.SparseAdam(params, lr=4e-2), [], True
        )
        self._test_rosenbrock_sparse(
            lambda params: optim.SparseAdam(params, lr=4e-2, maximize=True),
            scheduler_constructors=[],
            sparse_only=True,
            maximize=True,
        )
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            optim.SparseAdam(None, lr=1e-2, betas=(1.0, 0.0))
        with self.assertRaisesRegex(
            ValueError, "SparseAdam requires dense parameter tensors"
        ):
            optim.SparseAdam([torch.zeros(3, layout=torch.sparse_coo)])
        with self.assertRaisesRegex(
            ValueError, "SparseAdam requires dense parameter tensors"
        ):
            optim.SparseAdam([{"params": [torch.zeros(3, layout=torch.sparse_coo)]}])

    # ROCm precision is too low to pass this test
    def test_adadelta(self):
        # Handles https://github.com/pytorch/pytorch/issues/69698
        self.rel_tol = 4e-3
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adadelta(
                [weight, bias], maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adadelta(
                self._build_params_dict(weight, bias, rho=0.95),
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adadelta(
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
            lambda weight, bias, maximize, foreach: optim.Adadelta(
                [weight, bias], weight_decay=1, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        with self.assertRaisesRegex(ValueError, "Invalid rho value: 1.1"):
            optim.Adadelta(None, lr=1e-2, rho=1.1)

    def test_adadelta_complex(self):
        # Handles https://github.com/pytorch/pytorch/issues/69698
        self.rel_tol = 2e-2
        for optimizer in [optim.Adadelta]:
            self._test_complex_optimizer(lambda weight: optimizer([weight]))
            self._test_complex_optimizer(lambda weight: optimizer([weight], rho=0.95))
            self._test_complex_optimizer(
                lambda weight: optimizer([weight], rho=0.95, weight_decay=1)
            )

    def test_nadam(self):
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.NAdam(
                self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.NAdam(
                [weight, bias], lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.NAdam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                momentum_decay=6e-3,
                foreach=foreach,
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.NAdam(
                [weight, bias],
                lr=1e-3,
                weight_decay=0.1,
                momentum_decay=6e-3,
                foreach=foreach,
            ),
            [lambda opt: ExponentialLR(opt, gamma=0.9)],
            constructor_accepts_foreach=True,
        )
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            optim.NAdam(None, lr=1e-2, betas=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "Invalid momentum_decay value: -0.2"):
            optim.NAdam(None, lr=1e-2, momentum_decay=-0.2)

    def test_adagrad(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adagrad(
                [weight, bias], lr=1e-1, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adagrad(
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
            lambda weight, bias, maximize, foreach: optim.Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adagrad(
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
            lambda weight, bias, maximize, foreach: optim.Adagrad(
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
        with self.assertRaisesRegex(ValueError, "Invalid lr_decay value: -0.5"):
            optim.Adagrad(None, lr=1e-2, lr_decay=-0.5)

    def test_adagrad_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: optim.Adagrad(params, lr=1e-1, foreach=foreach)
            )
            self._test_rosenbrock_sparse(
                lambda params: optim.Adagrad(params, lr=0.1, foreach=foreach),
                scheduler_constructors=[
                    lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),
                    lambda opt: ReduceLROnPlateau(opt, threshold=1e-4),
                ],
            )

    def test_adagrad_complex(self):
        for foreach in (False, True):
            self._test_complex_optimizer(
                lambda param: optim.Adagrad([param], lr=1e-1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.Adagrad(
                    [param],
                    lr=1e-1,
                    initial_accumulator_value=0.1,
                    foreach=foreach,
                )
            )

    def test_adamax(self):
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adamax(
                [weight, bias], lr=1e-1, maximize=maximize, foreach=foreach
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adamax(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, maximize, foreach: optim.Adamax(
                [weight, bias],
                lr=1e-1,
                weight_decay=1,
                maximize=maximize,
                foreach=foreach,
            ),
            constructor_accepts_maximize=True,
            constructor_accepts_foreach=True,
        )
        self._test_complex_2d(optim.Adamax)
        self._test_complex_2d(functools.partial(optim.Adamax, foreach=True))
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 1: 1.0"
        ):
            optim.Adamax(None, lr=1e-2, betas=(0.0, 1.0))

    def test_radam(self):
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.RAdam(
                [weight, bias], lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.RAdam(
                self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.RAdam(
                [weight, bias], lr=1e-3, weight_decay=0.1, foreach=foreach
            ),
            constructor_accepts_foreach=True,
        )
        self._test_basic_cases(
            lambda weight, bias, foreach: optim.RAdam(
                [weight, bias], lr=1e-3, foreach=foreach
            ),
            [
                lambda opt: ExponentialLR(opt, gamma=0.9),
                lambda opt: ReduceLROnPlateau(opt),
            ],
            constructor_accepts_foreach=True,
        )
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            optim.RAdam(None, lr=1e-2, betas=(1.0, 0.0))

        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            optim.RAdam(None, lr=1e-2, weight_decay=-1)

    def test_rmsprop(self):
        for foreach in (False, True):
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.RMSprop(
                    [weight, bias], lr=1e-2, maximize=maximize, foreach=foreach
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.RMSprop(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2,
                    maximize=maximize,
                    foreach=foreach,
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.RMSprop(
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
                lambda weight, bias, maximize, foreach: optim.RMSprop(
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
                lambda weight, bias, maximize, foreach: optim.RMSprop(
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
                lambda weight, bias, maximize, foreach: optim.RMSprop(
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
            self._test_complex_2d(lambda param: optim.RMSprop(param, foreach=foreach))
            self._test_complex_2d(
                lambda param: optim.RMSprop(param, centered=True, foreach=foreach)
            )
            self._test_complex_2d(
                lambda param: optim.RMSprop(param, momentum=0.1, foreach=foreach)
            )
            self._test_complex_2d(
                lambda param: optim.RMSprop(param, maximize=True, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.RMSprop([param], foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.RMSprop([param], centered=True, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.RMSprop([param], momentum=0.1, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.RMSprop([param], maximize=True, foreach=foreach)
            )
            with self.assertRaisesRegex(ValueError, "Invalid momentum value: -1.0"):
                optim.RMSprop(None, lr=1e-2, momentum=-1.0, foreach=foreach)

    def test_asgd(self):
        for foreach in (False, True):
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.ASGD(
                    [weight, bias], lr=1e-3, t0=100, maximize=maximize, foreach=foreach
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.ASGD(
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
                lambda weight, bias, maximize, foreach: optim.ASGD(
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
                lambda params: optim.ASGD([params], foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda params: optim.ASGD([params], maximize=True, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda params: optim.ASGD(
                    [params], maximize=True, weight_decay=0.9, foreach=foreach
                )
            )
            self._test_complex_optimizer(
                lambda params: optim.ASGD(
                    [params], maximize=False, weight_decay=0.9, foreach=foreach
                )
            )
            with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -0.5"):
                optim.ASGD(None, lr=1e-2, weight_decay=-0.5, foreach=foreach)

    @skipIfRocm
    def test_rprop(self):
        is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(
            0
        ) == (8, 6)
        for foreach in (False, True):
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.Rprop(
                    [weight, bias], lr=2e-4, maximize=maximize, foreach=foreach
                ),
                constructor_accepts_maximize=True,
                constructor_accepts_foreach=True,
            )
            self._test_basic_cases(
                lambda weight, bias, maximize, foreach: optim.Rprop(
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
            self._test_complex_2d(lambda param: optim.Rprop(param, foreach=foreach))
            self._test_complex_optimizer(
                lambda param: optim.Rprop([param], lr=0.001, foreach=foreach)
            )
            self._test_complex_optimizer(
                lambda param: optim.Rprop(
                    [param], lr=0.001, maximize=True, foreach=foreach
                )
            )
            with self.assertRaisesRegex(ValueError, "Invalid eta values: 1.0, 0.5"):
                optim.Rprop(None, lr=1e-2, etas=(1.0, 0.5), foreach=foreach)

    def test_lbfgs(self):
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS([weight, bias]), ignore_multidevice=True
        )
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS(
                [weight, bias], line_search_fn="strong_wolfe"
            ),
            ignore_multidevice=True,
        )

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_lbfgs_returns_consistent_type(self):
        params = [torch.randn(10, 5), torch.randn(10)]
        opt1 = optim.LBFGS(params, 0.01, tolerance_grad=math.inf)
        opt2 = optim.LBFGS(params, 0.01, tolerance_grad=-math.inf)

        def closure():
            return torch.tensor([10])

        res1 = opt1.step(closure)
        res2 = opt2.step(closure)
        self.assertEqual(type(res1), type(res2))

    def test_invalid_param_type(self):
        self.assertRaisesRegex(
            TypeError,
            'params argument given to the optimizer should be an iterable of Tensors or dicts',
            lambda: optim.LBFGS(Parameter(torch.randn(5, 5)))
        )

    def test_duplicate_params_in_one_param_group(self):
        param = Parameter(torch.randn(1))
        with self.assertWarnsOnceRegex(UserWarning, '.*a parameter group with duplicate parameters.*'):
            optim.Adamax([param, param], lr=0.01)

    def test_duplicate_params_across_param_groups(self):
        param = Parameter(torch.randn(1))
        self.assertRaisesRegex(
            ValueError,
            'some parameters appear in more than one parameter group',
            lambda: optim.Adadelta([{'params': param}, {'params': param}])
        )

    def test_step_is_noop_when_params_have_no_grad(self):
        params = [torch.randn(2, 3, requires_grad=False) for _ in range(2)]
        old_params = [p.clone().detach() for p in params]

        def closure():
            return torch.tensor([1])

        optimizer_list = [
            optim.Adadelta,
            optim.AdamW,
            optim.Adam,
            optim.RAdam,
            optim.NAdam,
            optim.Adagrad,
            optim.Adamax,
            optim.RMSprop,
            optim.SGD,
            optim.SparseAdam,
            optim.ASGD,
            optim.LBFGS
        ]
        for optim_ctr in optimizer_list:
            opt = optim_ctr(params, lr=0.1)
            opt.step(closure)
        self.assertEqual(old_params, params)


    def test_step_is_noop_for_empty_grads(self):
        optimizers = [
            optim.Adadelta,
            optim.AdamW,
            optim.Adam,
            optim.RAdam,
            optim.NAdam,
            optim.Adagrad,
            optim.Adamax,
            optim.RMSprop,
            optim.SGD,
            optim.SparseAdam,
            optim.ASGD,
            optim.LBFGS
        ]
        param = torch.randn(5, 1, requires_grad=True)
        old_param = param.clone().detach()

        def closure():
            return torch.tensor([1])

        for optimizer in optimizers:
            opt = optimizer([param], lr=1e-5)
            param.grad = torch.zeros_like(param)
            if optimizer is optim.SparseAdam:
                # Intentionally construct a multidimensional empty v for the sparse grad
                # Single dim v passes the test while multidim correctly repros the issue
                # https://github.com/pytorch/pytorch/issues/82486
                i = torch.empty(1, 0)
                v = torch.empty(0, 1)
                param.grad = torch.sparse_coo_tensor(i, v, (5, 1))
            opt.step(closure)
            self.assertEqual(old_param, param)


    def test_fused_optimizer_does_not_step_if_foundinf(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required.")

        from torch.optim import adam, adamw

        num_tensors = 5
        for functional_optim, amsgrad, no_grad_scale in itertools.product((adam.adam, adamw.adamw), (False, True), (False, True)):
            params, grads, exp_avgs, exp_avg_sqs = [
                [torch.ones((1,), device="cuda") for _ in range(num_tensors)] for _ in range(4)]
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

    def test_fused_optimizer_raises(self):
        if not torch.cuda.is_available():
            self.skipTest("Requires CUDA devices")
        for optimizer_ctor in (torch.optim.Adam, torch.optim.AdamW):
            with self.assertRaisesRegex(RuntimeError, "`fused` and `foreach` cannot be `True` together."):
                optimizer_ctor([torch.empty((), device="cuda")], foreach=True, fused=True)
            with self.assertRaisesRegex(RuntimeError, "`fused` does not support `differentiable`"):
                optimizer_ctor([torch.empty((), device="cuda")], differentiable=True, fused=True)


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
                torch.optim.SGD,
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
                torch.optim.Adam,
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
                torch.optim.RMSprop,
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
                torch.optim.Adadelta,
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
                torch.optim.Adagrad,
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
                torch.optim.Adamax,
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
                torch.optim.ASGD,
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
                torch.optim.Rprop,
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
                torch.optim.AdamW,
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
                torch.optim.NAdam,
                {"lr": 0.9, "differentiable": True},
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
                torch.optim.RAdam,
                {"lr": 0.9, "differentiable": True},
                *state.values(),
            ),
        )


    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_defaults_changed_to_foreach(self):
        from torch.optim import (adam, adamw, nadam, sgd, radam, rmsprop, rprop,
                                 asgd, adamax, adadelta, adagrad)
        multi_optims = ((optim.Adam, adam, "_multi_tensor_adam"),
                        (optim.AdamW, adamw, "_multi_tensor_adamw"),
                        (optim.NAdam, nadam, "_multi_tensor_nadam"),
                        (optim.SGD, sgd, "_multi_tensor_sgd"),
                        (optim.RAdam, radam, "_multi_tensor_radam"),
                        (optim.RMSprop, rmsprop, "_multi_tensor_rmsprop"),
                        (optim.Rprop, rprop, "_multi_tensor_rprop"),
                        (optim.ASGD, asgd, "_multi_tensor_asgd"),
                        (optim.Adamax, adamax, "_multi_tensor_adamax"),
                        (optim.Adadelta, adadelta, "_multi_tensor_adadelta"),
                        (optim.Adagrad, adagrad, "_multi_tensor_adagrad"),)

        model = torch.nn.Linear(5, 5)
        model.to(dtype=torch.float64, device="cuda")
        input = torch.rand(2, 5, dtype=torch.float64, device="cuda")

        for opt, mod, func in multi_optims:
            defaults = {}
            if opt == optim.SGD:
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
    run_tests()
