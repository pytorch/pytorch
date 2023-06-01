# Owner(s): ["module: optimizer"]

import math
import unittest
import functools
import itertools
from copy import deepcopy

import torch
from torch import Tensor
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
    run_tests,
    TEST_WITH_UBSAN,
    load_tests,
    gradcheck,
    set_single_threaded_if_parallel_tbb,
    skipIfRocm,
    skipIfTorchDynamo
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_device_type import _TestParametrizer, largeTensorTest
from torch.testing._internal.common_methods_invocations import DecorateInfo
from typing import Callable, Dict, Any, List, Tuple, Union
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from unittest.mock import patch

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class OptimizerInfo:
    """ Optimizer information to be used in testing. """

    def __init__(self,
                 optim_cls: Optimizer,  # Class object for the Optimizer under test
                 *,
                 # Function to generate optimizer constructor configurations
                 optim_base_constructors_func: Callable,
                 # Implementation specific kwargs the optimizer supports, e.g., fused, foreach
                 supported_impl_kwargs: Tuple[str] = ('foreach', 'differentiable'),
                 # the devices on which the optim supports sparse tensors for params and grads, see SGD
                 supports_sparse_on: Tuple[str] = (),
                 # the optim only supports one config: sparse grads w/ dense params, see SparseAdam
                 only_supports_sparse_grads: bool = False,
                 # whether the optimizer.step() function requires a closure to be passed
                 step_requires_closure: bool = False,
                 # whether the optimizer supports per-param options with parameter groups
                 supports_param_groups: bool = True,
                 # whether the optimizer supports parameters on multiple devices
                 supports_multiple_devices: bool = True,
                 skips=(),  # Indicates which tests to skip
                 decorators=None,  # Additional decorators to apply to generated tests
                 ):
        self.optim_cls = optim_cls
        self.optim_base_constructors_func = optim_base_constructors_func
        self.supported_impl_kwargs = supported_impl_kwargs
        self.supports_sparse_on = supports_sparse_on
        self.only_supports_sparse_grads = only_supports_sparse_grads
        self.step_requires_closure = step_requires_closure
        self.supports_param_groups = supports_param_groups
        self.supports_multiple_devices = supports_multiple_devices
        self.decorators = (*(decorators if decorators else []), *(skips if skips else []))

    def get_decorators(self, test_class, test_name, device, param_kwargs):
        result = [set_single_threaded_if_parallel_tbb]
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(test_class, test_name, device, None, param_kwargs):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    @property
    def name(self):
        return self.optim_cls.__name__


class optims(_TestParametrizer):
    """ Decorator for specifying a list of optimizers over which to run a test. """

    def __init__(self, optim_info_iterable):
        self.optim_info_list = list(optim_info_iterable)

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError('The @optims decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        for optim_info in self.optim_info_list:
            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = optim_info.formatted_name

            # Construct parameter kwargs to pass to the test.
            param_kwargs = {'optim_info': optim_info}

            try:
                @functools.wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)

                decorator_fn = functools.partial(optim_info.get_decorators, generic_cls.__name__,
                                                 test.__name__, device_cls.device_type)

                yield (test_wrapper, test_name, param_kwargs, decorator_fn)
            except Exception as ex:
                # Provides an error message for debugging before rethrowing the exception
                print("Failed to instantiate {0} for module {1}!".format(test_name, optim_info.name))
                raise ex

# ----------------------------------------------------------------------------------------------------------------
# NOTE: The following optim_base_constructors_* sampling functions only return constructor combinations of NON-IMPLEMENTATION
# -CHANGING flags, i.e., flags that are not foreach, fused, capturable or differentiable.

def optim_base_constructors_adadelta(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        Adadelta(params),
        Adadelta(params, lr=0.01),  # TODO: Move out to testing in param_group?
        Adadelta(params, weight_decay=0.9),
        Adadelta(params, weight_decay=0.9, maximize=True),
        Adadelta(params, rho=0.95, weight_decay=0.9),  # TODO: Move out to testing in param_group?
    ]

def optim_base_constructors_adagrad(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        Adagrad(params),
        Adagrad(params, weight_decay=0.9),
        Adagrad(params, weight_decay=0.9, maximize=True),
        Adagrad(params, initial_accumulator_value=0.1, weight_decay=0.9),
        Adagrad(params, lr=0.1, lr_decay=0.5, weight_decay=0.9),  # TODO: Move out to testing in param_group?
    ]


def optim_base_constructors_adam(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        Adam(params),
        Adam(params, lr=0.01),  # TODO: Move out to testing in param_group?
        Adam(params, weight_decay=0.9),
        Adam(params, weight_decay=0.9, maximize=True),
        Adam(params, weight_decay=0.9, amsgrad=True),
    ]


def optim_base_constructors_adamax(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        Adamax(params),
        Adamax(params, lr=0.001),
        Adamax(params, weight_decay=0.9),
        Adamax(params, weight_decay=0.9, maximize=True),
    ]


def optim_base_constructors_adamw(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        AdamW(params),
        AdamW(params, lr=0.01),
        AdamW(params, weight_decay=0.9),
        AdamW(params, weight_decay=0.9, maximize=True),
        AdamW(params, weight_decay=0.9, amsgrad=True),
    ]


def optim_base_constructors_asgd(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        ASGD(params),
        ASGD(params, lr=0.02),
        ASGD(params, t0=100),
        ASGD(params, weight_decay=0.9),
        ASGD(params, weight_decay=0.9, maximize=True),
    ]


def optim_base_constructors_lbfgs(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        LBFGS(params),
        LBFGS(params, lr=0.01),
        LBFGS(params, tolerance_grad=math.inf),
        LBFGS(params, line_search_fn="strong_wolfe")
    ]


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_base_constructors_nadam(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        NAdam(params),
        NAdam(params, lr=1e-3),
        NAdam(params, momentum_decay=6e-3),
        NAdam(params, weight_decay=0.9, momentum_decay=6e-3),
    ]


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_base_constructors_radam(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        RAdam(params),
        RAdam(params, lr=1e-2),
        RAdam(params, weight_decay=0.9)
    ]


def optim_base_constructors_rmsprop(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        RMSprop(params),
        RMSprop(params, weight_decay=0.9),
        RMSprop(params, weight_decay=0.9, centered=True),
        RMSprop(params, weight_decay=0.9, centered=True, momentum=0.1),
        RMSprop(params, weight_decay=0.9, centered=True, momentum=0.1, maximize=True),
    ]


def optim_base_constructors_rprop(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        Rprop(params),
        Rprop(params, lr=2e-4),
        Rprop(params, etas=(1.5, 0.5)),
        Rprop(params, maximize=True),
    ]


def optim_base_constructors_sgd(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        SGD(params, lr=1e-3),
        SGD(params, lr=1e-3, momentum=0.5),
        SGD(params, lr=1e-3, momentum=0.5, weight_decay=0.9),
        SGD(params, lr=1e-3, momentum=0.5, nesterov=True, weight_decay=0.9),
        SGD(params, lr=1e-3, weight_decay=0.9, maximize=True),
    ]


def optim_base_constructors_sparseadam(optim_info: OptimizerInfo, params: Union[List[Parameter], List[Tensor], Dict[Any]]):
    return [
        SparseAdam(params),
        SparseAdam(params, lr=0.01),  # TODO: Move out to testing in param_group?
        SparseAdam(params, maximize=True)
    ]


# Database of OptimizerInfo entries in alphabetical order.
module_db: List[OptimizerInfo] = [
    OptimizerInfo(
        Adadelta,
        optim_base_constructors_func=optim_base_constructors_adadelta,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        Adagrad,
        optim_base_constructors_func=optim_base_constructors_adagrad,
        supported_impl_kwargs=('foreach', 'differentiable'),
        supports_sparse_on=('cpu'),
    ),
    OptimizerInfo(
        Adam,
        optim_base_constructors_func=optim_base_constructors_adam,
        supported_impl_kwargs=('foreach', 'differentiable', 'fused', 'capturable'),
    ),
    OptimizerInfo(
        Adamax,
        optim_base_constructors_func=optim_base_constructors_adamax,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        AdamW,
        optim_base_constructors_func=optim_base_constructors_adamw,
        supported_impl_kwargs=('foreach', 'differentiable', 'fused', 'capturable'),
    ),
    OptimizerInfo(
        ASGD,
        optim_base_constructors_func=optim_base_constructors_asgd,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        LBFGS,
        optim_base_constructors_func=optim_base_constructors_lbfgs,
        supported_impl_kwargs=(),
        step_requires_closure=True,
        supports_param_groups=False,
        supports_multiple_devices=False,
    ),
    OptimizerInfo(
        NAdam,
        optim_base_constructors_func=optim_base_constructors_nadam,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        RAdam,
        optim_base_constructors_func=optim_base_constructors_radam,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        RMSprop,
        optim_base_constructors_func=optim_base_constructors_rmsprop,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        Rprop,
        optim_base_constructors_func=optim_base_constructors_rprop,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        SGD,
        optim_base_constructors_func=optim_base_constructors_sgd,
        supported_impl_kwargs=('foreach', 'differentiable'),
        supports_sparse_on=('cpu', 'cuda'),
    ),
    OptimizerInfo(
        SparseAdam,
        optim_base_constructors_func=optim_base_constructors_sparseadam,
        supported_impl_kwargs=(),
        only_supports_sparse_grads=True,
    )
]


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
        with self.assertRaisesRegex(ValueError, "Invalid momentum value: -0.5"):
            SGD(None, lr=1e-2, momentum=-0.5)

    def test_sgd_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: SGD(params, lr=4.8e-3, foreach=foreach)
            )
            self._test_rosenbrock_sparse(
                lambda params: SGD(params, lr=0.0048, foreach=foreach),
                scheduler_constructors=[lambda opt: StepLR(opt, gamma=0.99999, step_size=300)],
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
            (Adam, dict(weight_decay=1.0, amsgrad=True, fused=False)),
            (Adam, dict(weight_decay=1.0, amsgrad=False, fused=False)),
            (Adam, dict(weight_decay=0.0, amsgrad=True, fused=False)),
            (Adam, dict(weight_decay=0.0, amsgrad=False, fused=False)),
            (AdamW, dict(weight_decay=1.0, amsgrad=True)),
            (AdamW, dict(weight_decay=1.0, amsgrad=False)),
            (AdamW, dict(weight_decay=0.0, amsgrad=True)),
            (AdamW, dict(weight_decay=0.0, amsgrad=False)),
            (NAdam, dict(weight_decay=0.0, momentum_decay=6e-3)),
            (NAdam, dict(weight_decay=1.0, momentum_decay=6e-3)),
            (NAdam, dict(weight_decay=0.0, momentum_decay=4e-3)),
            (NAdam, dict(weight_decay=0.01, momentum_decay=4e-3)),
            (
                SGD,
                dict(lr=0.2, momentum=1, dampening=0, weight_decay=1, nesterov=True),
            ),
            (
                SGD,
                dict(lr=0.2, momentum=1, dampening=0.5, weight_decay=1, nesterov=False),
            ),
            (RAdam, dict(weight_decay=0, eps=1e-6)),
            (RAdam, dict(weight_decay=0)),
            (RAdam, dict(weight_decay=1, eps=1e-6)),
            (RAdam, dict(weight_decay=1)),
            (RMSprop, dict(weight_decay=1, momentum=1, centered=True)),
            (RMSprop, dict(weight_decay=1, momentum=0, centered=True)),
            (RMSprop, dict(weight_decay=1, momentum=1, centered=False)),
            (RMSprop, dict(weight_decay=0, momentum=1, centered=False)),
            (Rprop, dict(lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50))),
            (ASGD, dict(weight_decay=0)),
            (ASGD, dict(weight_decay=1)),
            (Adamax, dict(weight_decay=0)),
            (Adamax, dict(weight_decay=1)),
            (Adadelta, dict(weight_decay=0)),
            (Adadelta, dict(weight_decay=1)),
            (Adagrad, dict(weight_decay=0)),
            (Adagrad, dict(weight_decay=1)),
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
            if optimizer_ctor == Adamax:
                continue
            params = [torch.ones(2 ** 32, device="cuda", dtype=torch.float16)]
            params[0].grad = torch.zeros_like(params[0])
            optimizer = optimizer_ctor(params, foreach=True, **optimizer_params)
            optimizer.step()

    @property
    def _fused_optimizer_configs(self):
        return tuple(itertools.product(
            (Adam, AdamW),
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
        self._test_complex_2d(Adam)
        self._test_complex_2d(functools.partial(Adam, foreach=True))
        self._test_complex_2d(functools.partial(Adam, foreach=True, weight_decay=0.2))

        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            Adam(None, lr=1e-2, betas=(1.0, 0.0))

        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            Adam(None, lr=1e-2, weight_decay=-1)

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
        self._test_complex_2d(AdamW)
        self._test_complex_2d(functools.partial(AdamW, foreach=True))
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            AdamW(None, lr=1e-2, weight_decay=-1)

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
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            SparseAdam(None, lr=1e-2, betas=(1.0, 0.0))
        with self.assertRaisesRegex(
            ValueError, "SparseAdam requires dense parameter tensors"
        ):
            SparseAdam([torch.zeros(3, layout=torch.sparse_coo)])
        with self.assertRaisesRegex(
            ValueError, "SparseAdam requires dense parameter tensors"
        ):
            SparseAdam([{"params": [torch.zeros(3, layout=torch.sparse_coo)]}])

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
        with self.assertRaisesRegex(ValueError, "Invalid rho value: 1.1"):
            Adadelta(None, lr=1e-2, rho=1.1)

    def test_adadelta_complex(self):
        # Handles https://github.com/pytorch/pytorch/issues/69698
        self.rel_tol = 2e-2
        for optimizer in [Adadelta]:
            self._test_complex_optimizer(lambda weight: optimizer([weight]))
            self._test_complex_optimizer(lambda weight: optimizer([weight], rho=0.95))
            self._test_complex_optimizer(
                lambda weight: optimizer([weight], rho=0.95, weight_decay=1)
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
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            NAdam(None, lr=1e-2, betas=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "Invalid momentum_decay value: -0.2"):
            NAdam(None, lr=1e-2, momentum_decay=-0.2)

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
        with self.assertRaisesRegex(ValueError, "Invalid lr_decay value: -0.5"):
            Adagrad(None, lr=1e-2, lr_decay=-0.5)

    def test_adagrad_sparse(self):
        for foreach in (False, True):
            self._test_rosenbrock_sparse(
                lambda params: Adagrad(params, lr=1e-1, foreach=foreach)
            )
            self._test_rosenbrock_sparse(
                lambda params: Adagrad(params, lr=0.1, foreach=foreach),
                scheduler_constructors=[
                    lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),
                    lambda opt: ReduceLROnPlateau(opt, threshold=1e-4),
                ],
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
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 1: 1.0"
        ):
            Adamax(None, lr=1e-2, betas=(0.0, 1.0))

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
        with self.assertRaisesRegex(
            ValueError, "Invalid beta parameter at index 0: 1.0"
        ):
            RAdam(None, lr=1e-2, betas=(1.0, 0.0))

        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            RAdam(None, lr=1e-2, weight_decay=-1)

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
            with self.assertRaisesRegex(ValueError, "Invalid momentum value: -1.0"):
                RMSprop(None, lr=1e-2, momentum=-1.0, foreach=foreach)

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
            with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -0.5"):
                ASGD(None, lr=1e-2, weight_decay=-0.5, foreach=foreach)

    @skipIfRocm
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
            with self.assertRaisesRegex(ValueError, "Invalid eta values: 1.0, 0.5"):
                Rprop(None, lr=1e-2, etas=(1.0, 0.5), foreach=foreach)

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

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_lbfgs_returns_consistent_type(self):
        params = [torch.randn(10, 5), torch.randn(10)]
        opt1 = LBFGS(params, 0.01, tolerance_grad=math.inf)
        opt2 = LBFGS(params, 0.01, tolerance_grad=-math.inf)

        def closure():
            return torch.tensor([10])

        res1 = opt1.step(closure)
        res2 = opt2.step(closure)
        self.assertEqual(type(res1), type(res2))

    def test_invalid_param_type(self):
        self.assertRaisesRegex(
            TypeError,
            'params argument given to the optimizer should be an iterable of Tensors or dicts',
            lambda: LBFGS(Parameter(torch.randn(5, 5)))
        )

    def test_duplicate_params_in_one_param_group(self):
        param = Parameter(torch.randn(1))
        with self.assertWarnsOnceRegex(UserWarning, '.*a parameter group with duplicate parameters.*'):
            Adamax([param, param], lr=0.01)

    def test_duplicate_params_across_param_groups(self):
        param = Parameter(torch.randn(1))
        self.assertRaisesRegex(
            ValueError,
            'some parameters appear in more than one parameter group',
            lambda: Adadelta([{'params': param}, {'params': param}])
        )

    def test_step_is_noop_when_params_have_no_grad(self):
        params = [torch.randn(2, 3, requires_grad=False) for _ in range(2)]
        old_params = [p.clone().detach() for p in params]

        def closure():
            return torch.tensor([1])

        optimizer_list = [
            Adadelta,
            AdamW,
            Adam,
            RAdam,
            NAdam,
            Adagrad,
            Adamax,
            RMSprop,
            SGD,
            SparseAdam,
            ASGD,
            LBFGS
        ]
        for optim_ctr in optimizer_list:
            opt = optim_ctr(params, lr=0.1)
            opt.step(closure)
        self.assertEqual(old_params, params)


    def test_step_is_noop_for_empty_grads(self):
        optimizers = [
            Adadelta,
            AdamW,
            Adam,
            RAdam,
            NAdam,
            Adagrad,
            Adamax,
            RMSprop,
            SGD,
            SparseAdam,
            ASGD,
            LBFGS
        ]
        param = torch.randn(5, 1, requires_grad=True)
        old_param = param.clone().detach()

        def closure():
            return torch.tensor([1])

        for optimizer in optimizers:
            opt = optimizer([param], lr=1e-5)
            param.grad = torch.zeros_like(param)
            if optimizer is SparseAdam:
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
        for optimizer_ctor in (Adam, AdamW):
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
    run_tests()
