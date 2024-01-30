# mypy: ignore-errors

import functools
import itertools
import unittest
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    LBFGS,
    NAdam,
    Optimizer,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_utils import (
    _TestParametrizer,
    set_single_threaded_if_parallel_tbb,
    skipIfMps,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
)
from torch.utils._foreach_utils import (
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
)


class OptimizerInput:
    """Contains args / kwargs to be passed to an optimizer constructor."""

    __slots__ = ["params", "kwargs", "desc"]

    def __init__(
        self,
        params: Union[List[Parameter], List[Tensor], Dict[Any, Any]],
        kwargs: Dict[str, Any],
        desc: str = "",
    ):
        # params can be a list of Tensors OR param_groups OR None
        self.params = params
        self.kwargs = kwargs
        self.desc = desc

    def __repr__(self):
        return f"params={self.params}, kwargs={self.kwargs}, desc={self.desc}"


class OptimizerErrorEnum(Enum):
    """Enumerates when an error is raised when testing optimizers."""

    CONSTRUCTION_ERROR = 0
    STEP_ERROR = 1


class ErrorOptimizerInput:
    """
    An OptimizerInput that will cause the optimizer to throw an error when constructed.
    Includes the type and string of the resulting error.
    """

    __slots__ = ["optimizer_error_input", "error_on", "error_type", "error_regex"]

    def __init__(
        self,
        optimizer_error_input,
        *,
        error_on=OptimizerErrorEnum.CONSTRUCTION_ERROR,
        error_type=RuntimeError,
        error_regex="",
    ):
        self.optimizer_error_input = optimizer_error_input
        self.error_on = error_on
        self.error_type = error_type
        self.error_regex = error_regex


class OptimizerInfo:
    """Optimizer information to be used in testing."""

    def __init__(
        self,
        optim_cls: Optimizer,  # Class object for the Optimizer under test
        *,
        # Function to generate optimizer inputs EXCLUDING params. We delegate params responsibility
        # to the test using the OptimizerInfo. OptimizerInput.params is likely None.
        # Can optionally take in device to filter out certain unsupported configs
        optim_inputs_func,
        # A subset of the global-cliquey flags (fused, foreach, differentiable) the optimizer
        # supports. See NOTE: [optimizer kwarg categories] for what global-cliquey means.
        supported_impls: Tuple[str] = ("foreach", "differentiable"),
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
        optim_error_inputs_func=None,  # Function to generate optim inputs that error
    ):
        self.optim_cls = optim_cls
        self.optim_inputs_func = optim_inputs_func
        self.supported_impls = supported_impls
        self.supports_sparse_on = supports_sparse_on
        self.only_supports_sparse_grads = only_supports_sparse_grads
        self.step_requires_closure = step_requires_closure
        self.supports_param_groups = supports_param_groups
        self.supports_multiple_devices = supports_multiple_devices
        self.decorators = (
            *(decorators if decorators else []),
            *(skips if skips else []),
        )
        self.optim_error_inputs_func = optim_error_inputs_func

    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        result = [set_single_threaded_if_parallel_tbb]
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(
                    test_class, test_name, device, dtype, param_kwargs
                ):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    @property
    def name(self):
        return self.optim_cls.__name__


class optims(_TestParametrizer):
    """Decorator for specifying a list of optimizers over which to run a test."""

    def __init__(self, optim_info_iterable, dtypes=None):
        self.optim_info_list = list(optim_info_iterable)

        # optimizers aren't limited to be one dtype as parameters can have different dtypes
        # We default to torch.float32, but dtypes should be specified through passed in
        # parameters.
        self.dtypes = dtypes if dtypes is not None else [torch.float32]

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError(
                "The @optims decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        for optim_info, dtype in itertools.product(self.optim_info_list, self.dtypes):
            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = optim_info.name

            # Construct parameter kwargs to pass to the test.
            param_kwargs = {"optim_info": optim_info, "dtype": dtype}

            try:

                @functools.wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)

                decorator_fn = functools.partial(
                    optim_info.get_decorators,
                    generic_cls.__name__,
                    test.__name__,
                    device_cls.device_type,
                    dtype,
                )

                yield (test_wrapper, test_name, param_kwargs, decorator_fn)
            except Exception as ex:
                # Provides an error message for debugging before rethrowing the exception
                print(
                    f"Failed to instantiate {test_name} for module {optim_info.name}!"
                )
                raise ex


# Helper function for generating error inputs for all optimizers, used below.
def get_error_inputs_for_all_optims(device, dtype):
    if str(device) == "cpu":
        sample_param = Parameter(torch.randn(1, device=device, dtype=dtype))
        return [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=sample_param,
                    kwargs={},
                    desc="invalid param type",
                ),
                error_type=TypeError,
                error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_param, sample_param],
                    kwargs={},
                    desc="a param group cannot have duplicate parameters",
                ),
                error_type=UserWarning,
                error_regex=".*a parameter group with duplicate parameters.*",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[{"params": sample_param}, {"params": sample_param}],
                    kwargs={},
                    desc="duplicate parameters should not occur across param groups either",
                ),
                error_type=ValueError,
                error_regex="some parameters appear in more than one parameter group",
            ),
        ]
    else:
        return []


# ------------------------------------------------------------------------------------------
# NOTE: [optimizer kwarg categories]
# We categorize optimizer kwargs as 3 types:
#  1. optimizer-specific flags are like amsgrad or rho or beta, flags that are specific to
#     algorithms and thus only show up for certain optimizers. There are many of these, so I
#     do not bother gathering them all and listing them here. The converse to these would be
#     global flags that every optimizer ideally _should_ support. We break global flags into
#     2 further categories and list them all below.
#  2. global-friendly = ["lr", "weight_decay", "maximize", "capturable"]
#     global-friendly flags are global flags who play nicely with all other global flags,
#     i.e., are mutually exclusive in function. This means that any pair of the following
#     flags can be toggled at once (e.g., maximize and weight_decay). Furthermore, any of the
#     following flags theoretically can be enabled with ANY other global flag, including the
#     cliquey ones (e.g, capturable and foreach).
#  3. global-cliquey = ["foreach", "fused", "differentiable"]
#     global-cliquey flags are global flags that do NOT coexist with other cliquey flags,
#     usually because they contradict each other in function. For example, one should not flip
#     both foreach AND fused to True, because they are two differing performance optimizations
#     in which you can only opt into one.
#
# The following optim_inputs_func_* sampling functions only return constructor combinations of
# optimizer-specific and global-friendly flags. This is because we are confident they would mesh
# well with additional kwargs. On the flip side of the same coin, we reserve setting the
# global-cliquey flags to individual tests and fully expect tests to edit OptimizerInput.kwargs.


def optim_inputs_func_adadelta(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=None, kwargs={"rho": 0.95, "weight_decay": 0.9}, desc="rho"
        ),
    ]


def optim_error_inputs_func_adadelta(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, rho=1.1),
                    desc="rho should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid rho value: 1.1",
            ),
        ]
    return error_inputs


def optim_inputs_func_adagrad(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(params=None, kwargs={"lr": 0.1}, desc="non-default lr"),
        OptimizerInput(
            params=None,
            kwargs={"initial_accumulator_value": 0.1, "weight_decay": 0.1},
            desc="initial_accumulator_value",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": 0.1, "lr_decay": 0.5, "weight_decay": 0.1},
            desc="lr_decay",
        ),  # TODO: Move out to testing in param_group?
    ]


def optim_error_inputs_func_adagrad(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, lr_decay=-0.5),
                    desc="lr_decay must be bigger than 0",
                ),
                error_type=ValueError,
                error_regex="Invalid lr_decay value: -0.5",
            ),
        ]
    return error_inputs


# TODO: consider tensor LR! See multi_tensor_optimizer_configs in test_optim.py --> tensor LR should work
# with all implementation code paths...
def optim_inputs_func_adam(device=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "amsgrad": True, "capturable": True},
            desc="capturable, amsgrad",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "amsgrad": True, "capturable": True},
            desc="Tensor lr with capturable and amsgrad",
        ),
    ]

    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1, "amsgrad": True}, desc="amsgrad"
        ),
    ] + (cuda_supported_configs if str(device) == "cuda" else [])


def optim_error_inputs_func_adam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-1),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -1",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=torch.tensor(0.001), foreach=True),
                    desc="lr as Tensor doesn't work with foreach & not capturable",
                ),
                error_type=ValueError,
                error_regex="lr as a Tensor is not supported for capturable=False and foreach=True",
            ),
        ]
    if str(device) == "cuda":
        sample_tensor = torch.empty((), device=device, dtype=dtype)
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_tensor],
                    kwargs={"foreach": True, "fused": True},
                    desc="`fused` and `foreach` cannot be `True` together",
                ),
                error_type=RuntimeError,
                error_regex="`fused` and `foreach` cannot be `True` together",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_tensor],
                    kwargs={"fused": True, "differentiable": True},
                    desc="`fused` does not support `differentiable`",
                ),
                error_type=RuntimeError,
                error_regex="`fused` does not support `differentiable`",
            ),
        ]
    return error_inputs


def optim_inputs_func_adamax(device=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "maximize": True, "capturable": True},
            desc="capturable, maximize, weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0, "maximize": True, "capturable": True},
            desc="capturable, maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "maximize": False, "capturable": True},
            desc="capturable, weight_decay",
        ),
    ]

    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.1}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ] + (cuda_supported_configs if str(device) == "cuda" else [])


def optim_error_inputs_func_adamax(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(0.0, 1.0)),
                    desc="beta2 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 1: 1.0",
            ),
        ]
    return error_inputs


def optim_inputs_func_adamw(device=None):
    return optim_inputs_func_adam(device=device)


def optim_error_inputs_func_adamw(device, dtype):
    return optim_error_inputs_func_adam(device, dtype)


def optim_inputs_func_asgd(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.02}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"t0": 100}, desc="t0"),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_asgd(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-0.5),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -0.5",
            ),
        ]
    return error_inputs


def optim_inputs_func_lbfgs(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"tolerance_grad": 1e-6}, desc="tolerance_grad"
        ),
        OptimizerInput(
            params=None,
            kwargs={"line_search_fn": "strong_wolfe"},
            desc="strong_wolfe",
        ),
    ]


def optim_error_inputs_func_lbfgs(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    return error_inputs


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_nadam(device=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "momentum_decay": 6e-3, "capturable": True},
            desc="weight_decay, capturable",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.9,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
                "capturable": True,
            },
            desc="decoupled_weight_decay, capturable",
        ),
    ]
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 1e-3}, desc="non-default lr"),
        OptimizerInput(
            params=None,
            kwargs={"momentum_decay": 6e-3},
            desc="non-zero momentum_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "momentum_decay": 6e-3},
            desc="weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
            },
            desc="decoupled_weight_decay",
        ),
    ] + (cuda_supported_configs if str(device) == "cuda" else [])


def optim_error_inputs_func_nadam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum_decay=-0.2),
                    desc="momentum_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum_decay value: -0.2",
            ),
        ]
    return error_inputs


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_radam(device=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={
                "capturable": True,
                "weight_decay": 0.1,
            },
            desc="capturable, weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "capturable": True,
                "weight_decay": 0.1,
                "decoupled_weight_decay": True,
            },
            desc="capturable, weight_decay, decoupled_weight_decay",
        ),
    ]
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 2e-3}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"eps": 1e-6}, desc="non-default eps"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "decoupled_weight_decay": True},
            desc="decoupled_weight_decay",
        ),
    ] + (cuda_supported_configs if "cuda" in str(device) else [])


def optim_error_inputs_func_radam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-1),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -1",
            ),
        ]
    return error_inputs


def optim_inputs_func_rmsprop(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 1e-3}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "centered": True},
            desc="centered",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "centered": True, "momentum": 0.1},
            desc="momentum",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
                "centered": True,
                "momentum": 0.1,
                "maximize": True,
            },
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_rmsprop(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum=-1.0),
                    desc="momentum should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum value: -1.0",
            ),
        ]
    return error_inputs


def optim_inputs_func_rprop(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 2e-4}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"etas": (0.5, 1.5)}, desc="non-default etas"
        ),
        OptimizerInput(
            params=None,
            kwargs={"step_sizes": (2e-6, 100)},
            desc="non-default step_sizes",
        ),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
    ]


def optim_error_inputs_func_rprop(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, etas=(1.0, 0.5)),
                    desc="0 < eta1 < 1 < eta2",
                ),
                error_type=ValueError,
                error_regex="Invalid eta values: 1.0, 0.5",
            ),
        ]
    return error_inputs


def optim_inputs_func_sgd(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 1e-2}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"momentum": 0.9}, desc="momentum"),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "dampening": 0.5},
            desc="dampening",
        ),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "weight_decay": 0.1},
            desc="non-zero weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "nesterov": True, "weight_decay": 0.1},
            desc="nesterov",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_sgd(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if str(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum=-0.5),
                    desc="momentum should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum value: -0.5",
            ),
        ]
    return error_inputs


def optim_inputs_func_sparseadam(device=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(
            params=None, kwargs={"lr": 0.01}, desc="non-default lr"
        ),  # TODO: Move out to testing in param_group?
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
    ]


def optim_error_inputs_func_sparseadam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)

    if str(device) == "cpu":
        # SparseAdam raises a warning and not an error for the first entry. We
        # update it here:
        error_inputs[0].error_type = FutureWarning
        error_inputs[
            0
        ].error_regex = "Passing in a raw Tensor as ``params`` to SparseAdam"

        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[
                        torch.zeros(
                            3, layout=torch.sparse_coo, device=device, dtype=dtype
                        )
                    ],
                    kwargs={},
                    desc="dense params required",
                ),
                error_type=ValueError,
                error_regex="SparseAdam requires dense parameter tensors",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[
                        {
                            "params": [
                                torch.zeros(
                                    3,
                                    layout=torch.sparse_coo,
                                    device=device,
                                    dtype=dtype,
                                )
                            ]
                        }
                    ],
                    kwargs={},
                    desc="dense params required in param_groups",
                ),
                error_type=ValueError,
                error_regex="SparseAdam requires dense parameter tensors",
            ),
        ]
    return error_inputs


def _get_optim_inputs_including_global_cliquey_kwargs(
    device, dtype, optim_info, skip=()
) -> List[OptimizerInput]:
    """
    Return a list of all configs for a given optimizer as a list of OptimizerInputs,
    including configs that have supported global cliquey kwargs (foreach, fused,
    differentiable) based on optim_info.supported_impls.

    The configs (optim_inputs) returned by optim_info.optim_inputs_func(...)
    intentionally do NOT include global cliquey kwargs to give flexibility to tests.
    For example, testing correctness between toggling foreach on and off is now
    trivial. That said, we sometimes want to test for all possible configs on an
    optimizer including all supported flags, so this helper returns all optim inputs.
    """
    assert all(
        x in ["foreach", "fused", "differentiable"] for x in skip
    ), "skip must be a subset of ['foreach', 'fused', 'differentiable']"

    optim_inputs = optim_info.optim_inputs_func(device=device)

    supported_impls = tuple(
        x
        for x in optim_info.supported_impls
        if x not in skip
        and (str(device) in _get_fused_kernels_supported_devices() or x != "fused")
        and (str(device) in _get_foreach_kernels_supported_devices() or x != "foreach")
    )

    all_optim_inputs = []
    for optim_input in optim_inputs:
        # Add the base config where all the flags are False
        base_kwargs = deepcopy(optim_input.kwargs)
        if len(supported_impls) != 0:
            for flag in supported_impls:
                base_kwargs[flag] = False
            all_optim_inputs.append(
                OptimizerInput(params=None, kwargs=base_kwargs, desc=optim_input.desc)
            )
        else:
            all_optim_inputs.append(optim_input)
        # Add a config for when each of the global cliquey kwargs is True
        # Note that in [optimizer kwarg categories], these kwargs are mutually
        # exclusive, so we do not need to product them together.
        for flag in supported_impls:
            new_kwargs = deepcopy(base_kwargs)
            new_kwargs[flag] = True
            all_optim_inputs.append(
                OptimizerInput(
                    params=None, kwargs=new_kwargs, desc=f"{optim_input.desc} & {flag}"
                )
            )
    return all_optim_inputs


# Database of OptimizerInfo entries in alphabetical order.
optim_db: List[OptimizerInfo] = [
    OptimizerInfo(
        Adadelta,
        optim_inputs_func=optim_inputs_func_adadelta,
        optim_error_inputs_func=optim_error_inputs_func_adadelta,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679 and #116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679"
                ),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        Adagrad,
        optim_inputs_func=optim_inputs_func_adagrad,
        optim_error_inputs_func=optim_error_inputs_func_adagrad,
        supported_impls=("foreach", "differentiable"),
        supports_sparse_on=("cpu"),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115607"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115607 and #116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        Adam,
        optim_inputs_func=optim_inputs_func_adam,
        optim_error_inputs_func=optim_error_inputs_func_adam,
        supported_impls=("foreach", "differentiable", "fused"),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Fixing #115607 should fix this test. fused is correct, but forloop is not."
                ),
                "TestOptimRenewed",
                "test_fused_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        Adamax,
        optim_inputs_func=optim_inputs_func_adamax,
        optim_error_inputs_func=optim_error_inputs_func_adamax,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo("Mismatched _foreach_addcdiv_ types, see #118159"),
                "TestOptimRenewed",
                "test_complex",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115607"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115607 and #116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                unittest.skip("Uses too much memory, even for H100, surprisingly."),
                "TestOptimRenewed",
                "test_foreach_large_tensor",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        AdamW,
        optim_inputs_func=optim_inputs_func_adamw,
        optim_error_inputs_func=optim_error_inputs_func_adamw,
        supported_impls=("foreach", "differentiable", "fused"),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Fixing #115607 should fix this test. fused is correct, but forloop is not."
                ),
                "TestOptimRenewed",
                "test_fused_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        ASGD,
        optim_inputs_func=optim_inputs_func_asgd,
        optim_error_inputs_func=optim_error_inputs_func_asgd,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See discrepancy in https://github.com/pytorch/pytorch/issues/115607"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(atol=1.5e-5, rtol=1e-5),
                    }
                ),
                "TestOptimRenewed",
                "test_step_is_noop_for_zero_grads",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        LBFGS,
        optim_inputs_func=optim_inputs_func_lbfgs,
        optim_error_inputs_func=optim_error_inputs_func_lbfgs,
        supported_impls=(),
        step_requires_closure=True,
        supports_param_groups=False,
        supports_multiple_devices=False,
        skips=(
            # Fails on MacOS 13.2.1 in CI https://github.com/pytorch/pytorch/issues/117094
            DecorateInfo(
                skipIfMps, "TestOptimRenewed", "test_can_load_older_state_dict"
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
            DecorateInfo(
                unittest.skip("Does not support param groups"),
                "TestOptimRenewed",
                "test_param_groups_lr",
            ),
            DecorateInfo(
                unittest.skip("Does not support param groups"),
                "TestOptimRenewed",
                "test_param_groups_weight_decay",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                unittest.skip("LBFGS doesn't support multidevice"),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
        ),
    ),
    OptimizerInfo(
        NAdam,
        optim_inputs_func=optim_inputs_func_nadam,
        optim_error_inputs_func=optim_error_inputs_func_nadam,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/116499"
                ),
                "TestOptimRenewed",
                "test_can_load_older_state_dict",
                device_type="cuda",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors, https://github.com/pytorch/pytorch/issues/117150"
                ),
                "TestOptimRenewed",
                "test_load_nontensor_step",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors, see https://github.com/pytorch/pytorch/issues/117150"
                ),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        RAdam,
        optim_inputs_func=optim_inputs_func_radam,
        optim_error_inputs_func=optim_error_inputs_func_radam,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115607"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        # previously atol=1e-7, rtol=1e-7
                        torch.float64: tol(atol=1.5e-7, rtol=1.1e-7)
                    }
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/115607"
                ),
                "TestOptimRenewed",
                "test_can_load_older_state_dict",
                device_type="cpu",
            ),
            DecorateInfo(
                toleranceOverride(
                    {  # previously atol=5-05, rtol=0.001, https://github.com/pytorch/pytorch/issues/116202
                        torch.float32: tol(atol=5e-04, rtol=0.01),
                    }
                ),
                "TestOptimRenewed",
                "test_mixed_device_dtype",
                active_if=TEST_WITH_TORCHDYNAMO,
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_complex",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_step_is_noop_when_params_have_no_grad",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_load_nontensor_step",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_param_groups_weight_decay",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_param_groups_lr",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_step_is_noop_for_zero_grads",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Should be fixed by https://github.com/pytorch/pytorch/issues/118230"
                ),
                "TestOptimRenewed",
                "test_mixed_device_dtype",
            ),
        ),
    ),
    OptimizerInfo(
        RMSprop,
        optim_inputs_func=optim_inputs_func_rmsprop,
        optim_error_inputs_func=optim_error_inputs_func_rmsprop,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfMps,  # addcdiv doesn't work for non-contiguous, see #118115
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679 and #116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                toleranceOverride(
                    {  # previously atol=5-05, rtol=0.001, https://github.com/pytorch/pytorch/issues/116202
                        torch.float32: tol(atol=5e-04, rtol=0.01),
                    }
                ),
                "TestOptimRenewed",
                "test_mixed_device_dtype",
                active_if=TEST_WITH_TORCHDYNAMO,
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679"
                ),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        Rprop,
        optim_inputs_func=optim_inputs_func_rprop,
        optim_error_inputs_func=optim_error_inputs_func_rprop,
        supported_impls=("foreach", "differentiable"),
        skips=(
            DecorateInfo(
                skipIfMps,  # Rprop doesn't update for non-contiguous, see #118117
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
                active_if=lambda kwargs: not kwargs["contiguous"],
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679"
                ),
                "TestOptimRenewed",
                "test_foreach_matches_forloop",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679 and #116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "See https://github.com/pytorch/pytorch/issues/115679"
                ),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        SGD,
        optim_inputs_func=optim_inputs_func_sgd,
        optim_error_inputs_func=optim_error_inputs_func_sgd,
        supported_impls=("foreach", "differentiable", "fused"),
        supports_sparse_on=("cpu", "cuda"),
        skips=(
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Dynamo memory usage is flaky, see https://github.com/pytorch/pytorch/issues/116046"
                ),
                "TestOptimRenewed",
                "test_peak_memory_foreach",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors w/ Global state changed, see https://github.com/pytorch/pytorch/issues/116028"
                ),
                "TestOptimRenewed",
                "test_set_default_dtype_works_with_foreach",
            ),
            DecorateInfo(
                toleranceOverride(
                    {  # previously atol=5-05, rtol=0.001, https://github.com/pytorch/pytorch/issues/116202
                        torch.float32: tol(atol=5e-04, rtol=0.007),
                    }
                ),
                "TestOptimRenewed",
                "test_mixed_device_dtype",
                active_if=TEST_WITH_TORCHDYNAMO,
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors with list out of range, see https://github.com/pytorch/pytorch/issues/116061"
                ),
                "TestOptimRenewed",
                "test_step_is_noop_for_zero_grads",
                device_type="cpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "No closure handling, https://github.com/pytorch/pytorch/issues/116494"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors with list out of range, see https://github.com/pytorch/pytorch/issues/116061"
                ),
                "TestOptimRenewed",
                "test_param_groups_weight_decay",
                device_type="cpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors with list out of range, see https://github.com/pytorch/pytorch/issues/116061"
                ),
                "TestOptimRenewed",
                "test_param_groups_lr",
                device_type="cpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "Errors with list out of range, see https://github.com/pytorch/pytorch/issues/116061"
                ),
                "TestOptimRenewed",
                "test_load_nontensor_step",
                device_type="cpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "momentum_buffer inconsistency, https://github.com/pytorch/pytorch/issues/117147"
                ),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo(
                    "fails, https://github.com/pytorch/pytorch/issues/117165"
                ),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
    OptimizerInfo(
        SparseAdam,
        optim_inputs_func=optim_inputs_func_sparseadam,
        optim_error_inputs_func=optim_error_inputs_func_sparseadam,
        supported_impls=(),
        only_supports_sparse_grads=True,
        skips=(
            DecorateInfo(
                skipIfMps,  # SparseAdam does not support MPS
                "TestOptimRenewed",
            ),
            DecorateInfo(
                unittest.skip(
                    "SparseAdam does not support dense gradients, see #116507"
                ),
                "TestOptimRenewed",
                "test_state_dict_deterministic",
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),
                "TestOptimRenewed",
                "test_param_groups_lr",
            ),
            DecorateInfo(
                unittest.skip(
                    "SparseAdam does not support dense gradients, see #116507"
                ),
                "TestOptimRenewed",
                "test_can_load_older_state_dict",
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),
                "TestOptimRenewed",
                "test_load_nontensor_step",
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction",
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),
                "TestOptimRenewed",
                "test_forloop_goes_right_direction_multigpu",
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),
                "TestOptimRenewed",
                "test_state_dict_with_cuda_params",
            ),
            DecorateInfo(
                skipIfTorchDynamo("cannot call to_sparse on p.grad, see #117184"),
                "TestOptimRenewed",
                "test_deepcopy_copies_all_public_attrs",
            ),
        ),
    ),
]
