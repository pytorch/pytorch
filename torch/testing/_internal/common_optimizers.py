import functools
import itertools
import math
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
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_utils import (
    _TestParametrizer,
    set_single_threaded_if_parallel_tbb,
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
        self.params = (
            params  # Here, params can be a list of Tensors OR param_groups as well.
        )
        self.kwargs = kwargs
        self.desc = desc


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
        # Function to generate optimizer inputs
        optim_inputs_func,
        # Implementation specific kwargs the optimizer supports, e.g., fused, foreach, differentiable
        # We consider capturable to be a base constructor flag since it is implemented across the board.
        supported_impl_kwargs: Tuple[str] = ("foreach", "differentiable"),
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
        self.supported_impl_kwargs = supported_impl_kwargs
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


# ----------------------------------------------------------------------------------------------------------------
# NOTE: The following optim_inputs_func_* sampling functions only return constructor combinations of NON-IMPLEMENTATION
# -CHANGING flags, i.e., flags that are not foreach, fused, or differentiable. The idea is that
# OptimizerInput.kwargs is editable, and these implementation flags can be added to kwargs during testing.


def optim_inputs_func_adadelta(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(
            params=params, kwargs={"lr": 0.01}, desc="non-default lr"
        ),  # TODO: Move out to testing in param_group?
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=params, kwargs={"rho": 0.95, "weight_decay": 0.9}, desc="rho"
        ),  # TODO: Move out to testing in param_group?
    ]


def optim_error_inputs_func_adadelta(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, rho=1.1),
                desc="rho should be between 0 and 1",
            ),
            error_type=ValueError,
            error_regex="Invalid rho value: 1.1",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_adagrad(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=params,
            kwargs={"initial_accumulator_value": 0.1, "weight_decay": 0.9},
            desc="initial_accumulator_value",
        ),
        OptimizerInput(
            params=params,
            kwargs={"lr": 0.1, "lr_decay": 0.5, "weight_decay": 0.9},
            desc="lr_decay",
        ),  # TODO: Move out to testing in param_group?
    ]


def optim_error_inputs_func_adagrad(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, lr_decay=-0.5),
                desc="lr_decay must be bigger than 0",
            ),
            error_type=ValueError,
            error_regex="Invalid lr_decay value: -0.5",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


# TODO: consider tensor LR! See multi_tensor_optimizer_configs in test_optim.py --> tensor LR should work
# with all implementation code paths...
def optim_inputs_func_adam(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(params=params, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9, "amsgrad": True}, desc="amsgrad"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "amsgrad": True, "capturable": True},
            desc="capturable, amsgrad",
        ),
    ]


def optim_error_inputs_func_adam(device, dtype):
    return [
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
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_adamax(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 0.001}, desc="non-default lr"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_adamax(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, betas=(0.0, 1.0)),
                desc="beta2 should be between 0 and 1",
            ),
            error_type=ValueError,
            error_regex="Invalid beta parameter at index 1: 1.0",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_adamw(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(params=params, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9, "amsgrad": True}, desc="amsgrad"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "amsgrad": True, "capturable": True},
            desc="amsgrad, capturable",
        ),
    ]


def optim_error_inputs_func_adamw(device, dtype):
    return [
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
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_asgd(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 0.02}, desc="non-default lr"),
        OptimizerInput(params=params, kwargs={"t0": 100}, desc="t0"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(params=params, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_asgd(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, weight_decay=-0.5),
                desc="weight_decay should > 0",
            ),
            error_type=ValueError,
            error_regex="Invalid weight_decay value: -0.5",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_lbfgs(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=params, kwargs={"tolerance_grad": math.inf}, desc="tolerance_grad"
        ),
        OptimizerInput(
            params=params,
            kwargs={"line_search_fn": "strong_wolfe"},
            desc="strong_wolfe",
        ),
    ]


def optim_error_inputs_func_lbfgs(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_nadam(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 1e-3}, desc="non-default lr"),
        OptimizerInput(
            params=params,
            kwargs={"momentum_decay": 6e-3},
            desc="non-zero momentum_decay",
        ),
        OptimizerInput(params=params, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "momentum_decay": 6e-3},
            desc="weight_decay",
        ),
        OptimizerInput(
            params=params,
            kwargs={
                "weight_decay": 0.9,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
            },
            desc="decoupled_weight_decay",
        ),
    ]


def optim_error_inputs_func_nadam(device, dtype):
    return [
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
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_radam(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 2e-3}, desc="non-default lr"),
        OptimizerInput(params=params, kwargs={"eps": 1e-6}, desc="non-default eps"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "decoupled_weight_decay": True},
            desc="decoupled_weight_decay",
        ),
    ]


def optim_error_inputs_func_radam(device, dtype):
    return [
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
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_rmsprop(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 1e-3}, desc="non-default lr"),
        OptimizerInput(
            params=params, kwargs={"weight_decay": 0.9}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "centered": True},
            desc="centered",
        ),
        OptimizerInput(
            params=params,
            kwargs={"weight_decay": 0.9, "centered": True, "momentum": 0.1},
            desc="momentum",
        ),
        OptimizerInput(
            params=params,
            kwargs={
                "weight_decay": 0.9,
                "centered": True,
                "momentum": 0.1,
                "maximize": True,
            },
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_rmsprop(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, momentum=-1.0),
                desc="momentum should be between 0 and 1",
            ),
            error_type=ValueError,
            error_regex="Invalid momentum value: -1.0",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_rprop(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(params=params, kwargs={"lr": 2e-4}, desc="non-default lr"),
        OptimizerInput(
            params=params, kwargs={"etas": (0.5, 1.5)}, desc="non-default etas"
        ),
        OptimizerInput(
            params=params,
            kwargs={"step_sizes": (2e-6, 100)},
            desc="non-default step_sizes",
        ),
        OptimizerInput(params=params, kwargs={"maximize": True}, desc="maximize"),
    ]


def optim_error_inputs_func_rprop(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, etas=(1.0, 0.5)),
                desc="0 < eta1 < 1 < eta2",
            ),
            error_type=ValueError,
            error_regex="Invalid eta values: 1.0, 0.5",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_sgd(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={"lr": 1e-2}, desc="default"),
        OptimizerInput(
            params=params, kwargs={"lr": 1e-2, "momentum": 0.9}, desc="momentum"
        ),
        OptimizerInput(
            params=params,
            kwargs={"lr": 1e-2, "momentum": 0.9, "dampening": 0.5},
            desc="dampening",
        ),
        OptimizerInput(
            params=params,
            kwargs={"lr": 1e-2, "momentum": 0.9, "weight_decay": 0.9},
            desc="non-zero weight_decay",
        ),
        OptimizerInput(
            params=params,
            kwargs={"lr": 1e-2, "momentum": 0.9, "nesterov": True, "weight_decay": 0.9},
            desc="nesterov",
        ),
        OptimizerInput(
            params=params,
            kwargs={"lr": 1e-2, "weight_decay": 0.9, "maximize": True},
            desc="maximize",
        ),
    ]


def optim_error_inputs_func_sgd(device, dtype):
    return [
        ErrorOptimizerInput(
            OptimizerInput(
                params=None,
                kwargs=dict(lr=1e-2, momentum=-0.5),
                desc="momentum should be between 0 and 1",
            ),
            error_type=ValueError,
            error_regex="Invalid momentum value: -0.5",
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


def optim_inputs_func_sparseadam(
    params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]
):
    return [
        OptimizerInput(params=params, kwargs={}, desc="default"),
        OptimizerInput(
            params=params, kwargs={"lr": 0.01}, desc="non-default lr"
        ),  # TODO: Move out to testing in param_group?
        OptimizerInput(params=params, kwargs={"maximize": True}, desc="maximize"),
    ]


def optim_error_inputs_func_sparseadam(device, dtype):
    return [
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
                    torch.zeros(3, layout=torch.sparse_coo, device=device, dtype=dtype)
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
                                3, layout=torch.sparse_coo, device=device, dtype=dtype
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
        ErrorOptimizerInput(
            OptimizerInput(
                params=Parameter(torch.randn(1, device=device, dtype=dtype)),
                kwargs={},
                desc="invalid param type",
            ),
            error_type=TypeError,
            error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
        ),
    ]


# Database of OptimizerInfo entries in alphabetical order.
optim_db: List[OptimizerInfo] = [
    OptimizerInfo(
        Adadelta,
        optim_inputs_func=optim_inputs_func_adadelta,
        optim_error_inputs_func=optim_error_inputs_func_adadelta,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        Adagrad,
        optim_inputs_func=optim_inputs_func_adagrad,
        optim_error_inputs_func=optim_error_inputs_func_adagrad,
        supported_impl_kwargs=("foreach", "differentiable"),
        supports_sparse_on=("cpu"),
    ),
    OptimizerInfo(
        Adam,
        optim_inputs_func=optim_inputs_func_adam,
        optim_error_inputs_func=optim_error_inputs_func_adam,
        supported_impl_kwargs=("foreach", "differentiable", "fused"),
    ),
    OptimizerInfo(
        Adamax,
        optim_inputs_func=optim_inputs_func_adamax,
        optim_error_inputs_func=optim_error_inputs_func_adamax,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        AdamW,
        optim_inputs_func=optim_inputs_func_adamw,
        optim_error_inputs_func=optim_error_inputs_func_adamw,
        supported_impl_kwargs=("foreach", "differentiable", "fused"),
    ),
    OptimizerInfo(
        ASGD,
        optim_inputs_func=optim_inputs_func_asgd,
        optim_error_inputs_func=optim_error_inputs_func_asgd,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        LBFGS,
        optim_inputs_func=optim_inputs_func_lbfgs,
        optim_error_inputs_func=optim_error_inputs_func_lbfgs,
        supported_impl_kwargs=(),
        step_requires_closure=True,
        supports_param_groups=False,
        supports_multiple_devices=False,
    ),
    OptimizerInfo(
        NAdam,
        optim_inputs_func=optim_inputs_func_nadam,
        optim_error_inputs_func=optim_error_inputs_func_nadam,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        RAdam,
        optim_inputs_func=optim_inputs_func_radam,
        optim_error_inputs_func=optim_error_inputs_func_radam,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        RMSprop,
        optim_inputs_func=optim_inputs_func_rmsprop,
        optim_error_inputs_func=optim_error_inputs_func_rmsprop,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        Rprop,
        optim_inputs_func=optim_inputs_func_rprop,
        optim_error_inputs_func=optim_error_inputs_func_rprop,
        supported_impl_kwargs=("foreach", "differentiable"),
    ),
    OptimizerInfo(
        SGD,
        optim_inputs_func=optim_inputs_func_sgd,
        optim_error_inputs_func=optim_error_inputs_func_sgd,
        supported_impl_kwargs=("foreach", "differentiable"),
        supports_sparse_on=("cpu", "cuda"),
    ),
    OptimizerInfo(
        SparseAdam,
        optim_inputs_func=optim_inputs_func_sparseadam,
        optim_error_inputs_func=optim_error_inputs_func_sparseadam,
        supported_impl_kwargs=(),
        only_supports_sparse_grads=True,
    ),
]
