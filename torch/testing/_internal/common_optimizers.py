from torch.optim import (
    Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD, SparseAdam, Optimizer
)
from torch import Tensor, Parameter
from torch.testing._internal.common_device_type import _TestParametrizer
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_utils import set_single_threaded_if_parallel_tbb
from typing import Callable, List, Tuple, Dict, Union, Any
import functools
import math

class OptimizerInput:
    """ Contains args / kwargs to be passed to an optimizer constructor. """
    __slots__ = ['params', 'kwargs', 'desc']

    def __init__(self, params: Union[List[Parameter], List[Tensor], Dict[Any, Any]], kwargs: Dict[str, Any], desc:str = ''):
        self.params = params  # Here, params can be a list of Tensors OR param_groups as well.
        self.kwargs = kwargs
        self.desc = desc


class OptimizerInfo:
    """ Optimizer information to be used in testing. """

    def __init__(self,
                 optim_cls: Optimizer,  # Class object for the Optimizer under test
                 *,
                 # Function to generate optimizer inputs
                 optim_inputs_func,
                 # Implementation specific kwargs the optimizer supports, e.g., fused, foreach, capturable
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
        self.decorators = (*(decorators if decorators else []), *(skips if skips else []))
        self.optim_error_inputs_func = optim_error_inputs_func  # TODO: make these for each optim, look at test_errors in test_modules.py

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
            # Actually, we don't have dtypes because optimizers should be able to handle differing
            # dtypes among params--though maybe "mixed" could just be an option or something.
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
# NOTE: The following optim_inputs_func_* sampling functions only return constructor combinations of NON-IMPLEMENTATION
# -CHANGING flags, i.e., flags that are not foreach, fused, capturable or differentiable. The idea is that
# OptimizerInput.kwargs is editable, and these implementation flags can be added to kwargs during testing.

def optim_inputs_func_adadelta(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'), 
        OptimizerInput(params=params, kwargs={'lr': 0.01}, desc='non-default lr'),  # TODO: Move out to testing in param_group?
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
        OptimizerInput(params=params, kwargs={'rho': 0.95, 'weight_decay': 0.9}, desc='rho'),  # TODO: Move out to testing in param_group?
    ]

def optim_inputs_func_adagrad(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
        OptimizerInput(params=params, kwargs={'initial_accumulator_value': 0.1, 'weight_decay': 0.9}, desc='initial_accumulator_value'),
        OptimizerInput(params=params, kwargs={'lr': 0.1, 'lr_decay': 0.5, 'weight_decay': 0.9}, desc='lr_decay'),  # TODO: Move out to testing in param_group?
    ]


def optim_inputs_func_adam(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 0.01}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'amsgrad': True}, desc='amsgrad'),
    ]


def optim_inputs_func_adamax(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 0.001}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
    ]


def optim_inputs_func_adamw(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 0.01}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'amsgrad': True}, desc='amsgrad'),
    ]


def optim_inputs_func_asgd(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 0.02}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'t0': 100}, desc='t0'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
    ]


def optim_inputs_func_lbfgs(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 0.01}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'tolerance_grad': math.inf}, desc='tolerance_grad'),
        OptimizerInput(params=params, kwargs={'line_search_fn': "strong_wolfe"}, desc='strong_wolfe'),
    ]


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_nadam(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 1e-3}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'momentum_decay': 6e-3}, desc='non-zero momentum_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'momentum_decay': 6e-3}, desc='weight_decay'),
    ]


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_radam(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 2e-3}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
    ]


def optim_inputs_func_rmsprop(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 1e-3}, desc='non-default lr')
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9}, desc='nonzero weight_decay'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'centered': True}, desc='centered'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'centered': True, 'momentum': 0.1}, desc='momentum'),
        OptimizerInput(params=params, kwargs={'weight_decay': 0.9, 'centered': True, 'momentum': 0.1, 'maximize': True}, desc='maximize'),
    ]


def optim_inputs_func_rprop(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 2e-4}, desc='non-default lr'),
        OptimizerInput(params=params, kwargs={'etas': (1.5, 0.5)}, desc='etas'),
        OptimizerInput(params=params, kwargs={'maximize': True}, desc='maximize'),
    ]


def optim_inputs_func_sgd(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={'lr': 1e-2}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 1e-2, 'momentum': 0.5}, desc='momentum'),
        OptimizerInput(params=params, kwargs={'lr': 1e-2, 'momentum': 0.5, 'weight_decay': 0.9}, desc='non-zero weight_decay'),
        OptimizerInput(params=params, kwargs={'lr': 1e-2, 'momentum': 0.5, 'nesterov': True, 'weight_decay': 0.9}, desc='nesterov'),
        OptimizerInput(params=params, kwargs={'lr': 1e-2, 'weight_decay': 0.9, 'maximize': True}, desc='maximize'),
    ]


def optim_inputs_func_sparseadam(params: Union[List[Parameter], List[Tensor], Dict[Any, Any]]):
    return [
        OptimizerInput(params=params, kwargs={}, desc='default'),
        OptimizerInput(params=params, kwargs={'lr': 0.01}, desc='non-default lr'),  # TODO: Move out to testing in param_group?
        OptimizerInput(params=params, kwargs={'maximize': True}, desc='maximize'),
    ]


# Database of OptimizerInfo entries in alphabetical order.
optim_db: List[OptimizerInfo] = [
    OptimizerInfo(
        Adadelta,
        optim_inputs_func_func=optim_inputs_func_adadelta,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        Adagrad,
        optim_inputs_func_func=optim_inputs_func_adagrad,
        supported_impl_kwargs=('foreach', 'differentiable'),
        supports_sparse_on=('cpu'),
    ),
    OptimizerInfo(
        Adam,
        optim_inputs_func_func=optim_inputs_func_adam,
        supported_impl_kwargs=('foreach', 'differentiable', 'fused', 'capturable'),
    ),
    OptimizerInfo(
        Adamax,
        optim_inputs_func_func=optim_inputs_func_adamax,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        AdamW,
        optim_inputs_func_func=optim_inputs_func_adamw,
        supported_impl_kwargs=('foreach', 'differentiable', 'fused', 'capturable'),
    ),
    OptimizerInfo(
        ASGD,
        optim_inputs_func_func=optim_inputs_func_asgd,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        LBFGS,
        optim_inputs_func_func=optim_inputs_func_lbfgs,
        supported_impl_kwargs=(),
        step_requires_closure=True,
        supports_param_groups=False,
        supports_multiple_devices=False,
    ),
    OptimizerInfo(
        NAdam,
        optim_inputs_func_func=optim_inputs_func_nadam,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        RAdam,
        optim_inputs_func_func=optim_inputs_func_radam,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        RMSprop,
        optim_inputs_func_func=optim_inputs_func_rmsprop,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        Rprop,
        optim_inputs_func_func=optim_inputs_func_rprop,
        supported_impl_kwargs=('foreach', 'differentiable'),
    ),
    OptimizerInfo(
        SGD,
        optim_inputs_func_func=optim_inputs_func_sgd,
        supported_impl_kwargs=('foreach', 'differentiable'),
        supports_sparse_on=('cpu', 'cuda'),
    ),
    OptimizerInfo(
        SparseAdam,
        optim_inputs_func_func=optim_inputs_func_sparseadam,
        supported_impl_kwargs=(),
        only_supports_sparse_grads=True,
    )
]