import torch
from copy import deepcopy
from functools import wraps, partial
from itertools import chain
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _dtype_test_suffix, _update_param_kwargs, skipIf)
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import freeze_rng_state
from types import ModuleType
from typing import List, Tuple, Type, Set, Dict


# List of all namespaces containing modules to test.
MODULE_NAMESPACES: List[ModuleType] = [
    torch.nn.modules,
    torch.nn.qat.modules,
    torch.nn.quantizable.modules,
    torch.nn.quantized.modules,
]

# Modules that shouldn't be tested for one reason or another.
MODULES_TO_SKIP: Set[Type] = {
    torch.nn.Module,  # abstract base class
    torch.nn.Container,  # deprecated
    torch.nn.NLLLoss2d,  # deprecated
    torch.nn.quantized.modules._ConvNd,  # abstract base class
    torch.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
}

# List of all module classes to test.
MODULE_CLASSES: List[Type] = list(chain(*[
    [getattr(namespace, module_name) for module_name in namespace.__all__]  # type: ignore[attr-defined]
    for namespace in MODULE_NAMESPACES]))
MODULE_CLASSES = [cls for cls in MODULE_CLASSES if cls not in MODULES_TO_SKIP]

# Dict of module class -> common name. Useful for making test names more intuitive.
# Example: torch.nn.modules.linear.Linear -> "nn.Linear"
MODULE_CLASS_NAMES: Dict[Type, str] = {}
for namespace in MODULE_NAMESPACES:
    for module_name in namespace.__all__:  # type: ignore[attr-defined]
        module_cls = getattr(namespace, module_name)
        namespace_name = namespace.__name__.replace('torch.', '').replace('.modules', '')
        MODULE_CLASS_NAMES[module_cls] = f'{namespace_name}.{module_name}'


class modules(_TestParametrizer):
    """ PROTOTYPE: Decorator for specifying a list of modules over which to run a test. """

    def __init__(self, module_info_list):
        super().__init__(handles_dtypes=True)
        self.module_info_list = module_info_list

    def _parametrize_test(self, test, generic_cls, device_cls):
        for module_info in self.module_info_list:
            # TODO: Factor some of this out since it's similar to OpInfo.
            for dtype in floating_types():
                # Construct the test name.
                test_name = '{}_{}{}'.format(module_info.name.replace('.', '_'),
                                             device_cls.device_type,
                                             _dtype_test_suffix(dtype))

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)

                try:
                    active_decorators = []
                    if module_info.should_skip(generic_cls.__name__, test.__name__, device_cls.device_type, dtype):
                        active_decorators.append(skipIf(True, "Skipped!"))

                    if module_info.decorators is not None:
                        for decorator in module_info.decorators:
                            # Can't use isinstance as it would cause a circular import
                            if decorator.__class__.__name__ == 'DecorateInfo':
                                if decorator.is_active(generic_cls.__name__, test.__name__,
                                                       device_cls.device_type, dtype):
                                    active_decorators += decorator.decorators
                            else:
                                active_decorators.append(decorator)

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    for decorator in active_decorators:
                        test_wrapper = decorator(test_wrapper)

                    yield (test_wrapper, test_name, param_kwargs)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print("Failed to instantiate {0} for module {1}!".format(test_name, module_info.name))
                    raise ex


def formatted_module_name(module_cls):
    """ Returns the common name of the module class formatted for use in test names. """
    return MODULE_CLASS_NAMES[module_cls].replace('.', '_')


class FunctionInput(object):
    """ Contains args and kwargs to pass as input to a function. """
    __slots__ = ['args', 'kwargs']

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ModuleInput(object):
    """ Contains args / kwargs for module instantiation + forward pass. """
    __slots__ = ['constructor_input', 'forward_input', 'desc', 'reference_fn']

    def __init__(self, constructor_input, forward_input=None, desc='', reference_fn=None):
        self.constructor_input = constructor_input  # Inputs to pass during construction
        self.forward_input = forward_input  # Inputs to pass to forward()
        self.desc = desc  # Description for this set of inputs
        self.reference_fn = reference_fn  # Reference with signature: reference_fn(module, parameters, *args, **kwargs)

        if reference_fn is not None:

            @wraps(reference_fn)
            def copy_reference_fn(m, *args, **kwargs):
                # Copy inputs to avoid undesired side effects from calling the reference.
                args, kwargs = deepcopy(args), deepcopy(kwargs)

                # Note that module parameters are passed in for convenience.
                return reference_fn(m, list(m.parameters()), *args, **kwargs)

            self.reference_fn = copy_reference_fn


class ModuleInfo(object):
    """ Module information to be used in testing. """

    def __init__(self,
                 module_cls,  # Class object for the module under test
                 *,
                 module_inputs_func,  # Function to generate module inputs
                 skips=(),  # Indicates which tests to skip
                 decorators=None,  # Additional decorators to apply to generated tests
                 ):
        self.module_cls = module_cls
        self.module_inputs_func = module_inputs_func
        self.skips = skips
        self.decorators = decorators

    def should_skip(self, cls_name, test_name, device_type, dtype):
        return any(si.is_active(cls_name, test_name, device_type, dtype) for si in self.skips)

    @property
    def name(self):
        return formatted_module_name(self.module_cls)

    @property
    def formatted_name(self):
        return self.name.replace('.', '_')


def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(10, 8),
                    forward_input=FunctionInput(make_input((4, 10))),
                    reference_fn=lambda m, p, i: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8)),
        ModuleInput(constructor_input=FunctionInput(10, 8, bias=False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='no_bias',
                    reference_fn=lambda m, p, i: torch.mm(i, p[0].t())),
        ModuleInput(constructor_input=FunctionInput(3, 5),
                    forward_input=FunctionInput(make_input(3)),
                    desc='no_batch_dim',
                    reference_fn=lambda m, p, i: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1])
    ]

    return module_inputs


def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('ignore_index', {'ignore_index': 2}),
        ('weights', {'weight': make_input(10)}),
        ('weights_ignore_index', {'weight': make_input(10), 'ignore_index': 2}),
        ('weights_ignore_index_neg', {'weight': make_input(10), 'ignore_index': -1})
    ]
    module_inputs = []
    for desc, constructor_kwargs in cases:

        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return nllloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10)).log_softmax(dim=1),
                                                    torch.empty(15, device=device).uniform_().mul(10).floor().long()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def no_batch_dim_reference_fn(m, p, *args, **kwargs):
    """Reference function for modules supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    single_batch_input_args = [input.unsqueeze(0) for input in args]
    with freeze_rng_state():
        return m(*single_batch_input_args).squeeze(0)


def no_batch_dim_reference_criterion_fn(m, *args, **kwargs):
    """Reference function for criterion supporting no batch dimensions."""
    output = no_batch_dim_reference_fn(m, *args, **kwargs)
    reduction = get_reduction(m)
    if reduction == 'none':
        return output.squeeze(0)
    # reduction is 'sum' or 'mean' which results in a 0D tensor
    return output


def generate_regression_criterion_inputs(make_input):
    return [
        ModuleInput(
            constructor_input=FunctionInput(reduction=reduction),
            forward_input=FunctionInput(make_input(shape=(4, )), make_input(shape=4,)),
            reference_fn=no_batch_dim_reference_criterion_fn,
            desc='no_batch_dim_{}'.format(reduction)
        ) for reduction in ['none', 'mean', 'sum']]


def module_inputs_torch_nn_AvgPool1d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(kernel_size=2),
                    forward_input=FunctionInput(make_input(shape=(3, 6))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_ELU(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(shape=(3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(shape=())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(shape=(3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_CELU(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(shape=(3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(shape=())),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(shape=(3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]

def module_inputs_torch_nn_ReLU(module_info, device, dtype, requires_grad):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    desc='no_batch_dim'),
    ]
    return module_inputs


def module_inputs_torch_nn_L1Loss(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(shape=(2, 3, 4)),
                                                make_input(shape=(2, 3, 4))),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * sum((a - b).abs().sum()
                                                                         for a, b in zip(i, t))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(shape=()), make_input(shape=())),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * (i - t).abs().sum(),
                    desc='scalar')] + generate_regression_criterion_inputs(make_input)


# Database of ModuleInfo entries in alphabetical order.
module_db: List[ModuleInfo] = [
    ModuleInfo(torch.nn.AvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AvgPool1d),
    ModuleInfo(torch.nn.ELU,
               module_inputs_func=module_inputs_torch_nn_ELU),
    ModuleInfo(torch.nn.L1Loss,
               module_inputs_func=module_inputs_torch_nn_L1Loss),
    ModuleInfo(torch.nn.Linear,
               module_inputs_func=module_inputs_torch_nn_Linear),
    ModuleInfo(torch.nn.NLLLoss,
               module_inputs_func=module_inputs_torch_nn_NLLLoss),
    ModuleInfo(torch.nn.ReLU,
               module_inputs_func=module_inputs_torch_nn_ReLU),
]
