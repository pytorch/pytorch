import torch
import unittest
from copy import deepcopy
from functools import wraps, partial
from itertools import chain
import itertools
import torch.nn.functional as F
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import floating_types
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _update_param_kwargs, skipIf, skipCUDAIfCudnnVersionLessThan, skipCUDAIfRocm, precisionOverride)
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import (
    freeze_rng_state, set_single_threaded_if_parallel_tbb, GRADCHECK_NONDET_TOL)
from torch.testing._internal.common_methods_invocations import DecorateInfo
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

    def __init__(self, module_info_list, allowed_dtypes=None):
        self.module_info_list = module_info_list
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError('The @modules decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        for module_info in self.module_info_list:
            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = module_info.name.replace('.', '_')

            dtypes = set(module_info.dtypes)
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            for dtype in dtypes:
                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)

                try:
                    active_decorators = [set_single_threaded_if_parallel_tbb]
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
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 supports_gradgrad=True,  # whether the op supports second order gradients
                 gradcheck_nondet_tol=0.0,  # tolerance for nondeterminism while performing gradcheck
                 ):
        self.module_cls = module_cls
        self.module_inputs_func = module_inputs_func
        self.skips = skips
        self.decorators = decorators
        self.dtypes = dtypes
        self.supports_gradgrad = supports_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol

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
                    forward_input=FunctionInput(input=make_input((4, 10))),
                    reference_fn=lambda m, p, input: torch.mm(input, p[0].t()) + p[1].view(1, -1).expand(4, 8)),
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
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_none', {'reduction': 'none'}),
        ('ignore_index', {'ignore_index': 2}),
        ('weights', {'weight': make_weight(10).abs()}),
        ('weights_ignore_index', {'weight': make_weight(10).abs(), 'ignore_index': 2}),
        ('weights_ignore_index_neg', {'weight': make_weight(10).abs(), 'ignore_index': -1})
    ]

    # TODO: Uncomment when negative weights is supported.
    # negative_weight = make_weight(10)
    # negative_weight[0] = -1
    # cases.append(('weights_negative', {'weight': negative_weight}))
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


def no_batch_dim_reference_mha(m, p, *args, **kwargs):
    """Reference function for MultiheadAttention supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    batch_dim = 0 if kwargs.get('batch_first', True) else 1
    if 'batch_first' in kwargs:
        kwargs.pop('batch_first')
    if 'key_padding_mask' in kwargs and kwargs['key_padding_mask'] is not None:
        kwargs['key_padding_mask'] = kwargs['key_padding_mask'].unsqueeze(0)
    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), output[1].squeeze(0))


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


def module_inputs_torch_nn_AdaptiveAvgPool2d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input(shape=(1, 3, 5, 6))),
                    desc='single')]


def module_inputs_torch_nn_BatchNorm2d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input(shape=(2, 3, 6, 6))))]


def module_inputs_torch_nn_BatchNorm3d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input(shape=(2, 3, 4, 4, 4))))]


def module_inputs_torch_nn_Conv2d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3, 4, 3),
                    forward_input=FunctionInput(make_input(shape=(2, 3, 7, 5))))]


def module_inputs_torch_nn_Conv3d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(2, 3, (2, 3, 2)),
                    forward_input=FunctionInput(make_input(shape=(1, 2, 4, 5, 4))))]


def module_inputs_torch_nn_ConvTranspose2d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3, 4, 3, (3, 2), 1, (1, 1)),
                    forward_input=FunctionInput(make_input(shape=(1, 3, 7, 6))))]


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
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(shape=(2, 3, 2, 5))),
                    desc='4d_input')]


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

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='channels_last_mem_format'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                    desc='channels_last_3d_mem_format')]


def module_inputs_torch_nn_Embedding(module_info, device, dtype, requires_grad, **kwargs):
    make_empty = partial(torch.empty, device=device, dtype=torch.long, requires_grad=False)
    return [
        ModuleInput(
            constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3),
            forward_input=FunctionInput(make_empty(2, 3).random_(4))
        ),
        ModuleInput(
            constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3),
            forward_input=FunctionInput(make_empty(1, 512).random_(4).expand(7, 512)),
            desc='discontiguous'
        ),
    ]


def module_inputs_torch_nn_Hardswish(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(shape=4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(shape=(2, 3, 2, 5))),
            desc='4d_input')
    ]


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


def module_inputs_torch_nn_MaxPool2d(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),
            forward_input=FunctionInput(make_input(shape=((3, 7, 7)))),
            desc='3d_input'),
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),
            forward_input=FunctionInput(make_input(shape=(1, 3, 7, 7))),
            desc='4d_input'),
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1), return_indices=True),
            forward_input=FunctionInput(make_input(shape=(1, 3, 7, 7))),
            desc='return_indices'),
    ]


def module_inputs_torch_nn_Sigmoid(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(shape=(2, 3, 4, 5))),
            desc='channels_last_mem_format'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(shape=(2, 3, 3, 4, 5))),
            desc='channels_last_3d_mem_format'
        )
    ]


def module_inputs_torch_nn_TransformerEncoderLayer(module_info, device, dtype, requires_grad, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),
            forward_input=FunctionInput(
                make_input(shape=(2, 3, 4))
            ),
            desc='relu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),
            forward_input=FunctionInput(
                make_input(shape=(2, 3, 4))
            ),
            desc='gelu_activation'
        ),
    ]


def module_inputs_torch_nn_MultiheadAttention(module_info, device, dtype, requires_grad, **kwargs):
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = []
    bool_vals = (True, False)
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3, 3)))
    products = itertools.product(bool_vals, bool_vals, bool_vals, key_padding_masks, attn_masks)
    for bias, add_bias_kv, add_zero_attn, key_padding_mask, attn_mask in products:
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=True,
                                                bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn),
                forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)),
                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask),
                reference_fn=no_batch_dim_reference_mha,
            )
        )
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(embed_dim=3, num_heads=3, batch_first=False,
                                                bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn),
                forward_input=FunctionInput(make_input((3, 3)), make_input((3, 3)), make_input((3, 3)),
                                            key_padding_mask=key_padding_mask, attn_mask=attn_mask),
                reference_fn=partial(no_batch_dim_reference_mha, batch_first=False),
            )
        )

    return samples


# Database of ModuleInfo entries in alphabetical order.
module_db: List[ModuleInfo] = [
    ModuleInfo(torch.nn.AdaptiveAvgPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool2d),
    ModuleInfo(torch.nn.AvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AvgPool1d,
               skips=(
                   # No channels_last support for AvgPool1d as it does not take 4D inputs
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.BatchNorm2d,
               module_inputs_func=module_inputs_torch_nn_BatchNorm2d,
               decorators=(
                   # Failure on ROCM for BatchNorm2d float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),)
               ),
    ModuleInfo(torch.nn.BatchNorm3d,
               module_inputs_func=module_inputs_torch_nn_BatchNorm3d,
               decorators=(
                   # Failure on ROCM for BatchNorm3d float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),)
               ),
    ModuleInfo(torch.nn.Conv2d,
               module_inputs_func=module_inputs_torch_nn_Conv2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # NHWC is disabled for float64 input in CudNN Conv.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', dtypes=[torch.float64]),
                   # No channels_last support for Conv2d on cpu currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='cpu'),),
               decorators=(
                   # Conv2d channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for Conv2d float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'))
               ),
    ModuleInfo(torch.nn.Conv3d,
               module_inputs_func=module_inputs_torch_nn_Conv3d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # NHWC is disabled for float64 input in CudNN Conv.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', dtypes=[torch.float64]),
                   # No channels_last support for Conv3d on cpu currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='cpu'),
                   # Greatest difference was 0.05072784423828125  > atol of 0.05
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),),
               decorators=(
                   # Conv3d channels_last support on cuda requires cudnn >= 8005
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for Conv3d float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]))
               ),
    ModuleInfo(torch.nn.ConvTranspose2d,
               module_inputs_func=module_inputs_torch_nn_ConvTranspose2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               skips=(
                   # NHWC is disabled for float64 input in CudNN Conv.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', dtypes=[torch.float64]),
                   # No channels_last support for ConvTranspose2d on cpu currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format', device_type='cpu'),),
               decorators=(
                   # ConvTranspose2d channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for ConvTranspose2d float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]))
               ),
    ModuleInfo(torch.nn.ELU,
               module_inputs_func=module_inputs_torch_nn_ELU),
    ModuleInfo(torch.nn.Embedding,
               module_inputs_func=module_inputs_torch_nn_Embedding,
               skips=(
                   # No channels_last support for Embedding.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Hardswish,
               module_inputs_func=module_inputs_torch_nn_Hardswish,
               supports_gradgrad=False),
    ModuleInfo(torch.nn.L1Loss,
               module_inputs_func=module_inputs_torch_nn_L1Loss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Linear,
               module_inputs_func=module_inputs_torch_nn_Linear,
               skips=(
                   # No channels_last support for Linear currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.MaxPool2d,
               module_inputs_func=module_inputs_torch_nn_MaxPool2d,
               skips=(
                   # TODO: test_non_contiguous_tensors doesn't handle case where output is not a singleton (such as
                   # return_indices=True for MaxPool2D), submit fix
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_non_contiguous_tensors'),
                   # TODO: test_cpu_gpu_parity doesn't handle case where output is not a singleton, submit fix
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),)
               ),
    ModuleInfo(torch.nn.MultiheadAttention,
               module_inputs_func=module_inputs_torch_nn_MultiheadAttention,
               skips=(
                   # No channels_last support for MultiheadAttention currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.NLLLoss,
               module_inputs_func=module_inputs_torch_nn_NLLLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.ReLU,
               module_inputs_func=module_inputs_torch_nn_ReLU),
    ModuleInfo(torch.nn.Sigmoid,
               module_inputs_func=module_inputs_torch_nn_Sigmoid),
    ModuleInfo(torch.nn.TransformerEncoderLayer,
               module_inputs_func=module_inputs_torch_nn_TransformerEncoderLayer,
               supports_gradgrad=False,
               skips=(
                   # No channels_last support for TransformerEncoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
]
