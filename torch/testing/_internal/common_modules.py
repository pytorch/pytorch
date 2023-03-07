import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import floating_types, floating_and_complex_types_and
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _update_param_kwargs, toleranceOverride, tol,
    skipCUDAIfCudnnVersionLessThan, skipCUDAIfRocm, precisionOverride, skipMeta, skipCUDAVersionIn)
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import (
    freeze_rng_state, set_single_threaded_if_parallel_tbb, skipIfMps, GRADCHECK_NONDET_TOL, TEST_WITH_ROCM)
from types import ModuleType
from typing import List, Tuple, Type, Set, Dict

# List of all namespaces containing modules to test.
MODULE_NAMESPACES: List[ModuleType] = [
    torch.nn.modules,
    torch.ao.nn.qat.modules,
    torch.ao.nn.quantizable.modules,
    torch.ao.nn.quantized.modules,
    torch.ao.nn.quantized.modules,
]

# Modules that shouldn't be tested for one reason or another.
MODULES_TO_SKIP: Set[Type] = {
    torch.nn.Module,  # abstract base class
    torch.nn.Container,  # deprecated
    torch.nn.NLLLoss2d,  # deprecated
    torch.ao.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
    torch.ao.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
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

        # Deal with any aliases by preferring earlier names.
        if module_cls not in MODULE_CLASS_NAMES:
            MODULE_CLASS_NAMES[module_cls] = f'{namespace_name}.{module_name}'


# Specifies the modes (i.e. train, eval) to test over.
TrainEvalMode = Enum('TrainEvalMode', ('train_only', 'eval_only', 'train_and_eval'))


class modules(_TestParametrizer):
    """ PROTOTYPE: Decorator for specifying a list of modules over which to run a test. """

    def __init__(self, module_info_iterable, allowed_dtypes=None, train_eval_mode=TrainEvalMode.train_and_eval):
        self.module_info_list = list(module_info_iterable)
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None
        self.train_eval_mode = train_eval_mode

    def _get_training_flags(self, module_info):
        training_flags = []
        if (self.train_eval_mode == TrainEvalMode.train_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(True)

        if (self.train_eval_mode == TrainEvalMode.eval_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(False)

        # If train and eval modes don't differ for the module, don't bother using more than one.
        if not module_info.train_and_eval_differ:
            training_flags = training_flags[:1]

        return training_flags

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError('The @modules decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        for module_info in self.module_info_list:
            dtypes = set(module_info.dtypes)
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            training_flags = self._get_training_flags(module_info)
            for (training, dtype) in product(training_flags, dtypes):
                # Construct the test name; device / dtype parts are handled outside.
                # See [Note: device and dtype suffix placement]
                test_name = module_info.formatted_name
                if len(training_flags) > 1:
                    test_name += f"_{'train_mode' if training else 'eval_mode'}"

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                _update_param_kwargs(param_kwargs, 'training', training)

                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorator_fn = partial(module_info.get_decorators, generic_cls.__name__,
                                           test.__name__, device_cls.device_type, dtype)

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print("Failed to instantiate {0} for module {1}!".format(test_name, module_info.name))
                    raise ex


def get_module_common_name(module_cls):
    if module_cls in MODULE_CLASS_NAMES:
        # Example: "nn.Linear"
        return MODULE_CLASS_NAMES[module_cls]
    else:
        return module_cls.__name__


class FunctionInput:
    """ Contains args and kwargs to pass as input to a function. """
    __slots__ = ['args', 'kwargs']

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ModuleInput:
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


class ModuleInfo:
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
                 module_memformat_affects_out=False,  # whether converting module to channels last will generate
                                                      # channels last output
                 train_and_eval_differ=False,  # whether the module has differing behavior between train and eval
                 ):
        self.module_cls = module_cls
        self.module_inputs_func = module_inputs_func
        self.decorators = (*(decorators if decorators else []), *(skips if skips else []))
        self.dtypes = dtypes
        self.supports_gradgrad = supports_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.module_memformat_affects_out = module_memformat_affects_out
        self.train_and_eval_differ = train_and_eval_differ

    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        result = [set_single_threaded_if_parallel_tbb]
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(test_class, test_name, device, dtype, param_kwargs):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    @property
    def name(self):
        return get_module_common_name(self.module_cls)

    @property
    def formatted_name(self):
        return self.name.replace('.', '_')


def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, training, **kwargs):
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


def module_inputs_torch_nn_Bilinear(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def bilinear_reference_fn(m, p, x1, x2, bias=True):
        result = torch.einsum('bn,anm,bm->ba', x1, p[0], x2)
        if bias:
            if x1.shape[0] == 1:
                result = result.view(-1) + p[1]
            else:
                result = result + p[1].view(1, -1).expand(x1.shape[0], p[0].shape[0])
        return result

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(2, 3, 4),
                    forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1, x2)),
        ModuleInput(constructor_input=FunctionInput(2, 3, 4, bias=False),
                    forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),
                    desc='no_bias',
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1, x2, bias=False)),
        ModuleInput(constructor_input=FunctionInput(2, 3, 4),
                    forward_input=FunctionInput(make_input((2)), make_input((3))),
                    desc='no_batch_dim',
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1.view(1, -1), x2.view(1, -1))),
    ]

    return module_inputs


def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    def make_input(shape, device=device, dtype=dtype, requires_grad=requires_grad):
        return make_tensor(shape, device=device, dtype=dtype,
                           requires_grad=False).log_softmax(dim=1).requires_grad_(requires_grad)
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
                        forward_input=FunctionInput(make_input((15, 10)),
                                                    torch.empty(15, device=device).uniform_().mul(10).floor().long()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_GaussianNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: List[Tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((3)),
                                                    make_target((3)),
                                                    make_input((1)).abs()),
                        desc=desc,
                        reference_fn=no_batch_dim_reference_fn)
        )

    return module_inputs


def no_batch_dim_reference_fn(m, p, *args, **kwargs):
    """Reference function for modules supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.

    Currently it only supports modules which return a single Tensor as output.
    You can bind the following kwargs.
    Kwargs:
        batch_first[bool] : If True, all the Tensors in `args` while be unsqueezed at dim `0` .
                        and output will be squeezed at dim `0` else dim `1` for both.
        kwargs_to_batchify[dict] : Dictionary specifying the name of the argument and dimension to unsqueeze.
                               Useful if there are few arguments whose batch dimension are different
                               from the ones selected by `batch_first`.
        is_criterion[bool] : Specify if the module is a criterion and handle the reduction for output accordingly.
    """
    def get_and_pop(key, default):
        v = kwargs.get(key, default)
        if key in kwargs:
            kwargs.pop(key)
        return v

    batch_dim = 0 if get_and_pop('batch_first', True) else 1
    kwargs_to_batchify = get_and_pop('kwargs_to_batchify', None)
    is_criterion = get_and_pop('is_criterion', False)

    if kwargs_to_batchify is not None:
        assert isinstance(kwargs_to_batchify, dict)
        for k, v in kwargs.items():
            if k in kwargs_to_batchify and v is not None:
                bdim = kwargs_to_batchify[k]
                kwargs[k] = v.unsqueeze(bdim)

    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs).squeeze(batch_dim)

    if is_criterion:
        reduction = get_reduction(m)
        if reduction == 'none':
            return output.squeeze(0)
    return output


def no_batch_dim_reference_mha(m, p, *args, **kwargs):
    """Reference function for MultiheadAttention supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
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


def no_batch_dim_reference_rnn_gru(m, p, *args, **kwargs):
    """Reference function for RNN and GRU supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = h.unsqueeze(1)

    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), output[1].squeeze(1))


def no_batch_dim_reference_lstm(m, p, *args, **kwargs):
    """Reference function for LSTM supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = (h[0].unsqueeze(1), h[1].unsqueeze(1))

    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), (output[1][0].squeeze(1), output[1][1].squeeze(1)))


def no_batch_dim_reference_lstmcell(m, p, *args, **kwargs):
    """Reference function for LSTMCell supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    inp, (h, c) = args
    single_batch_input_args = (inp.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(0), output[1].squeeze(0))


def generate_regression_criterion_inputs(make_input):
    return [
        ModuleInput(
            constructor_input=FunctionInput(reduction=reduction),
            forward_input=FunctionInput(make_input((4, )), make_input(4,)),
            reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True),
            desc='no_batch_dim_{}'.format(reduction)
        ) for reduction in ['none', 'mean', 'sum']]


def module_inputs_torch_nn_AvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(kernel_size=2),
                    forward_input=FunctionInput(make_input((3, 6))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_AdaptiveAvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single')]


def module_inputs_torch_nn_BatchNorm2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))))]


def module_inputs_torch_nn_BatchNorm3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))))]


def module_inputs_torch_nn_ConvNd(module_info, device, dtype, requires_grad, training, **kwargs):
    N = kwargs['N']
    lazy = kwargs.get('lazy', False)
    transposed = kwargs.get('transposed', False)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    conv_kwargs_list = [{}] if transposed else [{}, {'padding': 'same'}]
    kernel_size, C_in, C_out = 3, 4, 5
    input_no_batch_shape = (C_in,) + tuple((i + 3 for i in range(N)))
    input_batch_shape = (2,) + input_no_batch_shape
    return [
        ModuleInput(constructor_input=(FunctionInput(C_out, kernel_size, **conv_kwargs) if lazy else
                                       FunctionInput(C_in, C_out, kernel_size, **conv_kwargs)),
                    forward_input=FunctionInput(make_input(
                        input_batch_shape if with_batch else input_no_batch_shape)),
                    desc=('' if with_batch else 'no_batch_dim'),
                    reference_fn=(None if with_batch else no_batch_dim_reference_fn))
        for with_batch, conv_kwargs in itertools.product([True, False], conv_kwargs_list)
    ]


def module_inputs_torch_nn_ELU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((2, 3, 2, 5))),
                    desc='4d_input')]


def module_inputs_torch_nn_CELU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_ReLU(module_info, device, dtype, requires_grad, training, **kwargs):
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


def module_inputs_torch_nn_L1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4)),
                                                make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * sum((a - b).abs().sum()
                                                                         for a, b in zip(i, t))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(()), make_input(())),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * (i - t).abs().sum(),
                    desc='scalar')] + generate_regression_criterion_inputs(make_input)


def module_inputs_torch_nn_CrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    reductions = ['sum', 'mean', 'none']
    samples = []
    # Samples below are for validating the no-batch-dim support.
    for reduction in reductions:
        samples.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction),
                        forward_input=FunctionInput(make_input((9,)), make_target((), low=0, high=9)),
                        reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True))
        )
        samples.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, weight=make_weight((9,))),
                        forward_input=FunctionInput(make_input((9,)), make_target((), low=0, high=9)),
                        reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True))
        )
        samples.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, label_smoothing=0.5),
                        forward_input=FunctionInput(make_input((9,)), make_target((), low=0, high=9)),
                        reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True))
        )
        samples.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, label_smoothing=0.5,
                                                        weight=make_weight((9,))),
                        forward_input=FunctionInput(make_input((9,)), make_target((), low=0, high=9)),
                        reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True))
        )

    return samples


def module_inputs_torch_nn_Hardswish(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 2, 5))),
            desc='4d_input')
    ]


def module_inputs_torch_nn_MaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),
            forward_input=FunctionInput(make_input(((3, 7, 7)))),
            desc='3d_input'),
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1)),
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            desc='4d_input'),
        ModuleInput(
            constructor_input=FunctionInput((3, 3), (2, 2), (1, 1), return_indices=True),
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            desc='return_indices'),
    ]


def module_inputs_torch_nn_Sigmoid(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 4, 5))),
            desc='channels_last_mem_format'
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
            desc='channels_last_3d_mem_format'
        )
    ]


def module_inputs_torch_nn_TransformerEncoder(module_info, device, dtype, requires_grad, training, **kwargs):
    # Reuse the TransformerEncoderLayer samples since the forward args are nearly the same.
    for layer_module_input in module_inputs_torch_nn_TransformerEncoderLayer(
            None, device, dtype, requires_grad, training):
        # Construct a TransformerEncoderLayer object to pass to TransformerEncoder.
        l_args, l_kwargs = (layer_module_input.constructor_input.args,
                            layer_module_input.constructor_input.kwargs)
        encoder_layer = torch.nn.TransformerEncoderLayer(*l_args, **l_kwargs)
        num_layers = 2
        # Note: TransformerEncoderLayer takes a "src_mask" while
        # TransformerEncoder takes a "mask"; rename kwarg appropriately.
        forward_input = layer_module_input.forward_input
        if 'src_mask' in forward_input.kwargs:
            forward_input.kwargs['mask'] = forward_input.kwargs['src_mask']
            del forward_input.kwargs['src_mask']
        yield ModuleInput(
            constructor_input=FunctionInput(encoder_layer, num_layers),
            forward_input=forward_input,
            desc=layer_module_input.desc
        )

def module_inputs_torch_nn_TransformerEncoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    samples = [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),
            forward_input=FunctionInput(
                make_input((2, 3, 4))
            ),
            desc='relu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),
            forward_input=FunctionInput(
                make_input((2, 3, 4))
            ),
            desc='gelu_activation'
        ), ]

    # Samples below are for validating the no-batch-dim support.
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for src_mask, src_key_padding_mask, norm_first in itertools.product(attn_masks, key_padding_masks, (True, False)):
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=True, norm_first=norm_first),
                forward_input=FunctionInput(
                    make_input((3, 4)), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=True, kwargs_to_batchify={'src_key_padding_mask': 0}),
                desc='no_batch_dim_batch_first'
            ))

        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(4, 2, 8, dropout=0.0, batch_first=False, norm_first=norm_first),
                forward_input=FunctionInput(
                    make_input((3, 4)), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=False, kwargs_to_batchify={'src_key_padding_mask': 0}),
                desc='no_batch_dim'
            ))

    def fast_path_reference_fn(module, parameters, *args, **kwargs):
        assert not module.training
        module = module.train(True)
        output = module(*args, **kwargs)
        module = module.train(False)
        return output

    if not training:
        for norm_first in (True, False):
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(4, 2, 8, dropout=0.0, batch_first=True, norm_first=norm_first),
                    forward_input=FunctionInput(
                        make_input((2, 3, 4)),
                    ),
                    reference_fn=fast_path_reference_fn,
                    desc="fast_path_norm_first" if norm_first else "fast_path"
                )
            )

    return samples


def module_inputs_torch_nn_TransformerDecoderLayer(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    samples = [
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 16, 0.0),
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))
            ),
            desc='relu_activation'
        ),
        ModuleInput(
            constructor_input=FunctionInput(4, 2, 8, 0.0, F.gelu),
            forward_input=FunctionInput(
                make_input((2, 3, 4)), make_input((2, 3, 4))
            ),
            desc='gelu_activation'
        ), ]

    # Samples below are for validating the no-batch-dim support.
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for tgt_mask, tgt_key_padding_mask, norm_first in itertools.product(attn_masks, key_padding_masks, (True, False)):
        # Using same mask for tgt and memory
        memory_mask = tgt_mask
        memory_key_padding_mask = tgt_key_padding_mask
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                dropout=0.0, batch_first=True, norm_first=norm_first),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=True,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'memory_key_padding_mask': 0}),
                desc='no_batch_dim_batch_first'
            ))

        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(4, 2, 8, dropout=0.0, batch_first=False, norm_first=norm_first),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=False,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'memory_key_padding_mask': 0}),
                desc='no_batch_dim'
            ))

    return samples


def module_inputs_torch_nn_Transformer(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = []
    # Samples below are for validating the no-batch-dim support.
    key_padding_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool))
    attn_masks = (None, torch.tensor([False, False, True], device=device, dtype=torch.bool).expand((3, 3)))
    for mask, key_padding_mask, norm_first in itertools.product(attn_masks, key_padding_masks, (True, False)):
        # Using same mask for tgt and memory
        src_mask , tgt_mask = (mask,) * 2
        src_key_padding_mask, tgt_key_padding_mask = (key_padding_mask,) * 2
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                num_encoder_layers=1, num_decoder_layers=1,
                                                dropout=0.0, batch_first=True, norm_first=norm_first),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, src_mask=src_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=True,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'src_key_padding_mask': 0}),
                desc='no_batch_dim_batch_first'
            ))

        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(d_model=4, nhead=2, dim_feedforward=8,
                                                num_encoder_layers=1, num_decoder_layers=1,
                                                dropout=0.0, batch_first=False, norm_first=norm_first),
                forward_input=FunctionInput(
                    make_input((3, 4)), make_input((3, 4)), tgt_mask=tgt_mask, src_mask=src_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask
                ),
                reference_fn=partial(no_batch_dim_reference_fn,
                                     batch_first=False,
                                     kwargs_to_batchify={'tgt_key_padding_mask': 0, 'src_key_padding_mask': 0}),
                desc='no_batch_dim'
            ))

    return samples


def module_inputs_torch_nn_Embedding(module_info, device, dtype, requires_grad, training, **kwargs):
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


def module_inputs_torch_nn_MultiheadAttention(module_info, device, dtype, requires_grad, training, **kwargs):
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


def module_inputs_torch_nn_RNN_GRU_Cell(module_info, device, dtype, requires_grad, training, **kwargs):
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [
        ModuleInput(
            constructor_input=FunctionInput(5, 10),
            forward_input=FunctionInput(make_input(5), make_input(10)),
            reference_fn=no_batch_dim_reference_fn,
        ),
        ModuleInput(
            constructor_input=FunctionInput(5, 10, bias=True),
            forward_input=FunctionInput(make_input(5), make_input(10)),
            reference_fn=no_batch_dim_reference_fn,
        )
    ]

    is_rnn = kwargs.get('is_rnn', False)
    if is_rnn:
        # RNN also supports `nonlinearity` argument.
        # `tanh` is the default, so we check with `relu`
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(5, 10, bias=True, nonlinearity='relu'),
                forward_input=FunctionInput(make_input(5), make_input(10)),
                reference_fn=no_batch_dim_reference_fn,
            )
        )

    return samples


def module_inputs_torch_nn_LSTMCell(module_info, device, dtype, requires_grad, training, **kwargs):
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = (
        ModuleInput(
            constructor_input=FunctionInput(5, 10),
            forward_input=FunctionInput(make_input(5), (make_input(10), make_input(10))),
            reference_fn=no_batch_dim_reference_lstmcell,
        ),
        ModuleInput(
            constructor_input=FunctionInput(5, 10, bias=True),
            forward_input=FunctionInput(make_input(5), (make_input(10), make_input(10))),
            reference_fn=no_batch_dim_reference_lstmcell,
        ),
    )

    return samples

def make_packed_sequence(inp, batch_sizes):
    required_grad = inp.requires_grad
    inp.requires_grad_(False)  # user won't have access to inp so won't be able to get its grads
    seq = pack_padded_sequence(inp, batch_sizes)
    seq.data.requires_grad_(required_grad)
    return seq


def module_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, with_packed_sequence=False, **kwargs):
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    is_rnn = kwargs['is_rnn']
    nonlinearity = ('relu', 'tanh')
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)

    samples = []
    if is_rnn:
        prod_gen = product(nonlinearity, bias, batch_first, bidirectional)
    else:
        prod_gen = product(bias, batch_first, bidirectional)

    for args in prod_gen:
        if is_rnn:
            nl, b, b_f, bidir = args
        else:
            b, b_f, bidir = args

        cons_args = {'input_size': 2, 'hidden_size': 2, 'num_layers': 2,
                     'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': 3, 'num_layers': 2,
                            'batch_first': b_f, 'bias': b, 'bidirectional': bidir}

        if is_rnn:
            cons_args['nonlinearity'] = nl
            cons_args_hidden['nonlinearity'] = nl
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args),
                forward_input=FunctionInput(make_input((3, 2))),
                reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
            )
        )
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args_hidden),
                forward_input=FunctionInput(make_input((3, 2)), make_input((4 if bidir else 2, 3))),
                reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
            )
        )
        if with_packed_sequence:
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(**cons_args),
                    forward_input=FunctionInput(make_packed_sequence(make_input((5, 2, 2)), torch.tensor([5, 3]))),
                    reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
                )
            )
            samples.append(
                ModuleInput(
                    constructor_input=FunctionInput(**cons_args),
                    forward_input=FunctionInput(make_packed_sequence(make_input((5, 5, 2)), torch.tensor([5, 3, 3, 2, 2]))),
                    reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f),
                )
            )

    return samples


def module_inputs_torch_nn_LSTM(module_info, device, dtype, requires_grad, training, **kwargs):
    # Currently all samples below are for validating the no-batch-dim support.
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)
    proj_sizes = (0, 2)

    samples = []
    prod_gen = product(bias, batch_first, bidirectional, proj_sizes)

    for args in prod_gen:
        b, b_f, bidir, proj_size = args
        hidden_size = 3
        cons_args = {'input_size': 2, 'hidden_size': hidden_size, 'num_layers': 2, 'proj_size': proj_size,
                     'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': hidden_size, 'num_layers': 2, 'proj_size': proj_size,
                            'batch_first': b_f, 'bias': b, 'bidirectional': bidir}

        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args),
                forward_input=FunctionInput(make_input((2, 2))),
                reference_fn=partial(no_batch_dim_reference_lstm, batch_first=b_f),
            )
        )

        h_out = proj_size if proj_size > 0 else hidden_size
        hx = (make_input((4 if bidir else 2, h_out)), make_input((4 if bidir else 2, hidden_size)))
        samples.append(
            ModuleInput(
                constructor_input=FunctionInput(**cons_args_hidden),
                forward_input=FunctionInput(make_input((3, 2)), hx),
                reference_fn=partial(no_batch_dim_reference_lstm, batch_first=b_f),
            )
        )


    return samples


# All these operators share similar issues on cuDNN and MIOpen
rnn_gru_lstm_module_info_decorators = (
    # RuntimeError: Batching rule not implemented for aten::_cudnn_rnn_backward.
    # We could not generate a fallback
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_grad",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # NotImplementedError: the derivative for '_cudnn_rnn_backward' is not implemented.
    # Double backwards is not supported for CuDNN RNNs due to limitations in the CuDNN API
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_gradgrad",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # CUDNN GRU doesn't accept non-contiguous hx
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_non_contiguous_tensors",
        active_if=(TEST_CUDNN and not TEST_WITH_ROCM), device_type='cuda'
    ),
    # MIOPEN GRU doesn't accept non-contiguous hx (this is dispatched to miopen only for float).
    DecorateInfo(
        unittest.expectedFailure, "TestModule", "test_non_contiguous_tensors",
        active_if=(TEST_CUDNN and TEST_WITH_ROCM), dtypes=(torch.float,), device_type='cuda'
    ),
    DecorateInfo(
        skipCUDAVersionIn([(11, 7)]), "TestExpandedWeightModule", "test_module",
        device_type='cuda'
    ),
    DecorateInfo(
        skipCUDAVersionIn([(11, 7)]), "TestDecomp", "test_rnn_decomp_module",
        device_type='cuda'
    )
)

# Database of ModuleInfo entries in alphabetical order.
module_db: List[ModuleInfo] = [
    ModuleInfo(torch.nn.AdaptiveAvgPool2d,
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_inputs_func=module_inputs_torch_nn_AdaptiveAvgPool2d,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.AvgPool1d,
               module_inputs_func=module_inputs_torch_nn_AvgPool1d,
               skips=(
                   # No channels_last support for AvgPool1d as it does not take 4D inputs
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.BatchNorm2d,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_BatchNorm2d,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.BatchNorm3d,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_BatchNorm3d,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.Conv1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=False),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64])
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Conv2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=False),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='cuda', dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Conv3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=False),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 8005
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=False, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               dtypes=floating_and_complex_types_and(torch.chalf),
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # Not implmented for chalf on CPU
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_forward',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule',
                                'test_if_train_and_eval_modes_differ', dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_non_contiguous_tensors',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_multiple_device_transfer',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   # Ref: https://github.com/pytorch/pytorch/issues/73502
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_pickle', dtypes=(torch.chalf,)),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=False, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               dtypes=floating_and_complex_types_and(torch.chalf),
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.float64, torch.complex128]),
                   # These fail only on ROCm
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.complex32], active_if=TEST_WITH_ROCM),
                   # Not implmented for chalf on CPU
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_forward',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule',
                                'test_if_train_and_eval_modes_differ', dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_non_contiguous_tensors',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_multiple_device_transfer',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   # Ref: https://github.com/pytorch/pytorch/issues/73502
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_pickle', dtypes=(torch.chalf,)),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.ConvTranspose3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=False, transposed=True),
               dtypes=floating_and_complex_types_and(torch.chalf),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 8005
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
                   # These fail only on ROCm
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.complex32, torch.complex64], active_if=TEST_WITH_ROCM),
                   # Not implmented for chalf on CPU
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_forward',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_memory_format',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule',
                                'test_if_train_and_eval_modes_differ', dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_non_contiguous_tensors',
                                dtypes=(torch.chalf,), device_type='cpu'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_cpu_gpu_parity',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_multiple_device_transfer',
                                dtypes=(torch.chalf,), device_type='cuda'),
                   # Ref: https://github.com/pytorch/pytorch/issues/73502
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_pickle', dtypes=(torch.chalf,)),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
                   DecorateInfo(precisionOverride({torch.complex64: 1e-04}), 'TestModule', 'test_cpu_gpu_parity'),
               )),
    ModuleInfo(torch.nn.ELU,
               module_inputs_func=module_inputs_torch_nn_ELU,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.L1Loss,
               module_inputs_func=module_inputs_torch_nn_L1Loss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.LazyConv1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConv2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format",
                                device_type='cuda', dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConv3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 8005
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose1d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=1, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose2d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=2, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 7603
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=7603), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format", device_type='cuda',
                                dtypes=[torch.float64]),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.LazyConvTranspose3d,
               module_inputs_func=partial(module_inputs_torch_nn_ConvNd, N=3, lazy=True, transposed=True),
               gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
               module_memformat_affects_out=True,
               skips=(
                   # channels_last support on cuda requires cudnn >= 8005
                   DecorateInfo(skipCUDAIfCudnnVersionLessThan(version=8005), 'TestModule', 'test_memory_format'),
                   # Failure on ROCM for float32 issue #70125
                   DecorateInfo(skipCUDAIfRocm, 'TestModule', 'test_memory_format', dtypes=[torch.float32]),
                   # Lazy modules don't currently play well with ModuleInfo tests on the meta device.
                   # See https://github.com/pytorch/pytorch/issues/70505 for more info.
                   DecorateInfo(skipMeta),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # This was wrongly being skipped before and needs investigation.
                   # See https://github.com/pytorch/pytorch/issues/80247
                   DecorateInfo(unittest.expectedFailure, "TestModule", "test_memory_format"),
               ),
               decorators=(
                   DecorateInfo(precisionOverride({torch.float32: 1e-04}), 'TestModule', 'test_memory_format'),
               )),
    ModuleInfo(torch.nn.Linear,
               module_inputs_func=module_inputs_torch_nn_Linear,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # No channels_last support for Linear currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.Bilinear,
               module_inputs_func=module_inputs_torch_nn_Bilinear,
               decorators=[
                   DecorateInfo(
                       toleranceOverride({
                           torch.float32: tol(atol=1e-4, rtol=1e-4),
                           torch.float64: tol(atol=1e-4, rtol=1e-4)}),
                       'TestModule', 'test_forward', device_type='cpu')
               ],
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   # No channels_last support for Bilinear currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)
               ),
    ModuleInfo(torch.nn.MaxPool2d,
               module_inputs_func=module_inputs_torch_nn_MaxPool2d,
               skips=(
                   # TODO: test_non_contiguous_tensors doesn't handle case where output is not a singleton (such as
                   # return_indices=True for MaxPool2D), submit fix
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_non_contiguous_tensors'),
                   # TODO: test_cpu_gpu_parity doesn't handle case where output is not a singleton, submit fix
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_cpu_gpu_parity'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.NLLLoss,
               module_inputs_func=module_inputs_torch_nn_NLLLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.GaussianNLLLoss,
               module_inputs_func=module_inputs_torch_nn_GaussianNLLLoss,
               skips=(
                   # No channels_last support for loss functions.
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),)),
    ModuleInfo(torch.nn.CrossEntropyLoss,
               module_inputs_func=module_inputs_torch_nn_CrossEntropyLoss,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.Hardswish,
               module_inputs_func=module_inputs_torch_nn_Hardswish,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),),
               supports_gradgrad=False),
    # TransformerEncoder takes the same inputs as TransformerEncoderLayer
    ModuleInfo(torch.nn.TransformerEncoder,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_TransformerEncoder,
               skips=(
                   # No channels_last support for TransformerEncoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   # Doesn't support device / dtype kwargs directly because it is just a
                   # container of TransformerEncoderLayers.
                   DecorateInfo(unittest.expectedFailure, 'TestModule', 'test_factory_kwargs'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.TransformerEncoderLayer,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_TransformerEncoderLayer,
               skips=(
                   # No channels_last support for TransformerEncoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.TransformerDecoderLayer,
               module_inputs_func=module_inputs_torch_nn_TransformerDecoderLayer,
               skips=(
                   # No channels_last support for TransformerDecoderLayer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.Transformer,
               module_inputs_func=module_inputs_torch_nn_Transformer,
               skips=(
                   # No channels_last support for Transformer currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.MultiheadAttention,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_MultiheadAttention,
               skips=(
                   # No channels_last support for MultiheadAttention currently.
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.Embedding,
               module_inputs_func=module_inputs_torch_nn_Embedding,
               skips=(
                   DecorateInfo(unittest.skip("Skipped!"), 'TestModule', 'test_memory_format'),
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.ReLU,
               module_inputs_func=module_inputs_torch_nn_ReLU,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.RNNCell,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU_Cell, is_rnn=True),
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.GRUCell,
               module_inputs_func=module_inputs_torch_nn_RNN_GRU_Cell,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.LSTMCell,
               module_inputs_func=module_inputs_torch_nn_LSTMCell,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.Sigmoid,
               module_inputs_func=module_inputs_torch_nn_Sigmoid,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),)
               ),
    ModuleInfo(torch.nn.RNN,
               train_and_eval_differ=True,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU, is_rnn=True),
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),),
               decorators=rnn_gru_lstm_module_info_decorators
               ),
    ModuleInfo(torch.nn.GRU,
               train_and_eval_differ=True,
               module_inputs_func=partial(module_inputs_torch_nn_RNN_GRU, is_rnn=False),
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),),
               decorators=rnn_gru_lstm_module_info_decorators),
    ModuleInfo(torch.nn.LSTM,
               train_and_eval_differ=True,
               module_inputs_func=module_inputs_torch_nn_LSTM,
               skips=(
                   DecorateInfo(skipIfMps, 'TestModule', dtypes=[torch.float64]),),
               decorators=rnn_gru_lstm_module_info_decorators)
]
