from abc import abstractmethod
import tempfile
import unittest

from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi


import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
    gradcheck, gradgradcheck
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn

from typing import Dict, Callable, Tuple, List, Sequence, Union, Any

TemporaryFile = tempfile.TemporaryFile
PRECISION = 1e-5


def get_reduction(m):
    result = getattr(m, 'reduction', None)
    if result is None:
        result = _Reduction.legacy_get_string(getattr(m, 'sizeAverage', None), True, emit_warning=False)
    assert result is not None
    return result


def get_weight(m):
    result = getattr(m, 'weight', None)
    if result is not None:
        return result
    return getattr(m, 'weights', None)

# NOTE [How to check NN module / functional API parity between Python and C++ frontends]
#
# The way to check API parity is to add parity tests for the NN module / functional of interest.
# Here are the detailed steps:
#
# For NN module:
# 1. Make sure you already have a test dict with the module configuration you want to test.
# 2. Add `cpp_constructor_args` entry to the test dict, with its value exactly matching
#    the Python module constructor arguments. For example, if in the test dict we pass
#    `(10, 8)` to `torch.nn.Linear` constructor, then we should pass `torch::nn::LinearOptions(10, 8)`
#    as the corresponding C++ constructor argument to `torch::nn::Linear`.
# 3. If in the process of performing the above step you referenced any variables
#    in the `cpp_constructor_args` entry, you must add `cpp_var_map` entry
#    to the test dict to make sure that those variables are populated with the right Python values.
#    For example, if the Python constructor call is
#    `torch.nn.FractionalMaxPool2d(2, output_ratio=0.5, _random_samples=random_samples)`,
#    the corresponding C++ constructor argument is
#    `torch::nn::FractionalMaxPool2dOptions(2).output_ratio(0.5)._random_samples(random_samples)`,
#    and the `cpp_var_map` entry must be
#    `{'random_samples': random_samples}` in order to populate the C++ variable `random_samples`
#    used in the C++ constructor argument with the Python tensor value `random_samples`.
#
# For NN functional:
# 1. Make sure you already have a test dict with the functional configuration you want to test.
# 2. If the test dict's `constructor` entry looks like `wrap_functional(F.some_functional_name, ...)`,
#    then you must add `cpp_options_args` entry to the test dict, with its value exactly matching the Python
#    functional optional arguments. For example, if the test dict's `constructor` entry is
#    `wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest')`,
#    then the `cpp_options_args` entry should be
#    "F::InterpolateFuncOptions().size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)".
# 3. Otherwise, if the test dict's `constructor` entry looks like
#    `wrap_functional(lambda i: F.some_functional_name(...))`,
#    then you must add `cpp_function_call` entry to the test dict, with its value exactly matching the Python
#    functional function call. For example, if the test dict's `constructor` entry is
#    `wrap_functional(lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none'))`,
#    then the `cpp_function_call` entry should be
#    "F::poisson_nll_loss(i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))".
# 4. If in the process of performing the above two steps you referenced any variables
#    in the `cpp_options_args` or `cpp_function_call` entry, you must
#    add `cpp_var_map` entry to the test dict to make sure that those variables
#    are populated with the right Python values. For example, if the test dict's `constructor` entry is
#    `wrap_functional(lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none'))`,
#    then the `cpp_function_call` entry should be
#    "F::poisson_nll_loss(i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))".
#    Notice that there are two variables `i` and `t` that need to have their values provided,
#    and the way to do so is to add a `cpp_var_map` entry: `cpp_var_map={'i': '_get_input()', 't': t}`.
#    (Note that for `i`, since we want it to take the Python input value, we pass '_get_input()' string as value
#    and the C++ parity test mechanism will populate `i` with the Python input value correctly.)
#
# There are also a few optional flags in the test dict to control the C++ parity test behavior:
#
# - `test_cpp_api_parity`: if `False`, skips the C++ parity test for this test dict. Default: True.
# - `has_parity`: if `False`, expects this test dict to fail the C++ parity test. Default: True.


module_tests = [
    dict(
        module_name='Linear',
        constructor_args=(10, 8),
        cpp_constructor_args='torch::nn::LinearOptions(10, 8)',
        input_size=(4, 10),
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Linear',
        constructor_args=(10, 8, False),
        cpp_constructor_args='torch::nn::LinearOptions(10, 8).bias(false)',
        input_size=(4, 10),
        desc='no_bias',
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='RReLU',
        input_size=(1, 2, 2),
        test_cuda=False,
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)',
        input_size=(4, 4, 5),
        desc='with_up_down',
        test_cuda=False,
    ),
    dict(
        module_name='Flatten',
        input_size=(2, 3, 4, 5),
        reference_fn=lambda i, *_: torch.flatten(i, 1)
    ),
    # TODO: reference function
    dict(
        module_name='CrossMapLRN2d',
        constructor_args=(5, 5e-3, 1e-3, 2),
        cpp_constructor_args='torch::nn::CrossMapLRN2dOptions(5).alpha(5e-3).beta(1e-3).k(2)',
        input_size=(2, 3, 6, 6),
        check_gradgrad=False,
        # TODO(#50743): Figure out the error. "RuntimeError: Unrecognized tensor type ID: Batched"
        check_batched_grad=False,
    ),
]


# Generates rand tensor with non-equal values. This ensures that duplicate
# values won't be causing test failure for modules like MaxPooling.
# size should be small, otherwise randperm fails / long overflows.
def _rand_tensor_non_equal(*size):
    total = reduce(mul, size, 1)
    return torch.randperm(total).view(*size).double()


def wrap_functional(fn, **kwargs):
    class FunctionalModule(nn.Module):
        def forward(self, *args):
            return fn(*args, **kwargs)
    return FunctionalModule


def poissonnllloss_no_reduce_test():
    t = torch.randn(10, 10)
    return dict(
        fullname='PoissonNLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::poisson_nll_loss('
                          'i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(10, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: i.exp() - t.mul(i),
        pickle=False)


def bceloss_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).to(torch.get_default_dtype()))
    return dict(
        fullname='BCELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False,
        precision=7e-4)


def bceloss_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).to(torch.get_default_dtype())
    return dict(
        fullname='BCELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False)


def bceloss_weights_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).to(torch.get_default_dtype()))
    weights = torch.rand(10)
    return dict(
        fullname='BCELoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), '
                          'F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        reference_fn=lambda i, p, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False,
        precision=3e-4
    )


def bceloss_weights_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).to(torch.get_default_dtype())
    weights = torch.rand(())
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy(
            i, t.to(i.options()),
            F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False
    )


def bce_with_logistic_legacy_enum_test():
    t = Variable(torch.randn(15, 10).gt(0).to(torch.get_default_dtype()))
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_legacy_enum',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
    )


def bce_with_logistic_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).to(torch.get_default_dtype()))
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
    )


def bce_with_logistic_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).to(torch.get_default_dtype())
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False
    )


def kldivloss_with_target_no_reduce_test():
    t = torch.rand(10, 10)
    return dict(
        fullname='KLDivLoss_with_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def kldivloss_no_reduce_test():
    t = torch.rand(10, 10)
    return dict(
        fullname='KLDivLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
    )


def kldivloss_no_reduce_scalar_test():
    t = torch.rand(())
    return dict(
        fullname='KLDivLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.rand(()).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def kldivloss_with_log_target_no_reduce_test():
    t = torch.rand(10, 10).log()
    return dict(
        fullname='KLDivLoss_with_log_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def kldivloss_no_reduce_log_target_test():
    t = torch.rand(10, 10).log()
    return dict(
        fullname='KLDivLoss_no_reduce_log_target',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        input_fn=lambda: torch.rand(10, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False,
    )


def kldivloss_no_reduce_scalar_log_target_test():
    t = torch.rand(()).log()
    return dict(
        fullname='KLDivLoss_no_reduce_scalar_log_target',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        input_fn=lambda: torch.rand(()).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def l1loss_no_reduce_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='L1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        supports_forward_ad=True,
        pickle=False)


def l1loss_no_reduce_complex_test():
    t = torch.randn(2, 3, 4, dtype=torch.cdouble)
    return dict(
        fullname='L1Loss_no_reduce_complex',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.randn(2, 3, 4, dtype=torch.cdouble),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        supports_forward_ad=True,
        pickle=False)


def l1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='L1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        supports_forward_ad=True,
        pickle=False)


def mseloss_no_reduce_test():
    input_size = (2, 3, 4, 5)
    target = torch.randn(*input_size)
    return dict(
        fullname='MSELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),
        cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))',
        input_size=input_size,
        cpp_var_map={'i': '_get_input()', 'target': target},
        reference_fn=lambda i, *_: (i - target).pow(2),
        supports_forward_ad=True,
        pickle=False)


def mseloss_no_reduce_scalar_test():
    input_size = ()
    target = torch.randn(input_size)
    return dict(
        fullname='MSELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),
        cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))',
        input_size=input_size,
        cpp_var_map={'i': '_get_input()', 'target': target},
        reference_fn=lambda i, *_: (i - target).pow(2),
        supports_forward_ad=True,
        pickle=False)


def nllloss_no_reduce_test():
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss_no_reduce_ignore_index_test():
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    kwargs: Dict[str, Union[int, str]] = {'ignore_index': 2, 'reduction': 'none'}
    return dict(
        fullname='NLLLoss_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(2).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss_no_reduce_weights_test():
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLoss_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss_no_reduce_weights_ignore_index_test():
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': 2}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i.data))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone).ignore_index(2))''',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss_no_reduce_weights_ignore_index_neg_test():
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': -1}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index_neg',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone).ignore_index(-1))''',
        input=torch.rand(15, 10).add(1e-2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss2d_no_reduce_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss2d_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs: Dict[str, Union[int, str]] = {'ignore_index': 1, 'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(1).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss2d_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLoss2d_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nlllossNd_no_reduce_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLossNd_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nlllossNd_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs: Dict[str, Union[int, str]] = {'ignore_index': 1, 'reduction': 'none'}
    return dict(
        fullname='NLLLossNd_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(1).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nlllossNd_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLossNd_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def smoothl1loss_no_reduce_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='SmoothL1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def smoothl1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='SmoothL1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def smoothl1loss_beta_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='SmoothL1Loss_beta',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0.5)),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0.5)''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0.5),
        supports_forward_ad=True,
        pickle=False)


def smoothl1loss_zero_beta_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='SmoothL1Loss_zero_beta',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0)),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0)''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0),
        supports_forward_ad=True,
        pickle=False)


def huberloss_delta_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='HuberLoss_delta',
        constructor=wrap_functional(
            lambda i: F.huber_loss(i, t.type_as(i), reduction='none', delta=0.5)),
        cpp_function_call='''F::huber_loss(
            i, t.to(i.options()), F::HuberLossFuncOptions().reduction(torch::kNone).delta(0.5))''',
        input_fn=lambda: torch.randn(2, 3, 4),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['HuberLoss'](i, t.type_as(i), reduction='none', delta=0.5),
        supports_forward_ad=True,
        pickle=False)


def multilabelmarginloss_0d_no_reduce_test():
    t = torch.zeros(()).long()
    return dict(
        fullname='MultiLabelMarginLoss_0d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_1d_no_reduce_test():
    t = Variable(torch.rand(10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_index_neg_test():
    t = Variable(torch.clamp(torch.rand(5, 10).add(-.5).mul(20).floor().long(), min=-1))
    return dict(
        fullname='MultiLabelMarginLoss_index_neg',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def hingeembeddingloss_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).to(torch.get_default_dtype()).mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::hinge_embedding_loss(
            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), reduction='none'),
        check_sum_reduction=True,
        pickle=False)


def hingeembeddingloss_margin_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).to(torch.get_default_dtype()).mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), margin=0.5, reduction='none')),
        cpp_function_call='''F::hinge_embedding_loss(
            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().margin(0.5).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), margin=0.5, reduction='none'),
        check_sum_reduction=True,
        pickle=False)


def softmarginloss_no_reduce_test():
    t = torch.randn(5, 5)
    return dict(
        fullname='SoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.soft_margin_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::soft_margin_loss(
            i, t.to(i.options()), F::SoftMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 5),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['SoftMarginLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,
        pickle=False)


def multilabelsoftmarginloss_no_reduce_test():
    t = torch.rand(5, 10).mul(2).floor()
    return dict(
        fullname='MultiLabelSoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::multilabel_soft_margin_loss(
            i, t.to(i.options()), F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log())).sum(dim=1) / i.size(1),
        check_gradgrad=False,
        pickle=False)


def multilabelsoftmarginloss_weights_no_reduce_test():
    t = torch.rand(5, 10).mul(2).floor()
    weights = torch.rand(10)
    return dict(
        fullname='MultiLabelSoftMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i),
                                                    weight=weights.type_as(i), reduction='none')),
        cpp_function_call='''F::multilabel_soft_margin_loss(
            i, t.to(i.options()),
            F::MultilabelSoftMarginLossFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * weights).sum(dim=1) / i.size(1),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_1d_no_reduce_test():
    t = torch.rand(1).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_1d_input_0d_target_no_reduce_test():
    t = torch.rand(()).mul(8).floor().long()
    return dict(
        fullname='multimarginloss_1d_input_0d_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_p_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_p_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), p=2, reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().p(2).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10).clamp_(1e-2, 1 - 1e-2),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), p=2, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_margin_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), margin=0.5, reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::MultiMarginLossFuncOptions().margin(0.5).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  margin=0.5, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_weights_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    weights = torch.rand(10)
    return dict(
        fullname='MultiMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), weight=weights.type_as(i),
                                          reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::MultiMarginLossFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  weight=weights, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def single_batch_reference_fn(input, parameters, module):
    """Reference function for modules supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    def unsqueeze_inp(inp):
        if isinstance(inp, (list, tuple)):
            return [t.unsqueeze(0) for t in inp]
        return inp.unsqueeze(0)

    single_batch_input = unsqueeze_inp(input)
    single_batch_input = [single_batch_input] if isinstance(single_batch_input, torch.Tensor) else single_batch_input
    with freeze_rng_state():
        return module(*single_batch_input).squeeze(0)


new_module_tests = [
    poissonnllloss_no_reduce_test(),
    bceloss_no_reduce_test(),
    bceloss_weights_no_reduce_test(),
    bce_with_logistic_legacy_enum_test(),
    bce_with_logistic_no_reduce_test(),
    bceloss_no_reduce_scalar_test(),
    bceloss_weights_no_reduce_scalar_test(),
    bce_with_logistic_no_reduce_scalar_test(),
    kldivloss_with_target_no_reduce_test(),
    kldivloss_no_reduce_test(),
    kldivloss_no_reduce_scalar_test(),
    kldivloss_with_log_target_no_reduce_test(),
    kldivloss_no_reduce_log_target_test(),
    kldivloss_no_reduce_scalar_log_target_test(),
    l1loss_no_reduce_test(),
    l1loss_no_reduce_complex_test(),
    l1loss_no_reduce_scalar_test(),
    mseloss_no_reduce_test(),
    mseloss_no_reduce_scalar_test(),
    nllloss_no_reduce_test(),
    nllloss_no_reduce_ignore_index_test(),
    nllloss_no_reduce_weights_test(),
    nllloss_no_reduce_weights_ignore_index_test(),
    nllloss_no_reduce_weights_ignore_index_neg_test(),
    nllloss2d_no_reduce_test(),
    nllloss2d_no_reduce_weights_test(),
    nllloss2d_no_reduce_ignore_index_test(),
    nlllossNd_no_reduce_test(),
    nlllossNd_no_reduce_weights_test(),
    nlllossNd_no_reduce_ignore_index_test(),
    smoothl1loss_no_reduce_test(),
    smoothl1loss_no_reduce_scalar_test(),
    smoothl1loss_beta_test(),
    smoothl1loss_zero_beta_test(),
    huberloss_delta_test(),
    multilabelmarginloss_0d_no_reduce_test(),
    multilabelmarginloss_1d_no_reduce_test(),
    multilabelmarginloss_index_neg_test(),
    multilabelmarginloss_no_reduce_test(),
    hingeembeddingloss_no_reduce_test(),
    hingeembeddingloss_margin_no_reduce_test(),
    softmarginloss_no_reduce_test(),
    multilabelsoftmarginloss_no_reduce_test(),
    multilabelsoftmarginloss_weights_no_reduce_test(),
    multimarginloss_no_reduce_test(),
    multimarginloss_1d_no_reduce_test(),
    multimarginloss_1d_input_0d_target_no_reduce_test(),
    multimarginloss_p_no_reduce_test(),
    multimarginloss_margin_no_reduce_test(),
    multimarginloss_weights_no_reduce_test(),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)',
        input_size=(2, 4, 10),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3, 2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(2)',
        input_size=(2, 4, 10),
        cudnn=True,
        desc='stride',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3, 1, 1),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(1).padding(1)',
        input_size=(2, 4, 10),
        cudnn=True,
        desc='pad1',
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 5, 1, 2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 5).stride(1).padding(2)',
        input_size=(2, 4, 10),
        cudnn=True,
        desc='pad2',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 4, 3, 1, 1),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 3).stride(1).padding(1)',
        input_size=(1, 4, 1),
        cudnn=True,
        desc='pad1size1',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 4, 5, 1, 2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 5).stride(1).padding(2)',
        input_size=(1, 4, 1),
        cudnn=True,
        desc='pad2size1',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)',
        input_size=(0, 4, 10),
        cudnn=True,
        desc='zero_batch',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv1d_dilated',
        constructor=lambda: nn.Conv1d(4, 5, kernel_size=3, dilation=2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).dilation(2)',
        input_size=(2, 4, 10),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv1d_groups',
        constructor=lambda: nn.Conv1d(4, 6, kernel_size=3, groups=2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 6, 3).groups(2)',
        input_size=(2, 4, 6),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv1d_pad_valid',
        constructor=lambda: nn.Conv1d(4, 5, 3, padding="valid"),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kValid)',
        input_size=(2, 4, 10),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv1d_pad_same',
        constructor=lambda: nn.Conv1d(4, 5, 3, padding="same"),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kSame)',
        input_size=(2, 4, 10),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv1d_pad_same2',
        constructor=lambda: nn.Conv1d(4, 5, 4, padding="same"),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 4).padding(torch::kSame)',
        input_size=(2, 4, 10),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv1d_pad_same_dilated',
        constructor=lambda: nn.Conv1d(4, 5, 4, padding="same", dilation=2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kSame).dilation(2)',
        input_size=(2, 4, 10),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='ConvTranspose1d',
        constructor=lambda: nn.ConvTranspose1d(3, 4, kernel_size=3, stride=(3,), padding=1, output_padding=(1,)),
        cpp_constructor_args='torch::nn::ConvTranspose1dOptions(3, 4, 3).stride(3).padding(1).output_padding(1)',
        cudnn=True,
        input_size=(1, 3, 7),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, False),
        cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(3, 4, 3)
                                .stride(2).padding(1).output_padding(1).groups(1).bias(false)''',
        input_size=(1, 3, 6),
        cudnn=True,
        desc='no_bias',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2),
        cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(3, 4, 3)
                                .stride(2).padding(1).output_padding(1).groups(1).bias(true).dilation(2)''',
        input_size=(1, 3, 6),
        cudnn=True,
        desc='dilated',
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='ConvTranspose1d_groups',
        constructor=lambda: nn.ConvTranspose1d(4, 6, 3, stride=(3,), padding=1, output_padding=(1,), groups=2),
        cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(4, 6, 3)
                                .stride(3).padding(1).output_padding(1).groups(2)''',
        cudnn=True,
        input_size=(2, 4, 7),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2)),
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})',
        input_size=(2, 3, 7, 5),
        cudnn=True,
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2)),
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2})',
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='strided',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2}).padding({1, 1})',
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='padding',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 2, {3, 3}).stride({2, 2}).padding({1, 1}).dilation({2, 2})',
        input_size=(2, 3, 8, 8),
        cudnn=True,
        desc='dilated',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2), 1, 0, 1, 1, False),
        cpp_constructor_args='''torch::nn::Conv2dOptions(3, 4, {3, 2})
                                .stride(1).padding(0).dilation(1).groups(1).bias(false)''',
        input_size=(2, 3, 6, 5),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.015,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2)),
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})',
        input_size=(0, 3, 7, 5),
        cudnn=True,
        desc='zero_batch',
        check_with_long_tensor=True,
        with_tf32=True,
    ),
    dict(
        fullname='Conv2d_groups',
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)',
        input_size=(2, 4, 6, 5),
        cudnn=True,
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.015,
    ),
    dict(
        fullname='Conv2d_groups_thnn',
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)',
        input_size=(2, 4, 6, 5),
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.015,
    ),
    dict(
        fullname='Conv2d_pad_valid',
        constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="valid"),
        cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kValid)',
        input_size=(2, 2, 6, 5),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv2d_pad_same',
        constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="same"),
        cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kSame)',
        input_size=(2, 2, 6, 5),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        fullname='Conv2d_pad_same_dilated',
        constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="same", dilation=2),
        cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kSame).dilation(2)',
        input_size=(2, 2, 6, 5),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (3, 2), 1, (1, 1)),
        cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                .stride({3, 2}).padding(1).output_padding({1, 1})''',
        cudnn=True,
        input_size=(1, 3, 7, 6),
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)),
        cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                .stride({2, 3})
                                .padding(1)
                                .output_padding({1, 1})
                                .groups(1)
                                .bias(false)
                                .dilation({2, 2})''',
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='dilated',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False),
        cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                .stride({2, 3}).padding(1).output_padding({1, 1}).groups(1).bias(false)''',
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        fullname='ConvTranspose2d_groups',
        constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2),
        cpp_constructor_args='torch::nn::ConvTranspose2dOptions(2, 4, {2, 3}).groups(2)',
        input_size=(1, 2, 4, 5),
        cudnn=True,
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
    ),
    dict(
        fullname='Conv2d_depthwise',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), groups=4),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).groups(4)',
        input_size=(2, 4, 6, 6),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv2d_depthwise_with_multiplier',
        constructor=lambda: nn.Conv2d(4, 8, (3, 3), groups=4),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 8, {3, 3}).groups(4)',
        input_size=(2, 4, 6, 6),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv2d_depthwise_strided',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), stride=(2, 2), groups=4),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).stride({2, 2}).groups(4)',
        input_size=(2, 4, 6, 6),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv2d_depthwise_padded',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), padding=(1, 1), groups=4),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).padding({1, 1}).groups(4)',
        input_size=(2, 4, 6, 6),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv2d_depthwise_dilated',
        constructor=lambda: nn.Conv2d(4, 4, (2, 2), dilation=(2, 2), groups=4),
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {2, 2}).dilation({2, 2}).groups(4)',
        input_size=(2, 4, 5, 5),
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(2, 3, (2, 3, 2)),
        cpp_constructor_args='torch::nn::Conv3dOptions(2, 3, {2, 3, 2})',
        input_size=(1, 2, 4, 5, 4),
        cudnn=True,
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(2, 3, (2, 3, 4), 1, 0, 1, 1, False),
        cpp_constructor_args='''torch::nn::Conv3dOptions(2, 3, {2, 3, 4})
                                .stride(1).padding(0).dilation(1).groups(1).bias(false)''',
        input_size=(1, 2, 3, 4, 5),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(2, 3, (1, 1, 1), 1, 0, 1, 1, False),
        cpp_constructor_args='''torch::nn::Conv3dOptions(2, 3, {2, 3, 4})
                                .stride(1).padding(0).dilation(1).groups(1).bias(false)''',
        input_size=(1, 2, 3, 4, 5),
        cudnn=True,
        desc='1x1x1_no_bias',
        check_with_long_tensor=False,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2)',
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2, 1),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2).padding(1)',
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride_padding',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, (2, 3, 4)),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4})',
        input_size=(0, 3, 3, 4, 5),
        cudnn=True,
        check_with_long_tensor=True,
        desc='zero_batch',
        with_tf32=True,
    ),
    dict(
        fullname='Conv3d_groups',
        constructor=lambda: nn.Conv3d(2, 4, kernel_size=3, groups=2),
        cpp_constructor_args='torch::nn::Conv3dOptions(2, 4, 3).groups(2)',
        input_size=(1, 2, 4, 5, 4),
        cudnn=True,
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        fullname='Conv3d_dilated',
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2)',
        input_size=(2, 3, 5, 5, 5),
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        fullname='Conv3d_dilated_strided',
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2, stride=2),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2).stride(2)',
        input_size=(2, 3, 5, 5, 5),
        with_tf32=True,
        tf32_precision=0.05
    ),
    dict(
        fullname='Conv3d_pad_valid',
        constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="valid"),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kValid)',
        input_size=(2, 3, 6, 5, 4),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        fullname='Conv3d_pad_same',
        constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="same"),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kSame)',
        input_size=(2, 3, 6, 5, 4),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        fullname='Conv3d_pad_same_dilated',
        constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="same", dilation=2),
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kSame).dilation(2)',
        input_size=(2, 3, 6, 5, 4),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2)),
        cpp_constructor_args='torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})',
        cudnn=True,
        input_size=(1, 2, 4, 5, 4),
        with_tf32=True,
        tf32_precision=0.05
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)),
        cpp_constructor_args='''torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})
                                .stride(1).padding(0).output_padding(0).groups(1).bias(true).dilation({2, 2, 2})''',
        cudnn=True,
        input_size=(1, 2, 4, 5, 4),
        desc='dilated',
        with_tf32=True,
        tf32_precision=0.05
    ),
    dict(
        module_name='ReplicationPad3d',
        constructor_args=((1, 2, 3, 3, 2, 1),),
        cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',
        input_size=(2, 3, 2, 2, 2),
    ),
    dict(
        module_name='ReplicationPad3d',
        constructor_args=((1, 2, 3, 3, 2, 1),),
        cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',
        input_size=(3, 2, 2, 2),
        reference_fn=single_batch_reference_fn,
        desc='no_batch_dim',
    ),
    dict(
        module_name='ReplicationPad3d',
        constructor_args=((1, 2, 3, 3, 2, 1),),
        cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',
        input_fn=lambda: torch.rand(2, 3, 2, 2, 2, dtype=torch.complex128, requires_grad=True),
        skip_half=True,
        desc='complex'
    ),
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
    ),
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',
        input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),
        check_gradgrad=False,
        desc='discontiguous'
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
        desc='mean',
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',
        input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),
        check_gradgrad=False,
        desc='discontiguous',
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3, None, 2., False, 'sum'),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum)''',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
        desc='sum',
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3, None, 2., False, 'max'),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax)''',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
        desc='max',
    ),
    dict(
        fullname='EmbeddingBag_mean_padding_idx',
        constructor=lambda: nn.EmbeddingBag(4, 3, padding_idx=1),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3).padding_idx(1)',
        input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
        check_gradgrad=False,
    ),
    dict(
        fullname='EmbeddingBag_sum_padding_idx',
        constructor=lambda: nn.EmbeddingBag(4, 3, None, 2., False, 'sum', padding_idx=1),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum).padding_idx(1)''',
        input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
        check_gradgrad=False,
    ),
    dict(
        fullname='EmbeddingBag_max_padding_idx',
        constructor=lambda: nn.EmbeddingBag(4, 3, None, 2., False, 'max', padding_idx=1),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax).padding_idx(1)''',
        input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
        check_gradgrad=False,
    ),
    dict(
        fullname='EmbeddingBag_sparse',
        constructor=lambda: nn.EmbeddingBag(4, 3, sparse=True),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3).sparse(true)',
        input_fn=lambda: torch.randperm(2).repeat(1, 2),
        check_gradgrad=False,
        has_sparse_gradients=True,
    ),
    dict(
        constructor=lambda: nn.Embedding(4, 3, sparse=True),
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3).sparse(true)',
        input_fn=lambda: torch.randperm(2).repeat(1, 2),
        fullname='Embedding_sparse',
        check_gradgrad=False,
        has_sparse_gradients=True,
    ),
    dict(
        module_name='PixelShuffle',
        constructor_args=(3,),
        cpp_constructor_args='torch::nn::PixelShuffleOptions(3)',
        input_size=(1, 9, 4, 4),
    ),
    dict(
        module_name='PixelUnshuffle',
        constructor_args=(3,),
        cpp_constructor_args='torch::nn::PixelUnshuffleOptions(3)',
        input_size=(1, 1, 12, 12),
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)''',
        input_size=(1, 2, 4),
        fullname='interpolate_nearest_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)''',
        input_size=(0, 2, 4),
        fullname='interpolate_nearest_1d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)''',
        input_size=(1, 2, 3),
        fullname='interpolate_nearest_tuple_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt).scale_factor(std::vector<double>({4.})).mode(torch::kNearest)''',
        input_size=(1, 2, 4),
        fullname='interpolate_nearest_scale_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4),
        fullname='interpolate_linear_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, ), scale_factor=None, mode='linear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        input_size=(1, 2, 3),
        fullname='interpolate_linear_tuple_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4.}))
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4),
        fullname='interpolate_linear_scale_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        input_size=(0, 2, 4),
        fullname='interpolate_linear_1d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(true)''',
        input_size=(1, 2, 4),
        fullname='interpolate_linear_1d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4.}))
                            .mode(torch::kLinear)
                            .align_corners(true)''',
        input_size=(1, 2, 4),
        fullname='interpolate_linear_scale_1d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=2, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({2, 2}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(1, 128, 1, 1),
        fullname='interpolate_nearest_2d_launch_configs',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_nearest_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, 16), scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 16}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 3, 4),
        fullname='interpolate_nearest_tuple_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_nearest_scale_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(0, 2, 4, 4),
        fullname='interpolate_nearest_2d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(0, 2, 4, 4),
        fullname='interpolate_bilinear_2d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 2, 3),
        fullname='interpolate_bilinear_tuple_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.,
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 2.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_shared_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_skewed_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(true)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_tuple_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBilinear)
                            .align_corners(true)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_skewed_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        input_size=(0, 2, 4, 4),
        fullname='interpolate_bicubic_2d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                    mode='bicubic', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        input_size=(1, 2, 2, 3),
        fullname='interpolate_bicubic_tuple_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='bicubic', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bicubic', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 2.}))
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_tuple_shared_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bicubic', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_tuple_skewed_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bicubic', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(true)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_tuple_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bicubic', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBicubic)
                            .align_corners(true)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_tuple_skewed_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 4, 4, 4),
        fullname='interpolate_nearest_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(0, 2, 4, 4, 4),
        fullname='interpolate_nearest_3d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, 16, 16), scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 16, 16}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 3, 4, 4),
        fullname='interpolate_nearest_tuple_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4., 4.}))
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 4, 4, 4),
        fullname='interpolate_nearest_scale_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4, 4),
        fullname='interpolate_trilinear_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        input_size=(0, 2, 4, 4, 4),
        fullname='interpolate_trilinear_3d_zero_dim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6, 6),
                                    scale_factor=None, mode='trilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 2, 3, 3),
        fullname='interpolate_trilinear_tuple_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({3., 3., 3.}))
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 3, 4, 5),
        fullname='interpolate_trilinear_scale_3d',
        # See https://github.com/pytorch/pytorch/issues/5006
        precision=3e-4,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6, 6), scale_factor=None,
                                    mode='trilinear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(true)''',
        input_size=(1, 2, 2, 3, 3),
        fullname='interpolate_trilinear_tuple_3d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({3., 3., 3.}))
                            .mode(torch::kTrilinear)
                            .align_corners(true)''',
        input_size=(1, 2, 3, 4, 4),
        fullname='interpolate_trilinear_scale_3d_align_corners',
        # See https://github.com/pytorch/pytorch/issues/5006
        precision=3e-4,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        cpp_options_args='F::SoftmaxFuncOptions(-1)',
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
        input_size=(2, 128),
        fullname='softmax_lastdim_dtype',
        pickle=False,
        test_cuda=False
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        cpp_options_args='F::SoftmaxFuncOptions(1)',
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='softmax_spatial_special',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        cpp_options_args='F::SoftmaxFuncOptions(1)',
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial_dtype',
        pickle=False,
        test_cuda=False
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=0),
        cpp_options_args='F::SoftmaxFuncOptions(0)',
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim0',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=3),
        cpp_options_args='F::SoftmaxFuncOptions(3)',
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim3',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        cpp_options_args='F::SoftmaxFuncOptions(-1)',
        input_size=(),
        fullname='softmax_functional_scalar',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=-1),
        cpp_options_args='F::LogSoftmaxFuncOptions(-1)',
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='log_softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        cpp_options_args='F::LogSoftmaxFuncOptions(1)',
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='log_softmax_spatial_special',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        cpp_options_args='F::LogSoftmaxFuncOptions(1)',
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='log_softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=0),
        cpp_options_args='F::LogSoftmaxFuncOptions(0)',
        input_size=(2, 3, 4, 5),
        fullname='log_softmax_dim0',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=3),
        cpp_options_args='F::LogSoftmaxFuncOptions(3)',
        input_size=(2, 3, 4, 5),
        fullname='log_softmax_dim3',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=0),
        cpp_options_args='F::LogSoftmaxFuncOptions(0)',
        input_size=(),
        fullname='log_softmax_scalar',
        pickle=False,
    ),
    dict(
        fullname='Unfold',
        constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)),
        cpp_constructor_args='torch::nn::UnfoldOptions({2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold',
        constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
        cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        input_size=(2, 16, 4),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold_no_batch_dim_input',
        constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
        cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        input_size=(16, 4),
        check_gradgrad=False,
        ref=single_batch_reference_fn,
        test_cuda=True,
    ),
    dict(
        fullname='Unfold_int_input',
        constructor=lambda: nn.Unfold(2, 1, 0, 1),
        cpp_constructor_args='torch::nn::UnfoldOptions(2).dilation(1).padding(0).stride(1)',
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold_int_input',
        constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
        cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)',
        input_size=(2, 16, 4),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold_no_batch_dim_int_input',
        constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
        cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)',
        input_size=(16, 4),
        ref=single_batch_reference_fn,
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)',
        input_size=(),
        desc='with_up_down_scalar',
        test_cuda=False,
    ),
    dict(
        module_name='PairwiseDistance',
        input_fn=lambda: (torch.randn(10, 8), torch.randn(10, 8)),
    ),
    dict(
        module_name='PairwiseDistance',
        input_fn=lambda: (torch.randn(10, 1), torch.randn(10, 8)),
        desc='broadcast_lhs'
    ),
    dict(
        module_name='PairwiseDistance',
        input_fn=lambda: (torch.randn(10, 8), torch.randn(1, 8)),
        desc='broadcast_rhs'
    ),
    dict(
        module_name='PairwiseDistance',
        constructor_args=(1.5, 1e-05, True),
        cpp_constructor_args='torch::nn::PairwiseDistanceOptions().p(1.5).eps(1e-05).keepdim(true)',
        input_fn=lambda: (torch.randn(10, 8), torch.randn(10, 8)),
        desc='with_non_default_args',
    ),
    dict(
        module_name='PairwiseDistance',
        input_fn=lambda: (torch.randn(8), torch.randn(8)),
        reference_fn=single_batch_reference_fn,
        desc='no_batch_dim',
    ),
    dict(
        module_name='TransformerEncoderLayer',
        constructor_args=(4, 2, 16, 0.0),
        cpp_constructor_args='''torch::nn::TransformerEncoderLayerOptions(4, 2)
                                .dim_feedforward(16)
                                .dropout(0.0)''',
        input_size=(2, 3, 4),
        desc='relu_activation',
        with_tf32=True,
        tf32_precision=0.1,
        # TODO(#50743): figure out the error
        # RuntimeError: The size of tensor a (6) must match the size of tensor b (4)
        # at non-singleton dimension 2
        check_batched_grad=False,
        check_gradgrad=False,
    ),
    dict(
        module_name='TransformerEncoderLayer',
        constructor_args=(4, 2, 8, 0.0, F.gelu),
        cpp_constructor_args='''torch::nn::TransformerEncoderLayerOptions(4, 2)
                                .dim_feedforward(8)
                                .dropout(0.0)
                                .activation(torch::kGELU)''',
        input_size=(2, 3, 4),
        check_gradgrad=False,
        desc='gelu_activation',
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='TransformerDecoderLayer',
        constructor_args=(4, 2, 8, 0.0),
        cpp_constructor_args='''torch::nn::TransformerDecoderLayerOptions(4, 2)
                                .dim_feedforward(8)
                                .dropout(0.0)''',
        input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4)),
        check_gradgrad=False,
        desc='relu_activation',
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='TransformerDecoderLayer',
        constructor_args=(4, 2, 8, 0.0, F.gelu),
        cpp_constructor_args='''torch::nn::TransformerDecoderLayerOptions(4, 2)
                                .dim_feedforward(8)
                                .dropout(0.0)
                                .activation(torch::kGELU)''',
        input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4)),
        check_gradgrad=False,
        desc='gelu_activation',
        with_tf32=True,
        tf32_precision=0.05,
    ),
    dict(
        module_name='Transformer',
        constructor_args=(4, 2, 2, 2, 8, 0.0, F.relu),
        cpp_constructor_args='''torch::nn::TransformerOptions()
                                .d_model(4)
                                .nhead(2)
                                .num_encoder_layers(2)
                                .num_decoder_layers(2)
                                .dim_feedforward(8)
                                .dropout(0.0)
                                .activation(torch::kReLU)''',
        input_fn=lambda:(torch.rand(3, 3, 4), torch.rand(2, 3, 4), torch.rand(3, 3)),
        check_gradgrad=False,
        desc='multilayer_coder',
        with_tf32=True,
        tf32_precision=0.03,
    ),
    dict(
        module_name='Linear',
        constructor_args=(3, 5),
        cpp_constructor_args='torch::nn::LinearOptions(3, 5)',
        input_fn=lambda: torch.rand(3),
        reference_fn=lambda i, p, _: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1],
        desc="no_batch_dim",
        with_tf32=True,
        tf32_precision=0.005,
    ),
    dict(
        module_name='Flatten',
        cpp_constructor_args='torch::nn::FlattenOptions().start_dim(-3).end_dim(-1)',
        constructor_args=(-3, -1),
        input_size=(3, 4, 5),
        reference_fn=single_batch_reference_fn,
        desc="no_batch_dim",
    ),
    dict(
        module_name='Unflatten',
        cpp_constructor_args='torch::nn::UnflattenOptions(-2, {2, 2})',
        constructor_args=(-2, torch.Size([2, 2])),
        input_size=(3, 4, 5),
        reference_fn=single_batch_reference_fn,
        desc="no_batch_dim",
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([56, 56, 56], 1e-5, False),
        cpp_constructor_args='torch::nn::LayerNormOptions({56, 56, 56}).eps(1e-5).elementwise_affine(false)',
        input_size=(4, 56, 56, 56),
        cudnn=True,
        check_eval=True,
        gradcheck_fast_mode=True,
        check_half=True,
        desc='3d_no_affine_large_feature',
    ),
]

# add conv padding mode tests:
for padding_mode, cpp_padding_mode in zip(
        ['reflect', 'circular', 'replicate', 'zeros'],
        ['torch::kReflect', 'torch::kCircular', 'torch::kReplicate', 'torch::kZeros']):
    # conv signature:
    #     in_channels, out_channels, kernel_size, stride=1,
    #     padding=0, dilation=1, groups=1,
    #     bias=True, padding_mode='zeros'
    for d in (1, 2, 3):
        if d == 3 and padding_mode == 'reflect':
            # FIXME: remove after implementing reflection pad 3d
            #        https://github.com/pytorch/pytorch/issues/27655
            continue
        padding = tuple(range(1, d + 1))
        cpp_padding = '{' + ', '.join(map(str, padding)) + '}'
        input_size = (2, 2) + (4,) * d
        output_size = (2, 3) + tuple(p + 1 for p in padding)  # simplified from `(4 + 2 * p - 3) // 2 + 1`
        new_module_tests.append(
            dict(
                module_name=f'Conv{d}d',
                constructor_args=(2, 3, 3, 2, padding, 1, 1, True, padding_mode),
                cpp_constructor_args=f'''torch::nn::Conv{d}dOptions(2, 3, 3)
                                        .stride(2)
                                        .padding({cpp_padding})
                                        .dilation(1)
                                        .groups(1)
                                        .bias(true)
                                        .padding_mode({cpp_padding_mode})''',
                input_size=input_size,
                output_size=output_size,
                cudnn=True,
                desc=f'{padding_mode}_stride2_pad2',
                with_tf32=True,
                tf32_precision=0.05
            ),
        )

# Check that non linear activations work with no batch dimensions
non_linear_activations_no_batch = [
    'ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'Hardswish', 'LeakyReLU',
    'LogSigmoid', 'PReLU', 'ReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'GLU',
    'Sigmoid', 'SiLU', 'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh',
    'Tanhshrink', 'Threshold'
]
non_linear_activations_extra_info: Dict[str, dict] = {
    'CELU': {'constructor_args': (2.,)},
    'Threshold': {'constructor_args': (2., 1.)},
    'Hardsigmoid': {'check_gradgrad': False, 'check_jit': False},
    'Hardswish': {'check_gradgrad': False, 'check_jit': False},
    # For RRelu, test that compare CPU and GPU results fail because RNG
    # is different between CPU and GPU
    'RReLU': {'test_cuda': False},
}
for non_linear_activation in non_linear_activations_no_batch:
    activation_test_info = dict(
        module_name=non_linear_activation,
        input_size=(4,),
        reference_fn=single_batch_reference_fn,
        desc='no_batch_dim',
        test_cpp_api_parity=False,
    )
    extra_info = non_linear_activations_extra_info.get(non_linear_activation, {})
    activation_test_info.update(extra_info)
    new_module_tests.append(activation_test_info)


def kldivloss_reference(input, target, reduction='mean'):
    result = target * (target.log() - input)
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    elif reduction == 'batchmean' and result.dim() != 0:
        return result.sum() / result.size(0)
    return result


def kldivloss_log_target_reference(input, target, reduction='mean'):
    result = torch.exp(target) * (target - input)
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    elif reduction == 'batchmean' and result.dim() != 0:
        return result.sum() / result.size(0)
    return result


def nlllossNd_reference(input, target, weight=None, ignore_index=-100,
                        reduction='mean'):
    assert input.dim() >= 3
    N = input.size(0)
    C = input.size(1)
    out_size = (N,) + input.size()[2:]
    output = torch.zeros(out_size).type_as(input)

    if weight is None:
        weight = torch.ones(C).type_as(input)
    total_weight = 0
    for tup in product(*[range(size) for size in out_size]):
        t_nx = target[tup]
        norm = 0. if ignore_index == t_nx else weight[t_nx].item()
        input_index = list(tup)
        input_index.insert(1, t_nx)
        output[tup] = -input[tuple(input_index)] * norm
        total_weight += norm

    if reduction == 'mean':
        return output.sum() / total_weight
    elif reduction == 'sum':
        return output.sum()
    return output


def cross_entropy_loss_prob_target_reference(input, target, weight=None, reduction='mean',
                                             label_smoothing=0.0):
    assert input.dim() >= 2

    input = torch.log_softmax(input, 1)
    C = input.size(1)
    if weight is None:
        weight = torch.ones(C).type_as(input)
    weight = weight.view(1, C, *(1 for _ in input.shape[2:]))

    if label_smoothing > 0.0:
        assert label_smoothing <= 1.0
        target = (target * (1 - label_smoothing) + label_smoothing / C)

    output = -(input * target * weight).sum(dim=1)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def cross_entropy_loss_indices_target_reference(input, target, weight=None, ignore_index=-100,
                                                reduction='mean', label_smoothing=0.0):
    log_softmax_input = torch.log_softmax(input, 1)
    nllloss = F.nll_loss(
        log_softmax_input,
        target,
        weight,
        ignore_index=ignore_index,
        reduction=reduction)

    if label_smoothing == 0.0:
        return nllloss

    assert 0.0 < label_smoothing <= 1.0

    input = torch.log_softmax(input, 1)
    C = input.size(1)
    if weight is not None:
        input = input * weight.view(1, C, *(1 for _ in input.shape[2:]))

    smooth_loss = -torch.sum(input, 1)

    ignore_mask = target == ignore_index
    smooth_loss.masked_fill_(ignore_mask, 0.0)

    if reduction == 'mean':
        if weight is not None:
            # TODO: This code can path can be removed if #61309 is resolved
            # loss is normalized by the weights to be consistent with nll_loss_nd
            ret = torch.sum(smooth_loss) / weight.gather(0, target.masked_select(ignore_mask.logical_not()).flatten()).sum()
        else:
            ret = torch.mean(smooth_loss.masked_select(ignore_mask.logical_not()))
    elif reduction == 'sum':
        ret = torch.sum(smooth_loss)
    else:
        ret = smooth_loss

    return (1 - label_smoothing) * nllloss + ret * (label_smoothing / C)


def cross_entropy_loss_reference(input, target, weight=None, ignore_index=-100, reduction='mean',
                                 label_smoothing=0.0):
    if input.shape == target.shape:
        return cross_entropy_loss_prob_target_reference(
            input,
            target,
            weight=weight,
            reduction=reduction,
            label_smoothing=label_smoothing)
    else:
        return cross_entropy_loss_indices_target_reference(
            input, target, weight=weight, reduction=reduction,
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )


def nllloss_reference(input, target, weight=None, ignore_index=-100,
                      reduction='mean'):

    def nll_loss_helper(input, target, weight, ignore_index):
        if target == ignore_index:
            return (0, 0)
        norm = 1 if weight is None else weight[target]
        result = -input[target] * norm
        return (result, norm)

    losses_and_weights = [nll_loss_helper(i, t, weight, ignore_index)
                          for i, t in zip(input, target)]
    losses, weights = zip(*losses_and_weights)
    losses_tensor = input.new_tensor(losses)
    if reduction == 'mean':
        return sum(losses_tensor) / sum(weights)
    elif reduction == 'sum':
        return sum(losses_tensor)
    else:
        return losses_tensor


def smoothl1loss_reference(input, target, reduction='mean', beta=1.0):
    abs_diff = (input - target).abs()
    ge_beta_mask = (abs_diff >= beta).type_as(abs_diff)
    lt_beta_mask = (abs_diff < beta).type_as(abs_diff)
    # when beta <= 0 we should just use l1_loss
    if beta == 0:
        output = abs_diff
    else:
        output = ge_beta_mask * (abs_diff - 0.5 * beta) + lt_beta_mask * 0.5 * (abs_diff ** 2) / beta
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def huberloss_reference(input, target, reduction='mean', delta=1.0):
    abs_diff = (input - target).abs()
    ge_delta_mask = (abs_diff >= delta)
    lt_delta_mask = (abs_diff < delta)
    output = ge_delta_mask * delta * (abs_diff - 0.5 * delta) + lt_delta_mask * 0.5 * (abs_diff ** 2)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def _multilabelmarginloss_reference(input, target):
    targets = []
    for target_index in target:
        if target_index < 0:
            break
        targets.append(target_index)

    sum = 0
    for target_index in targets:
        for i in range(0, len(input)):
            if i not in targets:
                sum += max(0, 1 - input[target_index] + input[i])

    return sum


def multilabelmarginloss_reference(input, target, reduction='mean'):
    # make everything 2-dimensional
    input_dim = input.dim()
    if input.dim() < 2:
        assert target.dim() < 2
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0) if target.dim() == 1 else target.unsqueeze(0).unsqueeze(0)

    n = input.size(0)
    dim = input.size(1)
    output = input.new(n).zero_()
    for i in range(0, n):
        output[i] = _multilabelmarginloss_reference(input[i], target[i])

    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    elif input_dim < 2:
        # we know we have (1, C) X (1, C) -> (1,), so squeeze will get us
        # back to correct dimensionality
        return output.squeeze() / dim
    else:
        return output / dim


def hingeembeddingloss_reference(input, target, margin=1.0, reduction='mean'):
    margin_clamp = (margin - input).clamp(min=0).type_as(input)
    output = torch.where(target == 1, input, margin_clamp)

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def softmarginloss_reference(input, target, reduction='mean'):
    output = (1 + (-input * target).exp()).log()

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def _multimarginloss_reference(input, target_idx, p, margin, weight):
    if weight is None:
        weight = input.new(len(input)).fill_(1)

    output = 0
    for i in range(0, len(input)):
        if i != target_idx:
            output += max(0, weight[target_idx] * (margin - input[target_idx] + input[i]) ** p)
    return output


def multimarginloss_reference(input, target, p=1, margin=1, weight=None, reduction='mean'):
    if input.dim() < 2:
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)

    target_dim = target.dim()
    if target.dim() == 0:
        target = target.unsqueeze(0)

    n = input.size(0)
    dim = input.size(1)
    output = input.new(n)
    for x in range(0, n):
        output[x] = _multimarginloss_reference(input[x], target[x], p, margin, weight)

    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    elif target_dim == 0:
        return output.squeeze(0) / dim
    return output / dim


def cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    def _cos(a, b):
        cos = a.new(a.size(0))
        for i in range(0, a.size(0)):
            cos[i] = (a[i] * b[i]).sum() / ((((a[i] * a[i]).sum() + 1e-12) * ((b[i] * b[i]).sum() + 1e-12)) ** 0.5)
        return cos

    output = torch.where(target == 1, 1 - _cos(input1, input2), (_cos(input1, input2) - margin).clamp(min=0))

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False,
                                reduction='mean'):
    d_p = torch.pairwise_distance(anchor, positive, p, eps)
    d_n = torch.pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = torch.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    output = torch.clamp(margin + d_p - d_n, min=0.0)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def marginrankingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    output = (-target * (input1 - input2) + margin).clamp(min=0)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


# this directly follows Graves et al's paper, in contrast to the production implementation, it does not use log-space
def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        probs = log_probs[:input_length, i].exp()
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output


loss_reference_fns: Dict['str', Callable] = {
    'KLDivLoss': kldivloss_reference,
    'KLDivLoss_log_target': kldivloss_log_target_reference,
    'NLLLoss': nllloss_reference,
    'NLLLossNd': nlllossNd_reference,
    'SmoothL1Loss': smoothl1loss_reference,
    'HuberLoss': huberloss_reference,
    'MultiLabelMarginLoss': multilabelmarginloss_reference,
    'HingeEmbeddingLoss': hingeembeddingloss_reference,
    'SoftMarginLoss': softmarginloss_reference,
    'MultiMarginLoss': multimarginloss_reference,
    'CosineEmbeddingLoss': cosineembeddingloss_reference,
    'TripletMarginLoss': tripletmarginloss_reference,
    'MarginRankingLoss': marginrankingloss_reference,
    'CTCLoss': ctcloss_reference,
    'CrossEntropyLoss': cross_entropy_loss_reference
}


criterion_tests = [
    dict(
        module_name='L1Loss',
        input_size=(2, 3, 4),
        target_fn=lambda: torch.randn((2, 3, 4), requires_grad=True),
        reference_fn=lambda i, t, _: 1. / i.numel() *
        sum((a - b).abs().sum() for a, b in zip(i, t)),
        check_complex=True,
    ),
    dict(
        module_name='NLLLoss',
        input_fn=lambda: torch.rand(15, 10).log(),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, None, 2),
        cpp_constructor_args='torch::nn::NLLLossOptions().weight({}).ignore_index(2)',
        input_fn=lambda: torch.rand(15, 10).log(),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, _: nllloss_reference(i, t, ignore_index=2),
        desc='ignore_index',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(10))',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m)),
        desc='weights',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10), None, 2),
        cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(10)).ignore_index(2)',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m), ignore_index=2),
        desc='weights_ignore_index',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10), None, -1),
        cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(10)).ignore_index(-1)',
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.empty(15).uniform_().mul(10 + 1).floor().long() - 1,
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m), ignore_index=-1),
        desc='weights_ignore_index_neg',
        check_bfloat16=True,
    ),
    dict(
        module_name='KLDivLoss',
        input_fn=lambda: torch.rand(10, 10).log(),
        target_fn=lambda: torch.rand(10, 10),
        reference_fn=lambda i, t, m:
            kldivloss_reference(i, t, get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='KLDivLoss',
        constructor=wraps(nn.KLDivLoss)(partial(nn.KLDivLoss, log_target=True)),
        cpp_constructor_args='torch::nn::KLDivLossOptions().log_target(true)',
        input_fn=lambda: torch.rand(10, 10).log(),
        target_fn=lambda: torch.rand(10, 10).log(),
        reference_fn=lambda i, t, m:
            kldivloss_log_target_reference(i, t, get_reduction(m)),
        check_sum_reduction=True,
        desc='log_target',
    ),
    dict(
        module_name='MSELoss',
        input_size=(2, 3, 4, 5),
        target_fn=lambda: torch.randn((2, 3, 4, 5), requires_grad=True),
        reference_fn=lambda i, t, m: ((i - t).abs().pow(2).sum() / (i.numel()
                                      if get_reduction(m) == 'mean' else 1)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='BCELoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).to(torch.get_default_dtype()),
        reference_fn=lambda i, t, m: -(t * i.log() + (1 - t) * (1 - i).log()).sum() /
            (i.numel() if get_reduction(m) else 1),
        check_bfloat16=True,
    ),
    dict(
        module_name='BCELoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        cpp_constructor_args='torch::nn::BCELossOptions().weight(torch::rand(10))',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).to(torch.get_default_dtype()),
        reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() /
            (i.numel() if get_reduction(m) else 1),
        desc='weights',
        check_bfloat16=True,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(15, 10),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight(torch::rand(10))',
        input_size=(15, 10),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        desc='weights',
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        input_size=(10,),
        target_fn=lambda: torch.randn(10).gt(0).to(torch.get_default_dtype()).mul_(2).sub(1),
        reference_fn=lambda i, t, m:
            hingeembeddingloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        cpp_constructor_args='torch::nn::HingeEmbeddingLossOptions().margin(0.5)',
        input_size=(10,),
        target_fn=lambda: torch.randn(10).gt(0).to(torch.get_default_dtype()).mul_(2).sub(1),
        reference_fn=lambda i, t, m:
            hingeembeddingloss_reference(i, t, margin=0.5, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(10,),
        target_fn=lambda: torch.rand(10).mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            multilabelmarginloss_reference(i, t, reduction=get_reduction(m)),
        desc="1d",
        check_sum_reduction=True,
        check_gradgrad=False,
        check_bfloat16=True,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            multilabelmarginloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
        check_bfloat16=True,
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(2).floor(),
        reference_fn=lambda i, t, m: -(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()).sum() / i.numel(),
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(10,),
        target_fn=lambda: torch.rand(1).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, reduction=get_reduction(m)),
        desc='1d',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(2,),
        cpp_constructor_args='torch::nn::MultiMarginLossOptions().p(2)',
        input_fn=lambda: torch.rand(5, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, p=2, reduction=get_reduction(m)),
        desc='p',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(1, 0.5),
        cpp_constructor_args='torch::nn::MultiMarginLossOptions().p(1).margin(0.5)',
        legacy_constructor_args=(1, None, 0.5),
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, margin=0.5, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(1, 1., torch.rand(10).to(torch.get_default_dtype())),
        cpp_constructor_args='torch::nn::MultiMarginLossOptions().p(1).margin(1.).weight(torch::rand(10))',
        legacy_constructor_args=(1, torch.rand(10).to(torch.get_default_dtype())),
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, weight=get_weight(m), reduction=get_reduction(m)),
        desc='weights',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(5, 10),
        target_fn=lambda: torch.randn((5, 10), requires_grad=True),
        check_sum_reduction=True,
        reference_fn=lambda i, t, m, b=1.0:
            smoothl1loss_reference(i, t, reduction=get_reduction(m), beta=b),
    ),
    dict(
        module_name='HuberLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.randn((5, 10), requires_grad=True),
        check_sum_reduction=True,
        check_half=True,
        check_bfloat16=True,
        reference_fn=lambda i, t, m:
            huberloss_reference(i, t, reduction=get_reduction(m)),
    ),
    dict(
        module_name='SoftMarginLoss',
        input_size=(5, 5),
        target_fn=lambda: torch.randn(5, 5).sign(),
        reference_fn=lambda i, t, m:
            softmarginloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)),
        target_fn=lambda: torch.randn(15).sign(),
        reference_fn=lambda i, t, m:
            cosineembeddingloss_reference(i[0], i[1], t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        constructor_args=(0.7,),
        cpp_constructor_args='torch::nn::CosineEmbeddingLossOptions().margin(0.7)',
        input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)),
        target_fn=lambda: torch.randn(15).sign(),
        reference_fn=lambda i, t, m:
            cosineembeddingloss_reference(i[0], i[1], t, margin=0.7, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='MarginRankingLoss',
        input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target_fn=lambda: torch.randn(50).sign(),
        reference_fn=lambda i, t, m:
            marginrankingloss_reference(i[0], i[1], t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='MarginRankingLoss',
        constructor_args=(0.5,),
        cpp_constructor_args='torch::nn::MarginRankingLossOptions().margin(0.5)',
        input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target_fn=lambda: torch.randn(50).sign(),
        reference_fn=lambda i, t, m:
            marginrankingloss_reference(i[0], i[1], t, margin=0.5, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).to(torch.get_default_dtype()),
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        constructor_args=(torch.rand(10),),
        cpp_constructor_args='torch::nn::BCEWithLogitsLossOptions().weight(torch::rand(10))',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).to(torch.get_default_dtype()),
        desc='weights',
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        constructor_args=(torch.rand(()),),
        cpp_constructor_args='torch::nn::BCEWithLogitsLossOptions().weight(torch::rand({}))',
        input_fn=lambda: torch.rand(()).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(()).gt(0).to(torch.get_default_dtype()),
        desc='scalar_weights'
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='2d',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(3))',
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, weight=get_weight(m)),
        desc='2d_weights',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, None, 1),
        cpp_constructor_args='torch::nn::NLLLossOptions().weight({}).ignore_index(1)',
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, ignore_index=1),
        desc='2d_ignore_index',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5, 5, 2, 2),
        target_fn=lambda: torch.rand(2, 5, 5, 2, 2).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='higher_dim',
        check_bfloat16=True,
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='dim_is_3',
        check_bfloat16=True,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='2d',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight(torch::rand(3))',
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, weight=get_weight(m)),
        desc='2d_weights',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args=(None, None, 1),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight({}).ignore_index(1)',
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, ignore_index=1),
        desc='2d_ignore_index',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(2, 3, 5, 5, 2, 2),
        target_fn=lambda: torch.rand(2, 5, 5, 2, 2).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='higher_dim',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='dim_is_3',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(5, 3),
        target_fn=lambda: torch.rand(5, 3).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='2d_prob_target',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(5, 3, 4),
        target_fn=lambda: torch.rand(5, 3, 4).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='3d_prob_target',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(5, 3, 4, 2),
        target_fn=lambda: torch.rand(5, 3, 4, 2).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='4d_prob_target',
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_prob_target_smoothing_sum_reduction',
        constructor=lambda *args, **kwargs: nn.CrossEntropyLoss(reduction='sum',
                                                                label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).reduction(torch::kSum)',
        input_size=(5, 3),
        target_fn=lambda: torch.rand(5, 3).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_prob_target_smoothing',
        constructor=lambda *args: nn.CrossEntropyLoss(label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15)',
        input_size=(5, 3),
        target_fn=lambda: torch.rand(5, 3).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_prob_target_smoothing_weight',
        constructor_args_fn=lambda: (torch.rand(3).abs(),),
        constructor=lambda weight: nn.CrossEntropyLoss(weight, label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).weight(torch::rand(3).abs())',
        input_size=(5, 3),
        target_fn=lambda: torch.rand(5, 3).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), weight=get_weight(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_3d_prob_target_smoothing_sum_reduction',
        constructor=lambda *args: nn.CrossEntropyLoss(reduction='sum',
                                                                label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).reduction(torch::kSum)',
        input_size=(5, 3, 4),
        target_fn=lambda: torch.rand(5, 3, 4).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_3d_prob_target_smoothing',
        constructor=lambda *args: nn.CrossEntropyLoss(label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15)',
        input_size=(5, 3, 4),
        target_fn=lambda: torch.rand(5, 3, 4).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_3d_indices_target_smoothing',
        constructor=lambda *args: nn.CrossEntropyLoss(label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15)',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_3d_indices_target_smoothing_ignore_index',
        constructor=lambda *args: nn.CrossEntropyLoss(label_smoothing=0.15, ignore_index=1),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).ignore_index(1)',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15, ignore_index=1),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_3d_indices_target_smoothing_sum_reduction',
        constructor=lambda *args: nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).reduction(torch::kSum)',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_3d_indices_target_smoothing_sum_reduction_ignore_index',
        constructor=lambda *args: nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.15,
                                                      ignore_index=1),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).reduction(torch::kSum).ignore_index(1)',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15, ignore_index=1),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_indices_target_smoothing',
        constructor=lambda *args: nn.CrossEntropyLoss(label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15)',
        input_size=(15, 10),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_indices_target_smoothing_sum_reduction',
        constructor=lambda *args: nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).reduction(torch::kSum)',
        input_size=(15, 10),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_indices_target_smoothing_ignore_index',
        constructor=lambda *args: nn.CrossEntropyLoss(label_smoothing=0.15, ignore_index=3),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).ignore_index(3)',
        input_size=(15, 10),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), label_smoothing=0.15, ignore_index=3),
        check_bfloat16=False,
    ),
    dict(
        fullname='CrossEntropyLoss_2d_indices_target_smoothing_weight',
        constructor_args_fn=lambda: (torch.rand(10).abs(),),
        constructor=lambda weight: nn.CrossEntropyLoss(weight, label_smoothing=0.15),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().label_smoothing(0.15).weight(torch::rand(10).abs())',
        input_size=(15, 10),
        target_fn=lambda: torch.empty(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), weight=get_weight(m), label_smoothing=0.15),
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight(torch::rand(3))',
        input_size=(5, 3),
        target_fn=lambda: torch.rand(5, 3).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), weight=get_weight(m)),
        check_sum_reduction=True,
        desc='2d_prob_target_weights',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight(torch::rand(3))',
        input_size=(5, 3, 4),
        target_fn=lambda: torch.rand(5, 3, 4).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), weight=get_weight(m)),
        check_sum_reduction=True,
        desc='3d_prob_target_weights',
        check_bfloat16=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight(torch::rand(3))',
        input_size=(5, 3, 4, 2),
        target_fn=lambda: torch.rand(5, 3, 4, 2).softmax(dim=1),
        reference_fn=lambda i, t, m:
            loss_reference_fns['CrossEntropyLoss'](i, t, reduction=get_reduction(m), weight=get_weight(m)),
        check_sum_reduction=True,
        desc='4d_prob_target_weights',
        check_bfloat16=False,
    ),
    dict(
        module_name='PoissonNLLLoss',  # Default is log_input=True, full=False
        input_size=(2, 3, 4, 5),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _: (i.exp() - t.mul(i)).mean(),
        desc='no_full_loss',
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(False, False),  # log_input=False, full=False
        cpp_constructor_args='torch::nn::PoissonNLLLossOptions().log_input(false).full(false)',
        input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _: (i - t.mul((i + 1e-8).log())).mean(),
        desc='no_full_loss_no_log_input',
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(True, True),  # log_input=True, full=True
        cpp_constructor_args='torch::nn::PoissonNLLLossOptions().log_input(true).full(true)',
        input_size=(2, 3, 4, 5),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _:
            (i.exp() - t.mul(i) + (t.mul(t.log()) - t + 0.5 * (2. * pi * t).log()).masked_fill(t <= 1, 0)).mean(),
        desc='full_loss',
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(False, True),  # log_input=False, full=True
        cpp_constructor_args='torch::nn::PoissonNLLLossOptions().log_input(false).full(true)',
        input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _: (
            i - t.mul((i + 1e-8).log()) + (t.mul(t.log()) - t + 0.5 * (2. * pi * t).log()).masked_fill(t <= 1, 0)
        ).mean(),
        desc='full_loss_no_log_input',
    ),
    dict(
        module_name='L1Loss',
        input_size=(),
        target_fn=lambda: torch.randn((), requires_grad=True),
        reference_fn=lambda i, t, _: 1. / i.numel() * (i - t).abs().sum(),
        desc='scalar',
        check_complex=True,
    ),
    dict(
        module_name='KLDivLoss',
        input_fn=lambda: torch.rand(()).log(),
        target_fn=lambda: torch.rand(()),
        reference_fn=lambda i, t, m:
            kldivloss_reference(i, t, get_reduction(m)),
        check_sum_reduction=True,
        desc='scalar',
    ),
    dict(
        module_name='KLDivLoss',
        constructor=wraps(nn.KLDivLoss)(partial(nn.KLDivLoss, log_target=True)),
        cpp_constructor_args='torch::nn::KLDivLossOptions().log_target(true)',
        input_fn=lambda: torch.rand(()).log(),
        target_fn=lambda: torch.rand(()).log(),
        reference_fn=lambda i, t, m:
            kldivloss_log_target_reference(i, t, get_reduction(m)),
        check_sum_reduction=True,
        desc='scalar_log_target',
    ),
    dict(
        module_name='MSELoss',
        input_size=(),
        target_fn=lambda: torch.randn((), requires_grad=True),
        reference_fn=lambda i, t, m: ((i - t).abs().pow(2).sum() /
                                      (i.numel() if get_reduction(m) == 'mean' else 1)),
        check_sum_reduction=True,
        desc='scalar',
        check_bfloat16=True,
    ),
    dict(
        module_name='MSELoss',
        input_fn=lambda: torch.ones(5, 68, 64, 64, dtype=torch.float) / 10,
        target_fn=lambda: torch.zeros(5, 68, 64, 64, dtype=torch.float),
        reference_fn=lambda i, t, m: ((i - t).abs().pow(2).sum() /
                                      (i.numel() if get_reduction(m) == 'mean' else 1)),
        check_forward_only=True,
        desc='prec',
        check_bfloat16=True,
    ),
    dict(
        module_name='BCELoss',
        constructor_args_fn=lambda: (torch.rand(()),),
        cpp_constructor_args='torch::nn::BCELossOptions().weight(torch::rand({}))',
        input_fn=lambda: torch.rand(()).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.rand(()).gt(0).to(torch.get_default_dtype()),
        reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() /
            (i.numel() if get_reduction(m) == 'mean' else 1),
        desc='scalar_weights',
        check_bfloat16=True,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        cpp_constructor_args='torch::nn::HingeEmbeddingLossOptions().margin(0.5)',
        input_size=(),
        target_fn=lambda: torch.randn(()).gt(0).to(torch.get_default_dtype()).mul_(2).sub(1),
        desc='scalar_margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(),
        target_fn=lambda: torch.randn((), requires_grad=True),
        check_sum_reduction=True,
        reference_fn=lambda i, t, m, b=1.0:
            smoothl1loss_reference(i, t, reduction=get_reduction(m), beta=b),
        desc='scalar',
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        constructor_args=(torch.rand(10),),
        cpp_constructor_args='torch::nn::MultiLabelSoftMarginLossOptions().weight(torch::rand(10))',
        input_fn=lambda: torch.randn(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(2).floor(),
        reference_fn=lambda i, t, m: -((t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * get_weight(m)).sum() /
            (i.numel() if get_reduction(m) == 'mean' else i.size(1) if get_reduction(m) == 'sum' else 1),
        desc='weights',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='CTCLoss',
        constructor_args=(14,),  # blank=14
        extra_args=([50, 50, 50], [30, 25, 20]),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)),
        desc='lengths_intlists',
        check_forward_only=True,
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
        # `CTCLoss` in C++ frontend doesn't accept integer list for `input_lengths` or `target_lengths`
        test_cpp_api_parity=False,
        check_jit=False,
    ),
    dict(
        module_name='CTCLoss',
        constructor_args=(14,),  # blank=14
        cpp_constructor_args='torch::nn::CTCLossOptions().blank(14)',
        extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)),
        desc='lengths_tensors',
        check_forward_only=True,
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
    ),
    # Test is flaky
    # See https://github.com/pytorch/pytorch/issues/29380.
    # dict(
    #     module_name='CTCLoss',
    #     desc='1d_target',
    #     constructor_args=(14,),  # blank=14
    #     extra_args=([50, 50, 50], [30, 25, 20]),  # input_lengths, target_lengths
    #     input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
    #     target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long),
    #     reference_fn=lambda i, t, il, tl, m:
    #         ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)),
    #     check_sum_reduction=True,
    #     check_gradgrad=False,
    #     check_half=False,
    # ),
    dict(
        module_name='CTCLoss',
        desc='2d_int_target_lengths_intlists',
        constructor_args=(0,),  # blank=0
        extra_args=([50, 50, 50], [30, 25, 20]),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)),
        check_forward_only=True,
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
        # `CTCLoss` in C++ frontend doesn't accept integer list for `input_lengths` or `target_lengths`
        test_cpp_api_parity=False,
        check_jit=False,
    ),
    dict(
        module_name='CTCLoss',
        desc='2d_int_target_lengths_tensors',
        constructor_args=(0,),  # blank=0
        cpp_constructor_args='torch::nn::CTCLossOptions().blank(0)',
        extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)),
        check_forward_only=True,
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
    ),
    dict(
        module_name='CTCLoss',
        desc='2d_lengths_tensors',
        constructor_args=(0,),  # blank=0
        cpp_constructor_args='torch::nn::CTCLossOptions().blank(0)',
        extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)),
        check_forward_only=True,
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
    ),
]


def single_batch_reference_criterion_fn(*args):
    """Reference function for criterion supporting no batch dimensions.

    The criterion is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    criterion = args[-1]

    def unsqueeze_inp(inp):
        if isinstance(inp, (list, tuple)):
            return [t.unsqueeze(0) for t in inp]
        return inp.unsqueeze(0)

    def flatten(xs):
        result = []
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(flatten(x))
        else:
            result.append(xs)
        return result

    single_batch_input_args = flatten([unsqueeze_inp(input) for input in args[:-1]])

    output = criterion(*single_batch_input_args)
    reduction = get_reduction(criterion)

    if reduction == 'none':
        return output.squeeze(0)
    # reduction is 'sum' or 'mean' which results in a scalar
    return output


# Check that regression criterion work with no batch dimensions
regression_criterion_no_batch = [
    'L1Loss', 'MSELoss', 'PoissonNLLLoss', 'HuberLoss', 'SmoothL1Loss'
]
reductions = ['none', 'mean', 'sum']
for name, reduction in product(regression_criterion_no_batch, reductions):
    regression_test_info = dict(
        fullname=f"{name}_no_batch_dim_{reduction}",
        constructor=lambda *args, name=name: getattr(nn, name)(reduction=reduction),
        input_size=(3, ),
        target_size=(3, ),
        reference_fn=single_batch_reference_criterion_fn,
        test_cpp_api_parity=False,
    )
    criterion_tests.append(regression_test_info)


for reduction in reductions:
    regression_test_info = dict(
        fullname=f"KLDivLoss_no_batch_dim_{reduction}",
        constructor=lambda: nn.KLDivLoss(reduction=reduction),
        input_fn=lambda: torch.rand((3,)).log(),
        target_fn=lambda: torch.rand((3,)),
        reference_fn=single_batch_reference_criterion_fn,
        test_cpp_api_parity=False,
    )
    criterion_tests.append(regression_test_info)


# Check that classification criterion work with no batch dimensions
# List of tuples of (name, input_fn, target_fn)
classification_criterion_no_batch = [
    ('BCELoss', lambda: torch.sigmoid(torch.randn(9)), lambda: torch.randn(9).gt(0).to(torch.get_default_dtype())),
    ('BCEWithLogitsLoss', lambda: torch.randn(9), lambda: torch.randn(9)),
    ('HingeEmbeddingLoss', lambda: torch.randn(9), lambda: torch.tensor([-1, 1, 1] * 3)),
    ('MultiLabelMarginLoss', lambda: torch.randn(4), lambda: torch.tensor([3, 0, -1, 1])),
    ('SoftMarginLoss', lambda: torch.randn(9), lambda: torch.tensor([-1, 1, 1] * 3)),
    ('NLLLoss', lambda: F.log_softmax(torch.randn(3), dim=0), lambda: torch.tensor(1)),
    ('CosineEmbeddingLoss', lambda: (torch.randn(9), torch.randn(9)), lambda: torch.tensor(1)),
    # For MarginRankingLoss, input_fn : (x1, x2) and target_fn : target
    ('MarginRankingLoss', lambda: (torch.randn(()), torch.randn(())), lambda: torch.randn(()).sign()),
    # For TripletMarginLoss, input_fn : (anchor, positive) and target_fn : negative
    ('TripletMarginLoss', lambda: (torch.randn(9), torch.randn(9)), lambda: torch.randn(9)),
    ('MultiLabelSoftMarginLoss', lambda: torch.randn(9), lambda: torch.randn(9)),
]
classification_criterion_no_batch_extra_info: Dict[str, dict] = {
    'MultiLabelMarginLoss': {'check_gradgrad': False},
}
# TODO : Fix these discrepancies
classification_cpp_parity = {
    'BCELoss': False,
    'BCEWithLogitsLoss': False,
    'HingeEmbeddingLoss': False,
    'NLLLoss': False,
    'SoftMarginLoss': False,
}
reductions = ['none', 'mean', 'sum']
for (name, input_fn, target_fn), reduction in product(classification_criterion_no_batch,
                                                      reductions):
    classification_test_info = dict(
        fullname=f"{name}_no_batch_dim_{reduction}",
        constructor=lambda *args, name=name: getattr(nn, name)(reduction=reduction),
        input_fn=lambda f=input_fn: f(),
        target_fn=lambda f=target_fn: f(),
        reference_fn=single_batch_reference_criterion_fn,
        test_cpp_api_parity=True,
        has_parity=classification_cpp_parity.get(name, True)
    )
    extra_info = classification_criterion_no_batch_extra_info.get(name, {})
    classification_test_info.update(extra_info)
    criterion_tests.append(classification_test_info)


class NNTestCase(TestCase):

    # _forward is defined in classes inheriting from NNTestCase
    @abstractmethod
    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_parameters(self, module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        raise NotImplementedError

    @abstractmethod
    def _zero_grad_parameters(self, module: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def _backward(self, module: nn.Module,
                  input: _TensorOrTensors, output: torch.Tensor,
                  grad_output: Union[torch.Tensor, Sequence[torch.Tensor]],
                  create_graph: bool = False):
        raise NotImplementedError

    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)

    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        else:
            return tuple(self._flatten_tensors(a) for a in x)

    def _zero_grad_input(self, input):
        if isinstance(input, torch.Tensor):
            if input.requires_grad and input.grad is not None:
                input.grad.zero_()
                input.grad.detach_()
        else:
            for i in input:
                self._zero_grad_input(i)

    def _analytical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nelement()

        if jacobian_input:
            jacobian_inp = self._jacobian(input, output_size)
            flat_jacobian_input = list(_iter_tensors(jacobian_inp))

        if jacobian_parameters:
            num_param = sum(p.numel() for p in self._get_parameters(module)[0])
            jacobian_param = torch.zeros(num_param, output_size)

        for i in range(output_size):
            param, d_param = self._get_parameters(module)
            # make non grad zeros
            d_param = [torch.zeros_like(p) if d is None else d for (p, d) in zip(param, d_param)]

            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1

            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # Tensors will accumulate gradient from multiple steps
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)

            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, _iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x.contiguous().view(-1)
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)

        res: Tuple[torch.Tensor, ...] = tuple()
        if jacobian_input:
            res += jacobian_inp,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    def _numerical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        def fw(*input):
            return self._forward(module, input).detach()

        res: Tuple[torch.Tensor, ...] = tuple()
        if jacobian_input:
            res += _get_numerical_jacobian(fw, input, eps=1e-6),
        if jacobian_parameters:
            param, _ = self._get_parameters(module)
            to_cat = []
            for p in param:
                jacobian = _get_numerical_jacobian(fw, input, target=p, eps=1e-6)
                # get_numerical_jacobian returns a list of tuples but we require a tensor
                to_cat.append(jacobian[0][0])
            res += (torch.cat(to_cat, 0),)
        return res

    def check_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True):
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = list(_iter_tensors(analytical))
        numerical_t = list(_iter_tensors(numerical))

        differences = []
        for a, n in zip(analytical_t, numerical_t):
            if a.numel() != 0:
                differences.append(a.add(n, alpha=-1).abs().max())
            # TODO: compare structure (ensure analytic jacobian has correct shape)
        if len(differences) > 0:
            self.assertLessEqual(max(differences), PRECISION)  # type: ignore[type-var]


class TestBase:

    _required_arg_names = {'constructor_args', 'input', 'extra_args'}

    def __init__(self, constructor, desc='', reference_fn=None, fullname=None, **kwargs):
        self.desc = desc
        self.fullname = fullname
        self.constructor = constructor
        self.reference_fn = reference_fn
        for name in self._required_arg_names:
            if name not in kwargs and name + '_fn' not in kwargs and name + '_size' not in kwargs:
                if name in {'constructor_args', 'extra_args'}:
                    kwargs[name] = tuple()
                else:
                    raise ValueError("{}: Specify {} by a value, a function to generate it, or it's size!"
                                     .format(self.get_name(), name))
        self._extra_kwargs = kwargs
        self._arg_cache = {}

    def get_name(self):
        if self.fullname is not None:
            return 'test_' + self.fullname

        test_name = 'test_' + self.constructor.__name__
        if self.desc:
            test_name += '_' + self.desc
        return test_name

    def _unpack(self, value):
        if isinstance(value, torch.Tensor):
            return value
        elif is_iterable(value):
            return type(value)(self._unpack(v) for v in value)
        else:
            return value

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', True)

    @property
    def extra_args(self):
        return self._get_arg('extra_args', True)

    def _get_arg(self, name, unpack):
        assert name in self._required_arg_names

        if name not in self._arg_cache:
            fn_name = name + '_fn'
            size_name = name + '_size'

            if name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[name]
            elif fn_name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[fn_name]()
            else:
                assert size_name in self._extra_kwargs, \
                    f"Missing `{name}`, `{size_name}` or `{fn_name}` for {self.get_name()}"

                def map_tensor_sizes(sizes):
                    if isinstance(sizes, list):
                        return [map_tensor_sizes(s) for s in sizes]
                    elif isinstance(sizes, torch.Tensor):
                        return sizes.double()
                    else:
                        return torch.randn(sizes)

                self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])

        return self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name]

    def _get_input(self, unpack=True):
        return self._get_arg('input', unpack)

    def __call__(self, test_case):
        raise NotImplementedError


class ModuleTest(TestBase):

    @abstractmethod
    def _do_test(self, test_case: Any, module: nn.Module, input: Any) -> Any:
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.should_test_pickle = kwargs.get('pickle', True)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.FIXME_no_cuda_gradgrad_comparison = \
            kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        self.precision = kwargs.get('precision', 2e-4)
        self.check_forward_only = kwargs.get('check_forward_only', False)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        if self.reference_fn is not None:
            out = test_case._forward(module, input)
            ref_input = deepcopy(input)
            ref_module = deepcopy(module)
            expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0], ref_module)
            test_case.assertEqual(out, expected_out, exact_dtype=False)
        if self.check_forward_only:
            return
        self.test_noncontig(test_case, module, input)

        if self.should_test_pickle:
            # TODO: do this with in-memory files as soon as torch.save will support it
            with tempfile.TemporaryFile() as f:
                test_case._forward(module, input)
                torch.save(module, f)
                f.seek(0)
                module_copy = torch.load(f)
                test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))

        self._do_test(test_case, module, input)

    def noncontiguize(self, obj):
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        elif isinstance(obj, tuple):
            return tuple(self.noncontiguize(o) for o in obj)
        tensor = obj
        ndim = tensor.dim()
        # Always making only the last dimension noncontiguous is easy to hide
        # bugs because .view(-1) will still work. So try to find a dim with size
        # > 1 and make that non-contiguous, i.e., stack + select on the
        # dimension directly after that.
        dim = ndim
        for d in range(ndim):
            if tensor.size(d) > 1:
                dim = d + 1
                break
        noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
        assert noncontig.numel() == 1 or noncontig.numel() == 0 or not noncontig.is_contiguous()
        noncontig.requires_grad = tensor.requires_grad
        return noncontig

    def test_noncontig(self, test_case, module, input):
        # check no scalars, can't make non-contig
        if isinstance(input, torch.Tensor) and input.dim() == 0:
            return
        if any(i.dim() == 0 for i in input if isinstance(i, torch.Tensor)):
            return

        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            if getattr(module, "return_indices", False):
                output = output[0]
            grad_output = output.new(output.shape).normal_()
            output = output.clone()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])

        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for contig_i, contig_g in product((True, False), repeat=2):
            i = input if contig_i else nc_input
            # Some ops, e.g., nn.Flatten, return gradient that shares
            # storage with the grad_output. Hence we copy here.
            go = deepcopy(grad_output if contig_g else nc_grad_output)
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                out = test_case._forward(module, i)
                if getattr(module, "return_indices", False):
                    out = out[0]
                grad = test_case._backward(module, i, out, go)

                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, atol=1e-4, rtol=0)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')

        cpu_input = self._get_input()
        type_map = {torch.double: torch.float}
        cpu_input_tuple = cpu_input if isinstance(cpu_input, tuple) else (cpu_input,)

        is_any_input_complex = any(isinstance(t, torch.Tensor) and t.dtype.is_complex for t in cpu_input_tuple)

        gpu_input_tuple = to_gpu(cpu_input_tuple, type_map=type_map)

        cpu_module = self.constructor(*self.constructor_args)
        gpu_module = self.constructor(*self.constructor_args).float().cuda()
        cpu_param = test_case._get_parameters(cpu_module)
        gpu_param = test_case._get_parameters(gpu_module)
        for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0]):
            gpu_p.data.copy_(cpu_p)

        test_case._zero_grad_input(cpu_input_tuple)
        test_case._zero_grad_input(gpu_input_tuple)
        test_case._zero_grad_parameters(cpu_module)
        test_case._zero_grad_parameters(gpu_module)
        cpu_output = test_case._forward(cpu_module, cpu_input_tuple)
        gpu_output = test_case._forward(gpu_module, gpu_input_tuple)
        if getattr(cpu_module, "return_indices", False):
            cpu_output = cpu_output[0]
            gpu_output = gpu_output[0]
        test_case.assertEqual(cpu_output, gpu_output, atol=self.precision, rtol=0, exact_dtype=False)

        # Run backwards on CPU and GPU and compare results
        for _ in range(5):
            cpu_gradOutput = cpu_output.clone().normal_()
            gpu_gradOutput = cpu_gradOutput.type_as(gpu_output)
            cpu_gradInput = test_case._backward(cpu_module, cpu_input_tuple, cpu_output, cpu_gradOutput)
            gpu_gradInput = test_case._backward(gpu_module, gpu_input_tuple, gpu_output, gpu_gradOutput)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
            for cpu_d_p, gpu_d_p in zip(cpu_param[1], gpu_param[1]):
                test_case.assertEqual(cpu_d_p, gpu_d_p, atol=self.precision, rtol=0)

        # Run double-backwards on CPU and GPU and compare results
        if self.check_gradgrad and not self.FIXME_no_cuda_gradgrad_comparison:
            cpu_output = cpu_module(*cpu_input_tuple)
            gpu_output = gpu_module(*gpu_input_tuple)
            if getattr(cpu_module, "return_indices", False):
                cpu_output = cpu_output[0]
                gpu_output = gpu_output[0]

            cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
            gpu_gradOutput = cpu_gradOutput.type_as(gpu_output).detach()
            gpu_gradOutput.requires_grad = True

            cpu_gradInputs = torch.autograd.grad(
                cpu_output,
                cpu_input_tuple + tuple(cpu_module.parameters()),
                cpu_gradOutput,
                create_graph=True)
            gpu_gradInputs = torch.autograd.grad(
                gpu_output,
                gpu_input_tuple + tuple(gpu_module.parameters()),
                gpu_gradOutput,
                create_graph=True)

            for cpu_d_i, gpu_d_i in zip(cpu_gradInputs, gpu_gradInputs):
                test_case.assertEqual(cpu_d_i, gpu_d_i, atol=self.precision, rtol=0, exact_dtype=False)

            # We mix output into the second backwards computation so that
            # torch.autograd.grad doesn't complain that some inputs
            # are unreachable (which can happen if you differentiate
            # only on the gradient.
            if is_any_input_complex:
                outputs_cpu = cpu_output.sum().abs() + sum(x.sum().abs() for x in cpu_gradInputs)
                outputs_gpu = gpu_output.sum().abs() + sum(x.sum().abs() for x in gpu_gradInputs)
            else:
                outputs_cpu = cpu_output.sum() + sum(x.sum() for x in cpu_gradInputs)
                outputs_gpu = gpu_output.sum() + sum(x.sum() for x in gpu_gradInputs)

            cpu_gg = torch.autograd.grad(
                outputs_cpu,
                cpu_input_tuple + (cpu_gradOutput,) + tuple(cpu_module.parameters()),
                retain_graph=True)
            gpu_gg = torch.autograd.grad(
                outputs_gpu,
                gpu_input_tuple + (gpu_gradOutput,) + tuple(gpu_module.parameters()),
                retain_graph=True)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
            for cpu_d_p, gpu_d_p in zip(cpu_gg, gpu_gg):
                test_case.assertEqual(cpu_d_p, gpu_d_p, atol=self.precision, rtol=0, exact_dtype=False)

        self.test_noncontig(test_case, gpu_module, gpu_input_tuple)


class InputVariableMixin:
    def _get_input(self):
        input = TestBase._get_input(self, False)  # type: ignore[arg-type]

        def map_variables(i):
            if isinstance(i, torch.Tensor):
                if i.is_floating_point() or i.is_complex():
                    i.requires_grad = True
                return i
            else:
                return type(i)(map_variables(elem) for elem in i)

        return map_variables(input)


class NewModuleTest(InputVariableMixin, ModuleTest):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.skip_double = kwargs.get('skip_double', False)
        self.skip_half = kwargs.get('skip_half', False)
        self.with_tf32 = kwargs.get('with_tf32', False)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.has_sparse_gradients = kwargs.get('has_sparse_gradients', False)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        self.gradcheck_fast_mode = kwargs.get('gradcheck_fast_mode', None)
        self.supports_forward_ad = kwargs.get('supports_forward_ad', False)
        self.supports_fwgrad_bwgrad = kwargs.get('supports_fwgrad_bwgrad', False)

    def _check_gradients(self, test_case, module, input_tuple):
        params = tuple(x for x in module.parameters())
        num_inputs = len(input_tuple)

        def fn_to_gradcheck(*inputs_and_params, **kwargs):
            assert not kwargs
            return test_case._forward(module, inputs_and_params[:num_inputs])

        # gradcheck doesn't support operators that take in dense inputs but
        # return sparse parameters. This only happens in the case of nn.Embedding
        # and nn.EmbeddingBag. Instead, we call `self.check_jacobian`, which
        # is a slightly different version of gradcheck that can handle this.
        if self.has_sparse_gradients:
            assert num_inputs == 1
            test_input_jacobian = torch.is_floating_point(input_tuple[0])
            test_case.check_jacobian(module, input_tuple[0], test_input_jacobian)
        else:
            test_case.assertTrue(gradcheck(fn_to_gradcheck, input_tuple + params,
                                           check_batched_grad=self.check_batched_grad,
                                           fast_mode=self.gradcheck_fast_mode,
                                           check_forward_ad=self.supports_forward_ad))

        if self.check_gradgrad:
            test_case.assertTrue(gradgradcheck(fn_to_gradcheck, input_tuple + params,
                                               check_batched_grad=self.check_batched_grad,
                                               fast_mode=self.gradcheck_fast_mode,
                                               check_fwd_over_rev=self.supports_fwgrad_bwgrad))

    def _do_test(self, test_case, module, input):
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        input_tuple = input if isinstance(input, tuple) else (input,)

        self._check_gradients(test_case, module, input_tuple)

        # check if module can be printed
        module.__repr__()

        if self.check_inplace:
            # check if the inplace variant of the module gives the same result
            # as the out-of-place

            # check_inplace doesn't support multiple input tensors, since we don't have any modules
            # that modify the inputs in-place and that accept more than one input
            assert len(input_tuple) == 1
            input = input_tuple[0]

            module_ip = self.constructor(*self.constructor_args, inplace=True)

            input_version = input._version
            with freeze_rng_state():
                output = module(input)
            test_case.assertEqual(input._version, input_version)

            input_ip = deepcopy(input)
            input_ip_clone = input_ip.clone()
            with freeze_rng_state():
                output_ip = module_ip(input_ip_clone)
            test_case.assertNotEqual(input_ip_clone._version, input_version)
            test_case.assertEqual(output, output_ip)
            grad = output.data.clone().normal_()
            if input.grad is not None:
                with torch.no_grad():
                    input.grad.zero_()
            if input_ip.grad is not None:
                with torch.no_grad():
                    input_ip.grad.zero_()
            output.backward(grad)
            output_ip.backward(grad)
            test_case.assertEqual(input.grad, input_ip.grad)

        def assert_module_parameters_are(tensor_type, device_id=None):
            for p in module.parameters():
                test_case.assertIsInstance(p, tensor_type)
                if device_id is not None:
                    test_case.assertEqual(p.get_device(), device_id)

        if all(isinstance(t, torch.LongTensor) for t in input_tuple) and TEST_CUDA:
            # check that cuda() moves module parameters to correct GPU device,
            # and that float() casts parameters correctly
            input_tuple = tuple(t.cuda() for t in input_tuple)
            module.float().cuda()
            module(*input_tuple)
            assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

            if torch.cuda.device_count() > 1:
                input_tuple = tuple(t.cuda(1) for t in input_tuple)
                module.cuda(1)
                with torch.cuda.device(1):
                    module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 1)  # type: ignore[attr-defined]
        else:
            # check that float()/double() casters work correctly
            def to_type(tensor, real, complex):
                if tensor.is_complex():
                    return tensor.to(complex)
                elif tensor.is_floating_point():
                    return tensor.to(real)
                else:
                    return tensor

            def to_half(x):
                # TODO: torch.complex32 when properly supported
                return to_type(x, torch.float16, None)

            def to_single(x):
                return to_type(x, torch.float32, torch.complex64)

            def to_double(x):
                return to_type(x, torch.float64, torch.complex128)

            # to float
            input_tuple = tuple(to_single(t) for t in input_tuple)
            module.float()
            module(*input_tuple)
            assert_module_parameters_are(torch.FloatTensor)

            # and back to double
            input_tuple = tuple(to_double(t) for t in input_tuple)
            module.double()
            module(*input_tuple)
            assert_module_parameters_are(torch.DoubleTensor)

            if TEST_CUDA and self.should_test_cuda:
                # check that cuda() moves module parameters to correct GPU device,
                # and that float() casts parameters correctly

                # to GPU0
                input_tuple = tuple(to_single(t).cuda() for t in input_tuple)
                module.float().cuda()
                module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

                # to CPU
                input_tuple = tuple(t.cpu() for t in input_tuple)
                module.cpu()
                module(*input_tuple)
                assert_module_parameters_are(torch.FloatTensor)

                # back to GPU0
                input_tuple = tuple(t.cuda() for t in input_tuple)
                module.cuda()
                module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

                # test that forwards of module runs correctly without cuDNN
                if self.cudnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        module(*input_tuple)
                        assert_module_parameters_are(torch.cuda.FloatTensor, 0)  # type: ignore[attr-defined]

                if torch.cuda.device_count() >= 2:
                    # test cross-GPU transfer works
                    # to GPU1
                    input_tuple = tuple(t.cuda(1) for t in input_tuple)
                    module.cuda(1)
                    with torch.cuda.device(1):
                        module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.FloatTensor, 1)  # type: ignore[attr-defined]

                if not self.skip_double:
                    # test double()
                    input_tuple = tuple(to_double(t).cuda() for t in input_tuple)
                    module.double().cuda()
                    module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.DoubleTensor, 0)  # type: ignore[attr-defined]

                # test half()
                if not self.skip_half:
                    input_tuple = tuple(to_half(t).cuda() for t in input_tuple)
                    module.half().cuda()
                    module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.HalfTensor, 0)  # type: ignore[attr-defined]
        torch.set_num_threads(num_threads)

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)


class CriterionTest(InputVariableMixin, TestBase):  # type: ignore[misc]
    # TODO: check that criterions don't ignore grad_output

    _required_arg_names = TestBase._required_arg_names.union({'target'})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.check_forward_only = kwargs.get('check_forward_only', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.check_half = kwargs.get('check_half', True)
        self.check_bfloat16 = kwargs.get('check_bfloat16', False)
        self.check_complex = kwargs.get('check_complex', False)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.with_tf32 = kwargs.get('with_tf32', True)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        # Check that these methods don't raise errors
        module.__repr__()
        str(module)

        target = self._get_target()

        if self.reference_fn is not None:
            out = test_case._forward_criterion(module, input, target, extra_args=self.extra_args)
            ref_args = (deepcopy(input), deepcopy(target)) + self.extra_args + (module,)
            expected_out = self.reference_fn(*ref_args)
            test_case.assertEqual(out, expected_out)

        if self.check_forward_only:
            return

        params = tuple(x for x in module.parameters())
        if not isinstance(input, tuple):
            inputs = (input,) + params + (target,)

            def apply_fn(input, target, *params):
                return module(input, target)
        else:
            inputs = input + params + (target,)

            def apply_fn(input1, input2, target, *params):  # type: ignore[misc]
                return module(input1, input2, target)

        gradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)

        if self.check_gradgrad:
            gradgradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)

    def test_cuda(self, test_case, dtype, extra_args=None):
        def convert_dtype(obj, dtype, requires_grad=False):
            if isinstance(obj, torch.Tensor):
                return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
            elif isinstance(obj, tuple):
                return tuple(convert_dtype(o, dtype, requires_grad) for o in obj)
            else:
                return obj

        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')

        cpu_input = self._get_input()
        cpu_target = self._get_target()
        cpu_module = self.constructor(*self.constructor_args)
        gpu_module = self.constructor(*self.constructor_args)

        # Convert input, target and module parameters to dtype
        cpu_input = convert_dtype(cpu_input, dtype, True)
        if cpu_target.is_floating_point() or cpu_target.is_complex():
            cpu_target = convert_dtype(cpu_target, dtype)
        cpu_module.type(dtype)
        gpu_module.type(dtype)

        # GPU setup
        gpu_input = to_gpu(cpu_input)
        gpu_target = to_gpu(cpu_target)
        gpu_module.cuda()

        # torch.HalfTensor doesn't support most operations, converting back to default
        if dtype in {torch.half, torch.bfloat16}:
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            # Loss modules with weights require consistent input/module weight types
            cpu_module = self.constructor(*self.constructor_args)

        cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
        gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
        # dtype used to be able to be None, so set precision in this way instead of a precision map
        test_case.assertEqual(cpu_output, gpu_output,
                              atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4, rtol=0, exact_dtype=False)

        cpu_gradInput = test_case._backward_criterion(
            cpu_module, cpu_input, cpu_output, cpu_target, extra_args=extra_args)
        gpu_gradInput = test_case._backward_criterion(
            gpu_module, gpu_input, gpu_output, gpu_target, extra_args=extra_args)
        # dtype used to be able to be None, so set precision in this way instead of a precision map
        test_case.assertEqual(cpu_gradInput, gpu_gradInput,
                              atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4, rtol=0, exact_dtype=False)

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)

    @property
    def extra_args(self):
        return self._get_arg('extra_args', False)


def _test_bfloat16_ops(test_case, op, device, inp_dims=(), prec=1e-2, scale_factor=None):
    # fp32 compute
    input1 = torch.randn(inp_dims, dtype=torch.float32, device=device, requires_grad=True)
    if scale_factor is not None:
        input1 = (torch.rand(inp_dims, dtype=torch.bfloat16, device=device) * scale_factor).float().requires_grad_()
    out1 = op(input1)
    grad_input1 = torch.randn_like(out1, device=device)
    out1.backward(grad_input1)

    # bfloat16 compute
    op_bfp16 = op.bfloat16()
    input2 = input1.detach().bfloat16().requires_grad_()
    grad_input2 = grad_input1.bfloat16()
    out2 = op_bfp16(input2)
    out2.backward(grad_input2)

    test_case.assertEqual(out1, out2, atol=prec, rtol=prec, exact_dtype=False)
    test_case.assertEqual(input1.grad.data, input2.grad.data, atol=prec, rtol=prec, exact_dtype=False)

def _test_module_empty_input(test_case, module, inp, check_size=True, inference=False):
    if not inference:
        inp.requires_grad_(True)
    out = module(inp)
    if not inference:
        gO = torch.rand_like(out)
        out.backward(gO)
    if check_size:
        test_case.assertEqual(out.size(), inp.size())
    if not inference:
        for p in module.parameters():
            if p.requires_grad:
                test_case.assertEqual(p.grad, torch.zeros_like(p.grad))
        test_case.assertEqual(inp.grad, torch.zeros_like(inp))


def _create_basic_net():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_dummy_param = nn.Parameter(torch.empty(3, 5))
            self.layer_dummy_buf = nn.Buffer(torch.zeros(1, 3, 3, 7))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Layer()
            self.dummy_param = nn.Parameter(torch.empty(3, 5))
            self.dummy_buf = nn.Buffer(torch.zeros(7, 3, 3, 1))

    l = Layer()
    n = Net()
    s = nn.Sequential(n, n)

    return l, n, s
