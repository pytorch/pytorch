# Torch
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors

import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401

# Testing utils
from torch import inf

assert torch.get_default_dtype() == torch.float32

L = 20
M = 10
S = 5


def unpack_variables(args):
    if isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args

class dont_convert(tuple):
    pass

non_differentiable = collections.namedtuple('non_differentiable', ['tensor'])

def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None, dtype=torch.float, device=None):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        def maybe_non_contig(tensor):
            if not non_contiguous or tensor.numel() < 2:
                return tensor.clone()

            return noncontiguous_like(tensor)

        def conjugate(tensor):
            return tensor.conj()

        if isinstance(arg, (torch.Size, dont_convert)):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = conjugate(torch.randn((), dtype=dtype, device=device))
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return conjugate(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device))).requires_grad_(requires_grad)
        # double check casting
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
            return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
        elif isinstance(arg, torch.Tensor):
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError("User provided tensor is real for a test that runs with complex dtype, ",
                                   "which is not supported for now")
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of v afterwards
            v = conjugate(maybe_non_contig(arg)).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            return map_arg(arg(dtype=dtype, device=device))
        else:
            return arg
    args_out = tuple(map_arg(arg) for arg in call_args)
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    return args_out, kwargs_out

# NB: JIT script tests for all nn functional interfaces, script mode does
# not support in_place operations yet, so no inplace operation tests added.
# removed all the deprecated functions
#
# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name(will be used at test name suffix,
#       'inplace' skips grad tests),                         // optional
#   (True, nonfusible_nodes, fusible_nodes) for autodiff     // optional
#   fn to determine if test should be skipped,               // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
#   kwargs for function,                                     // optional
# )
nn_functional_tests = [
    ('conv1d', (S, S, S), ((S, S, S),)),
    ('conv2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_transpose1d', (S, S, S), ((S, S, S),)),
    ('conv_transpose2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv_transpose3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_tbc', (S, S, S), ((S, S, S), (S,), 2)),
    ('avg_pool1d', (S, S, S), (3,)),
    ('avg_pool2d', (S, S, S, S), (3,), '', (True,)),
    ('avg_pool3d', (S, S, S, S, S), (3,)),
    ('fractional_max_pool2d', (S, S, S, S), (3, [2, 3],)),
    ('max_pool1d', (S, S, S), (2, 1)),
    ('max_pool1d', (S, S, S), (2, 1, 1, 1, False, True), 'with_indices'),
    ('max_pool2d', (S, S, S, S), (2, 1), '', (True, 'aten::max_pool2d_with_indices')),
    ('max_pool2d', (S, S, S, S), (2, 1, 1, 1, False, True), 'with_indices', (True, 'aten::max_pool2d_with_indices')),
    ('max_pool3d', (S, S, S, S, S), (2, 1)),
    ('max_unpool1d', torch.tensor([[[2., 4]]]), (torch.tensor([[[1, 3]]]), 2, 2, 0)),
    ('max_unpool2d', torch.tensor([[[[2., 4]]]]), (torch.tensor([[[[1, 3]]]]), 2, 2, 0)),
    ('max_unpool3d', torch.tensor([[[[[2., 4]]]]]), (torch.tensor([[[[[1, 3]]]]]), 2, 2, 0)),
    ('lp_pool1d', (S, S, S), (2., 3, 2,)),
    ('lp_pool2d', (S, S, S, S), (2., 3, 2,)),
    ('adaptive_max_pool1d', (S, S, S), (5,)),
    ('adaptive_max_pool2d', (S, S, S, S), ([5, 7],)),
    ('adaptive_max_pool3d', (S, S, S, S, S), ([3, 2, 2],)),
    ('adaptive_avg_pool1d', (S, S, S), (5,), '', (True,)),
    ('adaptive_avg_pool2d', (S, S, S, S), ([5, 7],), '', (True,)),
    ('adaptive_avg_pool3d', (S, S, S, S, S), ([3, 2, 2],), '', (True,)),
    ('dropout', (S, S, S), (0.5,), '', (True, 'aten::native_dropout')),
    ('alpha_dropout', (S, S, S), (0.5,)),
    ('dropout2d', (S, S, S), (0.5,)),
    ('dropout2d', (S, S, S, S), (0.5,), 'batched'),
    ('dropout3d', (S, S, S, S), (0.5,)),
    ('dropout3d', (S, S, S, S, S), (0.5,), 'batched'),
    ('feature_alpha_dropout', (S, S, S), (0.5,)),
    ('threshold', (S, S, S), (0.1, 2.), '', (True,)),
    ('threshold', (S, S, S), (0.1, 2., True), 'inplace'),
    ('relu', (S, S, S), (), '', (True,)),
    ('relu', (S, S, S), (), 'inplace'),
    ('glu', (S - 1, S - 1, S - 1), (),),
    ('hardtanh', (S, S, S), (-0.5, 0.5), '', (True,)),
    ('hardtanh', (S, S, S), (-0.5, 0.5, True), 'inplace'),
    ('relu6', (S, S, S), (), '', (True,)),
    ('relu6', (S, S, S), (True), 'inplace'),
    ('elu', (S, S, S), (0.9,),),
    ('elu', (S, S, S), (0.9, True), 'inplace'),
    ('selu', (S, S, S), (),),
    ('selu', (S, S, S), (True), 'inplace'),
    ('celu', (S, S, S), (0.9,),),
    ('celu', (S, S, S), (0.9, True), 'inplace'),
    ('leaky_relu', (S, S, S), (0.02,), '', (True,)),
    ('leaky_relu', (S, S, S), (0.02,), 'inplace'),
    ('rrelu', (S, S), (0.1, 0.3, False),),
    ('rrelu', (S, S), (0.1, 0.3, False, True), 'inplace'),
    ('hardshrink', (S, S, S), (0.4,), '', (True,)),
    ('tanhshrink', (S, S, S), (),),
    ('softsign', (S, S, S), (),),
    ('softplus', (S, S, S), (), '', (True,)),
    ('softmin', (S, S, S), (0,),),
    ('softmax', (S, S, S), (0,), '', (True,)),
    ('softmax', (S, S, S), (0, 3, torch.double), 'with_all_args', (True,)),
    ('tanh', (S, S, S), (), '', (True,)),
    ('sigmoid', (S, S, S), (), '', (True,)),
    ('silu', (S, S, S), (), '', (True,)),
    ('log_softmax', (S, S, S), (0,), '', (True,)),
    ('linear', (S, S), ((M, S),), '', (True, ['aten::linear'])),
    ('linear', (S, S), ((M, S), (M,)), 'addmm', (True, ['aten::linear'])),
    ('bilinear', (S, S, S), ((S, S, M), torch.zeros(M, S, M),),),
    ('embedding', torch.tensor([[1, 2, 4, 5], [4, 3, 2, 5]]), (torch.rand(6, 3), ), '', (True,)),
    ('embedding_bag', torch.tensor([1, 2, 4, 2]), (torch.rand(5, 3), torch.tensor([0, 4]),),),
    ('batch_norm', (S, S),
        (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), None, None, True, ),
        'training', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (0, S, S, S),
        (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
         non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), True, ),
        'size_zero', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (0, S, S, S),
        (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
         non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), True, ),
        'size_zero_inference', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S),
        (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
         non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), True, ),
        'with_weight_and_bias_training', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                            None, non_differentiable(torch.ones(S)), True, ),
        'with_only_bias_training', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                            non_differentiable(torch.randn(S)), None, True, ),
        'with_only_weight_training', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                            None, None, False, ),
        'inference', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                            non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), False, ),
        'with_weight_and_bias_inference', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                            None, non_differentiable(torch.ones(S)), False, ),
        'with_only_bias_inference', (True, 'aten::_batch_norm_impl_index')),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)),
                            non_differentiable(torch.randn(S)), None, False, ),
        'with_only_weight_inference', (True, 'aten::_batch_norm_impl_index')),
    ('instance_norm', (S, S, S), (non_differentiable(torch.zeros(S)), non_differentiable(torch.ones(S))),),
    ('layer_norm', (S, S, S, S), ([5],), '',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    ('layer_norm', (S, S, S, S), ([5], non_differentiable(torch.rand(S)),), 'with_only_weight',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    ('layer_norm', (S, S, S, S), ([5], None, non_differentiable(torch.rand(S)),), 'with_only_bias',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    ('layer_norm', (S, S, S, S), ([5], non_differentiable(torch.rand(S)),
                                  non_differentiable(torch.rand(S))), 'with_weight_and_bias',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index', 'aten::addcmul'])),
    ('group_norm', (S, S, S), (1, torch.rand(5),),),
    ('local_response_norm', (S, S, S), (2, ),),
    ('nll_loss', F.log_softmax(torch.randn(3, 5), dim=0), (torch.tensor([1, 0, 4]),), '',),
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2),),),
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2), True, True), 'full'),
    ('kl_div', F.log_softmax(torch.randn(S, 10), 1), (F.softmax(torch.randn(S, 10), 1),),),
    ('cross_entropy', (3, S), (torch.randint(S, (3,), dtype=torch.int64),),),
    ('binary_cross_entropy_with_logits', (3,), (torch.empty(3).random_(2), ),),
    ('smooth_l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('huber_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('mse_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('smooth_l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('huber_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('mse_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('margin_ranking_loss', (S,), ((S,), (S,)),),
    ('hinge_embedding_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('multilabel_soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('cosine_embedding_loss', (S, S), ((S, S), non_differentiable(torch.rand(S,))),),
    ('pixel_shuffle', (1, 9, 4, 4), (3,),),
    ('pixel_unshuffle', (1, 1, 12, 12), (3,),),
    ('affine_grid', (S, 2, 3), (torch.Size([S, 1, 7, 7]),),),
    ('pad', (3, 3, 4, 2), ([1, 1],),),
    ('pairwise_distance', (S, S), ((S, S),),),
    ('pdist', (S, S), (),),
    ('cosine_similarity', (S, S), ((S, S),),),
    ('triplet_margin_loss', (S, S), ((S, S), (S, S)),),
    ('normalize', (S, S, S), (),),
    ('unfold', (S, S, S, S), ([2, 3]),),
    ('fold', (1, 3 * 2 * 2, 12), ([4, 5], [2, 2]),),
    ('grid_sample', (S, S, S, S), (non_differentiable(torch.rand(S, S, S, 2)),),),
    ('gumbel_softmax', (S, S), (2.,), '', (True, ['aten::softmax', 'aten::add', 'aten::div'], ['aten::neg'])),
    ('gumbel_softmax', (S, S), (2., True,), 'hard', (True, ['aten::softmax', 'aten::add', 'aten::div'], ['aten::neg'])),
    ('multilabel_margin_loss', torch.tensor([[0.2, -0.2, 0.07]]), (torch.tensor([[0, 0, 1]]),),),
    ('multi_margin_loss', (S, S), (non_differentiable(torch.randint(S, (S, ), dtype=torch.int64)),
                                   1, 1., non_differentiable(torch.randn(S))),),
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(), (non_differentiable(torch.rand(3, 2)),
                                                           non_differentiable(torch.randn(3, 2))),),
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(),
        (non_differentiable(torch.rand(3, 2)),
         non_differentiable(torch.randn(3, 2)), None, None, 'mean'), 'size_average'),
    ('ctc_loss', torch.rand(S, S, S).log_softmax(2).detach().requires_grad_(),
     (torch.randint(1, S, (S, S), dtype=torch.long), torch.full((S,), S, dtype=torch.long),
      torch.randint(1, S, (S,), dtype=torch.long))),
    ('upsample', torch.randn(S, S, M, M), (None, 2.), 'with_scale'),
    ('upsample', torch.randn(S, S, M, M), (4,), 'with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'nearest_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'nearest_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'nearest_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'area_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'area_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'area_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'bilinear_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'bilinear_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'bilinear_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'bicubic_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'bicubic_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'bicubic_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'nearest_3d'),
    ('interpolate', torch.randn(S, M, M), (None, 2.), 'nearest_3d_with_scale'),
    ('interpolate', torch.randn(S, M, M), (4,), 'nearest_3d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'area_3d'),
    ('interpolate', torch.randn(S, M, M), (None, 2.), 'area_3d_with_scale'),
    ('interpolate', torch.randn(S, M, M), (4,), 'area_3d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'linear_3d'),
    ('interpolate', torch.randn(S, M, M), (None, 2.), 'linear_3d_with_scale'),
    ('interpolate', torch.randn(S, M, M), (4,), 'linear_3d_with_size'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'nearest_5d_with_scale'),
    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'nearest_5d_with_size'),
    ('interpolate', torch.zeros(3, 3, 3).view(1, 1, 3, 3, 3), (2,), 'area_5d'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'area_5d_with_scale'),
    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'area_5d_with_size'),
    ('interpolate', torch.zeros(3, 3, 3).view(1, 1, 3, 3, 3), (2,), 'trilinear_5d'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'trilinear_5d_with_scale'),
    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'trilinear_5d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2, None, 'nearest', None, False),
     'nearest_4d_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (4, None, 'nearest', None, False),
     'nearest_4d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2., 'bilinear', None, False),
     'bilinear_4d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (4, None, 'bilinear', None, False),
     'bilinear_4d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2., 'bicubic', None, False),
     'bicubic_4d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (4, None, 'bicubic', None, False),
     'bicubic_4d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (None, 2., 'nearest', None, False),
     'nearest_3d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (4, None, 'nearest', None, False),
     'nearest_3d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (None, 2., 'linear', None, False),
     'linear_3d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (4, None, 'linear', None, False),
     'linear_3d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2., 'nearest', None, False),
     'nearest_5d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (4, None, 'nearest', None, False),
     'nearest_5d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2., 'trilinear', None, False),
     'trilinear_5d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (4, None, 'trilinear', None, False),
     'trilinear_5d_with_size_not_recompute_scale_factor'),
]

script_template = '''
def the_method({}):
    return {}
'''

def value_to_literal(value):
    if isinstance(value, str):
        # Quotes string and escapes special characters
        return ascii(value)
    if isinstance(value, torch.Tensor):
        return 'torch.' + str(value)
    else:
        return str(value)

def get_call(method_name, func_type, args, kwargs):
    kwargs_str = ', '.join([k + '=' + value_to_literal(v) for k, v in kwargs.items()])
    self_arg = args[0]
    if(func_type == 'method'):
        args = args[1:]

    argument_str = ', '.join(args)
    argument_str += ', ' if len(args) and len(kwargs) else ''
    argument_str += kwargs_str

    if func_type == 'functional' or func_type == 'function':
        call = f'torch.{method_name}({argument_str})'
    elif func_type == 'method':
        call = f'{self_arg}.{method_name}({argument_str})'
    elif func_type == 'nn_functional':
        call = f'torch.nn.functional.{method_name}({argument_str})'
    else:
        raise TypeError('Unsupported function type')

    return call

def get_constant(x):
    if x == inf:
        return 'math.inf'
    if x == -inf:
        return '-math.inf'
    return x

def get_script_args(args):
    formals: List[str] = []
    tensors: List[Union[torch.Tensor, List[torch.Tensor]]] = []
    actuals: List[str] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            name = f'i{len(formals)}'
            formals.append(name)
            actuals.append(name)
            tensors.append(arg)
        elif is_iterable_of_tensors(arg):
            name = f'i{len(formals)}'
            formals.append(name + ': List[torch.Tensor]')
            actuals.append(name)
            tensors.append(list(arg))
        elif isinstance(arg, str):
            actuals.append(f"'{arg}'")
        else:
            actuals.append(str(get_constant(arg)))
    return (formals, tensors, actuals)

# create a script function from (name, func_type, output_process_fn),
# and returns the compiled function and example inputs
def gen_script_fn_and_args(method_name, func_type, *args, **kwargs):
    formals, tensors, actuals = get_script_args(args)
    call = get_call(method_name, func_type, actuals, kwargs)
    script = script_template.format(', '.join(formals), call)
    CU = torch.jit.CompilationUnit(script)
    return CU.the_method, tensors

# create a script function from (name, func_type),
# returns a function takes in (args, kwargs) and runs the compiled function
def create_script_fn(self, method_name, func_type):
    # function returns tuple containing original output and
    # filtered output to be used in checking gradients
    def script_fn(*args, **kwargs):
        fn, tensors = gen_script_fn_and_args(method_name, func_type, *args, **kwargs)
        self.assertExportImport(fn.graph, tensors)
        output = fn(*tensors)
        # skip type annotate function attributes for now, see: https://github.com/python/mypy/issues/2087
        script_fn.last_graph = fn.graph_for(*tensors)  # type: ignore[attr-defined]
        return output
    return script_fn

class SplitInputs:
    all_tensors: List[Any]
    tensor_args: List[Any]
    nontensor_args: List[Any]
    arg_types: List[str]
    tensor_kwargs: Dict[str, Any]
    kwarg_order: List[str]
    nontensor_kwargs: Dict[str, Any]
    kwarg_types: Dict[str, Any]

    @staticmethod
    def _is_tensor_input(arg):
        return isinstance(arg, torch.Tensor) or is_iterable_of_tensors(arg)

    def __init__(self, args, kwargs):
        self.arg_types = ['t' if self._is_tensor_input(arg) else 's' for arg in args]
        self.kwarg_types = {k: 't' if self._is_tensor_input(v) else 's' for k, v in kwargs.items()}
        self.tensor_args = [arg for arg in args if self._is_tensor_input(arg)]
        self.nontensor_args = [arg for arg in args if not self._is_tensor_input(arg)]
        self.tensor_kwargs = {k: v for k, v in kwargs.items() if self._is_tensor_input(v)}
        self.nontensor_kwargs = {k: v for k, v in kwargs.items() if not self._is_tensor_input(v)}
        self.all_tensors = [*self.tensor_args, *[v for k, v in self.tensor_kwargs.items()]]
        self.kwarg_order = [k for k, v in kwargs.items()]

    def nontensors_match(self, other: 'SplitInputs'):
        if self.arg_types != other.arg_types:
            return False
        if self.kwarg_types != other.kwarg_types:
            return False
        if self.kwarg_order != other.kwarg_order:
            return False
        if self.nontensor_args != other.nontensor_args:
            return False
        if self.nontensor_kwargs != other.nontensor_kwargs:
            return False
        return True

# make a new function where all non-tensor arguments in 'args' have been partially
# applied, and all tensor arguments remain.
# used to trace functions when some arguments are not tensors
def partial_apply_nontensors(fn, args, kwargs):
    inputs = SplitInputs(args, kwargs)

    def new_fn(*tensors_):
        tensors = iter(tensors_)
        full_args = [args[i] if s == 's' else next(tensors) for i, s in enumerate(inputs.arg_types)]
        full_kwargs = {k: kwargs[k] if s == 's' else next(tensors) for k, s in inputs.kwarg_types.items()}
        return fn(*full_args, **full_kwargs)

    return new_fn, inputs

# create a trace function from input fn
def create_traced_fn(self, fn, cache_traced_fn=False):
    def traced_fn(*inputs, **kwargs):
        # `check_trace` is set to False because check_trace is run with @no_grad
        # Also, `check_against_reference` already does all the checks
        # against python function
        fn_tensors, split_inputs = partial_apply_nontensors(fn, inputs, kwargs)
        if not cache_traced_fn or not hasattr(traced_fn, 'traced'):
            traced = torch.jit.trace(fn_tensors, split_inputs.all_tensors, check_trace=False)
            self.assertExportImport(traced.graph, split_inputs.all_tensors)
            output = traced(*split_inputs.all_tensors)
            if cache_traced_fn:
                traced_fn.traced = traced
                traced_fn.split_inputs = split_inputs
        else:
            # Guard to check that nontensor inputs are the same as during tracing
            self.assertTrue(traced_fn.split_inputs.nontensors_match(split_inputs))
            output = traced_fn.traced(*split_inputs.all_tensors)
            traced = traced_fn.traced
        # skip type annotate function attributes for now, see: https://github.com/python/mypy/issues/2087
        traced_fn.last_graph = traced.graph_for(*split_inputs.all_tensors)  # type: ignore[attr-defined]
        traced_fn.graph = traced.graph  # type: ignore[attr-defined]
        return output
    return traced_fn

# known to be failing in script
EXCLUDE_SCRIPT = {
    'test_norm_fro_default',
    'test_norm_fro_cpu',
    'test_norm_nuc',
    'test_norm_fro',
    'test_norm_nuc_batched',

    # aten op has additional cudnn argument
    'test_nn_unfold',

    # flaky test - TODO fix
    'test_nn_ctc_loss',

    # unknown builtin op
    'test_nn_fold',

    # jit doesn't support sparse tensors.
    'test_to_sparse',
    'test_to_sparse_dim',
}

# generates a script function and set of example inputs
# from a specified test in the format of nn_functional_tests
def get_nn_functional_compiled_fn_and_inputs(name, self_size, args, variant_name='', *extra_args):
    test_name = 'test_nn_' + name

    if variant_name != '':
        test_name = test_name + '_' + variant_name

    no_grad = variant_name == 'inplace'

    self_variable = create_input((self_size,))[0][0]
    kwargs = None

    # need to record this because methods can change the size (e.g. unsqueeze)
    args_variable, kwargs_variable = create_input(args)

    self_tensor = deepcopy(self_variable.data)
    args_tensor = deepcopy(unpack_variables(args_variable))

    f_args_variable = (self_variable,) + args_variable
    f_args_tensor = (self_tensor,) + args_tensor
    with torch._jit_internal._disable_emit_hooks():
        script_fn, inputs = gen_script_fn_and_args(name, "nn_functional", *f_args_variable)
    return script_fn, inputs


# additional modules test
# TODO: delete this list once we make all nn_tests work
additional_module_tests = [
    {
        'module_name': 'Bilinear',
        'constructor_args': (S, S, M),
        'input_size': (S, S),
        'extra_args': ((S, S),)
    },
    {
        'module_name': 'RNNCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'LSTMCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'GRUCell',
        'constructor_args': (S, S),
        'input_size': (S, S),
    },
    {
        'module_name': 'MultiheadAttention',
        'constructor_args': (128, 8),
        'input_size': (10, 8, 128),
        'extra_args': (torch.randn(10, 8, 128), torch.randn(10, 8, 128)),
        'slowTest': True
    },
    {
        'module_name': 'Transformer',
        'constructor_args': (1, 1, 1, 1, 2),
        'input_size': (3, 1, 1),
        'extra_args': (torch.randn(1, 1, 1),),
        'slowTest': True
    }
]

EXCLUDE_SCRIPT_MODULES = {
    'test_nn_AdaptiveAvgPool2d_tuple_none',
    'test_nn_AdaptiveAvgPool3d_tuple_none',
    'test_nn_AdaptiveMaxPool2d_tuple_none',
    'test_nn_AdaptiveMaxPool3d_tuple_none',

    # Doesn't use future division, so this is not supported
    'test_nn_CrossMapLRN2d',
    # Derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented
    'test_nn_TransformerDecoderLayer_gelu_activation',
    'test_nn_TransformerDecoderLayer_relu_activation',
    'test_nn_TransformerEncoderLayer_gelu_activation',
    'test_nn_TransformerEncoderLayer_relu_activation',
    'test_nn_Transformer_multilayer_coder',
}

script_method_template = '''
def forward({}):
    return {}
'''

def create_script_module(self, nn_module, constructor_args, *args, **kwargs):
    def script_module(*args, **kwargs):
        formals, tensors, actuals = get_script_args(args)

        method_args = ', '.join(['self'] + actuals)
        call_args_str = ', '.join(actuals)
        call = f"self.submodule({call_args_str})"
        script = script_method_template.format(method_args, call)

        submodule_constants = []
        if kwargs.get('is_constant'):
            submodule_constants = ['submodule']

        # Create module to use the script method
        class TheModule(torch.jit.ScriptModule):
            __constants__ = submodule_constants

            def __init__(self):
                super().__init__()
                self.submodule = nn_module(*constructor_args)

        def make_module(script):
            module = TheModule()
            # check __repr__
            str(module)
            module.define(script)
            return module

        module = make_module(script)
        if self:
            self.assertExportImportModule(module, tensors)
            module(*args)
        # skip type annotate function attributes for now, see: https://github.com/python/mypy/issues/2087
        create_script_module.last_graph = module.graph  # type: ignore[attr-defined]
        return module
    return script_module

def check_alias_annotation(method_name, args, kwargs, *, aten_name, func_type='method'):
    formals, tensors, actuals = get_script_args(args)
    call = get_call(method_name, func_type, actuals, kwargs)
    script = script_template.format(', '.join(formals), call)
    CU = torch.jit.CompilationUnit(script)
    # to clean up IR
    torch._C._jit_pass_inline(CU.the_method.graph)
    torch._C._jit_pass_constant_propagation(CU.the_method.graph)
    torch._C._jit_check_alias_annotation(CU.the_method.graph, tuple(tensors), aten_name)

def get_nn_module_name_from_kwargs(**kwargs):
    if 'module_name' in kwargs:
        return kwargs['module_name']
    elif 'fullname' in kwargs:
        return kwargs['fullname']
    elif 'constructor' in kwargs:
        return kwargs['constructor'].__name__

def get_nn_mod_test_name(**kwargs):
    if 'fullname' in kwargs:
        test_name = kwargs['fullname']
    else:
        test_name = get_nn_module_name_from_kwargs(**kwargs)
        if 'desc' in kwargs:
            test_name = f"{test_name}_{kwargs['desc']}"
    return f'test_nn_{test_name}'

def get_nn_module_class_from_kwargs(**kwargs):
    name = get_nn_module_name_from_kwargs(**kwargs)
    index = name.find("_")
    if index == -1:
        return name
    else:
        return name[0:name.find("_")]

def try_get_nn_module_compiled_mod_and_inputs(*args, **kwargs):
    name = get_nn_module_name_from_kwargs(**kwargs)

    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        # eval() is not supported, so skip these tests
        return

    test_name = name
    if 'desc' in kwargs:
        test_name = f"{test_name}_{kwargs['desc']}"
    test_name = get_nn_mod_test_name(**kwargs)

    if test_name in EXCLUDE_SCRIPT_MODULES:
        return
    if 'constructor' in kwargs:
        nn_module = kwargs['constructor']
    else:
        nn_module = getattr(torch.nn, name)

    if "FunctionalModule" in str(nn_module):
        return

    if 'constructor_args_fn' in kwargs:
        constructor_args = kwargs['constructor_args_fn']()
    else:
        constructor_args = kwargs.get('constructor_args', ())

    # Set up inputs from tuple of sizes or constructor fn
    input_dtype = torch.double
    if 'input_fn' in kwargs:
        input = kwargs['input_fn']()
        if isinstance(input, torch.Tensor):
            input = (input,)

        if all(tensor.is_complex() for tensor in input):
            input_dtype = torch.cdouble
    else:
        input = (kwargs['input_size'],)

    # Extra parameters to forward()
    if 'extra_args' in kwargs:
        input = input + kwargs['extra_args']

    if 'target_size' in kwargs:
        input = input + (kwargs['target_size'],)
    elif 'target_fn' in kwargs:
        if torch.is_tensor(input):
            input = (input,)
        input = input + (kwargs['target_fn'](),)

    args_variable, kwargs_variable = create_input(input, dtype=input_dtype)
    f_args_variable = deepcopy(unpack_variables(args_variable))
    out_var = deepcopy(f_args_variable)

    args, mod = f_args_variable, create_script_module(None, nn_module, constructor_args, *f_args_variable)(*f_args_variable)

    return mod, out_var


def get_all_nn_module_tests():
    return module_tests + new_module_tests + additional_module_tests
