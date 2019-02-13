import torch
from torch._six import inf, nan, istuple
from functools import reduce, wraps
from operator import mul, itemgetter
from torch.autograd import Variable, Function, detect_anomaly
from torch.testing import make_non_contiguous
from common_utils import (skipIfNoLapack,
                          prod_single_zero, random_square_matrix_of_rank,
                          random_symmetric_matrix, random_symmetric_psd_matrix,
                          random_symmetric_pd_matrix, make_nonzero_det,
                          random_fullrank_matrix_distinct_singular_value, set_rng_seed)


def index_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape).mul_(max_indices).floor_().long()
    return index


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


def gather_variable(shape, index_dim, max_indices, duplicate=False):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.LongTensor(*shape)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            torch.randperm(max_indices)[:shape[batch_dim]])
    if duplicate:
        index.select(batch_dim, 0).copy_(index.select(batch_dim, 1))
    return index


def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


def mask_not_all_zeros(shape):
    assert len(shape) > 0
    while True:
        result = torch.randn(shape).gt(0)
        if result.sum() > 0:
            return result


def uniform_scalar(offset=0, requires_grad=False):
    v = torch.rand(()) + offset
    v.requires_grad = requires_grad
    return v


def normal_scalar_clamp(amin, amax, requires_grad=False):
    v = torch.randn(()).clamp(amin, amax)
    v.requires_grad = requires_grad
    return v


def prod_zeros(dim_size, dim_select):
    assert len(dim_select) == 2
    result = torch.randn(dim_size, dim_size, dim_size)
    result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
    result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
    result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()
    return result


class non_differentiable(object):
    def __init__(self, tensor):
        self.tensor = tensor


class dont_convert(tuple):
    pass


class NoArgsClass(object):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration()
    next = __next__  # Python 2 compatibility

    def __len__(self):
        return 0

NO_ARGS = NoArgsClass()
L = 20
M = 10
S = 5


# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name (will be used at test name suffix),    // optional
#   indices for possible dim arg,                            // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
# )
def method_tests():
    set_rng_seed(0)
    return [
        ('add', (S, S, S), ((S, S, S),)),
        ('add', (S, S, S), ((S, S),), 'broadcast_rhs'),
        ('add', (S, S), ((S, S, S),), 'broadcast_lhs'),
        ('add', (S, 1, S), ((M, S),), 'broadcast_all'),
        ('add', (), ((),), 'scalar'),
        ('add', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('add', (), ((S, S, S),), 'scalar_broadcast_lhs'),
        ('add', (S, S, S), (3.14,), 'constant'),
        ('add', (), (3.14,), 'scalar_constant'),
        ('__radd__', (S, S, S), (3.14,), 'constant'),
        ('__radd__', (), (3.14,), 'scalar_constant'),
        ('sub', (S, S, S), ((S, S, S),)),
        ('sub', (S, S, S), ((S, S),), 'broadcast_rhs'),
        ('sub', (S, S), ((S, S, S),), 'broadcast_lhs'),
        ('sub', (S, 1, S), ((M, S),), 'broadcast_all'),
        ('sub', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('sub', (), ((S, S, S),), 'scalar_broadcast_lhs'),
        ('sub', (S, S, S), (3.14,), 'constant'),
        ('sub', (), (3.14,), 'scalar_constant'),
        ('__rsub__', (S, S, S), (3.14,), 'constant'),
        ('__rsub__', (), (3.14,), 'scalar_constant'),
        ('mul', (S, S, S), ((S, S, S),)),
        ('mul', (), ((),), 'scalar'),
        ('mul', (S, S, S), ((S, S),), 'broadcast_rhs'),
        ('mul', (S, S), ((S, S, S),), 'broadcast_lhs'),
        ('mul', (S, 1, S), ((M, S),), 'broadcast_all'),
        ('mul', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('mul', (), ((S, S, S),), 'scalar_broadcast_lhs'),
        ('mul', (S, S, S), (3.14,), 'constant'),
        ('mul', (), (3.14,), 'scalar_constant'),
        ('__rmul__', (S, S, S), (3.14,), 'constant'),
        ('__rmul__', (), (3.14,), 'scalar_constant'),
        ('div', (S, S, S), (torch.rand(S, S, S) + 0.1,)),
        ('div', (S, S, S), (torch.rand(S, S) + 0.1,), 'broadcast_rhs'),
        ('div', (S, S), (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs'),
        ('div', (S, 1, S), (torch.rand(M, S) + 0.1,), 'broadcast_all'),
        ('div', (), (uniform_scalar(0.1),), 'scalar'),
        ('div', (S, S, S), (uniform_scalar(0.1),), 'scalar_broadcast_rhs'),
        ('div', (), (uniform_scalar(0.1),), 'scalar_broadcast_lhs'),
        ('div', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant'),
        ('__rdiv__', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant'),
        ('div', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant'),
        ('__rdiv__', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant'),
        ('pow', torch.rand(S, S, S) + 1e-3, (torch.rand(S, S, S) + 0.1,)),
        ('pow', torch.rand(S, S, S) + 1e-3, (torch.rand(1,) + 0.1,), 'broadcast_rhs'),
        ('pow', torch.rand(1,) + 1e-3, (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs'),
        ('pow', torch.rand(S, 1, S) + 1e-3, (torch.rand(1, S, 1) + 0.1,), 'broadcast_all'),
        ('pow', uniform_scalar(1e-3, requires_grad=True), (uniform_scalar(0.1),), 'scalar'),
        ('pow', torch.rand(S, S, S) + 1e-3, (uniform_scalar(0.1),), 'scalar_broadcast_rhs'),
        ('pow', uniform_scalar(1e-3, requires_grad=True), (torch.rand(S, S, S) + 0.1,), 'scalar_broadcast_lhs'),
        ('pow', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant'),
        ('__rpow__', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant'),
        ('pow', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant'),
        ('__rpow__', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant'),
        ('transpose', (1, 2, 3), (1, 2), 'dim', [0, 1]),
        ('transpose', (), (0, 0), 'scalar'),
        ('transpose', (1,), (0, 0), '1d'),
        ('transpose', torch.rand(L, L), (0, 1), '2d'),
        ('transpose', torch.rand(S, S, S), (2, 0), '3d'),
        ('t', (1, 2), NO_ARGS),
        ('view', (S, S, S), (S * S, S),),
        ('view', (S, S, S), (torch.Size([S * S, S]),), 'size'),
        ('view', (S,), (S,), '1d'),
        ('view', (), (dont_convert(()),), 'scalar_to_scalar'),
        ('view', (), (1,), 'scalar_to_1d'),
        ('reshape', (S, S, S), (S * S, S),),
        ('reshape', (S, S, S), (torch.Size([S * S, S]),), 'size'),
        ('reshape', (S,), (S,), '1d'),
        ('reshape', (), (dont_convert(()),), 'scalar_to_scalar'),
        ('reshape', (), (1,), 'scalar_to_1d'),
        ('reshape_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
        ('reshape_as', (), (non_differentiable(torch.tensor(42.)),), 'scalar'),
        ('reshape_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
        ('flip', (S, S, S), ([0],), 'd0'),
        ('flip', (S, S, S), ([0, 1, 2],), 'd012'),
        ('flip', (S, S, S), ([0, 2],), 'd02'),
        ('flip', (S, S, S), ([2, 0],), 'd20'),
        ('flip', (S, S, S), ([-1],), 'neg_d'),
        ('roll', (S, S, S), (0, 0), 'd0'),
        ('roll', (S, S, S), (1, 2), 'd12'),
        ('roll', (S, S, S), (0, 2,), 'd02'),
        ('roll', (S, S, S), (2, 0,), 'd20'),
        ('roll', (S, S, S), (-1, 0), 'neg_shift'),
        ('roll', (S, S, S), (10000, 1), 'loop_shift'),
        ('roll', (S, S, S), (2,), 'flattened'),
        ('roll', (S, S, S), ([1, 2, -1], [0, 1, 2]), 'three_dims'),
        ('rot90', (S, S, S), (1, [0, 1],), 'k1_d01'),
        ('rot90', (S, S, S), (1, [1, 2],), 'k1_d12'),
        ('rot90', (S, S, S), (1, [1, -1],), 'k1_neg_d'),
        ('rot90', (S, S, S), (), 'default'),
        ('view_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
        ('view_as', (), (non_differentiable(torch.tensor(5.5)),), 'scalar'),
        ('view_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
        ('expand', (S, 1, 1), (S, S, S)),
        ('expand', (torch.Size([S, 1, S]),), (S, S, S), 'size'),
        ('expand', (S, 1), (S, S, S), 'new_dim'),
        ('expand', (1,), (S, S, S), '1_element'),
        ('expand', (1, S), (1, 1, S), 'new_dim_front_old_front_1'),
        ('expand', (), (dont_convert(()),), 'scalar_to_scalar'),
        ('expand', (), (1, 3, 2), 'scalar_to_dims'),
        ('expand_as', (S, 1, 1), (torch.rand(S, S, S),)),
        ('exp', (S, S, S), NO_ARGS),
        ('exp', (), NO_ARGS, 'scalar'),
        ('expm1', (S, S, S), NO_ARGS),
        ('expm1', (), NO_ARGS, 'scalar'),
        ('erf', torch.rand(S, S, S), NO_ARGS),
        ('erf', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar'),
        ('erfc', torch.rand(S, S, S), NO_ARGS),
        ('erfc', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar'),
        ('erfinv', torch.rand(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
        ('erfinv', normal_scalar_clamp(-0.9, 0.9, requires_grad=True), NO_ARGS, 'scalar'),
        ('log', torch.rand(S, S, S) + 1e-2, NO_ARGS),
        ('log', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
        ('log10', torch.rand(S, S, S) + 1e-2, NO_ARGS),
        ('log10', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
        ('log1p', torch.rand(S, S, S), NO_ARGS),
        ('log1p', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar'),
        ('log2', torch.rand(S, S, S) + 1e-2, NO_ARGS),
        ('log2', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
        ('tanh', (S, S, S), NO_ARGS),
        ('tanh', (), NO_ARGS, 'scalar'),
        ('sigmoid', (S, S, S), NO_ARGS),
        ('sigmoid', (), NO_ARGS, 'scalar'),
        ('sinh', (S, S, S), NO_ARGS),
        ('sinh', (), NO_ARGS, 'scalar'),
        ('cosh', (S, S, S), NO_ARGS),
        ('cosh', (), NO_ARGS, 'scalar'),
        ('abs', (S, S, S), NO_ARGS),
        ('abs', (), NO_ARGS, 'scalar'),
        ('clamp', (S, S, S), (0, 1)),
        ('clamp', (S, S, S), (None, 0.5), 'min'),
        ('clamp', (S, S, S), (0.5, None), 'max'),
        ('clamp', (), (0, 1), 'scalar'),
        ('clamp', (), (None, 0.5), 'min_scalar'),
        ('clamp', (), (0.5, None), 'max_scalar'),
        ('sqrt', torch.rand(S, S, S) + 5e-4, NO_ARGS),
        ('sqrt', uniform_scalar(5e-4, requires_grad=True), NO_ARGS, 'scalar'),
        ('sin', (S, S, S), NO_ARGS),
        ('sin', (), NO_ARGS, 'scalar'),
        ('cos', (S, S, S), NO_ARGS),
        ('cos', (), NO_ARGS, 'scalar'),
        ('tan', torch.randn(S, S, S).clamp(-1, 1), NO_ARGS),
        ('asin', torch.randn(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
        ('acos', torch.randn(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
        ('atan', (S, S, S), NO_ARGS),
        ('atan', (), NO_ARGS, 'scalar'),
        ('atan2', (S, S, S), ((S, S, S),)),
        ('atan2', (), ((),), 'scalar'),
        ('atan2', (S, S, S), ((S,),), 'broadcast_rhs'),
        ('atan2', (S,), ((S, S, S),), 'broadcast_lhs'),
        ('atan2', (S, 1, S), ((S, S),), 'broadcast_all'),
        ('reciprocal', torch.rand(S, S, S) + 0.1, NO_ARGS),
        ('reciprocal', uniform_scalar(0.1, requires_grad=True), NO_ARGS, 'scalar'),
        ('round', (S, S, S), NO_ARGS),
        ('round', (), NO_ARGS, 'scalar'),
        ('sign', (S, S, S), NO_ARGS),
        ('sign', (), NO_ARGS, 'scalar'),
        ('trunc', (S, S, S), NO_ARGS),
        ('trunc', (), NO_ARGS, 'scalar'),
        ('floor', (S, S, S), NO_ARGS),
        ('floor', (), NO_ARGS, 'scalar'),
        ('ceil', (S, S, S), NO_ARGS),
        ('ceil', (), NO_ARGS, 'scalar'),
        ('rsqrt', torch.rand(S, S, S) + 1e-2, NO_ARGS),
        ('rsqrt', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
        ('frac', (S, S, S), NO_ARGS),
        ('frac', (), NO_ARGS, 'scalar'),
        ('fmod', (S, S, S), (1.5,)),
        ('fmod', (), (1.5,), 'scalar'),
        ('fmod', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
        ('fmod', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
        ('fmod', (S, S, S), (non_differentiable(torch.rand(S) + 1.5),), 'tensor_broadcast_rhs'),
        ('fmod', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
        ('fmod', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
        ('fmod', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
        ('fmod', (S, S, S), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor_broadcast_rhs'),
        ('remainder', (S, S, S), (1.5,)),
        ('remainder', (), (1.5,), 'scalar'),
        ('remainder', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
        ('remainder', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
        ('remainder', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
        ('remainder', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
        ('remainder', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
        ('lerp', (S, S, S), ((S, S, S), 0.4)),
        ('lerp', (S, S, S), ((S,), 0.4), 'broadcast_rhs'),
        ('lerp', (S,), ((S, S, S), 0.4), 'broadcast_lhs'),
        ('lerp', (S, 1, S), ((S, S), 0.4), 'broadcast_all'),
        ('lerp', (), ((), 0.4), 'scalar'),
        ('lerp', (S, S, S), ((), 0.4), 'scalar_broadcast_rhs'),
        ('lerp', (), ((S, S, S), 0.4), 'scalar_broadcast_lhs'),
        ('max', (S, S, S), NO_ARGS),
        ('max', (S, S, S), (1,), 'dim', [0]),
        ('max', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('max', (), NO_ARGS, 'scalar'),
        ('max', (), (0,), 'scalar_dim', [0]),
        ('max', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('max', (S, S, S), ((S, S, S),), 'elementwise'),
        ('max', (S, S, S), ((S,),), 'elementwise_broadcast_rhs'),
        ('max', (S,), ((S, S, S),), 'elementwise_broadcast_lhs'),
        ('max', (S, 1, S), ((S, S),), 'elementwise_broadcast_all'),
        ('max', (), ((),), 'scalar_elementwise'),
        ('max', (S, S, S), ((),), 'scalar_elementwise_broadcast_rhs'),
        ('max', (), ((S, S, S),), 'scalar_elementwise_broadcast_lhs'),
        ('min', (S, S, S), NO_ARGS),
        ('min', (S, S, S), (1,), 'dim', [0]),
        ('min', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('min', (), NO_ARGS, 'scalar'),
        ('min', (), (0,), 'scalar_dim', [0]),
        ('min', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('min', (S, S, S), ((S, S, S),), 'elementwise'),
        ('min', (S, S, S), ((S,),), 'elementwise_broadcast_rhs'),
        ('min', (S,), ((S, S, S),), 'elementwise_broadcast_lhs'),
        ('min', (S, 1, S), ((S, S),), 'elementwise_broadcast_all'),
        ('min', (), ((),), 'scalar_elementwise'),
        ('min', (S, S, S), ((),), 'scalar_elementwise_broadcast_rhs'),
        ('min', (), ((S, S, S),), 'scalar_elementwise_broadcast_lhs'),
        ('mean', (S, S, S), NO_ARGS),
        ('mean', (S, S, S), (1,), 'dim', [0]),
        ('mean', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('mean', (), NO_ARGS, 'scalar'),
        ('mean', (), (0,), 'scalar_dim', [0]),
        ('mean', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('kthvalue', (S, S, S), (2,)),
        ('kthvalue', (), (1,), 'scalar'),
        ('kthvalue', (S, S, S), (2, 1,), 'dim', [1]),
        ('kthvalue', (), (1, 0,), 'scalar_dim', [1]),
        ('kthvalue', (S, S, S), (2, 1, True,), 'keepdim_dim', [1]),
        ('kthvalue', (), (1, 0, True), 'scalar_keepdim_dim', [1]),
        ('kthvalue', (S,), (2, 0,), 'dim_1d', [1]),
        ('kthvalue', (S,), (2, 0, True,), 'keepdim_dim_1d', [1]),
        ('median', (S, S, S), NO_ARGS),
        ('median', (S, S, S), (1,), 'dim', [0]),
        ('median', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('median', (), NO_ARGS, 'scalar'),
        ('median', (), (0,), 'scalar_dim', [0]),
        ('median', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('mode', (S, S, S), NO_ARGS),
        ('mode', (S, S, S), (1,), 'dim', [0]),
        ('mode', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('mode', (), NO_ARGS, 'scalar'),
        ('mode', (), (0,), 'scalar_dim', [0]),
        ('mode', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('sum', (S, S, S), NO_ARGS),
        ('sum', (S, S, S), (1,), 'dim', [0]),
        ('sum', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('sum', (), NO_ARGS, 'scalar'),
        ('sum', (), (0,), 'scalar_dim', [0]),
        ('sum', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('sum', (S, S, S), ([1, 2],), 'multi_dim'),
        ('sum', (S, S, S), ([1, 2], True,), 'multi_dim_keepdim'),
        ('prod', (S, S, S), NO_ARGS),
        ('prod', (S, S, S), (1,), 'dim', [0]),
        ('prod', (S, S, S), (1, True,), 'keepdim_dim', [0]),
        ('prod', (), NO_ARGS, 'scalar'),
        ('prod', (), (0,), 'scalar_dim', [0]),
        ('prod', (), (0, True,), 'scalar_keepdim_dim', [0]),
        ('prod', prod_zeros(S, [0, 1]), NO_ARGS, 'zerodims2'),
        ('prod', prod_zeros(S, [0, 2]), NO_ARGS, 'zerodims1'),
        ('prod', prod_zeros(S, [1, 2]), NO_ARGS, 'zerodims0'),
        ('prod', prod_zeros(S, [0, 1]), (1,), 'zeros_dims2', [0]),
        ('prod', prod_zeros(S, [0, 2]), (1,), 'zeros_dims1', [0]),
        ('prod', prod_zeros(S, [1, 2]), (1,), 'zeros_dims0', [0]),
        ('prod', prod_zeros(S, [0, 1]), (1, True), 'keepdim_zeros_dims2', [0]),
        ('prod', prod_zeros(S, [0, 2]), (1, True), 'keepdim_zeros_dims1', [0]),
        ('prod', prod_zeros(S, [1, 2]), (1, True), 'keepdim_zeros_dims0', [0]),
        ('prod', prod_single_zero(S), NO_ARGS, 'single_zero'),
        ('prod', (torch.tensor(0., requires_grad=True)), NO_ARGS, 'scalar_zero'),
        ('prod', (torch.tensor(0., requires_grad=True)), (0,), 'scalar_dim_zero', [0]),
        ('prod', (torch.tensor(0., requires_grad=True)), (0, True,), 'scalar_keepdim_dim_zero', [0]),
        ('var', (S, S, S), NO_ARGS),
        ('var', (S, S, S), (1,), 'dim', [0]),
        ('var', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
        ('var', (S,), (0,), 'dim_1d', [0]),
        ('var', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
        ('std', (S, S, S), NO_ARGS),
        ('std', (S, S, S), (1,), 'dim', [0]),
        ('std', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
        ('std', (S,), (0,), 'dim_1d', [0]),
        ('std', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
        ('renorm', (S, S, S), (2, 1, 0.5), 'dim', [1]),
        ('renorm', (S, S, S), (1, 2, 3), 'norm_1'),
        ('renorm', (S, S, S), (inf, 2, 0.5), 'norm_inf'),
        ('repeat', (S,), (2,), 'single_number'),
        ('repeat', (), (2, 3), 'scalar'),
        ('repeat', (2, 2), (3, 2)),
        ('repeat', (2, 2), (1, 3, 1, 2), 'unsqueeze'),
        ('cumsum', (S, S, S), (0,), 'dim0', [0]),
        ('cumsum', (S, S, S), (1,), 'dim1', [0]),
        ('cumsum', (S, S, S), (1,), 'dim1_cast', [0], (), lambda x: x, {'dtype': torch.float64}),
        ('cumsum', (), (0,), 'dim0_scalar', [0]),
        ('cumprod', (S, S, S), (0,)),
        ('cumprod', (S, S, S), (1,), 'dim1', [0]),
        ('cumprod', (), (0,), 'scalar'),
        ('cumprod', (torch.tensor(0., requires_grad=True)), (0,), 'scalar_zeros'),
        ('cumprod', prod_zeros(S, [0, 1]), (1,), 'zeros_dim2', [0]),
        ('cumprod', prod_zeros(S, [0, 2]), (1,), 'zeros_dim1', [0]),
        ('cumprod', prod_zeros(S, [1, 2]), (1,), 'zeros_dim0', [0]),
        ('cumprod', prod_zeros(S, [1, 2]), (1,), 'zeros_dim0_cast', [0], (), lambda x: x, {'dtype': torch.float64}),
        ('unfold', (), (0, 1, 1), 'scalar', [0]),
        ('unfold', (S, S, S, S), (1, 3, 1), '', [0]),
        ('unfold', (S, S, S), (2, 3, 2), 'lastdim', [0]),
        ('addmm', (S, M), ((S, S), (S, M)),),
        ('addmm', (1,), ((S, S), (S, M)), 'broadcast_lhs'),
        ('addmm', (S, M), ((S, S), (S, M)), 'coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addmm', (1,), ((S, S), (S, M)), 'broadcast_lhs_coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addmm', (), ((S, S), (S, M)), 'scalar_broadcast_lhs'),
        ('addmm', (), ((S, S), (S, M)), 'scalar_broadcast_lhs_coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addbmm', (S, M), ((S, S, S), (S, S, M)),),
        ('addbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs'),
        ('addbmm', (S, M), ((S, S, S), (S, S, M)), 'coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs_coef',
         (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs'),
        ('addbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs_coef', (), (), lambda x: x,
         {'beta': 0.2, 'alpha': 0.6}),
        ('baddbmm', (S, S, M), ((S, S, S), (S, S, M)),),
        ('baddbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs'),
        ('baddbmm', (S, S, M), ((S, S, S), (S, S, M)), 'coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('baddbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs_coef',
         (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('baddbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs'),
        ('baddbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs_coef', (), (), lambda x: x,
         {'beta': 0.2, 'alpha': 0.6}),
        ('addmv', (S,), ((S, M), (M,)),),
        ('addmv', (1,), ((S, M), (M,)), 'broadcast_lhs'),
        ('addmv', (S,), ((S, M), (M,)), 'coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addmv', (1,), ((S, M), (M,)), 'broadcast_lhs_coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addmv', (), ((S, M), (M,)), 'scalar_broadcast_lhs'),
        ('addmv', (), ((S, M), (M,)), 'scalar_broadcast_lhs_coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addr', (S, M), ((S,), (M,)),),
        ('addr', (), ((S,), (M,)), 'broadcast_lhs'),
        ('addr', (S, M), ((S,), (M,)), 'coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('addr', (), ((S,), (M,)), 'broadcast_lhs_coef', (), (), lambda x: x, {'beta': 0.2, 'alpha': 0.6}),
        ('dot', (L,), ((L,),),),
        ('mm', (S, M), ((M, S),)),
        ('bmm', (M, S, M), ((M, M, S),)),
        ('mv', (S, M), ((M,),)),
        ('ger', (S,), ((M,),)),
        ('matmul', (L,), ((L,),),),
        ('matmul', (S, M), ((M,),), "2d_1d"),
        ('matmul', (M, ), ((M, S),), "1d_2d"),
        ('matmul', (S, M), ((M, S),), "2d_2d"),
        ('matmul', (S, S, M, M), ((S, S, M, S),), "4d_4d"),
        ('matmul', (S, S, M, M), ((M,),), "4d_1d"),
        ('matmul', (M,), ((S, S, M, S),), "1d_4d"),
        ('matrix_power', (S, S), [2], "n=2"),
        ('matrix_power', (S, S, S), [3], "n=3"),
        ('matrix_power', (S, S, S), [1], "n=1"),
        ('matrix_power', (S, S, S), [0], "n=0"),
        ('matrix_power', lambda: random_fullrank_matrix_distinct_singular_value(S), [-1], "n=-1",
         NO_ARGS, [skipIfNoLapack]),
        ('matrix_power', lambda: random_fullrank_matrix_distinct_singular_value(S), [-3], "n=-3",
         NO_ARGS, [skipIfNoLapack]),
        ('matrix_power', lambda: random_fullrank_matrix_distinct_singular_value(S, S), [-2], "n=-2",
         NO_ARGS, [skipIfNoLapack]),
        ('mvlgamma', torch.empty(S,).uniform_(0.5, 1), [1], "p=1"),
        ('mvlgamma', torch.empty(S,).uniform_(1, 2), [2], "p=2"),
        ('mvlgamma', torch.empty(S, S).uniform_(1.5, 3), [3], "p=3"),
        ('mvlgamma', torch.empty(S, S).uniform_(2.5, 5), [5], "p=5"),
        ('addcmul', (S, S), ((S, S), (S, S))),
        ('addcmul', (S, S), ((S, 1), (1, S)), 'broadcast_rhs'),
        ('addcmul', (1,), ((S, S, 1), (1, S)), 'broadcast_all'),
        ('addcmul', (S, S), ((S, S), (S, S)), 'scale', (), (), lambda x: x, {'value': 0.5}),
        ('addcmul', (S, S), ((S, 1), (1, S)), 'scale_broadcast_rhs', (), (), lambda x: x, {'value': 0.5}),
        ('addcmul', (1,), ((S, S, 1), (1, S)), 'scale_broadcast_all', (), (), lambda x: x, {'value': 0.5}),
        ('addcmul', (), ((), ()), 'scalar'),
        ('addcmul', (S, S), ((), ()), 'scalar_broadcast_rhs'),
        ('addcmul', (), ((S, S, 1), (1, S)), 'scalar_broadcast_lhs'),
        ('addcmul', (), ((), ()), 'scalar_scale', (), (), lambda x: x, {'value': 0.5}),
        ('addcmul', (S, S), ((), ()), 'scalar_scale_broadcast_rhs', (), (), lambda x: x, {'value': 0.5}),
        ('addcmul', (), ((S, S, 1), (1, S)), 'scalar_scale_broadcast_lhs', (), (), lambda x: x, {'value': 0.5}),
        ('addcdiv', (S, S), ((S, S), (S, S))),
        ('addcdiv', (S, S), ((S, 1), (1, S)), 'broadcast_rhs'),
        ('addcdiv', (1,), ((S, S, 1), (1, S)), 'broadcast_all'),
        ('addcdiv', (S, S), ((S, S), (S, S)), 'scale', (), (), lambda x: x, {'value': 0.5}),
        ('addcdiv', (S, S), ((S, 1), (1, S)), 'scale_broadcast_rhs', (), (), lambda x: x, {'value': 0.5}),
        ('addcdiv', (1,), ((S, S, 1), (1, S)), 'scale_broadcast_all', (), (), lambda x: x, {'value': 0.5}),
        ('addcdiv', (), ((), ()), 'scalar'),
        ('addcdiv', (S, S), ((), ()), 'scalar_broadcast_rhs'),
        ('addcdiv', (), ((S, S, 1), (1, S)), 'scalar_broadcast_lhs'),
        ('addcdiv', (), ((), ()), 'scalar_scale', (), (), lambda x: x, {'value': 0.5}),
        ('addcdiv', (S, S), ((), ()), 'scalar_scale_broadcast_rhs', (), (), lambda x: x, {'value': 0.5}),
        ('addcdiv', (), ((S, S, 1), (1, S)), 'scalar_scale_broadcast_lhs', (), (), lambda x: x, {'value': 0.5}),
        ('zero_', (S, S, S), NO_ARGS),
        ('zero_', (), NO_ARGS, 'scalar'),
        ('logsumexp', (S, S), (1,)),
        ('logsumexp', (), (0,), 'scalar'),
        ('norm', (S, S), (), 'default'),
        ('norm', (S, S), (2,), '2'),
        ('norm', (S, S), (0,), '0'),
        ('norm', (S, S), (0.5,), '0_5'),
        ('norm', (S, S), (1,), '1'),
        ('norm', (S, S), (3,), '3'),
        ('norm', (S, S), (inf,), 'inf'),
        ('norm', (S, S), (-inf,), '-inf'),
        ('norm', (S, S), ('fro',), 'fro_default'),
        ('norm', (S, S), ('fro', [0, 1],), 'fro'),
        ('norm', (S, S), ('nuc',), 'nuc', NO_ARGS, [skipIfNoLapack]),
        ('norm', (S, S), (-1,), 'neg_1'),
        ('norm', (S, S), (-2,), 'neg_2'),
        ('norm', (S, S), (-0.5,), 'neg_0_5'),
        ('norm', (S, S), (-1.5,), 'neg_1_5'),
        ('norm', (S, S), (-2, 1,), 'neg_2_2_dim', [1]),
        ('norm', (S, S), (-1, 1,), 'neg_1_2_dim', [1]),
        ('norm', (S, S), (0, 1,), '0_2_dim', [1]),
        ('norm', (S, S), (1, 1,), '1_2_dim', [1]),
        ('norm', (S, S), (2, 1,), '2_2_dim', [1]),
        ('norm', (S, S), (3, 1,), '3_2_dim', [1]),
        ('norm', (S, S), (inf, 1,), 'inf_2_dim'),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5,), '1_5_default'),
        ('norm', (S, S, S), (2, 1), '2_dim', [1]),
        ('norm', (S, S, S), (3, 1), '3_dim', [1]),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1), '1_5_dim', [1]),
        ('norm', (S, S, S), (2, 1, True), 'keepdim_2_dim', [1]),
        ('norm', (S, S, S), (3, 1, True), 'keepdim_3_dim', [1]),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1, True), 'keepdim_1_5_dim', [1]),
        ('norm', (), (2, 0), '2_dim_scalar', [1]),
        ('norm', (), (3, 0), '3_dim_scalar', [1]),
        ('norm', (), (2, 0, True), 'keepdim_2_dim_scalar', [1]),
        ('norm', (), (3, 0, True), 'keepdim_3_dim_scalar', [1]),
        ('clone', (S, M, S), NO_ARGS),
        ('clone', (), NO_ARGS, 'scalar'),
        ('dist', (S, S, S), ((S, S, S),)),
        ('dist', (S, S, S), ((S,),), 'broadcast_rhs'),
        ('dist', (S,), ((S, S, S),), 'broadcast_lhs'),
        ('dist', (S, 1, S), ((S, S),), 'broadcast_all'),
        ('dist', (), ((),), 'scalar'),
        ('dist', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('dist', (), ((S, S, S),), 'scalar_broadcast_lhs'),
        ('dist', (S, S, S), ((S, S, S), 4), '4'),
        ('dist', (S, S, S), ((S,), 4), '4_broadcast_rhs'),
        ('dist', (S,), ((S, S, S), 4), '4_broadcast_lhs'),
        ('dist', (S, 1, S), ((S, S), 4), '4_broadcast_all'),
        ('dist', (), ((), 4), 'scalar_4'),
        ('dist', (S, S, S), ((), 4), 'scalar_4_broadcast_rhs'),
        ('dist', (), ((S, S, S), 4), 'scalar_4_broadcast_lhs'),
        ('diag', (M, M), NO_ARGS, '2d'),
        ('diag', (3, 5), NO_ARGS, '2d_wide'),
        ('diag', (3, 5), (2,), '2d_wide_pos'),
        ('diag', (3, 5), (-2,), '2d_wide_neg'),
        ('diag', (5, 3), NO_ARGS, '2d_tall'),
        ('diag', (5, 3), (2,), '2d_tall_pos'),
        ('diag', (5, 3), (-2,), '2d_tall_neg'),
        ('diag', (M,), NO_ARGS, '1d'),
        ('diag', (M, M), (1,), '2d_1'),
        ('diag', (M, M), (2,), '2d_2'),
        ('diag_embed', (S, S), NO_ARGS),
        ('diagonal', (M, M), NO_ARGS, '2d'),
        ('diagonal', (3, 5), NO_ARGS, '2d_wide'),
        ('diagonal', (3, 5), (2,), '2d_wide_pos'),
        ('diagonal', (3, 5), (-2,), '2d_wide_neg'),
        ('diagonal', (5, 3), NO_ARGS, '2d_tall'),
        ('diagonal', (5, 3), (2,), '2d_tall_pos'),
        ('diagonal', (5, 3), (-2,), '2d_tall_neg'),
        ('diagonal', (M, M), (1,), '2d_1'),
        ('diagonal', (M, M), (2,), '2d_2'),
        ('diagonal', (M, M, M), (1, 1, 2), '3d_1'),
        ('diagonal', (M, M, M), (2, 0, 1), '3d_2'),
        ('diagonal', (M, M, M), (-2, 0, 1), '3d_3'),
        ('tril', (M, M), NO_ARGS),
        ('tril', (M, M), (2,), 'idx'),
        ('tril', (S, M, M), NO_ARGS, 'batched'),
        ('tril', (S, M, M), (2,), 'batched_idx'),
        ('tril', (3, 3, S, S), NO_ARGS, 'more_batched'),
        ('triu', (M, M), NO_ARGS),
        ('triu', (M, M), (2,), 'idx'),
        ('triu', (S, M, M), NO_ARGS, 'batched'),
        ('triu', (S, M, M), (2,), 'batched_idx'),
        ('triu', (3, 3, S, S), NO_ARGS, 'more_batched'),
        ('trace', (M, M), NO_ARGS),
        ('cross', (S, 3), ((S, 3),)),
        ('cross', (S, 3, S), ((S, 3, S), 1), 'dim'),
        ('index_select', (S, S, S), (0, index_variable(2, S)), 'dim', [0]),
        ('index_select', (), (0, torch.tensor([0], dtype=torch.int64)), 'scalar_mixed_dim', [0]),
        ('index_select', (), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_dim', [0]),
        ('index_add', (S, S), (0, index_variable(2, S), (2, S)), 'dim', [0]),
        ('index_add', (), (0, torch.tensor([0], dtype=torch.int64), (1,)), 'scalar_input_dim', [0]),
        ('index_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim', [0]),
        ('index_copy', (S, S), (0, index_perm_variable(2, S), (2, S)), 'dim', [0]),
        ('index_copy', (), (0, torch.tensor([0], dtype=torch.int64), (1,)), 'scalar_input_dim', [0]),
        ('index_copy', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim', [0]),
        ('index_fill', (S, S), (0, index_variable(2, S), 2), 'dim', [0]),
        ('index_fill', (S, S), (0, index_variable(2, S), ()), 'variable_dim', [0]),
        ('index_fill', (S, S), (0, torch.tensor(0, dtype=torch.int64), 2), 'scalar_index_dim', [0]),
        ('index_fill', (), (0, torch.tensor([0], dtype=torch.int64), 2), 'scalar_input_dim', [0]),
        ('index_fill', (), (0, torch.tensor(0, dtype=torch.int64), 2), 'scalar_both_dim', [0]),
        ('inverse', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
        ('inverse', lambda: random_fullrank_matrix_distinct_singular_value(S, 2, 3),
         NO_ARGS, 'batched', NO_ARGS, [skipIfNoLapack]),
        ('det', (S, S), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
        ('det', (1, 1), NO_ARGS, '1x1', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_symmetric_matrix(S), NO_ARGS, 'symmetric', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_symmetric_psd_matrix(S), NO_ARGS, 'symmetric_psd', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_symmetric_pd_matrix(S), NO_ARGS, 'symmetric_pd', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_square_matrix_of_rank(S, S - 2), NO_ARGS, 'dim2_null', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_square_matrix_of_rank(S, 1), NO_ARGS, 'rank1', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_square_matrix_of_rank(S, 2), NO_ARGS, 'rank2', NO_ARGS, [skipIfNoLapack]),
        ('det', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS,
         'distinct_singular_values', NO_ARGS, [skipIfNoLapack]),
        # For `logdet` and `slogdet`, the function at det=0 is not smooth.
        # We need to exclude tests with det=0 (e.g. dim2_null, rank1, rank2) and use
        # `make_nonzero_det` to make the random matrices have nonzero det. For
        # `logdet`, we also set `make_nonzero_det(matrix, sign=1)` to make the
        # matrix have positive det.
        ('logdet', lambda: make_nonzero_det(torch.randn(S, S), 1), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
        ('logdet', lambda: make_nonzero_det(torch.randn(1, 1), 1), NO_ARGS, '1x1', NO_ARGS, [skipIfNoLapack]),
        ('logdet', lambda: make_nonzero_det(random_symmetric_matrix(S), 1), NO_ARGS,
         'symmetric', NO_ARGS, [skipIfNoLapack]),
        ('logdet', lambda: make_nonzero_det(random_symmetric_pd_matrix(S), 1), NO_ARGS,
         'symmetric_pd', NO_ARGS, [skipIfNoLapack]),
        ('logdet', lambda: make_nonzero_det(random_fullrank_matrix_distinct_singular_value(S), 1, 0), NO_ARGS,
         'distinct_singular_values', NO_ARGS, [skipIfNoLapack]),
        ('slogdet', lambda: make_nonzero_det(torch.randn(1, 1), 1), NO_ARGS,
         '1x1_pos_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('slogdet', lambda: make_nonzero_det(torch.randn(1, 1), -1), NO_ARGS,
         '1x1_neg_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('slogdet', lambda: make_nonzero_det(torch.randn(S, S), 1), NO_ARGS,
         'pos_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('slogdet', lambda: make_nonzero_det(torch.randn(S, S), -1), NO_ARGS,
         'neg_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('slogdet', lambda: make_nonzero_det(random_symmetric_matrix(S)), NO_ARGS,
         'symmetric', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('slogdet', lambda: random_symmetric_pd_matrix(S), NO_ARGS,
         'symmetric_pd', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('slogdet', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS,
         'distinct_singular_values', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
        ('symeig', lambda: random_symmetric_matrix(S), (True, False), 'lower', NO_ARGS, [skipIfNoLapack]),
        ('symeig', lambda: random_symmetric_matrix(S), (True, True), 'upper', NO_ARGS, [skipIfNoLapack]),
        ('symeig', lambda: random_symmetric_matrix(M), (True, True), 'large', NO_ARGS, [skipIfNoLapack]),
        ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
        ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:(S - 2)], NO_ARGS,
         'wide', NO_ARGS, [skipIfNoLapack]),
        ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:, :(S - 2)], NO_ARGS,
         'tall', NO_ARGS, [skipIfNoLapack]),
        ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:(S - 2)], (False,),
         'wide_all', NO_ARGS, [skipIfNoLapack], lambda usv: (usv[0], usv[1], usv[2][:, :(S - 2)])),
        ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:, :(S - 2)], (False,),
         'tall_all', NO_ARGS, [skipIfNoLapack], lambda usv: (usv[0][:, :(S - 2)], usv[1], usv[2])),
        ('svd', lambda: random_fullrank_matrix_distinct_singular_value(M), NO_ARGS,
         'large', NO_ARGS, [skipIfNoLapack]),
        ('gesv', (S, S), (random_fullrank_matrix_distinct_singular_value(
            S, silent=True),), '', NO_ARGS, [skipIfNoLapack]),
        ('gesv', (S, S, S), (random_fullrank_matrix_distinct_singular_value(S, S, silent=True),),
         'batched', NO_ARGS, [skipIfNoLapack]),
        ('gesv', (2, 3, S, S), (random_fullrank_matrix_distinct_singular_value(S, 2, 3, silent=True),),
         'batched_dims', NO_ARGS, [skipIfNoLapack]),
        ('gesv', (2, 2, S, S), (random_fullrank_matrix_distinct_singular_value(S, 1, silent=True),),
         'batched_broadcast_A', NO_ARGS, [skipIfNoLapack]),
        ('gesv', (1, S, S), (random_fullrank_matrix_distinct_singular_value(S, 2, 2, silent=True),),
         'batched_broadcast_b', NO_ARGS, [skipIfNoLapack]),
        ('fill_', (S, S, S), (1,), 'number'),
        ('fill_', (), (1,), 'number_scalar'),
        ('fill_', (S, S, S), ((),), 'variable'),
        ('eq_', (S, S, S), ((S, S, S),)),
        ('eq_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('eq_', (), ((),), 'scalar'),
        ('eq_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('ne_', (S, S, S), ((S, S, S),)),
        ('ne_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('ne_', (), ((),), 'scalar'),
        ('ne_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('gt_', (S, S, S), ((S, S, S),)),
        ('gt_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('gt_', (), ((),), 'scalar'),
        ('gt_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('ge_', (S, S, S), ((S, S, S),)),
        ('ge_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('ge_', (), ((),), 'scalar'),
        ('ge_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('lt_', (S, S, S), ((S, S, S),)),
        ('lt_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('lt_', (), ((),), 'scalar'),
        ('lt_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('le_', (S, S, S), ((S, S, S),)),
        ('le_', (S, S, S), ((1,),), 'broadcast_rhs'),
        ('le_', (), ((),), 'scalar'),
        ('le_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
        ('eq_', (S, S, S), (0,), 'pyscalar'),
        ('ne_', (S, S, S), (0,), 'pyscalar'),
        ('gt_', (S, S, S), (0,), 'pyscalar'),
        ('ge_', (S, S, S), (0,), 'pyscalar'),
        ('le_', (S, S, S), (0,), 'pyscalar'),
        ('lt_', (), (0,), 'pyscalar'),
        ('eq_', (), (0,), 'pyscalar_scalar'),
        ('ne_', (), (0,), 'pyscalar_scalar'),
        ('gt_', (), (0,), 'pyscalar_scalar'),
        ('ge_', (), (0,), 'pyscalar_scalar'),
        ('lt_', (), (0,), 'pyscalar_scalar'),
        ('le_', (), (0,), 'pyscalar_scalar'),
        ('permute', (1, 2, 3, 4), (0, 2, 3, 1)),
        ('permute', (1, 2, 3, 4), (0, -2, -1, 1), 'neg_dim'),
        ('permute', (), (dont_convert(()),), 'scalar'),
        ('select', (S, S, S), (1, 2), 'dim', [0]),
        ('select', (S, S, S), (1, -1), 'wrap_dim', [0]),
        ('select', (S,), (0, 2), '1d'),
        ('narrow', (S, S, S), (1, 2, 2), 'dim', [0]),
        ('narrow', (S, S, S), (1, 0, 0), 'empty_dim', [0]),
        ('squeeze', (S, 1, S, 1), NO_ARGS),
        ('squeeze', (1, 1, 1, 1), NO_ARGS, 'input_sizes_are_ones'),
        ('squeeze', (S, 1, S, 1), (1,), '1_dim', [0]),
        ('squeeze', (S, 1, S, 1), (2,), 'not_1_dim', [0]),
        ('squeeze', (), (0,), 'scalar', [0]),
        ('unsqueeze', (S, S, S), (0,), 'first', [0]),
        ('unsqueeze', (S, S, S), (1,), 'middle', [0]),
        ('unsqueeze', (S, S, S), (3,), 'last', [0]),
        ('unsqueeze', (), (0,), 'scalar', [0]),
        ('chunk', (S, S, S), (2,)),
        ('chunk', (S, S, S), (S, 1), 'dim', [1]),
        ('split', (S, S, S), (2,)),
        ('split', (S, S, S), (S, 1), 'dim', [1]),
        ('split', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), 'size_list'),
        ('split', (S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], 2), 'size_list_dim', [1]),
        ('gather', (M, S), (0, gather_variable((S, S), 1, M, True)), 'dim0', [0]),
        ('gather', (M, S), (1, gather_variable((M, S // 2), 0, S, True)), 'dim1', [0]),
        ('gather', (), (0, torch.tensor([0], dtype=torch.int64)), 'scalar_input', [0]),
        ('gather', (S,), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_index', [0]),
        ('gather', (), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_both', [0]),
        ('scatter', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', [0]),
        ('scatter', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', [0]),
        ('scatter', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim0', [0]),
        ('scatter_add', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', [0]),
        ('scatter_add', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', [0]),
        ('scatter_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim0', [0]),
        ('masked_select', (M, M), (mask_not_all_zeros((M, M)),)),
        ('masked_select', (M, M), (mask_not_all_zeros((M,)),), 'broadcast_rhs'),
        ('masked_select', (M,), (mask_not_all_zeros((M, M)),), 'broadcast_lhs'),
        ('masked_select', (M, 1, M), (mask_not_all_zeros((M, M)),),
         'broadcast_all'),
        ('masked_select', (), (torch.tensor(1, dtype=torch.uint8),), 'scalar'),
        ('masked_select', (M, M), (torch.tensor(1, dtype=torch.uint8),), 'scalar_broadcast_rhs'),
        ('masked_select', (), (mask_not_all_zeros((M, M)),), 'scalar_broadcast_lhs'),
        ('masked_fill', (M, M), (torch.ByteTensor(M, M).bernoulli_(), 10)),
        ('masked_fill', (M, M), (torch.ByteTensor(M, M).bernoulli_(), torch.tensor(10)), 'tensor'),
        # no lhs or all broadcast on masked_fill or masked_scatter because it's always inplace
        ('masked_fill', (M, M), (torch.ByteTensor(M,).bernoulli_(), 10), 'broadcast_rhs'),
        ('masked_fill', (), (torch.tensor(0, dtype=torch.uint8, requires_grad=False).bernoulli_(), 10), 'scalar'),
        ('masked_fill', (), (torch.tensor(0, dtype=torch.uint8, requires_grad=False).bernoulli_(), torch.tensor(10)),
         'scalar_variable'),
        ('masked_fill', (M, M), (torch.tensor(0, dtype=torch.uint8, requires_grad=False).bernoulli_(), 10),
         'scalar_broadcast_rhs'),
        ('masked_scatter', (M, M), (torch.ByteTensor(M, M).bernoulli_(), (M, M))),
        ('masked_scatter', (M, M), (torch.ByteTensor(M,).bernoulli_(), (M, M)),
         'broadcast_rhs'),
        ('masked_scatter', (M, M), (bernoulli_scalar(), (M, M)), 'scalar'),
        ('masked_scatter', (M, M), (bernoulli_scalar(), (M, M)),
         'scalar_broadcast_rhs'),
        ('resize_', (S, S, S), (torch.Size([S * S, S])), 'fewer_dims'),
        ('resize_', (), (dont_convert(()),), 'scalar'),
        ('resize_', (), (torch.Size([1, 1, 1])), 'scalar_to_dims'),
        ('resize_as_', (), (non_differentiable(torch.tensor(5.)),), 'scalar'),
        ('resize_as_', (), (non_differentiable(torch.randn((1, 1, 1))),), 'scalar_to_dims'),
        ('resize_as_', (S, S, S), (non_differentiable(torch.randn(S * S, S)),)),
        ('sort', (S, M, S), NO_ARGS),
        ('sort', (S, M, S), (1,), 'dim'),
        ('sort', (S, M, S), (1, True), 'dim_desc'),
        ('sort', (), NO_ARGS, 'scalar'),
        ('sort', (), (0,), 'dim_scalar'),
        ('sort', (), (0, True), 'dim_desc_scalar'),
        ('topk', (S, M, S), (3,)),
        ('topk', (S, M, S), (3, 1), 'dim', [1]),
        ('topk', (S, M, S), (3, 1, True), 'dim_desc', [1]),
        ('topk', (S, M, S), (3, 1, True, True), 'dim_desc_sort', [1]),
        ('topk', (), (1,), 'scalar'),
        ('topk', (), (1, 0), 'dim_scalar', [1]),
        ('topk', (), (1, 0, True), 'dim_desc_scalar', [1]),
        ('topk', (), (1, 0, True, True), 'dim_desc_sort_scalar', [1]),
        ('take', (S, S, S), (torch.LongTensor([[-3, 2], [20, 2]]),)),
        ('take', (S, S, S), (torch.tensor(0, dtype=torch.int64),), 'scalar_index'),
        ('take', (), (torch.LongTensor([0]),), 'scalar_data'),
        ('take', (), (torch.tensor(0, dtype=torch.int64),), 'scalar_both'),
        ('where', (M, M), (mask_not_all_zeros((M, M)), (M, M))),
        ('where', (M, 1, M), (mask_not_all_zeros((M, M)), (M, M, 1)), 'broadcast_all'),
        ('where', (), (bernoulli_scalar(), ()), 'scalar'),
        ('where', (M, 1, M), (bernoulli_scalar(), (M, M, 1)), 'scalar_broadcast_mask'),
        ('where', (), (mask_not_all_zeros((M, M)), ()), 'scalar_broadcast_non_mask'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([1, 2]),)),
        ('__getitem__', torch.randn(S, S, S), (slice(0, 3),), 'slice'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(0, 3), 1]),), 'slice_index'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 2, 3], [1, 3, 3], [0, 0, 2]]),), 'adv_index'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 0, 3], [1, 1, 3], [0, 0, 2]]),), 'adv_index_dup'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(None), slice(None), [0, 3]]),), 'adv_index_end'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(None), [0, 3], slice(None)]),), 'adv_index_mid'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], slice(None), slice(None)]),), 'adv_index_beg'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], [1, 2], slice(None)]),), 'adv_index_comb'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], ]),), 'adv_index_sub'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], slice(None)]),), 'adv_index_sub_2'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], Ellipsis]),), 'adv_index_sub_3'),
        ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 2, 3], [1, 3, 3],
                                                             torch.LongTensor([0, 0, 2])]),), 'adv_index_var'),
    ]
# TODO: clamp with min/max


def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        def maybe_non_contig(tensor):
            return tensor if not non_contiguous else make_non_contiguous(tensor)

        if isinstance(arg, torch.Size) or isinstance(arg, dont_convert):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = torch.randn((), dtype=torch.double)
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return Variable(maybe_non_contig(torch.randn(*arg, dtype=torch.double)), requires_grad=requires_grad)
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                return maybe_non_contig(arg.tensor)
            return maybe_non_contig(arg.tensor)
        elif isinstance(arg, torch.Tensor):
            if arg.dtype == torch.float:
                arg = arg.double()
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of v afterwards
            v = maybe_non_contig(arg).detach().clone()
            v.requires_grad = requires_grad and v.is_floating_point()
            return v
        elif callable(arg):
            return map_arg(arg())
        else:
            return arg
    args_out = tuple(map_arg(arg) for arg in call_args)
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    return args_out, kwargs_out


def _compare_trilu_indices(
        self, row, col, offset=0, dtype=torch.long, device='cpu'):
    if row == 0 or col == 0:
        # have to handle this separately as tril and triu does not take
        # empty matrix as input
        self.assertEqual(
            torch.empty(0, 2, dtype=dtype, device=device).transpose(0, 1),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))

        self.assertEqual(
            torch.empty(0, 2, dtype=dtype, device=device).transpose(0, 1),
            torch.triu_indices(row, col, offset, dtype=dtype, device=device))

    else:
        self.assertEqual(
            torch.ones(row, col, dtype=dtype, device='cpu')
                 .tril(offset).nonzero().transpose(0, 1).to(device),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))

        self.assertEqual(
            torch.ones(row, col, dtype=dtype, device='cpu')
                 .tril(offset).nonzero().transpose(0, 1).to(device),
            torch.tril_indices(row, col, offset, dtype=dtype, device=device))


def _compare_large_trilu_indices(
        self, row, col, offset=0, dtype=torch.long, device='cpu'):
    l = torch.ones(row, col, dtype=dtype, device='cpu').tril(offset) \
             .nonzero()[-100:-1, :].transpose(0, 1).to(device)
    torch.cuda.empty_cache()

    r = torch.tril_indices(
        row, col, offset, dtype=dtype, device=device)[:, -100:-1]
    self.assertEqual(l, r)
    torch.cuda.empty_cache()

    l = torch.ones(row, col, dtype=dtype, device='cpu').triu(offset) \
             .nonzero()[-100:-1, :].transpose(0, 1).to(device)
    torch.cuda.empty_cache()

    r = torch.triu_indices(
        row, col, offset, dtype=dtype, device=device)[:, -100:-1]
    self.assertEqual(l, r)
    torch.cuda.empty_cache()

# (
#   row
#   col
#   offset (optional)
#   dtype (optional)
# )
tri_tests_args = [
    (1, 1),
    (3, 3),
    (3, 3, 1),
    (3, 3, 2),
    (3, 3, 200),
    (3, 3, -1),
    (3, 3, -2),
    (3, 3, -200),
    (0, 3, 0),
    (0, 3, 1),
    (0, 3, -1),
    (3, 0, 0),
    (3, 0, 1),
    (3, 0, -1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
    (3, 6, 0),
    (3, 6, 1),
    (3, 6, 3),
    (3, 6, 9),
    (3, 6, -1),
    (3, 6, -3),
    (3, 6, -9),
    (6, 3, 0),
    (6, 3, 1),
    (6, 3, 3),
    (6, 3, 9),
    (6, 3, -1),
    (6, 3, -3),
    (6, 3, -9),
    (258, 253, 1, torch.float32),
    (257, 258, 1, torch.float64),
    (258, 258, 1, torch.short),
    (3, 513, 1, torch.long),
    (513, 3, 1, torch.int),
    (513, 0, 1, torch.double),
    (1024, 1024),
    (1024, 1024, 500, torch.float32),
    (1024, 1024, 1023),
    (1024, 1024, -500),
    (1023, 1025),
    (1025, 1023, 1022),
    (1024, 1024, -500),
    (3, 2028),
    (3, 2028, 1),
    (3, 2028, -1),
    (2028, 3),
    (2028, 1),
    (2028, 1, -1)
]

tri_large_tests_args = [
    # Large test cases below are deliberately commented out to speed up CI
    # tests and to avoid OOM error. When modifying implementations of
    # tril_indices and triu_indices, please enable these tests and make sure
    # they pass.
    #
    # (1, 268435455),
    # (5000, 5000),
    # (10000, 10000),
    # (268435455, 1),
    # (134217727, 2, 1),
    # (2, 134217727, 1),
    # (536870901, 1),
    # (1, 536870901),
    # (268435455, 2, 1),
    # (2, 268435455, 1)
]


def run_additional_tri_tests(self, device):
    x = torch.ones(
        3, 3, dtype=torch.long, device=device, layout=torch.strided)
    l = x.tril(0).nonzero().transpose(0, 1)
    u = x.triu(0).nonzero().transpose(0, 1)
    self.assertEqual(l, torch.tril_indices(3, 3, device=device))
    self.assertEqual(
        l, torch.tril_indices(3, 3, device=device, layout=torch.strided))

    self.assertEqual(u, torch.triu_indices(3, 3, device=device))
    self.assertEqual(
        u, torch.triu_indices(3, 3, device=device, layout=torch.strided))

    self.assertRaises(
        RuntimeError,
        lambda: torch.triu_indices(
            1, 1, device=device, layout=torch.sparse_coo))

    self.assertRaises(
        RuntimeError,
        lambda: torch.tril_indices(
            1, 1, device=device, layout=torch.sparse_coo))


def unpack_variables(args):
    if istuple(args):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


EXCLUDE_FUNCTIONAL = {
    'addmm',
    'addmm_',
    'addbmm',
    'baddbmm',
    'addmv',
    'addmv_',
    'addr',
    'addr_',
    'reshape',
    'where'  # argument order
}
EXCLUDE_GRADCHECK = {
}
EXCLUDE_GRADGRADCHECK = {
}
EXCLUDE_GRADGRADCHECK_BY_TEST_NAME = {
    # *det methods uses svd in backward when matrix is not invertible. However,
    # svd backward is unstable unless the matrix has positive distinct singular
    # values. Generated random matrices satisfy this with high probability, but
    # we can't rely on it. So only test gradgrad on invertible test cases and
    # _distinct_singular_values.
    'test_det',
    'test_det_1x1',
    'test_det_symmetric',
    'test_det_symmetric_psd',
    'test_det_dim2_null',
    'test_det_rank1',
    'test_det_rank2',
    # `other` expand_as(self, other) is not used in autograd.
    'test_expand_as',
    'test_logdet',
    'test_logdet_1x1',
    'test_logdet_symmetric',
    'test_slogdet_1x1_neg_det',
    'test_slogdet_neg_det',
    'test_slogdet_symmetric',
}


def exclude_tensor_method(name, test_name):
    # there are no tensor equivalents for these (inplace or out)
    exclude_all_tensor_method_by_test_name = {
        'test_clamp_min',
        'test_clamp_max',
        'test_clamp_min_scalar',
        'test_clamp_max_scalar',
        'test_slice',
        'test_where',
        'test_where_broadcast_all',
        'test_where_scalar',
        'test_where_scalar_broadcast_mask',
        'test_where_scalar_broadcast_non_mask',
    }
    # there are no out-of-place tensor equivalents for these
    exclude_outplace_tensor_method = {
        'index_add',
        'index_copy',
        'index_fill',
        'masked_fill',
        'masked_scatter',
        'scatter',
        'scatter_add',
        'det',
    }
    if test_name in exclude_all_tensor_method_by_test_name:
        return True
    is_magic_method = name[:2] == '__' and name[-2:] == '__'
    is_inplace = name[-1] == "_" and not is_magic_method
    if not is_inplace and name in exclude_outplace_tensor_method:
        return True
    return False
