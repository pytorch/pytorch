import torch
import operator
from .compile_utils import get_aten_target

def _prod(x):
    s = 1
    for i in x:
        s *= i
    return s


def _size_of(metadata):
    sizes = {
        torch.float: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int: 4,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.bool: 1,
    }

    numel = _prod(metadata.shape)
    dtype = metadata.dtype

    if dtype not in sizes:
        raise NotImplementedError("Don't know the size of dtype ", dtype)

    return numel * sizes[dtype]

aten = torch.ops.aten

pointwise_ops = [aten.add, aten.sub, aten.div, aten.atan2, aten.mul, aten.max, aten.min, aten.pow, aten.remainder, aten.fmod, aten.__and__, aten.__or__, aten.__xor__, aten.__lshift__, aten.__rshift__, aten.eq, aten.ne, aten.ge, aten.gt, aten.le, aten.lt, aten.abs, aten.bitwise_not, aten.ceil, aten.floor, aten.frac, aten.neg, aten.relu, aten.round, aten.silu, aten.trunc, aten.log, aten.log10, aten.log1p, aten.log2, aten.lgamma, aten.exp, aten.expm1, aten.erf, aten.erfc, aten.cos, aten.acos, aten.cosh, aten.sin, aten.asin, aten.sinh, aten.tan, aten.atan, aten.tanh, aten.atanh, aten.sqrt, aten.rsqrt, aten.reciprocal, aten.sigmoid, aten.softplus, aten.threshold, aten.threshold_backward, aten.clamp, aten.where, aten.lerp, aten.addcmul, aten.gelu, aten.gelu_backward]  # noqa: E501
misc_ops = [aten.to, aten.type_as, operator.getitem]

reduction_ops = [aten.softmax, aten._softmax, aten._softmax_backward_data, aten.sum, aten.mean, aten._grad_sum_to_size, aten.sum_to_size, aten.amax]  # noqa: E501

# not recomputed by default since these are kinda expensive/hard to fuse into
# norm_ops = [aten.instance_norm, aten._batch_norm_impl_index, aten.native_batch_norm, aten.batch_norm, aten._batch_norm_impl_index_backward, aten.native_layer_norm, aten.layer_norm, aten.native_layer_norm_backward]  # noqa: E501

# Not used by default since NVFuser can't fuse view ops
# view_ops = [aten.expand, aten.clone, aten.transpose, aten.t, aten.view, aten._unsafe_view, aten.permute, aten.transpose, aten.t, aten._reshape_alias, aten.squeeze, aten.unsqueeze, aten.reshape, aten.cat, aten.slice, aten.split, aten.select, aten.repeat]  # noqa: E501

# These are the view ops that NVFuser can fuse
view_ops = [aten.squeeze, aten.unsqueeze]
random_ops = [aten.native_dropout, aten.rand_like, aten.randn_like]
compute_intensive_ops = [aten.mm, aten.convolution, aten.convolution_backward, aten.bmm, aten.addmm, aten.upsample_bilinear2d]  # noqa: E501
unrecomputable_ops = random_ops + compute_intensive_ops

recomputable_ops = set(
    pointwise_ops
    + misc_ops
    + reduction_ops
    + view_ops
)
fusible_ops = recomputable_ops | set(random_ops)

AGGRESSIVE_RECOMPUTATION = False


def ban_recomputation(node):
    if AGGRESSIVE_RECOMPUTATION:
        return (node.op == 'call_function' and get_aten_target(node) in unrecomputable_ops)
    else:
        if node.op != 'call_function':
            return False
        if get_aten_target(node) not in recomputable_ops:
            return True
        # If the output of the reduction is 4x smaller (arbitrary choice),
        # then we don't allow recomputation.
        if get_aten_target(node) in reduction_ops:
            input_tensors_size = sum(_size_of(i.meta['tensor_meta']) for i in node.args if isinstance(i, torch.fx.Node))
            output_size = _size_of(node.meta['tensor_meta'])
            return (output_size * 4 < input_tensors_size)
        return False
