import typing as t

import torch
import torch.fx
from torch.fx.passes.operator_support import OperatorSupport
# from torch.fx.passes.tools_common import CALLABLE_NODE_OPS, get_node_target
from torch._C._nvfuser import FusionDefinition as fd


class NvFuserOperatorSupport(OperatorSupport):
    """
    Operator support for nvFuser backend.

    Note: When adding a rule, please add it to the corresponding secion and follow the
    alphabetical order.
    """

    def __init__(self):

        support_dict = {

            # "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor"
            # "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor"
            # "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor"
            # "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor"
            # "aten::rsub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor"
            # "aten::rsub(Tensor self, Scalar other, Scalar alpha) -> Tensor"
            # "aten::div(Tensor self, Tensor other) -> Tensor"
            # "aten::div(Tensor self, Scalar other) -> Tensor"
            # "aten::atan2(Tensor self, Tensor other) -> Tensor"
            # "aten::mul(Tensor self, Tensor other) -> Tensor"
            # "aten::mul(Tensor self, Scalar other) -> Tensor"
            # "aten::max(Tensor self, Tensor other) -> Tensor"
            # "aten::min(Tensor self, Tensor other) -> Tensor"
            # "aten::pow(Tensor self, Tensor exponent) -> Tensor"
            # "aten::pow(Tensor self, Scalar exponent) -> Tensor"
            # "aten::pow(Scalar self, Tensor exponent) -> Tensor"
            # "aten::remainder(Tensor self, Tensor other) -> Tensor"
            # "aten::fmod(Tensor self, Tensor other) -> Tensor"
            # "aten::bitwise_and(Tensor self, Tensor other) -> Tensor"
            # "aten::__and__(Tensor self, Tensor other) -> Tensor"
            # "aten::bitwise_or(Tensor self, Tensor other) -> Tensor"
            # "aten::__or__(Tensor self, Tensor other) -> Tensor"
            # "aten::bitwise_xor(Tensor self, Tensor other) -> Tensor"
            # "aten::__xor__(Tensor self, Tensor other) -> Tensor"
            # "aten::bitwise_left_shift(Tensor self, Tensor other) -> Tensor"
            # "aten::__lshift__(Tensor self, Tensor other) -> Tensor"
            # "aten::bitwise_right_shift(Tensor self, Tensor other) -> Tensor"
            # "aten::__rshift__(Tensor self, Tensor other) -> Tensor"
            # "aten::eq(Tensor self, Tensor other) -> Tensor"
            # "aten::eq(Tensor self, Scalar other) -> Tensor"
            # "aten::ne(Tensor self, Tensor other) -> Tensor"
            # "aten::ne(Tensor self, Scalar other) -> Tensor"
            # "aten::ge(Tensor self, Tensor other) -> Tensor"
            # "aten::ge(Tensor self, Scalar other) -> Tensor"
            # "aten::gt(Tensor self, Tensor other) -> Tensor"
            # "aten::gt(Tensor self, Scalar other) -> Tensor"
            # "aten::le(Tensor self, Tensor other) -> Tensor"
            # "aten::le(Tensor self, Scalar other) -> Tensor"
            # "aten::lt(Tensor self, Tensor other) -> Tensor"
            # "aten::lt(Tensor self, Scalar other) -> Tensor"
            # "aten::abs(Tensor self) -> Tensor"
            # "aten::bitwise_not(Tensor self) -> Tensor"
            # "aten::ceil(Tensor self) -> Tensor"
            # "aten::floor(Tensor self) -> Tensor"
            # "aten::frac(Tensor self) -> Tensor"
            # "aten::neg(Tensor self) -> Tensor"
            # "aten::relu(Tensor self) -> Tensor"
            # "aten::round(Tensor self) -> Tensor"
            # "aten::silu(Tensor self) -> Tensor"
            # "aten::trunc(Tensor self) -> Tensor"
            # "aten::log(Tensor self) -> Tensor"
            # "aten::log10(Tensor self) -> Tensor"
            # "aten::log1p(Tensor self) -> Tensor"
            # "aten::log2(Tensor self) -> Tensor"
            # "aten::lgamma(Tensor self) -> Tensor"
            # "aten::exp(Tensor self) -> Tensor"
            # "aten::expm1(Tensor self) -> Tensor"
            # "aten::erf(Tensor self) -> Tensor"
            # "aten::erfc(Tensor self) -> Tensor"
            # "aten::cos(Tensor self) -> Tensor"
            # "aten::acos(Tensor self) -> Tensor"
            # "aten::cosh(Tensor self) -> Tensor"
            # "aten::sin(Tensor self) -> Tensor"
            # "aten::asin(Tensor self) -> Tensor"
            # "aten::sinh(Tensor self) -> Tensor"
            # "aten::tan(Tensor self) -> Tensor"
            # "aten::atan(Tensor self) -> Tensor"
            # "aten::tanh(Tensor self) -> Tensor"
            # "aten::atanh(Tensor self) -> Tensor"
            # "aten::sqrt(Tensor self) -> Tensor"
            # "aten::rsqrt(Tensor self) -> Tensor"
            # "aten::reciprocal(Tensor self) -> Tensor"
            # "aten::sigmoid(Tensor self) -> Tensor"
            # "aten::isfinite(Tensor self) -> Tensor"
            # "aten::isinf(Tensor self) -> Tensor"
            # "aten::isnan(Tensor self) -> Tensor"
            # "aten::isneginf(Tensor self) -> Tensor"
            # "aten::isposinf(Tensor self) -> Tensor"
            # "aten::isreal(Tensor self) -> Tensor"
            # "aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"
            # "aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor"
            # "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor"
            # "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor"
            # "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor"
            # "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor"
            # "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor"
            # "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor"
            # "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"
            # "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"
            # "aten::dropout(Tensor input, float p, bool train) -> Tensor"
            # "aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor"
            # "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor"
            # "aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)"
            # "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"
            # "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor"
            # "aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)"
            # "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"
            # "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"
            # "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"
            # "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"
            # "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"
            # "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"
            # "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"
            # "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"
            # "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"
            # "aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor"
            # "aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor"
            # "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)"
            # "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"
            # "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)"
            # "aten::sum_to_size(Tensor self, int[] size) -> Tensor"
            # "aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)"
            # "aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)"
            # "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor"
            # "aten::type_as(Tensor self, Tensor other) -> Tensor"
            # "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"
            # "aten::gelu(Tensor self, *, str approximate='none') -> Tensor"
            # "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor"
            # "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor"
            # "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"
            # "aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"
            # "aten::dropout(Tensor input, float p, bool train) -> Tensor"
            # "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"
            # "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"
            # "aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"
            # "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)"
            # "aten::sum_to_size(Tensor self, int[] size) -> Tensor"
            # "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)"
            # "aten::reshape(Tensor self, int[] shape) -> Tensor"
            # "aten::view(Tensor self, int[] size) -> Tensor"
            # "aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor"
            # "aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)"
            # "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)"
            # "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor"
            # "aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor"
            # "aten::gelu(Tensor self, *, str approximate='none') -> Tensor"
            # "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor"
            # "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"
            # "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor"
            # "aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)"
            # "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"
            # "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)"
            # "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor"
            # "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"
            # "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"
            # "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"
            # "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"

            # ===============================================================
            # call_function aten
            # ===============================================================
            # Following supported aten ops is copied from torch/csrc/jit/codegen/cuda/parser.cpp
            "aten.add": None,
            "aten.add": None,
            "aten.sub": None,
            "aten.sub": None,
            "aten.rsub": None,
            "aten.rsub": None,
            "aten.div": None,
            "aten.div": None,
            "aten.atan2": None,
            "aten.mul": None,
            "aten.mul": None,
            "aten.max": None,
            "aten.min": None,
            "aten.pow": None,
            "aten.pow": None,
            "aten.pow": None,
            "aten.remainder": None,
            "aten.fmod": None,
            "aten.bitwise_and": None,
            "aten.__and__": None,
            "aten.bitwise_or": None,
            "aten.__or__": None,
            "aten.bitwise_xor": None,
            "aten.__xor__": None,
            "aten.bitwise_left_shift": None,
            "aten.__lshift__": None,
            "aten.bitwise_right_shift": None,
            "aten.__rshift__": None,
            "aten.eq": None,
            "aten.eq": None,
            "aten.ne": None,
            "aten.ne": None,
            "aten.ge": None,
            "aten.ge": None,
            "aten.gt": None,
            "aten.gt": None,
            "aten.le": None,
            "aten.le": None,
            "aten.lt": None,
            "aten.lt": None,
            "aten.abs": None,
            "aten.bitwise_not": None,
            "aten.ceil": None,
            "aten.floor": None,
            "aten.frac": None,
            "aten.neg": None,
            "aten.relu": None,
            "aten.round": None,
            "aten.silu": None,
            "aten.trunc": None,
            "aten.log": None,
            "aten.log10": None,
            "aten.log1p": None,
            "aten.log2": None,
            "aten.lgamma": None,
            "aten.exp": None,
            "aten.expm1": None,
            "aten.erf": None,
            "aten.erfc": None,
            "aten.cos": None,
            "aten.acos": None,
            "aten.cosh": None,
            "aten.sin": None,
            "aten.asin": None,
            "aten.sinh": None,
            "aten.tan": None,
            "aten.atan": None,
            "aten.tanh": None,
            "aten.atanh": None,
            "aten.sqrt": None,
            "aten.rsqrt": None,
            "aten.reciprocal": None,
            "aten.sigmoid": None,
            "aten.isfinite": None,
            "aten.isinf": None,
            "aten.isnan": None,
            "aten.isneginf": None,
            "aten.isposinf": None,
            "aten.isreal": None,
            "aten.rand_like": None,
            "aten.softplus": None,
            "aten.threshold": None,
            "aten.threshold_backward": None,
            "aten.clamp": None,
            "aten.where": None,
            "aten.lerp": None,
            "aten.lerp": None,
            "aten.addcmul": None,
            "aten.native_dropout": None,
            "aten.dropout": None,
            "aten.native_dropout_backward": None,
            "aten.instance_norm": None,
            "aten._batch_norm_impl_index": None,
            "aten.native_batch_norm": None,
            "aten.batch_norm": None,
            "aten._batch_norm_impl_index_backward": None,
            "aten.native_batch_norm_backward": None,
            "aten.native_layer_norm": None,
            "aten.layer_norm": None,
            "aten.native_layer_norm_backward": None,
            "aten.softmax.int": None,
            "aten.log_softmax.int": None,
            "aten._softmax": None,
            "aten._log_softmax_backward_data": None,
            "aten._softmax_backward_data": None,
            "aten.var.dim": None,
            "aten.std.dim": None,
            "aten.sum.dim_IntList": None,
            "aten.mean.dim": None,
            "aten._grad_sum_to_size": None,
            "aten.sum_to_size": None,
            "aten._autocast_to_reduced_precision": None,
            "aten._autocast_to_full_precision": None,
            "aten.to.dtype": None,
            "aten.type_as": None,
            "aten.linear": None,
            "aten.gelu": None,
            "aten.gelu_backward": None,
            "aten.tanh_backward": None,
            "aten.amax": None,
            "aten.amin": None,
            "aten.dropout": None,
            "aten.native_dropout": None,
            "aten.amax": None,
            "aten.amin": None,
            "aten.sum.dim_IntList": None,
            "aten.sum_to_size": None,
            "aten._grad_sum_to_size": None,
            "aten.reshape": None,
            "aten.view": None,
            "aten.flatten.using_ints": None,
            "aten._batch_norm_impl_index": None,
            "aten.native_batch_norm": None,
            "aten.batch_norm": None,
            "aten.instance_norm": None,
            "aten.gelu": None,
            "aten.gelu_backward": None,
            "aten.native_layer_norm": None,
            "aten.layer_norm": None,
            "aten._batch_norm_impl_index_backward": None,
            "aten.native_batch_norm_backward": None,
            "aten.native_layer_norm_backward": None,
            "aten.to.dtype": None,
            "aten.log_softmax.int": None,
            "aten.softmax.int": None,
            "aten._log_softmax_backward_data": None,
            "aten._softmax_backward_data": None,
            # ===============================================================
            # call_function builtins and operator
            # ===============================================================
            "getattr": None,
            #     "_operator.add": None,
            #     "_operator.div": None,
            #     "_operator.getitem": None,
            #     "_operator.mul": None,
            #     "_operator.sub": None,
            #     "_operator.truediv": None,
        }

        prim_nvfuser_ops = set(torch._prims.__all__).intersection(dir(fd.Ops))

        ops_with_nvfuser_impl = {
            "torch.ops.prims." + name + ".default" : None
            for name in prim_nvfuser_ops
            if getattr(torch.ops.prims, name).default.impl_nvfuser is not None
        }

        merged_support_dict = {**support_dict, **ops_with_nvfuser_impl}

        # print(merged_support_dict)

        super().__init__(merged_support_dict)

    # Extension point: can further override operator_support() to skip node if input/ouput tensor types is scalar
    #
    # def is_node_supported(
    #     self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    # ) -> bool: