import typing as t

import torch
import torch.fx
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS, get_node_target
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
            "torch.ops.aten.add": None,
            "torch.ops.aten.sub": None,
            "torch.ops.aten.rsub": None,
            "torch.ops.aten.div": None,
            "torch.ops.aten.atan2": None,
            "torch.ops.aten.mul": None,
            "torch.ops.aten.max": None,
            "torch.ops.aten.min": None,
            "torch.ops.aten.pow": None,
            "torch.ops.aten.remainder": None,
            "torch.ops.aten.fmod": None,
            "torch.ops.aten.bitwise_and": None,
            "torch.ops.aten.__and__": None,
            "torch.ops.aten.bitwise_or": None,
            "torch.ops.aten.__or__": None,
            "torch.ops.aten.bitwise_xor": None,
            "torch.ops.aten.__xor__": None,
            "torch.ops.aten.bitwise_left_shift": None,
            "torch.ops.aten.__lshift__": None,
            "torch.ops.aten.bitwise_right_shift": None,
            "torch.ops.aten.__rshift__": None,
            "torch.ops.aten.eq": None,
            "torch.ops.aten.ne": None,
            "torch.ops.aten.ge": None,
            "torch.ops.aten.gt": None,
            "torch.ops.aten.le": None,
            "torch.ops.aten.lt": None,
            "torch.ops.aten.abs": None,
            "torch.ops.aten.bitwise_not": None,
            "torch.ops.aten.ceil": None,
            "torch.ops.aten.floor": None,
            "torch.ops.aten.frac": None,
            "torch.ops.aten.neg": None,
            "torch.ops.aten.relu": None,
            "torch.ops.aten.round": None,
            "torch.ops.aten.silu": None,
            "torch.ops.aten.trunc": None,
            "torch.ops.aten.log": None,
            "torch.ops.aten.log10": None,
            "torch.ops.aten.log1p": None,
            "torch.ops.aten.log2": None,
            "torch.ops.aten.lgamma": None,
            "torch.ops.aten.exp": None,
            "torch.ops.aten.expm1": None,
            "torch.ops.aten.erf": None,
            "torch.ops.aten.erfc": None,
            "torch.ops.aten.cos": None,
            "torch.ops.aten.acos": None,
            "torch.ops.aten.cosh": None,
            "torch.ops.aten.sin": None,
            "torch.ops.aten.asin": None,
            "torch.ops.aten.sinh": None,
            "torch.ops.aten.tan": None,
            "torch.ops.aten.atan": None,
            "torch.ops.aten.tanh": None,
            "torch.ops.aten.atanh": None,
            "torch.ops.aten.sqrt": None,
            "torch.ops.aten.rsqrt": None,
            "torch.ops.aten.reciprocal": None,
            "torch.ops.aten.sigmoid": None,
            "torch.ops.aten.isfinite": None,
            "torch.ops.aten.isinf": None,
            "torch.ops.aten.isnan": None,
            "torch.ops.aten.isneginf": None,
            "torch.ops.aten.isposinf": None,
            "torch.ops.aten.isreal": None,
            "torch.ops.aten.rand_like": None,
            "torch.ops.aten.softplus": None,
            "torch.ops.aten.threshold": None,
            "torch.ops.aten.threshold_backward": None,
            "torch.ops.aten.clamp": None,
            "torch.ops.aten.where": None,
            "torch.ops.aten.lerp": None,
            "torch.ops.aten.addcmul": None,
            "torch.ops.aten.native_dropout": None,
            "torch.ops.aten.dropout": None,
            "torch.ops.aten.native_dropout_backward": None,
            "torch.ops.aten.instance_norm": None,
            "torch.ops.aten._batch_norm_impl_index": None,
            "torch.ops.aten.native_batch_norm": None,
            "torch.ops.aten.batch_norm": None,
            "torch.ops.aten._batch_norm_impl_index_backward": None,
            "torch.ops.aten.native_batch_norm_backward": None,
            "torch.ops.aten.native_layer_norm": None,
            "torch.ops.aten.layer_norm": None,
            "torch.ops.aten.native_layer_norm_backward": None,
            "torch.ops.aten.softmax.int": None,
            "torch.ops.aten.log_softmax.int": None,
            "torch.ops.aten._softmax": None,
            "torch.ops.aten._log_softmax_backward_data": None,
            "torch.ops.aten._softmax_backward_data": None,
            "torch.ops.aten.var.dim": None,
            "torch.ops.aten.std.dim": None,
            "torch.ops.aten.sum.dim_IntList": None,
            "torch.ops.aten.mean.dim": None,
            "torch.ops.aten._grad_sum_to_size": None,
            "torch.ops.aten.sum_to_size": None,
            "torch.ops.aten._autocast_to_reduced_precision": None,
            "torch.ops.aten._autocast_to_full_precision": None,
            "torch.ops.aten.to.dtype": None,
            "torch.ops.aten.type_as": None,
            "torch.ops.aten.linear": None,
            "torch.ops.aten.gelu": None,
            "torch.ops.aten.gelu_backward": None,
            "torch.ops.aten.tanh_backward": None,
            "torch.ops.aten.amax": None,
            "torch.ops.aten.amin": None,
            "torch.ops.aten.reshape": None,
            "torch.ops.aten.view": None,
            "torch.ops.aten.flatten.using_ints": None,


            # ===============================================================
            # call_function aten: inplace variants
            # ===============================================================

            "torch.ops.aten.add_": None,
            "torch.ops.aten.relu_": None,


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

        super().__init__(merged_support_dict)

    # Extension point: can further override operator_support() to skip node if input/ouput tensor types is scalar
    #
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:

        # nvFuser subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            return False

        return super().is_node_supported(submodules, node)

