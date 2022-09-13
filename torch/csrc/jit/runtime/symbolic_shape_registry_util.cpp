#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>
#include <unordered_map>

namespace torch {
namespace jit {

const OperatorMap<std::string>& get_tensorexpr_elementwise_set() {
  // clang-format off
 static const OperatorMap<std::string> tensorexpr_elementwise_set{
      {"aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", "unary"},
      {"aten::_cast_Float(Tensor self, bool non_blocking) -> Tensor", "unary"},
      {"aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", "unary"},
      {"aten::mul.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::div.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::eq.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::ne.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::ge.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::gt.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::le.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::lt.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor", "unary"},
      {"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor", "unary"},
      {"aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor", "unary"},
      {"aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor", "unary"},
      {"aten::to.dtype_layout(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor", "unary"},
      {"aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)", "unary"},
      {"aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)", "unary"},
      {"aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)", "unary"},
      {"aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)", "unary"},
      {"aten::isnan(Tensor self) -> Tensor", "unary"},
      {"aten::lgamma(Tensor self) -> Tensor", "unary"},
      {"aten::log10(Tensor self) -> Tensor", "unary"},
      {"aten::log(Tensor self) -> Tensor", "unary"},
      {"aten::log2(Tensor self) -> Tensor", "unary"},
      {"aten::log1p(Tensor self) -> Tensor", "unary"},
      {"aten::exp(Tensor self) -> Tensor", "unary"},
      {"aten::erf(Tensor self) -> Tensor", "unary"},
      {"aten::erfc(Tensor self) -> Tensor", "unary"},
      // TODO: uncomment when we properly support pow
      // "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor",
      // "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor",
      // TODO: support clamp_min, clamp_max
      // "aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor",
      // "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor", TODO: requires 0-dim Tensor
      // "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor",
      // TODO: uncomment once we can handle rand+broadcasts
      // "aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      {"aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::cos(Tensor self) -> Tensor", "unary"},
      {"aten::sin(Tensor self) -> Tensor", "unary"},
      {"aten::tan(Tensor self) -> Tensor", "unary"},
      {"aten::acos(Tensor self) -> Tensor", "unary"},
      {"aten::asin(Tensor self) -> Tensor", "unary"},
      {"aten::atan(Tensor self) -> Tensor", "unary"},
      {"aten::cosh(Tensor self) -> Tensor", "unary"},
      {"aten::sinh(Tensor self) -> Tensor", "unary"},
      {"aten::tanh(Tensor self) -> Tensor", "unary"},
      {"aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor", "unary"},
      {"aten::hardsigmoid(Tensor self) -> Tensor", "unary"},
      {"aten::hardswish(Tensor self) -> Tensor", "unary"},
      {"aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor", "unary"},
      {"aten::sqrt(Tensor self) -> Tensor", "unary"},
      {"aten::rsqrt(Tensor self) -> Tensor", "unary"},
      {"aten::abs(Tensor self) -> Tensor", "unary"},
      {"aten::floor(Tensor self) -> Tensor", "unary"},
      {"aten::ceil(Tensor self) -> Tensor", "unary"},
      {"aten::round(Tensor self) -> Tensor", "unary"},
      {"aten::trunc(Tensor self) -> Tensor", "unary"},
      {"aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor", "unary"},
      {"aten::sigmoid(Tensor self) -> Tensor", "unary"},
      {"aten::relu(Tensor self) -> Tensor", "unary"},
      {"aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor", "unary"},
      {"aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor", "unary"},
      {"aten::mish(Tensor self) -> Tensor", "unary"},
      {"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor", "unary"},
      {"aten::relu6(Tensor self) -> Tensor", "unary"},
      {"aten::gelu(Tensor self, *, str approximate='none') -> Tensor", "unary"},
      {"aten::silu(Tensor self) -> Tensor", "unary"},
      {"aten::neg(Tensor self) -> Tensor", "unary"},
      {"aten::reciprocal(Tensor self) -> Tensor", "unary"},
      {"aten::expm1(Tensor self) -> Tensor", "unary"},
      {"aten::frac(Tensor self) -> Tensor", "unary"},
      {"aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor", "unary"},
      {"aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor", "unary"},
      {"aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", "broadcast"},
      {"aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor", "broadcast"},
      {"aten::type_as(Tensor self, Tensor other) -> Tensor", "unary"},
      {"aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", "broadcast"},
      {"aten::mul.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::div.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::eq.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::ne.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::ge.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::gt.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::le.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::lt.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor", "broadcast"},
      {"aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::atan2(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      // TODO: enable other min/max variants, operators that can be both
      // elementwise or reductions:
      {"aten::min.other(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::max.other(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor", "broadcast_three"},
      {"aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor", "broadcast_three"},
      {"aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor", "broadcast_three"},
      {"aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor", "broadcast_one_three"},
      // TODO: enable slice, shape inference is not implemented for this op yet
  };
  // clang-format on
  return tensorexpr_elementwise_set;
}

} // namespace jit
} // namespace torch
