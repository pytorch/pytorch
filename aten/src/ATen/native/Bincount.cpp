// Returns the frequency of elements of input non-negative integer tensor.

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"

#include <tuple>

namespace at { namespace native {

#define AT_DISPATCH_BINCOUNT_INT_CASE(scalar, type)                \
  case scalar: {                                                   \
    return _bincount_cpu_template<type>(self, weights, minlength); \
  }

namespace {
template <typename integral_t>
Tensor _bincount_cpu_template(
    const Tensor& self,
    const Tensor& weights = {},
    int64_t minlength = 0) {
  if (minlength < 0) {
    throw std::domain_error("minlength should be >= 0");
  }
  if (self.dim() != 1 || !isIntegralType(self.type().scalarType()) ||
      *self.min().data<integral_t>() < 0) {
    throw std::domain_error(
        "bincount only supports 1-d non-negative integral inputs.");
  }

  bool isWeights = weights.defined();
  if (isWeights && weights.numel() != self.numel()) {
    throw std::runtime_error("input and weights should have the same length");
  }

  Tensor output;
  int64_t nbins = static_cast<int64_t>(*self.max().data<integral_t>()) + 1L;
  nbins = std::max(nbins, minlength); // at least minlength # of bins

  integral_t* self_p = self.contiguous().data<integral_t>();
  if (isWeights) {
    output = zeros(CPU(kDouble), {nbins});
    double* output_p = output.data<double>();
    const Tensor& w_t = weights.toType(ScalarType::Double).contiguous();
    const double* weights_p = w_t.data<double>();
    for (uint64_t i = 0; i < self.numel(); i++) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    output = zeros(CPU(kLong), {nbins});
    int64_t* output_p = output.data<int64_t>();
    for (uint64_t i = 0; i < self.numel(); i++) {
      output_p[self_p[i]] += 1;
    }
  }
  return output;
}
} // namespace

Tensor _bincount_cpu(
    const Tensor& self,
    const Tensor& weights = {},
    int64_t minlength = 0) {
  const auto& type = self.type();
  switch (type.scalarType()) {
    AT_DISPATCH_BINCOUNT_INT_CASE(ScalarType::Byte, uint8_t);
    AT_DISPATCH_BINCOUNT_INT_CASE(ScalarType::Char, int8_t);
    AT_DISPATCH_BINCOUNT_INT_CASE(ScalarType::Int, int32_t);
    AT_DISPATCH_BINCOUNT_INT_CASE(ScalarType::Long, int64_t);
    AT_DISPATCH_BINCOUNT_INT_CASE(ScalarType::Short, int16_t);
    default:
      AT_ERROR("bincount not supoorted for '%s'", type.toString());
  }
  return {};
}

#undef AT_DISPATCH_BINCOUNT_INT_CASE
}} // namespace at::native
