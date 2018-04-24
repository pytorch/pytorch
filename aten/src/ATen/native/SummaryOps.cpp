// Returns the frequency of elements of input non-negative integer tensor.

#include "ATen/ATen.h"
#include "ATen/Dispatch.h"

#include <tuple>

namespace at { namespace native {

///////////////// bincount /////////////////
namespace {
template <typename integral_t, typename T>
Tensor _bincount_cpu_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() != 1 || self.numel() == 0 ||
      !isIntegralType(self.type().scalarType()) ||
      *self.min().data<integral_t>() < 0) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.numel() != self.numel()) {
    AT_ERROR("input and weights should have the same length");
  }

  Tensor output;
  int64_t nbins = static_cast<int64_t>(*self.max().data<integral_t>()) + 1L;
  nbins = std::max(nbins, minlength); // at least minlength # of bins

  integral_t* self_p = self.contiguous().data<integral_t>();
  if (has_weights) {
    output = zeros(weights.type(), {nbins});
    T* output_p = output.data<T>();
    const T* weights_p = weights.contiguous().data<T>();
    for (uint64_t i = 0; i < self.numel(); i++) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    output = zeros(CPU(kLong), {nbins});
    int64_t* output_p = output.data<int64_t>();
    for (uint64_t i = 0; i < self.numel(); i++) {
      output_p[self_p[i]] += 1L;
    }
  }
  return output;
}
} // namespace

Tensor
_bincount_cpu(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.type(), "bincount", [&] {
    if (weights.type().scalarType() == ScalarType::Float)
      return _bincount_cpu_template<scalar_t, float>(self, weights, minlength);
    return _bincount_cpu_template<scalar_t, double>(self, weights, minlength);
  });
}

}} // namespace at::native
