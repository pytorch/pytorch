#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/item_native.h>
#endif

namespace at {
namespace native {

Scalar item(const Tensor& self) {
  int64_t numel = self.numel();
  TORCH_CHECK(numel == 1, "a Tensor with ", numel, " elements cannot be converted to Scalar");
  if (self.is_sparse()) {
    if (self._nnz() == 0) return Scalar(0);
    if (self.is_coalesced()) return at::_local_scalar_dense(self._values());
    return at::_local_scalar_dense(self._values().sum());
  } else if (self.is_quantized()) {
    return self.dequantize().item();
  } else {
    return _local_scalar_dense(self);
  }
}

Scalar _local_scalar_dense_cpu(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
    kComplexHalf, kHalf, kBool, kBFloat16, self.scalar_type(), "_local_scalar_dense_cpu", [&] {
        scalar_t value = *self.data_ptr<scalar_t>();
        r = Scalar(value);
      });
  return r;
}

}} // at::native
