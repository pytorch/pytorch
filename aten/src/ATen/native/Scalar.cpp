#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <ATen/ops/item_native.h>
#endif

namespace at::native {

Scalar item(const Tensor& self) {
  auto numel = self.sym_numel();
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

#define AT_SD_BASE_TYPES AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_COMPLEX_TYPES), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
#if !defined(C10_MOBILE)
#define AT_SD_TYPES AT_EXPAND(AT_SD_BASE_TYPES), AT_EXPAND(AT_FLOAT8_TYPES)
#else
#define AT_SD_TYPES AT_EXPAND(AT_SD_BASE_TYPES)
#endif

Scalar _local_scalar_dense_cpu(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_V2(
    self.scalar_type(),
    "_local_scalar_dense_cpu",
    AT_WRAP([&] {
      scalar_t value = *self.const_data_ptr<scalar_t>();
      r = Scalar(value);
    }),
    AT_EXPAND(AT_SD_TYPES)
  );
  return r;
}

} // at::native
