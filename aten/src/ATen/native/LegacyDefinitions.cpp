#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>

namespace at { namespace native {

// Methods

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

}} // namespace at::native
