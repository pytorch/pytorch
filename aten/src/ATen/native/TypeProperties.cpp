#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include <type_traits>

namespace at { namespace native {

bool is_cuda(const Tensor& self) {
  return self.type().is_cuda();
}

bool is_distributed(const Tensor& self) {
  return self.type().is_distributed();
}

bool is_floating_point(const Tensor& self) {
  return at::isFloatingType(self.type().scalarType());
}

bool is_signed(const Tensor &self) {
  if (self.type().scalarType() == ScalarType::Half) {
    return true;
  }
  return AT_DISPATCH_ALL_TYPES(self.type(), "is_signed", [&]() -> bool {
    return std::is_signed<scalar_t>();
  });
}

bool is_sparse(const Tensor& self) {
  return self.type().is_sparse();
}

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.toType(other.type());
}

}} // namespace at::native
