#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <type_traits>

namespace at { namespace native {

bool is_cuda(const Tensor& self) {
  return self.is_cuda();
}

bool is_distributed(const Tensor& self) {
  return self.is_distributed();
}

bool is_complex(const Tensor& self) {
  return at::isComplexType(self.scalar_type());
}

bool is_floating_point(const Tensor& self) {
  return at::isFloatingType(self.scalar_type());
}

bool is_signed(const Tensor &self) {
  if (self.scalar_type() == ScalarType::Half) {
    return true;
  }
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "is_signed", [&]() -> bool {
    return std::is_signed<scalar_t>();
  });
}

bool is_sparse(const Tensor& self) {
  return self.is_sparse();
}

bool is_quantized(const Tensor& self) {
  return self.is_quantized();
}

// True if `self` has the same derived type of TensorImpl as `other`.
bool _has_same_tensorimpl_type(const Tensor& self, const Tensor& other) {
  return typeid(*(self.unsafeGetTensorImpl())) == typeid(*(other.unsafeGetTensorImpl()));
}

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.toType(other.type());
}

}} // namespace at::native
