#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {
Tensor null(const Tensor& self) {
  return self;
}
Tensor null_c10(Tensor self) {
  return self;
}
Tensor null4(const Tensor& self, const Tensor& a, const Tensor& b, const Tensor& c) {
  return self;
}
Tensor null4_c10(Tensor self, Tensor a, Tensor b, Tensor c) {
  return self;
}
Tensor null_back(const Tensor& self) {
  return self;
}
Tensor null_back_c10(Tensor self) {
  return self;
}

}}  // namespace at::native
