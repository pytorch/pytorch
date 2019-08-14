#ifdef BUILD_NAMEDTENSOR
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

Tensor& set_names_(Tensor& self, optional<DimnameList> names) {
  return at::internal_set_names_inplace(self, names);
}

Tensor set_names(const Tensor& self, optional<DimnameList> names) {
  auto result = self.alias();
  at::internal_set_names_inplace(result, names);
  return result;
}

}}  // namespace at::native
#endif
