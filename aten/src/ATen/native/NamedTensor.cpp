#ifdef BUILD_NAMEDTENSOR
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

Tensor& set_names_(Tensor& self, optional<DimnameList> names) {
  return at::internal_set_names_inplace(self, names);
}

Tensor align_to(const Tensor& tensor, DimnameList names) {
  auto tensor_sizes = tensor.sizes();
  TORCH_CHECK(
      tensor.has_names() || tensor_sizes.size() == 0,
      "align_to: input tensor must have named dimensions.");
  TORCH_CHECK(
      names.size() >= tensor.dim(),
      "Cannot align tensor with dims ", tensor.names().value(),
      " to a shorter list of dims ", names, ".");

  std::vector<int64_t> expanded_sizes(names.size(), 1);
  auto tensor_names = *tensor.names();
  ptrdiff_t dim = (ptrdiff_t)tensor.dim() - 1;
  ptrdiff_t idx = (ptrdiff_t)names.size() - 1;
  for (; idx >= 0 && dim >= 0; --idx) {
    TORCH_INTERNAL_ASSERT(!tensor_names[dim].is_tagged() && !names[idx].is_tagged(), "Tagged names NYI");
    if (tensor_names[dim] != names[idx]) {
      continue;
    }
    if (tensor_names[dim].is_wildcard()) {
      TORCH_CHECK(
          tensor_sizes.size() - dim == names.size() - idx,
          "Aligning tensor `a` with dims ", tensor_names, " to `names` ", names,
          "would change the absolute position from the right of an unnamed dimension. ",
          "Please name unnamed dimensions to avoid ambiguity.")
    }
    expanded_sizes[idx] = tensor_sizes[dim];
    --dim;
  }
  TORCH_CHECK(
      dim == -1,
      "Could not align tensor `a` with dims ", tensor_names,
      " to `names` ", names, " because `a.names` is not a subsequence of `names`.");

  auto result = tensor.view(expanded_sizes);
  at::internal_set_names_inplace(result, names);
  return result;
}

}}  // namespace at::native
#endif
