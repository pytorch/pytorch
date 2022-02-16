#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <ATen/ExpandUtils.h>

#include <ATen/native/Cross.h>

namespace at {
namespace meta {

TORCH_PRECOMPUTE_META_FUNC(linalg_cross)
(const Tensor & input, const Tensor & other, const int64_t dimension) {
  auto out_size = infer_size(input.sizes(), other.sizes());
  Tensor input_broadcasted = input.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);

  int64_t dim = maybe_wrap_dim(dimension, input.dim()); // default dim = -1
  TORCH_CHECK(input_broadcasted.size(dim) == 3, "dimension ", dimension, " does not have size 3");

  set_output(out_size, input.options());
  return TORCH_PRECOMPUTE_STRUCT(linalg_cross)().set_dim(dim);
}

}
namespace native {

DEFINE_DISPATCH(cross_stub);

int64_t _default_cross_dim(const c10::optional<int64_t> &dimension, IntArrayRef sizes) {
  // If dimension is not given, it defaults to the first dimension found with the size 3.
  // Note that this behaviour might be unexpected.
  // _default_cross_dim is called internally inside the cross implementation to calculate
  // the dim and finally cross delegates to the linalg_cross implementation with this dim
  if(dimension.has_value()) {
    return *dimension;
  }

  for(auto i : c10::irange(sizes.size())) {
    if(sizes[i] == 3) {
      return i;
    }
  }
  TORCH_CHECK(false, "no dimension of size 3 in input");
}

Tensor cross(const Tensor & input, const Tensor & other, const c10::optional<int64_t> dimension) {
  auto dim = _default_cross_dim(dimension, input.sizes());
  return at::linalg_cross(input, other, dim);
}

Tensor & cross_out(const Tensor & input, const Tensor & other, const c10::optional<int64_t> dimension, Tensor & out) {
  auto dim = _default_cross_dim(dimension, input.sizes());
  return at::linalg_cross_out(out, input, other, dim);
}

TORCH_IMPL_FUNC(linalg_cross_out)
(const Tensor & input, const Tensor & other, const int64_t dim, const Tensor & out) {
  auto out_size = infer_size(input.sizes(), other.sizes());
  Tensor input_broadcasted = input.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);

  cross_stub(input.device().type(), out, input_broadcasted, other_broadcasted, dim);
}

}} // namespace at::native
