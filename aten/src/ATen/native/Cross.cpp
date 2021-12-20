#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <ATen/ExpandUtils.h>

#include <ATen/native/Cross.h>

namespace at { namespace native {

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

Tensor linalg_cross(const Tensor & input, const Tensor & other, const int64_t dimension) {
  Tensor out = at::empty({0}, input.options());
  native::linalg_cross_out(input, other, dimension, out);
  return out;
}

Tensor & linalg_cross_out(const Tensor & input, const Tensor & other, const int64_t dimension, Tensor & out) {
  // Broadcast inputs
  auto out_size = infer_size(input.sizes(), other.sizes());
  Tensor input_broadcasted = input.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);

  // default dimension=-1
  int64_t dim = maybe_wrap_dim(dimension, input.dim());
  TORCH_CHECK(input_broadcasted.size(dim) == 3, "dimension ", dimension, " does not have size 3");

  // check if resizing output is required
  // raise a warning while resizing if output has one or more elements
  at::native::resize_output(out, out_size);

  cross_stub(input.device().type(), out, input_broadcasted, other_broadcasted, dim);
  return out;
}

}} // namespace at::native
