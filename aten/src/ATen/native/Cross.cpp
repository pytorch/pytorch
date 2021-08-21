#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

#include <ATen/native/Cross.h>

namespace at { namespace native {

DEFINE_DISPATCH(cross_stub);

int64_t _default_cross_dim(const c10::optional<int64_t> &dimension, IntArrayRef sizes) {
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
  auto device1 = input.device().type();
  TORCH_CHECK(input.dim() == other.dim(), "inconsistent tensors dimensions input: ", input.dim(), " other: ", other.dim());
  TORCH_CHECK(input.sizes() == other.sizes(), "inconsistent tensors sizes input: ", input.sizes(), " other: ", other.sizes());

  // default dimension=-1
  int64_t dim = maybe_wrap_dim(dimension, input.dim());
  TORCH_CHECK(input.size(dim) == 3, "dimension ", dimension, " does not have size 3");

  // check if resizing output is required
  // raise a warning while resizing if output has one or more elements
  at::native::resize_output(out, input.sizes());

  cross_stub(device1, out, input, other, dim);
  return out;
}

}} // namespace at::native
