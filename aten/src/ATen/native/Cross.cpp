#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Cross.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorMeta.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>


#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cross_native.h>
#include <ATen/ops/linalg_cross.h>
#include <ATen/ops/linalg_cross_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(linalg_cross)
(const Tensor & input, const Tensor & other, int64_t dim) {
  auto x_d = input.dim();
  auto y_d = other.dim();
  // This is to avoid things like
  // linalg.cross(torch.randn(2, 3), torch.randn(5, 2, 3), dim=2)
  TORCH_CHECK(x_d == y_d, "linalg.cross: inputs must have the same number of dimensions.");
  TORCH_CHECK(input.size(dim) == 3 && other.size(dim) == 3, "linalg.cross: inputs dimension ", dim, " must have length 3. Got ", input.size(dim), " and ", other.size(dim));

  // Broadcast the batch dimension of input and other.
  // Since the non-batch dimensions agree, this is the same as broadcast all the inputs
  auto out_size = infer_size(input.sizes(), other.sizes());

  set_output_raw_strided(0, out_size, {}, input.options());
}

} // namespace at::meta
namespace at::native {

DEFINE_DISPATCH(cross_stub);

static int64_t _default_cross_dim(const std::optional<int64_t> &dimension, SymIntArrayRef sizes) {
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

Tensor cross(const Tensor & input, const Tensor & other, const std::optional<int64_t> dimension) {
  if (!dimension) {
    TORCH_WARN_ONCE(
      "Using torch.cross without specifying the dim arg is deprecated.\n",
      "Please either pass the dim explicitly or simply use torch.linalg.cross.\n",
      "The default value of dim will change to agree with that of linalg.cross in a future release."
    );
  }
  auto dim = _default_cross_dim(dimension, input.sym_sizes());
  return at::linalg_cross(input, other, dim);
}

Tensor & cross_out(const Tensor & input, const Tensor & other, const std::optional<int64_t> dimension, Tensor & out) {
  auto dim = _default_cross_dim(dimension, input.sym_sizes());
  return at::linalg_cross_out(out, input, other, dim);
}


TORCH_IMPL_FUNC(linalg_cross_out)
(const Tensor & input, const Tensor & other, int64_t dim, const Tensor & out) {
  dim = maybe_wrap_dim(dim, input.dim());
  auto out_size = out.sizes();
  Tensor input_broadcasted = input.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);

  cross_stub(input.device().type(), out, input_broadcasted, other_broadcasted, dim);
}

} // namespace at::native
