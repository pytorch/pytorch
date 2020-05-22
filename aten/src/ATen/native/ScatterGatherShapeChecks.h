#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>

namespace at { namespace native {

namespace {

// Used for `scatter` and `scatter_add`
// Tests:
//  1. index.size(d) <= self.size(d) for all d != dim
//  2. index.size(d) <= src.size(d) for all d if src is a Tensor
static void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const c10::optional<Tensor>& src_opt = c10::nullopt
) {
  bool is_wrong_shape = false;
  int64_t self_dims = ensure_nonempty_dim(self.dim());

  //  Check: index.size(d) <= self.size(d) for all d != dim
  for (int64_t d = 0; d < self_dims; ++d) {
    int64_t index_d_size = ensure_nonempty_size(index, d);
    if (d == dim) continue;
    if (index_d_size > ensure_nonempty_size(self, d)) {
      is_wrong_shape = true;
      break;
    }
  }

  //  Check: index.size(d) <= src.size(d) for all d if src is Tensor
  if (!is_wrong_shape && src_opt.has_value()) {
    auto src = src_opt.value();
    for (int64_t d = 0; d < self_dims; ++d) {
      int64_t index_d_size = ensure_nonempty_size(index, d);
      if (index_d_size > ensure_nonempty_size(src, d)) {
        is_wrong_shape = true;
        break;
      }
    }
  }

  if (src_opt.has_value()) {
    auto src = src_opt.value();
    TORCH_CHECK(!is_wrong_shape,
      "Expected index sizes ", index.sizes(),
      " to be smaller than self sizes ", self.sizes(),
      " apart from dimension ", dim,
      " and to be smaller size than src sizes ", src.sizes()
    );
  }
  else {
    TORCH_CHECK(!is_wrong_shape,
      "Expected index sizes ", index.sizes(),
      " to be smaller than self sizes ", self.sizes(),
      " apart from dimension ", dim
    );
  }
}

// If is_external_self == false, then
// used for `gather`-like methods with no preallocated `self` tensor,
// which means that `self`, a tensors to `gather` into,
// is not allocated outside of these `gather`-like methods.
// Aka check for standard `gather`.
//
// Test:
// 1. index.size(d) == self.size(d) for all d != dim
// 2. index.size(d) <= src.size(d) for all d != dim
//
//
// If is_external_self == true, then
// used for `gather`-like methods with preallocated `self` tensor,
// which means that `self`, a tensors to `gather` into,
// is allocated outside of these `gather`-like methods.
//
// Test:
// 1. index.size(d) <= self.size(d) for all d != dim
// 2. index.size(d) <= src.size(d) for all d 
static void gather_shape_check(const Tensor& self, int64_t dim,
  const Tensor& index, const Tensor& src,
  bool is_external_self = false
) {
  auto self_dims = ensure_nonempty_dim(self.dim());

  TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as input tensor"
  );

  if (is_external_self) {
    // note, that checks in this case are like for scatter,
    // but with the roles of `src` and `self` interchanged.
    scatter_shape_check(src, dim, index, self);

    return;
  }

  for (int64_t i = 0; i < self_dims; ++i) {
    if (i != dim) {
      TORCH_CHECK(
        ensure_nonempty_size(index, i) == ensure_nonempty_size(self, i),
        "Size does not match at dimension ", i,
        " get ", ensure_nonempty_size(self, i),
        " vs ", ensure_nonempty_size(index, i)
      );

      TORCH_CHECK(
        ensure_nonempty_size(index, i) <= ensure_nonempty_size(src, i),
        "Size does not match at dimension ", i,
        " expected index sizes ", index.sizes(),
        " to be smaller than src sizes ", src.sizes(),
        " apart from dimension ", dim
      );
    }
  }
}

} // anonymous namespace

}} // namespace at::native
