#pragma once

#include <vector>
#include <ATen/ATen.h>
#include <ATen/native/ReduceOpsUtils.h>

namespace at { namespace native {

namespace {

// checks whether index.dtype == int64
// and self.dtyp == src.dtype if src is a Tensor
static void scatter_gather_dtype_check(
  const std::string& method_name,
  const Tensor& self,
  const Tensor& index,
  const c10::optional<Tensor>& src_opt = c10::nullopt
) {
  TORCH_CHECK(
    index.scalar_type() == at::ScalarType::Long,
    method_name, "(): Expected dtype int64 for index"
  );

  if (src_opt.has_value()) {
    auto src = src_opt.value();
    TORCH_CHECK(
      self.scalar_type() == src.scalar_type(),
      method_name, "(): Expected self.dtype to be equal to src.dtype"
    );
  }
}

// Used for `gather`-like methods
// Test:
// 1. index.size(d) == self.size(d) for all d != dim
// 2. index.size(d) <= src.size(d) for all d != dim
// 3. index.dim() == self.dim() == src.dim()
static void gather_shape_check(const Tensor& self, int64_t dim,
  const Tensor& index, const Tensor& src
) {
  auto self_dims = ensure_nonempty_dim(self.dim());
  TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as out tensor"
  );

  auto src_dims = ensure_nonempty_dim(src.dim());
  TORCH_CHECK(src_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as input tensor"
  );

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
        " expected index ", index.sizes(),
        " to be smaller than src ", src.sizes(),
        " apart from dimension ", dim
      );
    }
  }
}
// Used for `scatter` and `scatter_add`
// Tests:
//  1. index.size(d) <= self.size(d) for all d != dim
//  2. index.size(d) <= src.size(d) for all d if src is a Tensor
//  3. index.dim() == self.dim() == src.dim()
static void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const c10::optional<Tensor>& src_opt = c10::nullopt
) {
  TORCH_CHECK(
    ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as self tensor"
  );

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

    TORCH_CHECK(
      ensure_nonempty_dim(src.dim()) == ensure_nonempty_dim(index.dim()),
      "Index tensor must have the same number of dimensions as src tensor"
    );

    TORCH_CHECK(!is_wrong_shape,
      "Expected index ", index.sizes(),
      " to be smaller than self ", self.sizes(),
      " apart from dimension ", dim,
      " and to be smaller size than src ", src.sizes()
    );
  }
  else {
    TORCH_CHECK(!is_wrong_shape,
      "Expected index ", index.sizes(),
      " to be smaller than self ", self.sizes(),
      " apart from dimension ", dim
    );
  }
}

} // anonymous namespace

}} // namespace at::native
