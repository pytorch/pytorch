#pragma once

#include <vector>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

// checks whether index.dtype == int64
// and self.dtype == src.dtype if src is a Tensor
inline void scatter_gather_dtype_check(
  const std::string& method_name,
  const Tensor& self,
  const Tensor& index,
  const std::optional<Tensor>& src_opt = std::nullopt
) {
  if (index.numel() != 0) {
    TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long,
      method_name, "(): Expected dtype int64 for index"
    );
  }

  if (src_opt.has_value()) {
    const auto& src = src_opt.value();
    TORCH_CHECK(
      self.scalar_type() == src.scalar_type(),
      method_name, "(): Expected self.dtype to be equal to src.dtype"
    );
  }
}

// Used for `gather`-like methods
// Note: self means the input tensor here
// Test:
// 1. index.size(d) <= self.size(d) for all d != dim
// 2. index.dim() == self.dim()
inline void gather_shape_check(const Tensor& self, int64_t dim,
  const Tensor& index
) {
  auto self_dims = ensure_nonempty_dim(self.dim());
  TORCH_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as input tensor"
  );

  for (const auto i : c10::irange(self_dims)) {
    if (i != dim) {
      TORCH_CHECK(
        ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
        "Size does not match at dimension ", i,
        " expected index ", index.sizes(),
        " to be smaller than self ", self.sizes(),
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
inline void scatter_shape_check(
  const Tensor& self, int64_t dim, const Tensor& index,
  const std::optional<Tensor>& src_opt = std::nullopt
) {
  if (index.numel() == 0) return;
  TORCH_CHECK(
    ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
    "Index tensor must have the same number of dimensions as self tensor"
  );

  bool is_wrong_shape = false;
  int64_t self_dims = ensure_nonempty_dim(self.dim());

  //  Check: index.size(d) <= self.size(d) for all d != dim
  for (const auto d : c10::irange(self_dims)) {
    int64_t index_d_size = ensure_nonempty_size(index, d);
    if (d == dim) continue;
    if (index_d_size > ensure_nonempty_size(self, d)) {
      is_wrong_shape = true;
      break;
    }
  }

  //  Check: index.size(d) <= src.size(d) for all d if src is Tensor
  if (!is_wrong_shape && src_opt.has_value()) {
    const auto& src = src_opt.value();
    for (const auto d : c10::irange(self_dims)) {
      int64_t index_d_size = ensure_nonempty_size(index, d);
      if (index_d_size > ensure_nonempty_size(src, d)) {
        is_wrong_shape = true;
        break;
      }
    }
  }

  if (src_opt.has_value()) {
    const auto& src = src_opt.value();

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

} // namespace at::native
