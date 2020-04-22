#pragma once

#include <limits>

namespace at { namespace native {

// Maximum and minimum possible scalar values, including infinities
template <typename scalar_t>
constexpr scalar_t upper_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? lim::infinity() : lim::max();
}

template <typename scalar_t>
constexpr scalar_t lower_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? -lim::infinity() : lim::lowest();
}

static inline int64_t ensure_nonempty_dim(int64_t dim) {
  return std::max<int64_t>(dim, 1);
}

static inline int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}

static inline int64_t ensure_nonempty_stride(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.stride(dim);
}

using IdxVec = std::vector<int64_t>;
static inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.size() == 0) {
    vec.push_back(1);
  }
  return vec;
}

static inline Tensor restride_dim(
  const Tensor& src, int64_t dim,
  IntArrayRef replacement_shape
) {
  auto strides = ensure_nonempty_vec(src.strides().vec());
  strides[dim] = 0;
  return src.as_strided(replacement_shape, strides);
}

inline Tensor &_dimreduce_setup(Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntArrayRef self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
  return result;
}

inline bool _dimreduce_return_trivial(Tensor &result, const Tensor &self,
                                      Scalar ident, int64_t dim, bool keepdim) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  // Return identity
  if (self.numel() == 0) {
    _dimreduce_setup(result, self, dim);
    result.fill_(ident);
    if (!keepdim) result.squeeze_(dim);
    return true;
  }
  return false;
}

inline bool _dimreduce_return_trivial_no_ident(Tensor &result, const Tensor &self,
                                               int64_t dim, bool keepdim, const char *fn_name) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }

  if (self.numel() == 0) {
    AT_ERROR("cannot perform reduction function ", fn_name,
             " on tensor with no elements because the operation does not have an identity");
  }
  return false;
}

inline c10::optional<Tensor> _allreduce_return_trivial(
    const Tensor& self,
    Scalar ident) {
  // Return identity
  if (self.numel() == 0) {
    return at::scalar_tensor(ident, self.options());
  }
  return c10::nullopt;
}
}}  // at::native
