#pragma once

#include <ATen/ATen.h>

namespace at {
namespace caching {

TORCH_API bool is_cached_tensor(const at::Tensor& t);
TORCH_API void add_cached_tensor(const at::Tensor& t);
TORCH_API void remove_cached_tensor(const at::Tensor& t);
TORCH_API void set_cached_tensors_enabled(bool enable);

// For gradient buffer stealing we will adjust the use count of tensors
// which are persisted by cudagraphs, just as we need to adjust reference
// count of tensors with hooks.

#ifndef C10_MOBILE

// For gradient buffer stealing we will adjust the use count of tensors
// which are persisted by cudagraphs, just as we need to adjust reference
// count of tensors with hooks.
TORCH_API size_t adjusted_use_count(const at::Tensor& t);

#else

inline size_t adjusted_use_count(const at::Tensor& t) {
  return t.use_count();
}

#endif

} // namespace caching
} // namespace at
