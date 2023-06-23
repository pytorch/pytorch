#pragma once

#include <ATen/ATen.h>

namespace at {
namespace caching {

// Some systems (just cudagraphs currently) will persist a static tensor output
// whose TensorImpl does not change across iterations. For these tensors caching
// dtype conversions is invalid. Additionally, there will be an extra reference
// count to these cached tensors that would prevent buffer inplacing and other
// checks on tensor uniqueness. If we are not using these systems the enabled
// flag will be false and we will avoid the hash lookup.

TORCH_API bool is_cached_tensor(const at::Tensor& t);
TORCH_API void add_cached_tensor(const at::Tensor& t);
TORCH_API void remove_cached_tensor(const at::Tensor& t);
TORCH_API void set_cached_tensors_enabled(bool enable);

// For gradient buffer stealing we will adjust the use count of tensors
// which are persisted by cudagraphs, just as we need to adjust reference
// count of tensors with hooks.
TORCH_API size_t adjusted_use_count(const at::Tensor& t);

} // namespace caching
} // namespace at
