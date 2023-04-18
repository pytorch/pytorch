#pragma once

#include <ATen/ATen.h>

namespace at {
namespace caching {


TORCH_API bool is_cached_tensor(const at::Tensor& t);
TORCH_API void add_cached_tensor(const at::Tensor& t);
TORCH_API void remove_cached_tensor(const at::Tensor& t);
TORCH_API void set_cached_tensors_enabled(bool enable);

}
}