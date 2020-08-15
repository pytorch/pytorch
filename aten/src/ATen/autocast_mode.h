#pragma once

namespace at {
namespace autocast {

TORCH_API bool is_enabled();
TORCH_API void set_enabled(bool enabled);
TORCH_API void clear_cache();
TORCH_API int increment_nesting();
TORCH_API int decrement_nesting();

} // namespace autocast
} // namespace at
