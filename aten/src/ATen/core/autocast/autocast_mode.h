#pragma once

namespace at {
namespace autocast {

bool is_enabled();
void set_enabled(bool enabled);
void clear_cache();
int increment_nesting();
int decrement_nesting();

} // namespace autocast
} // namespace at
