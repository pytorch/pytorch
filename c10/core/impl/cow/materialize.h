#pragma once

#include <c10/macros/Macros.h>

namespace c10 {
struct Storage;
}; // namespace c10

namespace c10::impl::cow {

auto C10_API materialize(Storage const& storage) -> void;

} // namespace c10::impl::cow
