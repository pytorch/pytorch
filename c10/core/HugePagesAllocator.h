#pragma once

#include <c10/core/Allocator.h>

namespace c10 {

//C10_API bool installHugePagesAllocator();
C10_API bool installOversizeAllocator();

} // namespace c10
