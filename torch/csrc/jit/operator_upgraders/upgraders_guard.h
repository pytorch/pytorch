#pragma once
#include <c10/macros/Export.h>

namespace torch {
namespace jit {

// Flag that controls if we want to enable upgraders
// in the server side. When this flag is set to False,
// it will switch to old dynamic versioning approach
#define ENABLE_UPGRADERS true

TORCH_API bool is_upgraders_enabled();

} // namespace jit
} // namespace torch
