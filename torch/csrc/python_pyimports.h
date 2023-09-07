#pragma once
#include <c10/core/pyimports.h>

namespace torch {
namespace detail {

// See NOTE: [torch::Library and Python imports] for more details.
// Returns a list of (module name, context) of modules that were ignored
// by previous calls to `request_pyimport`.
TORCH_API c10::impl::IgnoredPyImports initialize_pyimports_handler();

} // namespace detail
} // namespace torch
