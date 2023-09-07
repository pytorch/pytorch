#pragma once
#include <c10/core/pyimports.h>

namespace torch {
namespace detail {

TORCH_API const c10::impl::IgnoredPyImports& initialize_pyimports_handler();

}
} // namespace torch
