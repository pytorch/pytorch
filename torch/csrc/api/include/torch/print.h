#pragma once
#include <c10/macros/Export.h>

namespace torch {
/// Set print options for tensor printing
/// 
/// Controls how tensors are displayed when printed to streams.
/// 
/// \param sci_mode Whether to use scientific notation for floating point numbers
TORCH_API void set_printoptions(bool sci_mode = true);

} // namespace torch
