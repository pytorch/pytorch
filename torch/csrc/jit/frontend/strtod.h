#pragma once

#include <ATen/core/Macros.h>

namespace torch {
namespace jit {

TORCH_API double strtod_c(const char* nptr, char** endptr);
TORCH_API float strtof_c(const char* nptr, char** endptr);

} // namespace jit
} // namespace torch
