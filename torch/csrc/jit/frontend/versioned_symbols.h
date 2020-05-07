#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/api/module.h>

#include <cstdint>

namespace torch {
namespace jit {

// Maps the given symbol into an implementation of its behavior at the
// given version.
// See note [Versioned Symbols]
TORCH_API Symbol
get_symbol_for_version(const Symbol name, const uint64_t version);

} // namespace jit
} // namespace torch
