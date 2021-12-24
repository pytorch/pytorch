#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_guard.h>

#include <cstdint>

namespace torch {
namespace jit {
#if !ENABLE_UPGRADERS
// Maps the given symbol into an implementation of its behavior at the
// given version.
// See note [Versioned Symbols]
TORCH_API Symbol
get_symbol_for_version(const Symbol name, const uint64_t version);

// Maps the given kind to the minimum version that supports it.
// See note [Dynamic Versions and torch.jit.save vs. torch.save]
TORCH_API uint64_t get_min_version_for_kind(const NodeKind& kind);
#endif
} // namespace jit
} // namespace torch
