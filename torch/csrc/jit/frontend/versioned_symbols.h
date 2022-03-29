#pragma once

#include <caffe2/serialize/versions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>

#include <cstdint>

namespace torch {
namespace jit {
// Maps the given symbol into an implementation of its behavior at the
// given version.
// See note [Versioned Symbols]
TORCH_API Symbol
get_symbol_for_version(const Symbol name, const uint64_t version);

// Maps the given kind to the minimum version that supports it.
// See note [Dynamic Versions and torch.jit.save vs. torch.save]
TORCH_API uint64_t get_min_version_for_kind(const NodeKind& kind);
} // namespace jit
} // namespace torch
