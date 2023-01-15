#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/kernel_spec.h>
#include <torch/csrc/jit/ir/ir.h>

#include <cstdint>
#include <functional>

namespace torch {
namespace jit {
namespace fuser {

// A thread-safe cache interface.

// Normalizes the graph by canonicalizing and erasing shape information
TORCH_API std::shared_ptr<Graph> normalizeGraphForCache(
    const std::shared_ptr<Graph>& graph);

// Stores the given graph, returning the key used to access it
TORCH_API int64_t store(std::shared_ptr<Graph> graph);

// Given a graph, find a KernelSpec based on it
TORCH_API at::optional<KernelSpec*> lookupGraph(std::shared_ptr<Graph> graph);

// Returns the graph corresponding to the given key (if it exists)
TORCH_API at::optional<KernelSpec*> retrieve(const int64_t key);

// Returns the size of the fusion key -> KernelSpec cache.
// Only used for testing.
TORCH_API int64_t debugNumCachedKernelSpecs();

} // namespace fuser
} // namespace jit
} // namespace torch
