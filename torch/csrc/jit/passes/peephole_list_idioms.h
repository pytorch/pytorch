#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Peephole Optimizes List Ops such as len(li) and li[1].
// Currently this is invoked as part of PeepholeOptimize
// return true if graph is modified.
// If `refine_list_len` is true will attempt to refine the len of lists through
// len comparisons and assertions. This does not generally optimize pytorch
// programs so it is not called by default in PeepholeOptimize.
TORCH_API bool PeepholeOptimizeListIdioms(
    const std::shared_ptr<Graph>& graph,
    bool refine_list_len = false);

} // namespace jit
} // namespace torch
