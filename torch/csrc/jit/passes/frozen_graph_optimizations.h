#pragma once

#include <torch/csrc/jit/ir/ir.h>

/** \brief Runs a set of Optimizations that Optimize Frozen Graphs
 *
 * Currently this set of optimizations is:
 * - FoldFrozenConvBatchnorm
 * - FoldFrozenConvAddOrSub
 * - FoldFrozenConvMulOrDiv
 */

namespace torch {
namespace jit {

TORCH_API void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,
    bool optimize_numerics = true);

} // namespace jit
} // namespace torch
