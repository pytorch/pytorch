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

bool nonConstantParameters(Node* n);
// Checks if the parameters, not including the
// first param are all constants.

TORCH_API void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,
    bool optimize_numerics = true);

} // namespace jit
} // namespace torch
