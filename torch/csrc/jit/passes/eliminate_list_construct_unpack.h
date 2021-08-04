#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/* Given:
 *          %3:Tensor[] = prim::ListConstruct(%0, %1)
 *          %4:Tensor, %5:Tensor = prim ::ListUnpack(%3)
 *          some op (%4, %5)
 *
 * Result:
 *          %3:Tensor[] = prim::ListConstruct(%0, %1)
 *          %4:Tensor, %5:Tensor = prim ::ListUnpack(%3)
 *          some op (%0, %1)
 */
void EliminateListConstructUnpack(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch
