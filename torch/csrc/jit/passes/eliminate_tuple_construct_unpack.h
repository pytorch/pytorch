#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/* Given:
 *          %3:Tensor[] = prim::TupleConstruct(%0, %1)
 *          %4:Tensor, %5:Tensor = prim ::TupleUnpack(%3)
 *          some op (%4, %5)
 *
 * Result:
 *          %3:Tensor[] = prim::TupleConstruct(%0, %1)
 *          %4:Tensor, %5:Tensor = prim ::TupleUnpack(%3)
 *          some op (%0, %1)
 */
void EliminateTupleConstructUnpack(std::shared_ptr<torch::jit::Graph>& graph);

/* Given:
 *          %1:Tensor, %2:Tensor = prim ::TupleUnpack(%0)
 *          %3:Tensor[] = prim::TupleConstruct(%1, %2)
 *          some op (%3)
 *
 * Result:
 *          %1:Tensor, %2:Tensor = prim ::TupleUnpack(%0)
 *          %3:Tensor[] = prim::TupleConstruct(%1, %2)
 *          some op (%0)
 */
void EliminateTupleUnpackConstruct(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace jit
} // namespace torch
