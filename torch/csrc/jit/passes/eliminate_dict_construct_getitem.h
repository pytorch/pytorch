#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/* Given:
 *   %2 : Dict = prim::DictConstruct(%key, %value)
 *   %3 : Tensor = aten::__getitem__(%2, %key)
 *   %4 : Tensor = op(%3)
 * Produces:
 *   %2 : Dict = prim::DictConstruct(%key, %value)
 *   %3 : Tensor = aten::__getitem__(%2, %key)
 *   %4 : Tensor = op(%value)
 *
 * This only happens if the only uses of the DictConstruct output are
 * __getitem__s; the optimization is not safe if the dict is mutated!
 *
 * The calls to DictConstruct / __getitem__ can then be removed via
 * dead code elimination.
 */
TORCH_API void EliminateDictConstructGetItem(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
