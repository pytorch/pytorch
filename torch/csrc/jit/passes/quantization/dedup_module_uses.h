#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

/** Recursively deduplicate multiple uses of the same module by
 *  creating an instance clone for each use of the module, which means
 *  the type will be the same as before and all the attributes will be
 *  copied, then we'll change the use of the original module to the use
 *  of cloned module in the Graph.
 *
 *  This is done to ensure that modules can survive destructive passes
 *  without changing model behavior. For example, here:
 *
 *    x = self.conv1(x)
 *    x = self.relu(x)
 *    x = self.conv2(x)
 *    x = self.relu(x)
 *
 *  self.relu needs to be deduplicated for potential future destructive passes
 *  to work properly.
 */
TORCH_API void DedupModuleUses(Module& module);

} // namespace jit
} // namespace torch
