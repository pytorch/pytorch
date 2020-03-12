#pragma once

#include <torch/csrc/jit/ir/ir.h>

/* `getCustomPreFusionPasses()` returns a vector of passes that will be executed
 * after differentiation but before any fusion. This is the de-facto location
 * for compiler backends to insert passes.
 *
 * `getCustomPostFusionPasses()` returns a vector of passes that will be
 * executed after differentiation and after fusion (if any). This is the
 * location for fusion cleanup passes if they are needed.
 *
 * Static registration of a pass can be done by creating a global
 * `Register{Pre,Post}FusionPass r(Pass)` variable in a compilation unit.
 *
 * pass_manager.h uses a Meyer's singleton to store a vector of `Pass`es, which
 * modify the IR graph in place.
 */

namespace torch {
namespace jit {

// A pass modifies a Graph in place.
using GraphPass = std::function<void(std::shared_ptr<Graph>&)>;

TORCH_API std::vector<GraphPass>& getCustomPostFusionPasses();
TORCH_API std::vector<GraphPass>& getCustomPreFusionPasses();

struct TORCH_API RegisterPostFusionPass {
  RegisterPostFusionPass(GraphPass p);
};

using RegisterPass = RegisterPostFusionPass;

struct TORCH_API RegisterPreFusionPass {
  RegisterPreFusionPass(GraphPass p);
};

} // namespace jit
} // namespace torch
