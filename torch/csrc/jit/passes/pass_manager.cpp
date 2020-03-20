#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

std::vector<GraphPass>& getCustomPostFusionPasses() {
  static std::vector<GraphPass> passes;
  return passes;
}

std::vector<GraphPass>& getCustomPreFusionPasses() {
  static std::vector<GraphPass> passes;
  return passes;
}

RegisterPostFusionPass::RegisterPostFusionPass(GraphPass p) {
  getCustomPostFusionPasses().emplace_back(std::move(p));
}

RegisterPreFusionPass::RegisterPreFusionPass(GraphPass p) {
  getCustomPreFusionPasses().emplace_back(std::move(p));
}

} // namespace jit
} // namespace torch
