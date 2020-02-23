#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

std::vector<Pass>& getCustomPostPasses() {
  static std::vector<Pass> passes;
  return passes;
}

std::vector<Pass>& getCustomPrePasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPostFusionPass::RegisterPostFusionPass(Pass p) {
  getCustomPostPasses().emplace_back(std::move(p));
}

RegisterPreFusionPass::RegisterPreFusionPass(Pass p) {
  getCustomPrePasses().emplace_back(std::move(p));
}

} // namespace jit
} // namespace torch
