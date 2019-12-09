#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

std::vector<Pass>& getCustomPasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPass::RegisterPass(Pass p) {
  getCustomPasses().emplace_back(std::move(p));
}

} // namespace jit
} // namespace torch
