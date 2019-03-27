#include <torch/csrc/jit/pass_manager.h>

namespace torch {
namespace jit {

std::vector<Pass>& getPasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPass::RegisterPass(Pass p) {
  getPasses().emplace_back(std::move(p));
}

}
}
