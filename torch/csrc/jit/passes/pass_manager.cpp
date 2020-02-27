
copy: fbcode/caffe2/torch/csrc/jit/passes/pass_manager.cpp
copyrev: 34dd1a8534b0a639e22828718fa438f2c66d26c4

#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

std::vector<Pass>& getCustomPostFusionPasses() {
  static std::vector<Pass> passes;
  return passes;
}

std::vector<Pass>& getCustomPreFusionPasses() {
  static std::vector<Pass> passes;
  return passes;
}

RegisterPostFusionPass::RegisterPostFusionPass(Pass p) {
  getCustomPostFusionPasses().emplace_back(std::move(p));
}

RegisterPreFusionPass::RegisterPreFusionPass(Pass p) {
  getCustomPreFusionPasses().emplace_back(std::move(p));
}

} // namespace jit
} // namespace torch
