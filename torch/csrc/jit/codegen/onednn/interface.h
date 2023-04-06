#pragma once
#include <ATen/Config.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

static std::atomic<bool> onednn_enabled{false};

std::atomic<bool>& getLlgaEnabled() {
  return onednn_enabled;
}

C10_EXPORT void fuseGraph(std::shared_ptr<Graph>& g);

} // namespace onednn
} // namespace fuser

struct C10_EXPORT RegisterLlgaFuseGraph
    : public PassManager<RegisterLlgaFuseGraph> {
  static bool setEnabled(bool enabled) {
    TORCH_CHECK(
        AT_MKLDNN_ENABLED(),
        "Running oneDNN Graph fuser is only supported with MKLDNN builds.");
    bool oldState = fuser::onednn::getLlgaEnabled();
    fuser::onednn::getLlgaEnabled() = enabled;
    if (enabled) {
      registerPass(fuser::onednn::fuseGraph);
    } else {
      clearPass();
    }
    return oldState;
  }

  static bool isEnabled() {
    return fuser::onednn::getLlgaEnabled();
  }

  // override PassManager::registerPass to register pre-pass
  static bool registerPass(GraphPass p) {
    if (!isRegistered()) {
      passID(registerPrePass(std::move(p)), true);
      isRegistered(true);
      return false;
    }
    return true;
  }

  // override PassManager::clearPass to clear pre-pass
  static void clearPass() {
    if (isRegistered()) {
      clearPrePass(passID());
      isRegistered(true);
    }
  }
};

} // namespace jit
} // namespace torch
