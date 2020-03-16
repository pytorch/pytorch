#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

RegisterPostPass::RegisterPostPass(GraphPass p) {
  registerPostPass(p);
}

RegisterPrePass::RegisterPrePass(GraphPass p) {
  registerPrePass(p);
}

std::vector<GraphPassEntry>& getCustomPostPasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

std::vector<GraphPassEntry>& getCustomPrePasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

GraphPassNameType RegisterPostPass::registerPostPass(GraphPass p) {
  getCustomPostPasses().emplace_back(GraphPassEntry{std::move(p), graphPassID});
  return graphPassID++;
}

GraphPassNameType RegisterPrePass::registerPrePass(GraphPass p) {
  getCustomPrePasses().emplace_back(GraphPassEntry{std::move(p), graphPassID});
  return graphPassID++;
}

ClearPostPass::ClearPostPass(GraphPassNameType pid) {
  auto& passes = getCustomPostPasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

ClearPrePass::ClearPrePass(GraphPassNameType pid) {
  auto& passes = getCustomPrePasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

ClearAllPostPasses::ClearAllPostPasses() {
  auto& passes = getCustomPostPasses();
  passes.erase(passes.begin(), passes.end());
}

ClearAllPrePasses::ClearAllPrePasses() {
  auto& passes = getCustomPrePasses();
  passes.erase(passes.begin(), passes.end());
}

GraphPassNameType PassManager::name(GraphPassNameType PassName, bool set) {
  static GraphPassNameType name = 0;
  if (set)
    name = PassName;
  return name;
}

bool PassManager::flipRegistered(bool flip) {
  static bool val = false;
  if (flip)
    val = !val;
  return val;
}
void PassManager::registerPass(GraphPass pass) {
  if (!flipRegistered()) {
    name(RegisterPostPass::registerPostPass(pass), true);
    flipRegistered(true);
  }
}

void PassManager::clearPass() {
  if (flipRegistered()) {
    ClearPostPass pass(name());
    flipRegistered(true);
  }
}

} // namespace jit
} // namespace torch
