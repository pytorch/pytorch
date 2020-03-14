#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

RegisterPostFusionPass::RegisterPostFusionPass(GraphPass p){
  registerPostFusionPass(p);
}

RegisterPreFusionPass::RegisterPreFusionPass(GraphPass p){
  registerPreFusionPass(p);
}

std::vector<GraphPassEntry>& getCustomPostFusionPasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

std::vector<GraphPassEntry>& getCustomPreFusionPasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

GraphPassNameType RegisterPostFusionPass::registerPostFusionPass(GraphPass p) {
  getCustomPostFusionPasses().emplace_back(GraphPassEntry{std::move(p), graphPassID});
  return graphPassID++;
}

GraphPassNameType RegisterPreFusionPass::registerPreFusionPass(GraphPass p) {
  getCustomPreFusionPasses().emplace_back(GraphPassEntry{std::move(p), graphPassID});
  return graphPassID++;
}

ClearPostFusionPass::ClearPostFusionPass(GraphPassNameType pid) {
  auto& passes = getCustomPostFusionPasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

ClearPreFusionPass::ClearPreFusionPass(GraphPassNameType pid) {
  auto& passes = getCustomPreFusionPasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

ClearAllPostFusionPasses::ClearAllPostFusionPasses() {
  auto& passes = getCustomPostFusionPasses();
  passes.erase(passes.begin(), passes.end());
}

ClearAllPreFusionPasses::ClearAllPreFusionPasses() {
  auto& passes = getCustomPreFusionPasses();
  passes.erase(passes.begin(), passes.end());
}

GraphPassNameType PassManager::name(GraphPassNameType PassName, bool set){
  static GraphPassNameType name = 0;
  if(set)
    name = PassName;
  return name;
}

bool PassManager::flipRegistered(bool flip){
  static bool val = false;
  if(flip) val = !val;
  return val;
}
void PassManager::registerPass(GraphPass pass) {
  if (!flipRegistered()) {
    name( RegisterPostFusionPass::registerPostFusionPass(pass), true );
    flipRegistered(true);
  }
}

void PassManager::clearPass() {
  if (flipRegistered()) {
    ClearPostFusionPass pass(name());
    flipRegistered(true);
  }
}


} // namespace jit
} // namespace torch
