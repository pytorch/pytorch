#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

std::vector<GraphPassEntry>& getCustomPostPasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

std::vector<GraphPassEntry>& getCustomPrePasses() {
  static std::vector<GraphPassEntry> passes;
  return passes;
}

GraphPassNameType registerPostPass(GraphPass p) {
  getCustomPostPasses().emplace_back(GraphPassEntry{std::move(p), graphPassID});
  return graphPassID++;
}

GraphPassNameType registerPrePass(GraphPass p) {
  getCustomPrePasses().emplace_back(GraphPassEntry{std::move(p), graphPassID});
  return graphPassID++;
}

void ClearPostPass(GraphPassNameType pid) {
  auto& passes = getCustomPostPasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

void ClearPrePass(GraphPassNameType pid) {
  auto& passes = getCustomPrePasses();
  auto it = passes.begin();
  for (; it != passes.end(); it++) {
    if (pid == (*it).second)
      break;
  }
  if (it != passes.end())
    passes.erase(it);
}

void ClearAllPostPasses() {
  auto& passes = getCustomPostPasses();
  passes.erase(passes.begin(), passes.end());
}

void ClearAllPrePasses() {
  auto& passes = getCustomPrePasses();
  passes.erase(passes.begin(), passes.end());
}

template<typename DerivedType>
GraphPassNameType PassManager<DerivedType>::name(GraphPassNameType PassName, bool set) {
  static GraphPassNameType name = 0;
  if (set)
    name = PassName;
  return name;
}

template<typename DerivedType>
bool PassManager<DerivedType>::isRegistered(bool flip_bit) {
  static bool val = false;
  if (flip_bit)
    val = !val;
  return val;
}

template<typename DerivedType>
void PassManager<DerivedType>::registerPass(GraphPass pass) {
  if (!isRegistered()) {
    // If we don't already have a registered pass, register pass
    // hold on to its name, change isRegistered to true
    name(RegisterPostPass::registerPostPass(std::move(pass)), true);
    isRegistered(true);
  }
}

template<typename DerivedType>
void PassManager<DerivedType>::clearPass() {
  //If the pass is registered, clear it and change isRegistered to false.
  if (isRegistered()) {
    ClearPostPass pass(name());
    isRegistered(true);
  }
}

} // namespace jit
} // namespace torch
