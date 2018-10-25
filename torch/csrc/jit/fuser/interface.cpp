#include "torch/csrc/jit/fuser/interface.h"

#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/compiler.h"
#include "torch/csrc/jit/fuser/executor.h"
#include "torch/csrc/jit/fuser/fallback.h"

namespace torch { namespace jit {

namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
bool cpu_fuser_enabled = false;

} // namespace detail

void registerFusion(int64_t& key, const Node* fusion_group) {
  fuser::registerFusion(key, fusion_group); 
}

void runFusion(const int64_t key, Stack& stack) {
  const auto result = fuser::runFusion(key, stack);
  if (!result) fuser::runFallback(key, stack);
}

bool canFuseOnCPU() {
  #if USE_CPU_FUSER
    return detail::cpu_fuser_enabled;
  #endif // USE_CPU_FUSER

  return false;
}

bool canFuseOnGPU() {
  #if USE_CUDA_FUSER
    return true;
  #endif  // USE_CUDA_FUSER

  return false;
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

// Uses the above interface by stuffing the graph into a node and treating that
// node as a fusion group.
std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = wrapper_graph->insertNode(wrapper_graph->createFusionGroup(device));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  int64_t key;
  fuser::registerFusion(key, fusion_group);
  fuser::runFusion(key, stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

size_t nCompiledKernels() { return fuser::nCompiledKernels(); }

} // namespace jit
} // namespace torch
