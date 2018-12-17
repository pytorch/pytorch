#include "torch/csrc/jit/fuser/interface.h"

#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER
  #include "torch/csrc/jit/fuser/compiler.h"
  #include "torch/csrc/jit/fuser/executor.h"
  #include "torch/csrc/jit/fuser/fallback.h"
#endif // USE_CUDA_FUSER || USE_CPU_FUSER

#include <stdexcept>

namespace torch { namespace jit {

namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
bool cpu_fuser_enabled = false;

} // namespace detail

int64_t registerFusion(const Node* fusion_group) {
  #if USE_CUDA_FUSER || USE_CPU_FUSER
    return fuser::registerFusion(fusion_group);
  #else
    throw std::runtime_error("Fusion not supported for this build.");
  #endif // USE_CUDA_FUSER || USE_CPU_FUSER
}

void runFusion(const int64_t key, Stack& stack) {
  #if USE_CUDA_FUSER || USE_CPU_FUSER
    const auto result = fuser::runFusion(key, stack);
    if (!result) fuser::runFallback(key, stack);
  #else 
    throw std::runtime_error("Fusion not supported for this build.");
  #endif // USE_CUDA_FUSER || USE_CPU_FUSER
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
, at::ArrayRef<at::Tensor> inputs) {
  #if USE_CUDA_FUSER || USE_CPU_FUSER
    // Creates a fusion group node
    auto wrapper_graph = std::make_shared<Graph>();
    Node* fusion_group = wrapper_graph->insertNode(wrapper_graph->createFusionGroup());
    fusion_group->g_(attr::Subgraph, graph.copy());
    for (size_t i = 0; i < graph.inputs().size(); ++i) {
      fusion_group->addInput(wrapper_graph->addInput());
    }
    for (size_t i = 0; i < graph.outputs().size(); ++i) {
      wrapper_graph->registerOutput(fusion_group->addOutput());
    }

    // Creates the stack, registers and runs the fusion
    Stack stack = fmap<IValue>(inputs);
    const auto key = fuser::registerFusion(fusion_group);
    fuser::runFusion(key, stack);
    return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
  #else 
    throw std::runtime_error("Fusion not supported for this build.");
  #endif // USE_CUDA_FUSER || USE_CPU_FUSER
}

size_t nCompiledKernels() { 
  #if USE_CUDA_FUSER || USE_CPU_FUSER
    return fuser::nCompiledKernels(); 
  #else
    return 0;
  #endif // USE_CUDA_FUSER || USE_CPU_FUSER
}

} // namespace jit
} // namespace torch
