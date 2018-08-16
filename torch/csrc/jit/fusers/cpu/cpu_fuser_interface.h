#if !(defined _WIN32)
#pragma once

#include "torch/csrc/jit/fusers/fuser_interface.h"

#include <memory>

namespace torch { namespace jit { 

inline std::shared_ptr<FusionFunction> getCPUFusionFunction(Node* fusion_group) {
  auto& graph = *fusion_group->g(attr::Subgraph);
  cpufuser::AnnotatedGraph agraph(graph, fusion_group->i(attr::device));
  
  for (auto& input : graph.inputs()) {
    auto t = input->type()->expect<TensorType>();
    agraph.input_desc.emplace_back(t);
  }

  for (auto& output : graph.outputs()) {
    auto t = output->type()->expect<TensorType>();
    agraph.output_desc.emplace_back(t);
  }

  return cpufuser::getCompiler().getOrCompile(agraph);
}

inline bool canCompileOnCPU() {
  return cpufuser::getCompiler().canCompileOnCPU();
}

} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)