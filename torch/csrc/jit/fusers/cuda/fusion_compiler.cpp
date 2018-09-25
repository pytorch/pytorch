#include "torch/csrc/jit/fusers/cuda/fusion_compiler.h"

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_handle_impl.h"

#include "torch/csrc/jit/passes/shape_analysis.h" // EraseShapeInformation
#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/ivalue.h" // IValue

#include "torch/csrc/jit/assertions.h"

#include <sstream>
#include <string>
#include <tuple>
#include <cstdlib>

namespace torch { namespace jit { namespace cudafuser {
CUDAFusionCompiler& getFusionCompiler() {
  static CUDAFusionCompiler compiler;
  return compiler;
}

std::shared_ptr<FusionHandle> CUDAFusionCompiler::getFusionHandle(
  Node* fusion_group) {
  // verifies on GPU
  const auto device = fusion_group->i(attr::device);
  JIT_ASSERT(device != kCPUDevice);

  auto graph = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);
  std::stringstream key;
  key << "device " << device << "\n";
  key << *graph << "\n";
  std::string key_ = key.str();
  auto it = cache_map.find(key_);
  if (it == cache_map.end()) {
    std::tie(it, std::ignore) = 
      cache_map.emplace(
        key_
      , std::make_shared<FusionHandleImpl>(graph, device));
  }

  return it->second;
}

std::vector<at::Tensor> CUDAFusionCompiler::debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group = 
    wrapper_graph->insertNode(wrapper_graph->createFusionGroup(device));
  fusion_group->g_(attr::Subgraph, graph.copy());
  
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  auto cache = getFusionHandle(fusion_group);
  Stack stack = fmap<IValue>(inputs);
  cache->run(stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

} // namespace cudafuser
} // namespace jit 
} // namespace torch
