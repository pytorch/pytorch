#include "torch/csrc/jit/fuser/cuda/fusion_compiler.h"

#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/ivalue.h" // IValue
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/common/fusion_handle_impl.h"

#include <sstream>
#include <string>
#include <tuple>
#include <cstdlib>

namespace torch { namespace jit { namespace fuser { namespace cuda {
CUDAFusionCompiler& getFusionCompiler() {
  static CUDAFusionCompiler compiler;
  return compiler;
}

std::shared_ptr<FusionHandle> CUDAFusionCompiler::getFusionHandle(
  const KernelSpec& spec
, const int device) {
  std::stringstream ss;
  ss << *(spec.graph()) << "\n";
  std::string key = ss.str();
  auto it = cache_map.find(key);
  if (it == cache_map.end()) {
    std::tie(it, std::ignore) = 
      cache_map.emplace(
        key
      , std::make_shared<FusionHandleImpl>(spec.graph(), device));
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

  auto graph_copy = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph_copy);
  KernelSpec spec{0, graph_copy};
  auto cache = getFusionHandle(spec, device);
  Stack stack = fmap<IValue>(inputs);
  cache->run(stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

} // namespace cuda
} // namespace fuser
} // namespace jit 
} // namespace torch
