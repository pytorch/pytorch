#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)

#include "torch/csrc/jit/fusers/cuda/cuda_fuser_interface.h"
#include "torch/csrc/jit/fusers/cuda/annotated_graph.h"
#include "torch/csrc/jit/fusers/cuda/cuda_fuser.h"

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

std::shared_ptr<CompiledFusionFunction> getCUDAFusionFunction(Node* fusion_group) {
  auto& graph = *fusion_group->g(attr::Subgraph);
  cudafuser::AnnotatedGraph agraph(graph, fusion_group->i(attr::device));
  
  for (auto& input : graph.inputs()) {
    auto t = input->type()->expect<TensorType>();
    agraph.input_desc.emplace_back(t);
  }

  for (auto& output : graph.outputs()) {
    auto t = output->type()->expect<TensorType>();
    agraph.output_desc.emplace_back(t);
  }

  return cudafuser::getCUDAFuser().getOrCompile(agraph);
}

void debugCUDALaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs) {
    cudafuser::getCUDAFuser().debugLaunchGraph(graph, device, inputs, outputs);
}

} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
