#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/common/tensor_desc.h"

namespace torch { namespace jit { namespace fuser {

struct AnnotatedGraph {
  // short-term storage only, so it borrows Graph.
  AnnotatedGraph(Graph& graph, int device)
  : graph(&graph), device(device) {}
  
  Graph* graph = nullptr; // TODO: this should really be const
  int device = kCPUDevice;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
};

} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
