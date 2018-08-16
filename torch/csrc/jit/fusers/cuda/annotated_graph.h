#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "torch/csrc/jit/ir.h"

#include "torch/csrc/jit/fusers/cuda/tensor_desc.h"

#include <vector>

namespace torch { namespace jit { namespace cudafuser {

struct AnnotatedGraph {
  // short-term storage only, so it borrows Graph.
  AnnotatedGraph(Graph& graph, int device)
  : graph{&graph}, device{device} {}

  Graph* graph = nullptr;
  int device = kCPUDevice;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
};

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)