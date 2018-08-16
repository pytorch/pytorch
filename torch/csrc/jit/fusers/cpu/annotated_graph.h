#if !(defined _WIN32)
#pragma once

#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/WindowsTorchApiMacro.h"

#include "torch/csrc/jit/ir.h"

#include <vector>

namespace torch { namespace jit { namespace cpufuser {

TORCH_API struct AnnotatedGraph {
  // short-term storage only, so it borrows Graph.
  AnnotatedGraph(Graph& graph, int device)
  : graph(&graph), device(device) {}

  Graph* graph = nullptr;
  int device = kCPUDevice;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
};

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)