// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <torch/torch.h>
#include <xnnpack.h>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNGraph {
 private:
  // xnn_subgraph
  xnn_subgraph_t _subgraph_ptr;

  // Graph passes for optimizing and tracing torchscript graph
  // Essentially massaging the graph into a digestiable format for
  // xnnpack graph lowering.
  std::shared_ptr<torch::jit::Graph> optimizeAndTraceGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      std::vector<c10::IValue>& example_inputs);

  // Makes a pass through the graph and throws if any ops are unsupported
  void checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph);

 public:
  XNNGraph() : _subgraph_ptr(nullptr) {
    xnn_status status = xnn_initialize(/*allocator =*/nullptr);
    TORCH_CHECK(xnn_status_success == status, "Failed to initialize xnnpack");
  }

  ~XNNGraph() {
    xnn_deinitialize();
    if (_subgraph_ptr != nullptr) {
      xnn_delete_subgraph(_subgraph_ptr);
    }
  }

  void buildXNNGraph(
      std::shared_ptr<torch::jit::Graph>& graph,
      std::vector<c10::IValue> example_inputs);

  void runGraphOnInputs(
      std::vector<at::Tensor> tensor_inputs,
      std::vector<at::Tensor> tensor_outputs);
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
