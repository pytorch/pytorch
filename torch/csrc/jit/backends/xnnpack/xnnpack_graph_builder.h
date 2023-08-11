// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <torch/torch.h>
#include <xnnpack.h>
#include <unordered_set>
#include <vector>

#include <torch/csrc/jit/backends/xnnpack/serialization/serializer.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNGraph {
 private:
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  // serializer class
  XNNSerializer _serializer;
  // xnn subgraph
  xnn_subgraph_t _subgraph_ptr;
  // Set of all the tensor values throughout the jit graph
  std::unordered_set<torch::jit::Value*> _intermediate_tensors;
  // Set of all the tensor values mapped to the xnnpack ids
  std::unordered_map<torch::jit::Value*, uint32_t> _val_to_ids;
  // Vector containing the torch valued inputs/outputs,
  // must be ordered to preserve the order of input/outputs
  std::vector<torch::jit::Value*> _inputs;
  std::vector<torch::jit::Value*> _outputs;

  // Graph passes for optimizing and tracing torchscript graph
  // Essentially massaging the graph into a digestiable format for
  // xnnpack graph lowering.
  std::shared_ptr<torch::jit::Graph> optimizeAndTraceGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      std::vector<c10::IValue>& example_inputs);

  // Gather all the intermediate tensor values within a graph. This
  // skips through all prim constants. The purpose of this is for defining
  // the tensor values beforehand for the xnnpack subgraph.
  void gatherTensorValues(std::shared_ptr<torch::jit::Graph>& graph);

  // Gathers the tensor values in a give node
  void gatherNodeInputs(torch::jit::Node& node);

  // Helper function to determine if a jit value is a graph input
  bool isGraphInput(torch::jit::Value* val);

  // Helper function to determine if a jit value is a graph output
  bool isGraphOutput(torch::jit::Value* val);

  // Defines all xnnpack nodes for the nodes in the graph
  void defineAllNodes(std::shared_ptr<torch::jit::Graph>& graph);

  // Defines all xnn tensor values used throughout the graph
  void defineAllTensorValues();

  // Makes a pass through the graph and throws if any ops are unsupported
  void checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph);

 public:
  XNNGraph() : _serializer(), _subgraph_ptr(nullptr) {
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

  std::string serializedXNNGraph();

  std::vector<std::vector<long>> getGraphOutputShapes();
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
