#pragma once

#include <sstream>

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>

namespace torch {
namespace lazy {

using TSOpVector = std::vector<torch::jit::Value*>;

class TORCH_API TSComputation : public Computation {
 public:
  TSComputation(const std::shared_ptr<torch::jit::Graph>& graph)
      : graph_(graph), graph_executor_(graph, "") {
    for (torch::jit::Value* input : graph_->inputs()) {
      parameter_names_.push_back(input->debugName());
    }
  }

  int parameters_size() const override {
    return parameter_names_.size();
  }

  const std::vector<Shape>& parameter_shapes() const override {
    throw std::runtime_error(
        "TODO(whc) implement TS computation shapes or change interface");
    return parameter_shapes_;
  }

  const std::vector<std::string>& parameter_names() const override {
    return parameter_names_;
  }

  const Shape& result_shape() const override {
    throw std::runtime_error(
        "TODO(whc) implement TS computation shapes or change interface");
    return result_shape_;
  }

  const std::string to_string() const override {
    std::ostringstream oss;
    oss << *graph_;
    return oss.str();
  }

  std::shared_ptr<torch::jit::Graph> graph() const {
    return graph_;
  }

  torch::jit::GraphExecutor& graph_executor() {
    return graph_executor_;
  }

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::GraphExecutor graph_executor_;
  std::vector<std::string> parameter_names_;
  std::vector<Shape> parameter_shapes_;
  Shape result_shape_;
};

class TORCH_API TSLoweringContext : public LoweringContext {
 public:
  TSLoweringContext(const std::string& name, const BackendDevice device);

  TSLoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<Node*> post_order,
      Util::EmissionMap emit_status);

  size_t AddResult(const Output& output) override {
    return AddResult(GetOutputOp(output));
  }

  void AddParameter(
      const torch::lazy::Output& output,
      size_t index,
      const Shape& shape,
      const std::string& name) override {
    TORCH_INTERNAL_ASSERT(false, "not implemented");
  }

  void Lower(const Node* node);

  ComputationPtr Build() override {
    for (torch::jit::Value* output : root_tuple_) {
      graph_->block()->registerOutput(output);
    }
    return std::shared_ptr<Computation>(new TSComputation(graph_));
  }

  // Retrieves the lowered operation for an output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const Output& output) {
    auto it = emitted_outputs_.find(output);
    if (it == emitted_outputs_.end()) {
      auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
      for (auto node : post_order) {
        Lower(node);
      }
      // At this point the output better be present, otherwise there is an issue
      // with the lowering code.
      it = emitted_outputs_.find(output);
      TORCH_CHECK(
          it != emitted_outputs_.end(),
          "No TS operation emitted for output: ",
          output.ToString());
    }
    return it->second;
  }

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const Output& output, torch::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value* GetParameter(BackendDataPtr data);

  std::shared_ptr<torch::jit::Graph> graph() const {
    return graph_;
  }

 private:
  struct Parameter {
    torch::jit::Value* param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value* op) {
    root_tuple_.push_back(std::move(op));
    return root_tuple_.size() - 1;
  }

  std::shared_ptr<torch::jit::Graph> graph_;
  std::shared_ptr<torch::jit::GraphFunction> function_;
  std::unordered_map<BackendData::Handle, Parameter> parameters_map_;
  std::vector<torch::jit::Value*> root_tuple_;
  OutputMap<torch::jit::Value*> emitted_outputs_;
};

} // namespace lazy
} // namespace torch
