#pragma once

#include <torch/jit.h>

#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"
#include "torch/csrc/jit/runtime/graph_executor.h"
namespace torch_lazy_tensors {
namespace compiler {

using TSOpVector = std::vector<torch::jit::Value*>;

class TSNodeLoweringInterface {
  /**
   * This interface is only needed for legacy ops, and can be removed once all
   * ops implement TSNode->lower().
   * */
 public:
  TSNodeLoweringInterface() = default;

  virtual ~TSNodeLoweringInterface() = default;

  virtual bool Lower(const torch::lazy::Node* node) = 0;

  static std::unique_ptr<TSNodeLoweringInterface> Create(
      ir::LoweringContext* loctx);
};

namespace ts_backend {

class TSComputation : public Computation {
 public:
  TSComputation(std::shared_ptr<torch::jit::Graph> graph)
      : graph_(graph), graph_executor_(std::move(graph), "") {
    for (torch::jit::Value* input : graph_->inputs()) {
      parameter_names_.push_back(input->debugName());
    }
  }

  int parameters_size() const override { return parameter_names_.size(); }

  const std::vector<torch::lazy::Shape>& parameter_shapes() const override {
    throw std::runtime_error(
        "TODO(whc) implement TS computation shapes or change interface");
    return parameter_shapes_;
  }

  const std::vector<std::string>& parameter_names() const override {
    return parameter_names_;
  }

  const torch::lazy::Shape& result_shape() const override {
    throw std::runtime_error(
        "TODO(whc) implement TS computation shapes or change interface");
    return result_shape_;
  }

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

  torch::jit::GraphExecutor& graph_executor() { return graph_executor_; }

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::GraphExecutor graph_executor_;
  std::vector<std::string> parameter_names_;
  std::vector<torch::lazy::Shape> parameter_shapes_;
  torch::lazy::Shape result_shape_;
};

class TSLoweringContext : public ir::LoweringContext {
 public:
  TSLoweringContext(const std::string& name, torch::lazy::BackendDevice device);

  TSLoweringContext(const std::string& name, torch::lazy::BackendDevice device,
                    c10::ArrayRef<torch::lazy::Node*> post_order,
                    torch::lazy::Util::EmissionMap emit_status);

  torch::lazy::Shape GetResultShape(size_t index) const override;

  size_t AddResult(const torch::lazy::Output& output) override;

  ComputationPtr Build() override;

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const torch::lazy::Output& output);

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const torch::lazy::Output& output, torch::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value* GetParameter(torch::lazy::BackendDataPtr data);

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

 private:
  struct Parameter {
    torch::jit::Value* param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value* op);

  std::shared_ptr<torch::jit::Graph> graph_;
  std::unordered_map<torch::lazy::BackendData::Handle, Parameter>
      parameters_map_;
  std::vector<torch::jit::Value*> root_tuple_;
  torch::lazy::OutputMap<torch::jit::Value*> emitted_outputs_;
  std::unique_ptr<TSNodeLoweringInterface> lowering_;
};

}  // namespace ts_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
