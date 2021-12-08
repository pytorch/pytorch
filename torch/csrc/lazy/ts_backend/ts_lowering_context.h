#pragma once

#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/api/include/torch/jit.h>

namespace torch {
namespace lazy {

using TSOpVector = std::vector<torch::jit::Value*>;

class TORCH_API TSNodeLoweringInterface {
  /**
   * This interface is only needed for legacy ops, and can be removed once all
   * ops implement TSNode->lower().
   * */
 public:
  TSNodeLoweringInterface() = default;

  virtual ~TSNodeLoweringInterface() = default;

  virtual bool Lower(const Node* node) = 0;

  static std::unique_ptr<TSNodeLoweringInterface> Create(LoweringContext* loctx);
};

class TORCH_API TSComputation : public Computation {
 public:
  explicit TSComputation(const std::shared_ptr<torch::jit::Graph>& graph)
      : graph_(graph), graph_executor_(graph, "") {
    for (torch::jit::Value* input : graph_->inputs()) {
      parameter_names_.push_back(input->debugName());
    }
  }

  int parameters_size() const override { return parameter_names_.size(); }

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

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

  torch::jit::GraphExecutor& graph_executor() { return graph_executor_; }

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

  TSLoweringContext(const std::string& name, BackendDevice device,
                    c10::ArrayRef<Node*> post_order,
                    Util::EmissionMap emit_status);

  // TODO(whc) replace these when real impl lands;
  // I am just landing the interface in this diff, but MSVC won't allow undefined virtual funcs
  Shape GetResultShape(size_t index) const override { TORCH_INTERNAL_ASSERT(false, "not implemented"); }

  size_t AddResult(const Output& output) override { TORCH_INTERNAL_ASSERT(false, "not implemented"); }

  ComputationPtr Build() override { TORCH_INTERNAL_ASSERT(false, "not implemented"); }

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const Output& output);

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const Output& output, torch::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value* GetParameter(BackendDataPtr data);

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

 private:
  struct Parameter {
    torch::jit::Value* param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value* op);

  std::shared_ptr<torch::jit::Graph> graph_;
  std::unordered_map<BackendData::Handle, Parameter> parameters_map_;
  std::vector<torch::jit::Value*> root_tuple_;
  OutputMap<torch::jit::Value*> emitted_outputs_;
  std::unique_ptr<TSNodeLoweringInterface> lowering_;
};

}  // namespace lazy
}  // namespace torch
