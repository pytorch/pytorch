#pragma once

#include <torch/jit.h>

#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace compiler {

using TSOpVector = std::vector<torch::jit::Value*>;

class NodeLowering;

namespace ts_backend {

class GenericComputationTS : public lazy_tensors::GenericComputation {
 public:
  GenericComputationTS(std::shared_ptr<torch::jit::Graph> graph)
      : graph_(std::move(graph)) {}

  lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GetProgramShape()
      const override {
    std::vector<std::string> parameter_names;
    for (torch::jit::Value* input : graph_->inputs()) {
      parameter_names.push_back(input->debugName());
    }
    // NB: The return type is only used by certain backends to assing a physical
    // layout. This backend doesn't use it for anything, so it's ok to leave it
    // empty.
    std::vector<lazy_tensors::Shape> parameters(parameter_names.size());
    return lazy_tensors::ProgramShape(parameters, parameter_names,
                                      lazy_tensors::Shape());
  }

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
};

class TSLoweringContext : public ir::LoweringContext {
 public:
  TSLoweringContext(const std::string& name, Device device);

  TSLoweringContext(const std::string& name, Device device,
                    lazy_tensors::Span<const ir::Node* const> post_order,
                    ir::Util::EmissionMap emit_status);

  lazy_tensors::Shape GetResultShape(size_t index) const override;

  size_t AddResult(const ir::Output& output) override;

  lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>>
  Build() override;

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const ir::Output& output);

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const ir::Output& output, torch::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value* GetParameter(
      const std::shared_ptr<lazy_tensors::client::Data>& data);

  std::shared_ptr<torch::jit::Graph> graph() const { return graph_; }

 private:
  struct Parameter {
    torch::jit::Value* param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value* op);

  std::shared_ptr<torch::jit::Graph> graph_;
  std::unordered_map<lazy_tensors::client::Data::OpaqueHandle, Parameter>
      parameters_map_;
  std::vector<torch::jit::Value*> root_tuple_;
  ir::OutputMap<torch::jit::Value*> emitted_outputs_;
  std::unique_ptr<NodeLowering> lowering_;
};

}  // namespace ts_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
