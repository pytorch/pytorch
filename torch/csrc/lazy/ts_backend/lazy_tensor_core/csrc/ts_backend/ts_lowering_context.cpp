#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace ts_backend {

TSLoweringContext::TSLoweringContext(const std::string& name, Device device)
    : ir::LoweringContext(name, device),
      graph_(std::make_shared<torch::jit::Graph>()) {
  lowering_ = NodeLowering::Create(this);
}

TSLoweringContext::TSLoweringContext(
    const std::string& name, Device device,
    lazy_tensors::Span<const ir::Node* const> post_order,
    ir::Util::EmissionMap emit_status)
    : ir::LoweringContext(name, device, post_order, emit_status),
      graph_(std::make_shared<torch::jit::Graph>()) {
  lowering_ = NodeLowering::Create(this);
  for (auto node : post_order) {
    bool ok = lowering_->Lower(node);
    LTC_CHECK(ok) << "Failed to lower: " << *node;
  }
}

lazy_tensors::Shape TSLoweringContext::GetResultShape(size_t index) const {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

size_t TSLoweringContext::AddResult(const ir::Output& output) {
  return AddResult(GetOutputOp(output));
}

lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>>
TSLoweringContext::Build() {
  for (torch::jit::Value* output : root_tuple_) {
    graph_->block()->registerOutput(output);
  }
  return std::shared_ptr<lazy_tensors::GenericComputation>(
      new GenericComputationTS(graph_));
}

torch::jit::Value* TSLoweringContext::GetOutputOp(const ir::Output& output) {
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = ir::Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      bool ok = lowering_->Lower(node);
      LTC_CHECK(ok) << "Failed to lower: " << *node;
    }
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    LTC_CHECK(it != emitted_outputs_.end())
        << "No TS operation emitted for output: " << output;
  }
  return it->second;
}

void TSLoweringContext::AssignOutputOp(const ir::Output& output,
                                       torch::jit::Value* op) {
  emitted_outputs_[output] = std::move(op);
}

torch::jit::Value* TSLoweringContext::GetParameter(
    const std::shared_ptr<lazy_tensors::client::Data>& data) {
  lazy_tensors::client::Data::OpaqueHandle handle = data->GetOpaqueHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    torch::jit::Value* param =
        graph_->addInput(lazy_tensors::StrCat("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

size_t TSLoweringContext::AddResult(torch::jit::Value* op) {
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

}  // namespace ts_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
