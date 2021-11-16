#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch {
namespace lazy {

TSLoweringContext::TSLoweringContext(const std::string& name,
                                     BackendDevice device)
    : torch::lazy::LoweringContext(name, device),
      graph_(std::make_shared<torch::jit::Graph>()) {
  lowering_ = TSNodeLoweringInterface::Create(this);
}

TSLoweringContext::TSLoweringContext(const std::string& name,
                                     BackendDevice device,
                                     c10::ArrayRef<Node*> post_order,
                                     Util::EmissionMap emit_status)
    : torch::lazy::LoweringContext(name, device, post_order, emit_status),
      graph_(std::make_shared<torch::jit::Graph>()) {
  lowering_ = TSNodeLoweringInterface::Create(this);
  for (auto node : post_order) {
    bool ok = lowering_->Lower(node);
    CHECK(ok) << "Failed to lower: " << *node;
  }
}

void TSLoweringContext::AssignOutputOp(const Output& output,
                                       torch::jit::Value* op) {
  emitted_outputs_[output] = std::move(op);
}

torch::jit::Value* TSLoweringContext::GetParameter(BackendDataPtr data) {
  BackendData::Handle handle = data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    torch::jit::Value* param =
        graph_->addInput(c10::str("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

}  // namespace lazy
}  // namespace torch
