#include <c10/core/ScalarType.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

TSLoweringContext::TSLoweringContext(
    const std::string& name,
    BackendDevice device)
    : torch::lazy::LoweringContext(name, device),
      graph_(std::make_shared<torch::jit::Graph>()) {
  lowering_ = TSNodeLoweringInterface::Create(this);
}

TSLoweringContext::TSLoweringContext(
    const std::string& name,
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

void TSLoweringContext::AssignOutputOp(
    const Output& output,
    torch::jit::Value* op) {
  auto ts_node = NodeCast<TsNode>(output.node, output.node->op());
  if (!ts_node->getPythonStacktrace().empty()) {
    op->node()->s_(c10::Symbol::attr("source"), ts_node->getPythonStacktrace());
  }
  emitted_outputs_[output] = op;
}

torch::jit::Value* TSLoweringContext::GetParameter(BackendDataPtr data) {
  const auto ts_data =
      std::static_pointer_cast<TSData>(data);
  BackendData::Handle handle = ts_data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    torch::jit::Value* param =
        graph_->addInput(c10::str("p", parameters_.size()));
    if (ts_data->scalar.has_value()) {
      auto scalarType = ts_data->scalar.value().type();
      if (isFloatingType(scalarType)) {
        param->setType(c10::FloatType::get());
      } else if (isIntegralType(scalarType, /*includeBool=*/true)) {
        param->setType(c10::IntType::get());
      } else {
        TORCH_CHECK(
            false, "Unhandled scalar type: ", c10::toString(scalarType));
      }
    }
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(ts_data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

} // namespace lazy
} // namespace torch
