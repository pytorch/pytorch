#include <torch/csrc/lazy/backend/lowering_context.h>

namespace torch {
namespace lazy {

LoweringContext::LoweringContext(const std::string& name, BackendDevice device)
    : device_(std::move(device)) {}

LoweringContext::LoweringContext(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<torch::lazy::Node*> post_order,
    Util::EmissionMap emit_status)
    : device_(std::move(device)), emit_status_(std::move(emit_status)) {}

const std::vector<BackendDataPtr>& LoweringContext::GetParametersData() const {
  return parameters_;
}

} // namespace lazy
} // namespace torch
