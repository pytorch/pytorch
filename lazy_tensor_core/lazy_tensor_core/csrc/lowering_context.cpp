#include "lazy_tensor_core/csrc/lowering_context.h"

#include <sstream>
#include <stdexcept>


namespace torch_lazy_tensors {
namespace ir {

LoweringContext::LoweringContext(const std::string& name, Device device)
    : device_(std::move(device)) {}

LoweringContext::LoweringContext(const std::string& name, Device device,
                                 c10::ArrayRef<torch::lazy::Node*> post_order,
                                 Util::EmissionMap emit_status)
    : device_(std::move(device)), emit_status_(std::move(emit_status)) {}

const std::vector<compiler::DataPtr>&
LoweringContext::GetParametersData() const {
  return parameters_;
}

void LoweringContext::LowerNodeToResult(const torch::lazy::Node* node) {
  LOG(FATAL) << "Not implemented.";
}

void LoweringContext::AddParameter(const torch::lazy::Output& output, size_t index,
                                   const lazy_tensors::Shape& shape,
                                   const std::string& name) {
  LOG(FATAL) << "Not implemented.";
}

}  // namespace ir
}  // namespace torch_lazy_tensors
