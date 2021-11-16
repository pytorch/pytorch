#include "lazy_tensor_core/csrc/lowering_context.h"

#include <sstream>
#include <stdexcept>

#include <c10/util/Logging.h>

namespace torch_lazy_tensors {
namespace ir {

LoweringContext::LoweringContext(const std::string& name, torch::lazy::BackendDevice device)
    : device_(std::move(device)) {}

LoweringContext::LoweringContext(const std::string& name,
                                 torch::lazy::BackendDevice device,
                                 c10::ArrayRef<torch::lazy::Node*> post_order,
                                 torch::lazy::Util::EmissionMap emit_status)
    : device_(std::move(device)), emit_status_(std::move(emit_status)) {}

const std::vector<torch::lazy::BackendDataPtr>&
LoweringContext::GetParametersData() const {
  return parameters_;
}

void LoweringContext::AddParameter(const torch::lazy::Output& output, size_t index,
                                   const torch::lazy::Shape& shape,
                                   const std::string& name) {
  LOG(FATAL) << "Not implemented.";
}

}  // namespace ir
}  // namespace torch_lazy_tensors
