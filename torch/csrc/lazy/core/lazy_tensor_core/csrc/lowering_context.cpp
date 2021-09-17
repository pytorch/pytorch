#include "lazy_tensor_core/csrc/lowering_context.h"

#include <sstream>
#include <stdexcept>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/python_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/sys_util.h"

namespace torch_lazy_tensors {
namespace ir {

LoweringContext::LoweringContext(const std::string& name, Device device)
    : device_(std::move(device)) {}

LoweringContext::LoweringContext(
    const std::string& name, Device device,
    lazy_tensors::Span<const Node* const> post_order,
    Util::EmissionMap emit_status)
    : device_(std::move(device)), emit_status_(std::move(emit_status)) {}

const std::vector<lazy_tensors::ComputationClient::DataPtr>&
LoweringContext::GetParametersData() const {
  return parameters_;
}

void LoweringContext::LowerNodeToResult(const Node* node) {
  LTC_LOG(FATAL) << "Not implemented.";
}

void LoweringContext::AddParameter(const Output& output, size_t index,
                                   const lazy_tensors::Shape& shape,
                                   const std::string& name) {
  LTC_LOG(FATAL) << "Not implemented.";
}

void LoweringContext::SetUpAlias(const lazy_tensors::ShapeIndex& output_index,
                                 lazy_tensors::int64 param_number,
                                 const lazy_tensors::ShapeIndex& param_index) {}

}  // namespace ir
}  // namespace torch_lazy_tensors
