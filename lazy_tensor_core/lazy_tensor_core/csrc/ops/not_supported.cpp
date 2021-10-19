#include "lazy_tensor_core/csrc/ops/not_supported.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NotSupported::NotSupported(std::string description, lazy_tensors::Shape shape)
    : TsNode(ltc_not_supported, std::move(shape), /*num_outputs=*/1,
           torch::lazy::MHash(description)),
      description_(std::move(description)) {}

NodePtr NotSupported::Clone(OpList operands) const {
  return torch::lazy::MakeNode<NotSupported>(description_, shape());
}

std::string NotSupported::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", description=" << description_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
