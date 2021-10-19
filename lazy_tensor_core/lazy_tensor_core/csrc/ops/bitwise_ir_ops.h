#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Value has implicit cast to bool, operator overloads would be confusing.
torch::lazy::Value BitwiseOr(const torch::lazy::Value& node1, const torch::lazy::Value& node2);
torch::lazy::Value BitwiseXor(const torch::lazy::Value& node1, const torch::lazy::Value& node2);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
