#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {

torch::lazy::NodePtr operator+(const torch::lazy::Value& node1, const torch::lazy::Value& node2);
torch::lazy::NodePtr operator-(const torch::lazy::Value& node1, const torch::lazy::Value& node2);
torch::lazy::NodePtr operator*(const torch::lazy::Value& node1, const torch::lazy::Value& node2);
torch::lazy::NodePtr operator/(const torch::lazy::Value& node1, const torch::lazy::Value& node2);

}  // namespace ir
}  // namespace torch_lazy_tensors
