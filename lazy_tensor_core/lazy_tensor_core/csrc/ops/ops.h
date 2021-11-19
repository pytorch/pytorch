#pragma once

// This header can depend on ops/ and ts_backend/torch::lazy::TsNode.h, as well
// as system/c++, PT,... but not on other lazy tensor core headers.

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <memory>

#include "lazy_tensor_core/csrc/ops/generic.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     torch::lazy::Shape shape) {
  return torch::lazy::MakeNode<Scalar>(value, std::move(shape));
}
inline torch::lazy::NodePtr ScalarOp(const at::Scalar& value,
                                     c10::ScalarType type) {
  return torch::lazy::MakeNode<Scalar>(value, type);
}

inline torch::lazy::NodePtr GenericOp(torch::lazy::OpKind op, torch::lazy::OpList operands,
                         torch::lazy::Shape shape, size_t num_outputs = 1,
                         torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, std::move(shape),
                           num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, torch::lazy::OpList operands,
    const std::function<torch::lazy::Shape()>& shape_fn, size_t num_outputs = 1,
    torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, shape_fn, num_outputs,
                           hash_seed);
}

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, torch::lazy::OpList operands,
    size_t num_outputs = 1,
    torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(std::move(op), operands, num_outputs, hash_seed);
}

inline torch::lazy::NodePtr GenericOp(torch::lazy::OpKind op,
                                      torch::lazy::Shape shape,
                                      size_t num_outputs,
                                      torch::lazy::hash_t hash_seed) {
  return torch::lazy::MakeNode<Generic>(std::move(op), std::move(shape), num_outputs,
                           hash_seed);
}

torch::lazy::NodePtr Pow(const torch::lazy::Value& input, const torch::lazy::Value& exponent);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
