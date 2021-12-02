#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Generic IR Node implementation for nodes which can simply be described by a
// specific torch::lazy::OpKind and a lowering function. IR nodes carrying
// metadata should not be using this class (and have the metadata captured by
// the LowerFn), but they should instead create a dedicated IR node. Doing the
// former would limit IR introspection.
class Generic : public torch::lazy::TsNode {
 public:
  Generic(torch::lazy::OpKind op, torch::lazy::OpList operands,
          torch::lazy::Shape shape, size_t num_outputs = 1,
          torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(torch::lazy::OpKind op, torch::lazy::OpList operands,
          const std::function<torch::lazy::Shape()>& shape_fn,
          size_t num_outputs = 1,
          torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(torch::lazy::OpKind op, torch::lazy::OpList operands,
          size_t num_outputs = 1,
          torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(torch::lazy::OpKind op, torch::lazy::Shape shape, size_t num_outputs,
          torch::lazy::hash_t hash_seed);

 private:
  torch::lazy::hash_t hash_seed_;
};

inline torch::lazy::NodePtr GenericOp(
    torch::lazy::OpKind op, torch::lazy::OpList operands,
    torch::lazy::Shape shape, size_t num_outputs = 1,
    torch::lazy::hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return torch::lazy::MakeNode<Generic>(
      std::move(op), operands, std::move(shape), num_outputs, hash_seed);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
