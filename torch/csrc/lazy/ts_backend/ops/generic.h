#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <torch/csrc/lazy/core/ir_builder.h>

namespace torch::lazy {

// Generic IR Node implementation for nodes which can simply be described by a
// specific OpKind and a lowering function. IR nodes carrying
// metadata should not be using this class TORCH_API (and have the metadata
// captured by the LowerFn), but they should instead create a dedicated IR node.
// Doing the former would limit IR introspection.
class TORCH_API Generic : public TsNode {
 public:
  Generic(
      OpKind op,
      OpList operands,
      Shape shape,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(
      OpKind op,
      OpList operands,
      const std::function<Shape()>& shape_fn,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(
      OpKind op,
      OpList operands,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed);

 private:
  hash_t hash_seed_;
};

inline NodePtr GenericOp(
    OpKind op,
    OpList operands,
    Shape shape,
    size_t num_outputs = 1,
    hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return MakeNode<Generic>(
      op, operands, std::move(shape), num_outputs, hash_seed);
}

} // namespace torch::lazy
