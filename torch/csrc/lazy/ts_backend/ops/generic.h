#pragma once

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

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

  bool Equal(
      OpKind op,
      OpList operands,
      Shape shape,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) const {
    TORCH_INTERNAL_ASSERT(false, "Reusing Generic nodes is unsupported")
    return false;
  }

 private:
  hash_t hash_seed_;
};

inline NodePtr GenericOp(
    OpKind op,
    OpList operands,
    Shape shape,
    size_t num_outputs = 1,
    hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return ReuseOrMakeNode<Generic>(
      torch::lazy::OpKind(ltc_tensor_data),
      op,
      operands,
      std::move(shape),
      num_outputs,
      hash_seed);
}

} // namespace lazy
} // namespace torch
