#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Generic IR Node implementation for nodes which can simply be described by a
// specific OpKind and a lowering function. IR nodes carrying metadata should
// not be using this class (and have the metadata captured by the LowerFn), but
// they should instead create a dedicated IR node. Doing the former would limit
// IR introspection.
class Generic : public Node {
 public:
  Generic(OpKind op, lazy_tensors::Span<const Value> operands,
          lazy_tensors::Shape shape, size_t num_outputs = 1,
          lazy_tensors::hash_t hash_seed = 0x5a2d296e9);

  Generic(OpKind op, lazy_tensors::Span<const Value> operands,
          const std::function<lazy_tensors::Shape()>& shape_fn,
          size_t num_outputs = 1, lazy_tensors::hash_t hash_seed = 0x5a2d296e9);

  Generic(OpKind op, lazy_tensors::Span<const Value> operands,
          size_t num_outputs = 1, lazy_tensors::hash_t hash_seed = 0x5a2d296e9);

  Generic(OpKind op, lazy_tensors::Shape shape, size_t num_outputs,
          lazy_tensors::hash_t hash_seed);

  NodePtr Clone(OpList operands) const override;

 private:
  lazy_tensors::hash_t hash_seed_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
