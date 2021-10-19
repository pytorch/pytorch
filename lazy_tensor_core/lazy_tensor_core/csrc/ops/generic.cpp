#include "lazy_tensor_core/csrc/ops/generic.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Generic::Generic(OpKind op, OpList operands,
                 lazy_tensors::Shape shape, size_t num_outputs,
                 torch::lazy::hash_t hash_seed)
    : TsNode(std::move(op), operands, std::move(shape), num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, OpList operands,
                 const std::function<lazy_tensors::Shape()>& shape_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : TsNode(std::move(op), operands, shape_fn, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, OpList operands,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : TsNode(std::move(op), operands, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, lazy_tensors::Shape shape, size_t num_outputs,
                 torch::lazy::hash_t hash_seed)
    : TsNode(std::move(op), std::move(shape), num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

NodePtr Generic::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Generic>(op(), operands, shape(), num_outputs(), hash_seed_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
