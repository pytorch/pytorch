#include "lazy_tensor_core/csrc/ops/generic.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Generic::Generic(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 torch::lazy::Shape shape, size_t num_outputs,
                 torch::lazy::hash_t hash_seed)
    : torch::lazy::TsNode(std::move(op), operands, {std::move(shape)},
                          num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 const std::function<torch::lazy::Shape()>& shape_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : torch::lazy::TsNode(std::move(op), operands, shape_fn, num_outputs,
                          hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(torch::lazy::OpKind op, torch::lazy::OpList operands,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : torch::lazy::TsNode(std::move(op), operands, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

Generic::Generic(torch::lazy::OpKind op, torch::lazy::Shape shape,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : torch::lazy::TsNode(std::move(op), std::move(shape), num_outputs,
                          hash_seed),
      hash_seed_(hash_seed) {}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
