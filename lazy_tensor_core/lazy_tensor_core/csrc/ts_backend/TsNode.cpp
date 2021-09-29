#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {

TsNode::TsNode(OpKind op, OpList operands, lazy_tensors::Shape shape,
               size_t num_outputs, torch::lazy::hash_t hash_seed)
    : Node(op, operands, shape, num_outputs, hash_seed) {}

TsNode::TsNode(OpKind op, OpList operands,
               const std::function<lazy_tensors::Shape()>& shape_fn,
               size_t num_outputs, torch::lazy::hash_t hash_seed)
    : Node(op, operands, shape_fn, num_outputs, hash_seed) {}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs,
               torch::lazy::hash_t hash_seed)
    : Node(op, operands, num_outputs, hash_seed) {}

TsNode::TsNode(OpKind op, lazy_tensors::Shape shape, size_t num_outputs,
               torch::lazy::hash_t hash_seed)
    : Node(op, shape, num_outputs, hash_seed) {}

}  // namespace ir
}  // namespace torch_lazy_tensors