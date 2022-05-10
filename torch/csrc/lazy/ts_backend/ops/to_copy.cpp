#include <torch/csrc/lazy/ts_backend/ops/to_copy.h>

namespace torch {
namespace lazy {

const OpKind ToCopy::class_op_kind(at::aten::_to_copy);

}  // namespace lazy
}  // namespace torch
