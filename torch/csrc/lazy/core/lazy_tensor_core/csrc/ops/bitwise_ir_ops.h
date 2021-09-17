#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Value has implicit cast to bool, operator overloads would be confusing.
Value BitwiseAnd(const Value& node1, const Value& node2);
Value BitwiseOr(const Value& node1, const Value& node2);
Value BitwiseXor(const Value& node1, const Value& node2);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
