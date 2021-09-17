#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2);
NodePtr operator-(const Value& node1, const Value& node2);
NodePtr operator*(const Value& node1, const Value& node2);
NodePtr operator/(const Value& node1, const Value& node2);

}  // namespace ir
}  // namespace torch_lazy_tensors
