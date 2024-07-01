#pragma once

#include <torch/csrc/lazy/core/ir.h>

namespace torch {
namespace lazy {

TORCH_API NodePtr operator+(const Value& node1, const Value& node2);
TORCH_API NodePtr operator-(const Value& node1, const Value& node2);
TORCH_API NodePtr operator*(const Value& node1, const Value& node2);
TORCH_API NodePtr operator/(const Value& node1, const Value& node2);

} // namespace lazy
} // namespace torch
