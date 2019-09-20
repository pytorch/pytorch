#pragma once
#include <ATen/core/ivalue.h>
#include <Aten/core/operator_name.h>
#include <torch/csrc/jit/instruction.h>
#include <vector>

namespace torch{
namespace jit{
namespace mobile {
struct Bytecode {
  std::vector<Instruction> instructions_;
  std::vector<c10::OperatorName> op_names_;
  std::vector<c10::IValue> constants_;
  size_t agg_size_; // Aggregated output size.
};
} // namespace mobile
} // namespace torch
} // namespace jit
