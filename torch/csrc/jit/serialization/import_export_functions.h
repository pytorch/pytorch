#pragma once
#include <ATen/core/ivalue.h>

// Functions that are used in both import and export processes

namespace torch::jit {
using c10::IValue;
IValue expect_field(
    c10::ivalue::TupleElements& elements,
    const std::string& expected_name,
    size_t entry);
std::string operator_str(
    const std::string& name,
    const std::string& overloadname);
} // namespace torch::jit
