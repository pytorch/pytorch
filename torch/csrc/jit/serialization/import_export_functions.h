#pragma once

// Functions that are used in both import and export processes
namespace torch {
namespace jit {
using c10::IValue;
const IValue& expect_field(
    const IValue& tup,
    const std::string& expected_name,
    size_t entry);
std::string operator_str(
    const std::string& name,
    const std::string& overloadname);
} // namespace jit
} // namespace torch
