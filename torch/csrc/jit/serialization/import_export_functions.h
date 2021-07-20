#pragma once

// Functions that are used in both import and export processes
namespace torch {
namespace jit {
IValue expect_field(IValue tup, const std::string& expected_name, size_t entry);
std::string operator_str(
    const std::string& name,
    const std::string& overloadname);

IValue Tup(std::vector<IValue> ivalues);
IValue Table(const std::vector<std::pair<std::string, IValue>>& entries);

} // namespace jit
} // namespace torch
