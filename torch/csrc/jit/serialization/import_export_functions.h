#pragma once

// Functions that are used in both import and export processes
namespace torch {
namespace jit {
void moduleMethodsTuple(
    const Module& module,
    std::vector<c10::IValue>& elements);
IValue expect_field(IValue tup, const std::string& expected_name, size_t entry);
} // namespace jit
} // namespace torch
