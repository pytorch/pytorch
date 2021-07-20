#pragma once

// Functions that are used in both import and export processes
namespace torch {
namespace jit {
IValue expect_field(IValue tup, const std::string& expected_name, size_t entry);
std::string operator_str(
    const std::string& name,
    const std::string& overloadname);

IValue Tup(std::vector<IValue> ivalues) {
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

IValue Table(const std::vector<std::pair<std::string, IValue>>& entries) {
  std::vector<IValue> ivalue_entries;
  ivalue_entries.reserve(entries.size());
  for (const auto& e : entries) {
    ivalue_entries.push_back(Tup({e.first, e.second}));
  }
  return Tup(std::move(ivalue_entries));
}

} // namespace jit
} // namespace torch
