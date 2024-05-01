#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Simple data structure for containing a type T in nested control blocks
// Should only be used after initial compilation where type checking and
// loads and stores are emitted

template <typename T>
struct MiniEnvironment {
  MiniEnvironment(Block* b, std::shared_ptr<MiniEnvironment> next = nullptr)
      : next(std::move(next)) {}

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<MiniEnvironment<T>> next;

  T findInThisFrame(const std::string& name) {
    auto it = table.find(name);
    if (it != table.end()) {
      return it->second;
    }
    return nullptr;
  }

  T findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  void setVar(const std::string& name, T value) {
    table[name] = value;
  }

  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    result.reserve(table.size());
    for (auto& kv : table) {
      result.push_back(kv.first);
    }
    std::sort(result.begin(), result.end());
    return result;
  }

 private:
  std::unordered_map<std::string, T> table;
};

} // namespace jit
} // namespace torch
