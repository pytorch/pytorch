#pragma once

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

/// Registration class for new operators. Effectively calls
/// `torch::jit::registerOperator` for every supplied operator, but allows doing
/// so in the global scope when a `RegisterOperators` object is assigned to a
/// static variable.
/// Note: This is *not* the custom operator API. If you want to register custom
/// operators, take a look at torch::RegisterOperators.
struct TORCH_API RegisterOperators {
  RegisterOperators() = default;

  /// Registers a vector of already created `Operator`s.
  /// The operator element is now optional to filter null ops. It's backward
  /// compatible and works for selective operator registration.
  RegisterOperators(std::vector<c10::optional<Operator>> operators) {
    for (c10::optional<Operator>& o : operators) {
      if (o) {
        registerOperator(std::move(o.value()));
      }
    }
  }
};

} // namespace jit
} // namespace torch
