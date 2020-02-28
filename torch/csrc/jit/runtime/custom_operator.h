#pragma once

#include <torch/csrc/jit/runtime/operator.h>
#include <ATen/core/stack.h>
#include <ATen/core/op_registration/op_registration.h>

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
  RegisterOperators(std::vector<Operator> operators) {
    for (Operator& o : operators) {
      registerOperator(std::move(o));
    }
  }
};

} // namespace jit
} // namespace torch
