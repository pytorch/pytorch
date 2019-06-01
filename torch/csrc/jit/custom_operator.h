#pragma once

#include <torch/csrc/jit/operator.h>
#include <ATen/core/stack.h>
#include <ATen/core/op_registration/op_registration.h>

namespace torch {
namespace jit {

/// Registration class for new operators. Effectively calls
/// `torch::jit::registerOperator` for every supplied operator, but allows doing
/// so in the global scope when a `RegisterOperators` object is assigned to a
/// static variable. Also handles registration of user-defined, "custom"
/// operators.
struct TORCH_API RegisterOperators {
  RegisterOperators() = default;

  /// Registers a vector of already created `Operator`s.
  RegisterOperators(std::vector<Operator> operators) {
    for (Operator& o : operators) {
      registerOperator(std::move(o));
    }
  }

  /// Calls `op(...)` with the given operator name and implementation.
  template <typename Implementation>
  RegisterOperators(const std::string& name, Implementation&& implementation) {
    op(name, std::forward<Implementation>(implementation));
  }

  template <typename Implementation>
  RegisterOperators& op(
      const std::string& name,
      Implementation&& implementation) {
    registrars_.emplace_back(std::make_shared<c10::RegisterOperators>(name, std::forward<Implementation>(implementation)));

    return *this;
  }

private:
  // A c10::RegisterOperators instance is not copyable, so to make
  // torch::jit::RegisterOperators copyable, we use shared_ptrs.
  // We need to keep the c10::RegisterOperators instances around
  // because this is an RAII pattern. In the destructor, the registered
  // ops get de-registered.
  std::vector<std::shared_ptr<c10::RegisterOperators>> registrars_;
};

} // namespace jit

using RegisterOperators = c10::RegisterOperators;

} // namespace torch
