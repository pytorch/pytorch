#include "torch/csrc/autograd/function_port.h"

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

#include <cstdint>
#include <functional>
#include <memory>

namespace torch {
namespace autograd {
FunctionPort FunctionPort::for_gradient(const Variable& variable) {
  if (variable.defined()) {
    if (variable.grad_fn() != nullptr) {
      return FunctionPort(variable.grad_fn(), variable.output_nr());
    } else if (variable.requires_grad()) {
      return FunctionPort(variable.grad_accumulator());
    }
  }
  return FunctionPort();
}

FunctionPort::FunctionPort(
    const std::shared_ptr<Function>& function_,
    uint32_t port_)
    : function(function_), port(port_) {}

FunctionPort::~FunctionPort() = default;

bool FunctionPort::operator==(const FunctionPort& other) const noexcept {
  return this->function == other.function && this->port == other.port;
}

bool FunctionPort::operator!=(const FunctionPort& other) const noexcept {
  return !(*this == other);
}
} // namespace autograd
} // namespace torch
