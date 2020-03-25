#include "function.h"
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/custom_class_detail.h>
#include "interpreter.h"

namespace torch{
namespace jit{

char const * toString(OpCode op);
namespace mobile {
Function::Function(c10::QualifiedName name)
    : name_(name), code_(std::make_shared<Code>()) {}

void Function::append_instruction(OpCode op, int X, int N) {
  TORCH_CHECK(isOpSupportedInMobile(op), toString(op),
              " is not supported in mobile module.");
  code_->instructions_.emplace_back(op, X, N);
}

bool Function::append_operator(const std::string& name,
                               const std::string& overload_name) {
  // Keep the original opname in code_
  code_->op_names_.emplace_back(name, overload_name);
  auto opname = code_->op_names_.back();
  // Add "_" prefix to work around the double registration both of jit/generated
  // and here. TODO: remove it when we have separate build for lite interpreter.
  if (opname.name != "aten::Int") {
    opname.name = "_" + opname.name;
  }
  auto op = c10::Dispatcher::singleton().findSchema(opname);
  if (!op.has_value()) {
    return false;
  }
  // TODO: operator.h now does not depend on Node* so we can also look up operators from
  // that registry for use in mobile as a way to share implementations.
  auto fn = [op](Stack& stack) {
    c10::Dispatcher::singleton().callBoxed(*op, &stack);
  };
  code_->operators_.emplace_back(fn);
  return true;
}

void Function::append_constant(const c10::IValue& constant) {
  code_->constants_.push_back(constant);
}

void Function::append_type(const at::TypePtr& type) {
  code_->types_.push_back(type);
}

void Function::set_register_size(size_t size) {
  code_->register_size_ = size;
}

bool Function::run(Stack& stack) const {
  InterpreterState interp_state(code_);
  return interp_state.run(stack);
}
} // namespace mobile
} // namespace torch
} // namespace jit
