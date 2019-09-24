#include "function.h"
#include "interpreter.h"

namespace torch{
namespace jit{
namespace mobile {
Function::Function(c10::QualifiedName name)
    : name_(name), code_(std::make_shared<Code>()) {}

void Function::append_instruction(OpCode op, int N, int X) {
  code_->instructions_.emplace_back(op, N, X);
}

void Function::append_opname(const std::string& name,
                             const std::string& overload_name) {
  code_->op_names_.emplace_back(name, overload_name);
}

void Function::append_constant(const c10::IValue& constant) {
  code_->constants_.push_back(constant);
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
