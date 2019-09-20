#include "mobile_module.h"
#include <torch/csrc/jit/script/jit_exception.h>
#include <torch/csrc/jit/lite_interpreter/lite_interpreter.h>

namespace torch {
namespace jit {
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {

const c10::QualifiedName& Function::qualname() const {
  return name_;
}

const std::string& Function::name() const {
  return name_.name();
}

void Function::append_instruction(OpCode op, int N, int X) {
  bytecode_.instructions_.emplace_back(op, N, X);
}

void Function::append_opname(const std::string& name,
                           const std::string& overload_name) {
  bytecode_.op_names_.emplace_back(name, overload_name);
}

void Function::append_constant(const c10::IValue& constant) {
  bytecode_.constants_.push_back(constant);
}

bool Function::run(Stack& stack) const {
  InterpreterState interp_state(bytecode_);
  return interp_state.run(stack);
}

void CompilationUnit::register_function(std::unique_ptr<Function> fn) {
  methods_.emplace_back(std::move(fn));
}

c10::IValue Module::run_method(const std::string& method_name, Stack& stack) {
  auto m = find_method(method_name);
  stack.insert(stack.begin(), object_);
  m->run(stack);
  return stack.front();
}

Function* Module::find_method(const std::string& basename) const {
  for (auto& fn : cu_->methods()) {
    if (fn->name() == basename) {
      return fn.get();
    }
  }
  return nullptr;
}

} // namespace mobile
} // namespace torch
} // namespace jit
