#include "function.h"
#include "interpreter.h"
#include <torch/csrc/jit/instruction.h>
#include <ATen/core/op_registration/op_registration.h>

namespace torch{
namespace jit{
char const * toString(OpCode op);
namespace mobile {
Function::Function(c10::QualifiedName name)
    : name_(name), code_(std::make_shared<Code>()) {}

void Function::append_instruction(OpCode op, int N, int X) {
  TORCH_CHECK(isOpSupportedInMobile(op), toString(op),
              " is not supported in mobile module.");
  code_->instructions_.emplace_back(op, N, X);
}

void Function::append_operator(const std::string& name,
                               const std::string& overload_name) {
  // Keep the original opname in code_
  code_->op_names_.emplace_back(name, overload_name);

  //
  auto opname = code_->op_names_.back();
  // Special treatments for some prim ops.
  if (opname.name == "prim::ListConstruct") {
    // Currently the c10 boxed op registration does not support lambda captures
    // so we cannot register an op to pass N. We register a dummy operator so
    // that the index of operator table is kept. The instruction itself will be
    // handled in interpreter directly.
    opname.name = "aten::dummy";
    opname.overload_name = "";
  }
  // Add "_" prefix to work around the double registration both of jit/generated
  // and here. TODO: remove it when we have separate build for lite interpreter.
  opname.name = "_" + opname.name;
  auto op = c10::Dispatcher::singleton().findSchema(opname);
  TORCH_CHECK(op.has_value(), opname.name, ".", opname.overload_name, " cannot be found.");
  code_->operators_.emplace_back(op);
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
