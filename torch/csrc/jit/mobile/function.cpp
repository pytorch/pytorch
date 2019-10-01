#include "function.h"
#include "interpreter.h"
#include <torch/csrc/jit/instruction.h>
#include <ATen/core/op_registration/op_registration.h>

namespace torch{
namespace jit{

namespace {
template <typename dtype> // int64_t, bool, double
void listConstruct(int num_inputs, Stack& stack) {
  auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
  c10::List<dtype> vals =
    c10::impl::toList(fmap(inputs, [](const IValue& v) { return v.to<dtype>(); }));
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}

void tensorListConstruct(int num_inputs, Stack& stack) {
  const size_t stack_size = stack.size();
  c10::List<at::Tensor> vals;
  vals.reserve(num_inputs);
  for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
    vals.emplace_back(std::move(stack[i]).toTensor());
  }
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}
}

char const * toString(OpCode op);
namespace mobile {
Function::Function(c10::QualifiedName name)
    : name_(name), code_(std::make_shared<Code>()) {}

void Function::append_instruction(OpCode op, int X, int N) {
  TORCH_CHECK(isOpSupportedInMobile(op), toString(op),
              " is not supported in mobile module.");
  code_->instructions_.emplace_back(op, X, N);
}

void Function::append_operator(const std::string& name,
                               const std::string& overload_name) {
  // Keep the original opname in code_
  code_->op_names_.emplace_back(name, overload_name);
  auto opname = code_->op_names_.back();
  // Add "_" prefix to work around the double registration both of jit/generated
  // and here. TODO: remove it when we have separate build for lite interpreter.
  opname.name = "_" + opname.name;
  auto op = c10::Dispatcher::singleton().findSchema(opname);
  TORCH_CHECK(op.has_value(), opname.name, ".", opname.overload_name, " cannot be found.");
  code_->operators_.emplace_back(op);
}

void Function::build_vararg_operator_table() {
  for (auto& ins : code_->instructions_) {
    if (ins.op == OPN) {
      auto opname = code_->op_names_[ins.X];
      if (opname.name == "prim::ListConstruct") {
        if (opname.overload_name == "int") {
          code_->vararg_operators_.emplace_back(listConstruct<int64_t>);
        } else if (opname.overload_name == "float") {
          code_->vararg_operators_.emplace_back(listConstruct<double>);
        } else if (opname.overload_name == "bool") {
          code_->vararg_operators_.emplace_back(listConstruct<bool>);
        } else if (opname.overload_name == "tensor") {
          code_->vararg_operators_.emplace_back(tensorListConstruct);
        } else {
          AT_ERROR("Type of ListConstruct is not supported.");
        }
      } else {
        AT_ERROR("OPN operator ", opname.name, " is not supported.");
      }
      ins.X = code_->vararg_operators_.size() - 1;
    }
  }
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
