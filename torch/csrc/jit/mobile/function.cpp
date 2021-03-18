#include <torch/csrc/jit/mobile/function.h>

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/custom_class_detail.h>

namespace torch {
namespace jit {

char const* toString(OpCode op);
namespace mobile {
Function::Function(c10::QualifiedName name)
    : name_(std::move(name)), code_(std::make_shared<Code>()) {}

const c10::QualifiedName& Function::qualname() const {
  return name_;
}

const std::string& Function::name() const {
  return name_.name();
}

void Function::append_instruction(OpCode op, int X, int N, int64_t dbg_handle) {
  TORCH_CHECK(
      isOpSupportedInMobile(op),
      toString(op),
      " is not supported in mobile module.");
  code_->instructions_.emplace_back(op, X, N);
  code_->debug_handles_.emplace_back(dbg_handle);
}

bool Function::append_operator(
    const std::string& name,
    const std::string& overload_name,
    int64_t model_version) {
  // Keep the original opname in code_
  code_->op_names_.emplace_back(name, overload_name);
  auto opname = code_->op_names_.back();

  const auto& opname_c10 = opname;
  std::function<void(Stack&)> fn;

  auto jit_op = findOperatorFor(opname);
  if (jit_op) {
    fn = [jit_op](Stack& stack) { jit_op->getOperation()(&stack); };
  } else {
    auto op = c10::Dispatcher::singleton().findSchema(opname_c10);
    if (op.has_value()) {
      fn = [op](Stack& stack) { op->callBoxed(&stack); };
    } else {
      return false;
    }
  }

  if (model_version == 0x3LL &&
      opname == c10::OperatorName("aten::_convolution", "")) {
    // Since byte-code versions 0x4L, convolution has an additional
    // default-value argument (allow_tf32=True, see
    // https://github.com/pytorch/pytorch/pull/40737). This wrapper handles
    // backward compatibility with models of byte-code version <= 0x3L, where
    // this bool argument does not yet exist.
    fn = [fn](Stack& stack) {
      stack.push_back(true);
      fn(stack);
    };
  }

  code_->operators_.emplace_back(fn);
  return true;
}

void Function::set_module_debug_info_list_size(size_t size) {
  pc_to_module_debug_info_.resize(size);
  for (size_t i = 0; i < size; ++i) {
    pc_to_module_debug_info_[i] = "<no module info>";
  }
}

void Function::set_module_info(const std::string& module_info, size_t pc) {
  TORCH_CHECK(
      pc < pc_to_module_debug_info_.size(),
      "Module debug info index out of boundary.");
  pc_to_module_debug_info_[pc] = module_info;
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

std::string Function::get_module_debug_info(size_t pc) const {
  TORCH_CHECK(
      pc < pc_to_module_debug_info_.size(),
      "Module debug info index out of boundary.");
  return pc_to_module_debug_info_[pc];
}

void Function::setSchema(c10::FunctionSchema schema) {
  schema_ = std::move(schema);
}

const at::optional<c10::FunctionSchema>& Function::getSchema() const {
  return schema_;
}

bool Function::run(Stack& stack) const {
  const auto& schema = getSchema();
  if (schema) { // if we have a schema then resolve optional args if any
    schema->checkAndNormalizeInputs(
        stack, std::unordered_map<std::string, IValue>{} /*kwargs*/);
  }
  InterpreterState interp_state(code_);
  return interp_state.run(stack);
}

c10::IValue Function::operator()(Stack& stack) const {
  run(stack);
  return stack.front();
}

const std::shared_ptr<Code> Function::get_code() const {
  return code_;
}

int64_t Function::getCurrentDebugHandle() const {
  size_t pc = getInterpretersCurrentPC();
  return (pc < code_->debug_handles_.size()) ? code_->debug_handles_[pc] : -1;
}

} // namespace mobile
} // namespace jit
} // namespace torch
