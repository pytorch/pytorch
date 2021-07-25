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

bool Function::
    append_operator(const std::string& name, const std::string& overload_name, const c10::optional<int>& num_specified_args, int64_t model_version /* TODO: T90339189 deprecate all v3 when v3 models are removed */) {
  // Keep the original opname in code_
  code_->op_names_.emplace_back(name, overload_name);
  auto opname = code_->op_names_.back();

  const auto& opname_c10 = opname;
  std::function<void(Stack&)> fn;

  auto jit_op = findOperatorFor(opname);
  std::vector<c10::Argument> args;
  if (jit_op) {
    fn = [jit_op](Stack& stack) { jit_op->getOperation()(&stack); };
    args = jit_op->schema().arguments();
  } else {
    auto op = c10::Dispatcher::singleton().findSchema(opname_c10);
    if (op.has_value()) {
      fn = [op](Stack& stack) { op->callBoxed(&stack); };
      if (op->hasSchema()) {
        args = op->schema().arguments();
      } else {
        TORCH_CHECK(false, "arguments are missing for operator ", opname);
      }
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
  } else {
    // num_specified_args >= 0 indicates number of arguments are available
    // from model. We can use it to handle backward compatibility.
    if (num_specified_args &&
        num_specified_args.value() < static_cast<int64_t>(args.size())) {
      // Sanity check at load time, to save perf at runtime
      for (size_t i = num_specified_args.value(); i < args.size(); ++i) {
        auto default_val = args[i].default_value();
        TORCH_CHECK(
            default_val.has_value(),
            "Error happened at preparing for default values for the argument. The ",
            i,
            "th arguement of operator",
            opname,
            " does not have a specified value or default value. ");
      }
      fn = [fn, num_specified_args, args](Stack& stack) {
        for (size_t i = num_specified_args.value(); i < args.size(); ++i) {
          stack.push_back(args[i].default_value());
        }
        fn(stack);
      };
    }
  }
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

int64_t Function::get_debug_handle(size_t pc) const {
  TORCH_CHECK(code_, "Valid code must exist.");
  TORCH_CHECK(
      pc < code_->debug_handles_.size(),
      "Module debug info index out of boundary.");
  return code_->debug_handles_[pc];
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

int64_t Function::getExceptionDebugHandle() const {
  size_t pc = getInterpretersExceptionPC();
  return (pc < code_->debug_handles_.size()) ? code_->debug_handles_[pc] : -1;
}

} // namespace mobile
} // namespace jit
} // namespace torch
