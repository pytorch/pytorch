#include <ATen/core/dynamic_type.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/parse_operators.h>
#include <torch/csrc/jit/mobile/prim_ops_registery.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

char const* toString(OpCode op);
namespace mobile {
Function::Function(c10::QualifiedName name) : name_(std::move(name)) {}

Function::Function(
    c10::QualifiedName name,
    Code code,
    at::optional<c10::FunctionSchema> schema)
    : name_(std::move(name)),
      code_(std::move(code)),
      schema_(std::move(schema)) {}

const c10::QualifiedName& Function::qualname() const {
  return name_;
}

void Function::append_instruction(OpCode op, int X, int N, int64_t dbg_handle) {
  TORCH_CHECK(
      isOpSupportedInMobile(op),
      toString(op),
      " is not supported in mobile module.");
  code_.instructions_.emplace_back(op, X, N);
  code_.debug_handles_.emplace_back(dbg_handle);
}

void Function::append_instruction(OpCode op, int X, int N) {
  TORCH_CHECK(
      isOpSupportedInMobile(op),
      toString(op),
      " is not supported in mobile module.");
  code_.instructions_.emplace_back(op, X, N);
}

void Function::append_operator(
    const std::string& name,
    const std::string& overload_name,
    const c10::optional<int>& num_specified_args) {
  // Keep the original opname in code_
  code_.op_names_.emplace_back(name, overload_name);
  code_.operator_input_sizes_.emplace_back(num_specified_args.value_or(-1));
}

std::string operator_str(const c10::OperatorName& opname) {
  std::string result = opname.name;
  if (!opname.overload_name.empty()) {
    result += "." + opname.overload_name;
  }
  return result;
}

bool Function::initialize_operators(bool should_check_operators) {
  if (code_.initialized) {
    return true;
  }
  std::unordered_set<std::string> unsupported_op_names;
  code_.operators_.resize(code_.op_names_.size());
  bool all_ops_supported = true;
  for (unsigned i = 0; i < code_.op_names_.size(); i++) {
    const auto& opname = code_.op_names_[i];
    int num_args = code_.operator_input_sizes_[i];
    c10::optional<int> num_specified_args =
        num_args < 0 ? c10::nullopt : c10::optional<int>(num_args);
    auto func = makeOperatorFunction(opname, num_specified_args);
    if (!func.has_value()) {
      unsupported_op_names.insert(operator_str(opname));
      all_ops_supported = false;
    } else {
      code_.operators_[i] = *func;
    }
  }
  if (should_check_operators) {
    TORCH_CHECK(
        unsupported_op_names.empty(),
        "Following ops cannot be found: [",
        c10::Join(", ", unsupported_op_names),
        "]. Please check if the operator library is included in the build. If built with selected ops, check if these ops are in the list. If you are a Meta employee, please see fburl.com/missing_ops for a fix. Or post it in https://discuss.pytorch.org/c/mobile/");
  }
  code_.initialized = all_ops_supported;
  return all_ops_supported;
}

void Function::append_constant(const c10::IValue& constant) {
  code_.constants_.push_back(constant);
}

void Function::append_type(const at::TypePtr& type) {
  code_.types_.push_back(type);
}

void Function::append_function(mobile::Function& function) {
  code_.functions_.push_back(&function);
}

void Function::set_register_size(size_t size) {
  code_.register_size_ = size;
}

int64_t Function::get_debug_handle(size_t pc) const {
  TORCH_CHECK(
      pc < code_.debug_handles_.size(),
      "Module debug info index out of boundary.");
  return code_.debug_handles_[pc];
}

torch::jit::Function& Function::setSchema(c10::FunctionSchema schema) {
  schema_ = std::move(schema);
  return *this;
}

bool Function::hasSchema() const {
  return schema_.has_value();
}

const c10::FunctionSchema& Function::getSchema() const {
  return *schema_;
}

void Function::run(Stack& stack) {
  initialize_operators(/* should_check_operators */ true);
  if (hasSchema()) { // if we have a schema then resolve optional args if any
    getSchema().checkAndNormalizeInputs<c10::DynamicType>(
        stack, std::unordered_map<std::string, IValue>{} /*kwargs*/);
  }
  InterpreterState interp_state(code_);
  interp_state.run(stack);
}

at::IValue Function::operator()(Stack& stack) {
  run(stack);
  return stack.front();
}

size_t Function::num_inputs() const {
  return schema_->arguments().size();
}

bool Function::call(Stack&, c10::function_ref<void(const mobile::Code&)> f) {
  initialize_operators(true);
  f(code_);
  return true;
}

const Code& Function::get_code() const {
  return code_;
}

Code& Function::get_code() {
  return code_;
}

const std::vector<int64_t>& Function::getExceptionDebugHandles() const {
  return getInterpretersExceptionDebugHandles();
}

c10::optional<std::function<void(Stack&)>> makeOperatorFunction(
    c10::OperatorName opname,
    c10::optional<int> num_specified_args) {
  std::function<void(Stack&)> fn;
  const auto full_name = c10::toString(opname);
  const std::vector<c10::Argument>* pArgs = nullptr;
  bool promoted_op = mobile::hasPrimOpsFn(full_name);
  if (promoted_op) {
    fn = mobile::getPrimOpsFn(full_name);
  } else {
    std::shared_ptr<Operator> jit_op = findOperatorFor(opname);
    if (jit_op) {
      fn = [jit_op](Stack& stack) { jit_op->getOperation()(stack); };
      pArgs = &jit_op->schema().arguments();
    } else {
      auto op = c10::Dispatcher::singleton().findSchema(opname);
      if (op.has_value()) {
        fn = [op](Stack& stack) { op->callBoxed(&stack); };
        if (op->hasSchema()) {
          pArgs = &op->schema().arguments();
        } else {
          TORCH_CHECK(false, "arguments are missing for operator ", opname);
        }
      } else {
        return c10::nullopt;
      }
    }
  }

  if (!promoted_op) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(pArgs);
    const auto& args = *pArgs;
    // num_specified_args >= 0 indicates number of arguments are available
    // from model. We can use it to handle backward compatibility.
    if (num_specified_args &&
        num_specified_args.value() < static_cast<int64_t>(args.size())) {
      fn = [fn, num_specified_args, &args](Stack& stack) {
        std::vector<IValue> out_args;
        // The following logic pops and temporarily stores all out arguments
        // from the stack (which can be 0 or more, and always appended to the
        // schema), in order to push the necessary default values. Finally,
        // the out arguments are pushed back into the stack.
        for (size_t i = args.size() - 1; i > 0 && args.at(i).is_out(); i--) {
          out_args.push_back(stack.back());
          stack.pop_back();
        }
        TORCH_CHECK(
            static_cast<size_t>(num_specified_args.value()) >= out_args.size(),
            "The number of output arguments is: ",
            out_args.size(),
            ", which is more then the number of specified arguments: ",
            num_specified_args.value());
        size_t start_index = num_specified_args.value() - out_args.size();
        for (size_t i = start_index; i < (args.size() - out_args.size()); ++i) {
          TORCH_CHECK(
              args[i].default_value().has_value(),
              "Error happened at preparing for default values for the argument. The ",
              i,
              "th argument ",
              args[i].name(),
              " does not have a specified value or default value. ");

          stack.emplace_back(args[i].default_value());
        }
        stack.insert(stack.end(), out_args.rbegin(), out_args.rend());
        fn(stack);
      };
    }
  }
  return fn;
}

Function& Function::registerFunc(
    const std::string& qualified_name,
    const std::vector<Instruction>& instructions,
    const std::vector<c10::IValue>& constants,
    const std::vector<c10::TypePtr>& types,
    const size_t register_size) {
  static std::unordered_map<c10::QualifiedName, Function>
      upgrader_function_holder;
  c10::QualifiedName name = c10::QualifiedName(qualified_name);
  auto found = upgrader_function_holder.find(name);
  // Register the function if it's not found in the map.
  if (found == upgrader_function_holder.end()) {
    auto name_function_pair =
        upgrader_function_holder.emplace(name, Function(name));
    auto& func = name_function_pair.first->second;
    for (auto const& inst : instructions) {
      func.append_instruction(inst.op, inst.X, inst.N);
    }
    for (auto const& constant : constants) {
      func.append_constant(constant);
    }
    for (auto const& type : types) {
      func.append_type(type);
    }
    func.set_register_size(register_size);
    return func;
  }
  auto& upgrader_function_in_holder = found->second;
  return upgrader_function_in_holder;
}

} // namespace mobile
} // namespace jit
} // namespace torch
