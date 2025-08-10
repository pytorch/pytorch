#include <torch/nativert/kernels/C10Kernel.h>

#include <fmt/ostream.h>

#include <c10/util/Enumerate.h>

#ifdef __SIGRID_USE_GPU__
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#endif

namespace torch::nativert {

C10Kernel::C10Kernel(
    const Node* node,
    OpKernelKind kind,
    AliasingSpec&& aliasingSpec)
    : OpKernel(node, kind),
      op_(getOperatorForTarget(node->target(), node)),
      schema_(op_.schema(), std::move(aliasingSpec), kind_),
      arguments_(prefillStackWithStaticArgs(node, op_.schema())) {}

void C10Kernel::computeInternal(ExecutionFrame& executionFrame) const {
  // Make a copy of the stack
  std::vector<c10::IValue> stack = arguments_.getStackWithStaticArgs();

  fillDynamicInputs(executionFrame, arguments_, stack);

  // Call the op with the prepared stack.
  try {
    op_.callBoxed(stack);
  } catch (const std::exception& ex) {
    auto stackTrace = node_->getMetadata("stack_trace");
    throw std::runtime_error(fmt::format(
        "Exception while executing node: {}\n"
        "with args:\n{}\n"
        "{}\n"
        "Original Python stacktrace:\n{}",
        fmt::streamed(*node_),
        readableArgs(op_.schema(), stack),
        ex.what(),
        stackTrace ? *stackTrace : "<no stack trace>"));
  }

  // Write out results
  // TODO: we store intermediates in a single table (symint and tensor alike).
  // This can theoretically lead to name collisions, although based on how
  // these are named I don't think it will ever happen in practice. We need to
  // enforce it though.
  const auto& outputValues = node_->outputs();
  TORCH_CHECK(
      outputValues.size() == stack.size(),
      "Output size mismatch for ",
      node_->toString());
  for (auto&& [i, actualOutput] : c10::enumerate(stack)) {
    executionFrame.setIValue(outputValues[i]->id(), std::move(actualOutput));
  }
}

namespace {
std::unordered_map<std::string, c10::IValue> getSymInputs(
    const ExecutionFrame& executionFrame,
    const Node& node) {
  std::unordered_map<std::string, c10::IValue> inputs;
  for (const auto& input : node.inputs()) {
    const auto& val = executionFrame.getIValue(input.value->id());
    if (val.isInt() || val.isDouble() || val.isBool()) {
      inputs[input.name] = val;
    } else {
      throw std::runtime_error("unsupported type for symbolic input");
    }
  }
  for (const auto& attribute : node.attributes()) {
    if (std::holds_alternative<int64_t>(attribute.value)) {
      inputs[attribute.name] = std::get<int64_t>(attribute.value);
    } else if (std::holds_alternative<double>(attribute.value)) {
      inputs[attribute.name] = std::get<double>(attribute.value);
    } else if (std::holds_alternative<bool>(attribute.value)) {
      inputs[attribute.name] = std::get<bool>(attribute.value);
    } else {
      throw std::runtime_error("unsupported type for symbolic input");
    }
  }
  return inputs;
}

template <typename T>
void computeScalarBinaryOp(
    ExecutionFrame& executionFrame,
    const Node& node,
    std::enable_if_t<true, T> a,
    std::enable_if_t<true, T> b) {
  std::string_view target = node.target();
  T out;

  if (target == "_operator.add") {
    out = a + b;
  } else if (target == "_operator.sub") {
    out = a - b;
  } else if (target == "_operator.mul") {
    out = a * b;
  } else if (target == "_operator.pow") {
    out = std::pow(a, b);
  } else {
    throw std::runtime_error(
        fmt::format("unsupported operator for symbolic values: {}", target));
  }

  executionFrame.setIValue(node.outputs()[0]->id(), out);
  VLOG(2) << fmt::format(
      "Completed executing node: {} with a={}, b={}, out={}",
      fmt::streamed(node),
      a,
      b,
      out);
}

} // namespace

void ScalarBinaryOpKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  const auto& a = inputs.at("a");
  const auto& b = inputs.at("b");

  auto coerceToDouble = [](const c10::IValue& x) -> double {
    if (x.isInt()) {
      return static_cast<double>(x.toInt());
    } else if (x.isDouble()) {
      return x.toDouble();
    } else {
      throw std::runtime_error("unsupported type for symbolic input");
    }
  };

  if (a.isInt() && b.isInt()) {
    computeScalarBinaryOp<int64_t>(
        executionFrame, *node_, a.toInt(), b.toInt());
  } else {
    computeScalarBinaryOp<double>(
        executionFrame, *node_, coerceToDouble(a), coerceToDouble(b));
  }
}

void SymIntOpKernel::computeInternal(ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  int64_t a = inputs.at("a").toInt();
  std::string_view target = node_->target();
  if (target == "torch.sym_float") {
    double out = static_cast<double>(a);
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
    VLOG(2) << fmt::format(
        "Completed executing node: {} with a={}, out={}",
        fmt::streamed(*node_),
        a,
        out);
    return;
  }
  int64_t b = inputs.at("b").toInt();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t out;

  if (target == "_operator.floordiv") {
    out = a / b;
  } else if (target == "_operator.mod") {
    out = a % b;
  } else if (target == "torch.sym_max") {
    out = std::max(a, b);
  } else if (target == "torch.sym_min") {
    out = std::min(a, b);
  } else {
    throw std::runtime_error(
        fmt::format("unsupported operator for SymInt: {}", node_->target()));
  }

  executionFrame.setIValue(node_->outputs()[0]->id(), out);
  VLOG(2) << fmt::format(
      "Completed executing node: {} with a={}, b={}, out={}",
      fmt::streamed(*node_),
      a,
      b,
      out);
}

void SymBoolOpKernel::computeInternal(ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool out;

  const std::string_view target = node_->target();
  if (target == "torch.sym_not") {
    bool a = inputs.at("a").toBool();
    out = !a;
  } else if (target == "_operator.ge") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a >= b;
  } else if (target == "_operator.le") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a <= b;
  } else if (target == "_operator.eq") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a == b;
  } else if (target == "_operator.gt") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a > b;
  } else if (target == "_operator.lt") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a < b;
  } else if (target == "_operator.and_") {
    bool a = inputs.at("a").toBool();
    bool b = inputs.at("b").toBool();
    out = a && b;
  } else {
    throw std::runtime_error(
        fmt::format("unsupported operator for SymBool: {}", node_->target()));
  }

  executionFrame.setIValue(node_->outputs()[0]->id(), out);
}

void SymFloatOpKernel::computeInternal(ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  const std::string_view target = node_->target();
  if (target == "math.trunc") {
    double x = inputs.at("x").toDouble();
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    int64_t out = trunc(x);
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else if (target == "torch._sym_sqrt") {
    double a = inputs.at("a").toDouble();
    double out = std::sqrt(a);
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else if (target == "_operator.neg") {
    auto a = inputs.at("a");
    c10::IValue out;
    if (a.isInt()) {
      out = -a.toInt();
    } else if (a.isDouble()) {
      out = -a.toDouble();
    } else {
      throw std::runtime_error("unsupported type for symbolic input");
    }
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else if (target == "_operator.truediv") {
    auto ia = inputs.at("a");
    double a = ia.isInt() ? static_cast<double>(ia.toInt()) : ia.toDouble();
    auto ib = inputs.at("b");
    double b = ib.isInt() ? static_cast<double>(ib.toInt()) : ib.toDouble();
    double out = a / b;
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else {
    throw std::runtime_error(
        fmt::format("unsupported operator for SymFloat: {}", node_->target()));
  }
}

} // namespace torch::nativert
