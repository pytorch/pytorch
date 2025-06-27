#pragma once

#include <c10/core/Device.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/OpKernelKind.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

c10::OperatorHandle getOperatorForTarget(
    std::string_view target,
    const Node* node = nullptr);
/**
 * @brief Manages static and dynamic arguments for kernel execution.
 *
 * The `Arguments` class encapsulates both static and dynamic arguments
 * used during the execution of operators in a graph.
 * Static arguments are the inputs that were specialized to a fixed value
 * during graph capture phase. For example, scalar inputs and device are
 * considered static arguments.
 * Dynamic arguments are the inputs that were not baked in the graph
 * during graph capture, i.e. all the tensor inputs to operators
 */
class Arguments {
 public:
  Arguments(
      std::vector<c10::IValue> stackWithStaticArgs,
      std::vector<Value*> dynamicArgs)
      : stackWithStaticArgs_(std::move(stackWithStaticArgs)),
        dynamicArgs_(std::move(dynamicArgs)) {
    for (size_t i = 0; i < dynamicArgs_.size(); i++) {
      if (dynamicArgs_[i]) {
        indices_.push_back(i);
      }
    }
  }

  // Returns a view of pairs consist of the argument index and
  // the corresponding Value pointer from the graph.
  auto getDynamicArgs() const {
    std::vector<std::pair<size_t, Value*>> ret;
    ret.reserve(indices_.size());
    for (auto i : indices_) {
      ret.emplace_back(i, dynamicArgs_[i]);
    }
    return ret;
  }

  // Argument i means the i-th input to the operator in the argument list.
  // Will return nullptr if the argument is not dynamic.
  Value* findDynamic(size_t i) const {
    DCHECK(i < dynamicArgs_.size()) << "Invalid input index: " << i;
    return dynamicArgs_[i];
  }

  // Argument i means the i-th input to the operator in the argument list.
  // Will return None as IValue if the argument is not static.
  const c10::IValue& getStatic(size_t i) const {
    DCHECK(i < stackWithStaticArgs_.size()) << "Invalid input index: " << i;
    return stackWithStaticArgs_[i];
  }

  const std::vector<c10::IValue>& getStackWithStaticArgs() const {
    return stackWithStaticArgs_;
  }

 private:
  // stack pre-populated with attributes, aka static arguments
  const std::vector<c10::IValue> stackWithStaticArgs_;

  // Argument can only be asTensor, asTensors, asSymInt, asSymInts
  const std::vector<Value*> dynamicArgs_;
  std::vector<size_t> indices_;
};

void fillDynamicInputs(
    const ExecutionFrame& executionFrame,
    const Arguments& arguments,
    std::vector<c10::IValue>& stack);

Arguments prefillStackWithStaticArgs(
    const Node* node,
    const c10::FunctionSchema& schema);

std::string readableArgs(
    const c10::FunctionSchema& schema,
    const std::vector<c10::IValue>& stack);

/**
 * @brief Abstract interface representing a kernel, which is responsible for
 * executing a single Node in the graph.
 *
 * The OpKernel class is responsible for executing a single Node in the graph.
 * It provides an interface for accessing node inputs and outputs, determining
 * the execution kind, and executing the node's computation.
 */
class OpKernel {
 public:
  explicit OpKernel(
      const Node* node,
      std::optional<c10::Device> device = std::nullopt,
      OpKernelKind kind = OpKernelKind::kInterpreterFallbackKernel)
      : node_(node), device_(device), kind_(kind) {
    VLOG(1) << "Initializing kernel for node: " << *node_;
  }

  const Node* node() const {
    return node_;
  }
  void compute(ExecutionFrame& executionFrame) const;

  OpKernelKind kind() const {
    return kind_;
  }

  bool hasPrimKernel() const {
    return kind() == OpKernelKind::kPrimKernel;
  }

  bool hasStaticDispatch() const {
    return kind() == OpKernelKind::kStaticDispatchKernel ||
        kind() == OpKernelKind::kNativeStaticDispatchKernel;
  }

  size_t numInputs() const {
    return node_->inputs().size();
  }

  size_t numOutputs() const {
    return node_->outputs().size();
  }

  // Input is readonly
  [[nodiscard]] virtual const c10::IValue& input(
      uint32_t i,
      ExecutionFrame& executionFrame) const {
    TORCH_CHECK(i < numInputs(), "Invalid input index: ", i);
    return executionFrame.getIValue(node_->inputs()[i].value->id());
  }

  // Output is read/write
  c10::IValue& output(uint32_t i, ExecutionFrame& executionFrame) const {
    TORCH_CHECK(i < numOutputs(), "Invalid output index: ", i);
    return executionFrame.getIValue(node_->outputs()[i]->id(), true);
  }

  virtual ~OpKernel() = default;

 protected:
  virtual void computeInternal(ExecutionFrame& executionFrame) const = 0;

  const Node* node_;
  std::optional<c10::Device> device_;
  const static bool blockingEnabled_;
  // this should be set in the ctor!
  const OpKernelKind kind_;
};

} // namespace torch::nativert
