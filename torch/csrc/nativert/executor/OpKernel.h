#pragma once

#include <torch/script.h>
#include "c10/core/Device.h"
#include "torch/csrc/nativert/executor/ExecutionFrame.h"
#include "torch/csrc/nativert/graph/Graph.h"

namespace torch::nativert {

c10::OperatorHandle getOperatorForTarget(
    std::string_view target,
    const Node* node = nullptr);

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

  /**
   * Dynamic arguments are the inputs that were not baked in the graph
   * during graph capture, i.e. all the tensor inputs to operators.
   *
   * This API will return a view of pairs consist of the argument index
   * and the corresponding Value pointer from the graph.
   */
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

  /**
   * Static arguments are the inputs that were specialized to a fixed value
   * during graph capture phase. For example, scalar inputs and device
   * are considered arguments.
   */
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

// Abstract interface representing a kernel, which is responsible for executing
// a single Node.
class OpKernel {
 public:
  explicit OpKernel(
      const Node* node,
      std::optional<c10::Device> device = std::nullopt)
      : node_(node), device_(device) {
    VLOG(1) << "Initializing kernel for node: " << *node_;
  }

  enum class Kind : uint8_t {
    kPrimKernel,
    kStaticDispatchKernel,
    kInterpreterFallbackKernel,
  };

  const Node* node() const {
    return node_;
  }
  void compute(ExecutionFrame& executionFrame) const;

  Kind kind() const {
    return kind_;
  }

  bool hasPrimKernel() const {
    return kind() == Kind::kPrimKernel;
  }

  bool hasStaticDispatch() const {
    return kind() == Kind::kStaticDispatchKernel;
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
    CHECK(i < numInputs()) << "Invalid input index: " << i;
    return executionFrame.getIValue(node_->inputs()[i].value->id());
  }

  // Output is readwrite
  c10::IValue& output(uint32_t i, ExecutionFrame& executionFrame) const {
    CHECK(i < numOutputs()) << "Invalid output index: " << i;
    return executionFrame.getIValue(node_->outputs()[i]->id(), true);
  }

  virtual ~OpKernel() = default;

 protected:
  virtual void computeInternal(ExecutionFrame& executionFrame) const = 0;

  const Node* node_;
  std::optional<c10::Device> device_;
  const static bool blockingEnabled_;
  Kind kind_ = Kind::kInterpreterFallbackKernel;
};

} // namespace torch::nativert
