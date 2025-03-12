#pragma once

#include "c10/core/Device.h"
#include "torch/csrc/nativert/executor/GraphExecutorBase.h"
#include "torch/csrc/nativert/graph/Graph.h"

namespace torch::nativert {

class HigherOrderKernel : public OpKernel {
  enum class OpType {
    UNKNOWN,
    COND,
    WHILE_LOOP,
    RUN_CONST_GRAPH,
  };

 public:
  HigherOrderKernel(
      const Node* node,
      std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors);
  void computeInternal(ExecutionFrame& executionFrame) const override final;

 private:
  std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors_;
  OpType opType_;
};

} // namespace torch::nativert
