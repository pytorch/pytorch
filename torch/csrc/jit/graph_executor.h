#pragma once

#include <memory>
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

struct GraphExecutorImpl;
struct GraphExecutor {
  GraphExecutor() {}
  GraphExecutor(std::shared_ptr<Graph> graph);
  GraphExecutor(std::shared_ptr<Graph> graph, bool symbolically_differentiable);
  std::vector<at::Tensor> run(std::vector<at::Tensor> inputs);
  operator bool() const {
    return pImpl != nullptr;
  }
private:
  std::shared_ptr<GraphExecutorImpl> pImpl;
};

}}
