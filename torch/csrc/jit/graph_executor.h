#pragma once

#include <memory>
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/variable_tensor_list.h"

namespace torch { namespace jit {

struct GraphExecutorImpl;
struct GraphExecutor {
  GraphExecutor() {}
  GraphExecutor(std::shared_ptr<Graph> graph, bool optimize = true);
  // note: if not specified, symbolically_differentiable is computed from the graph.
  GraphExecutor(std::shared_ptr<Graph> graph, bool optimize, bool symbolically_differentiable);
  variable_tensor_list run(variable_tensor_list && inputs);
  operator bool() const {
    return pImpl != nullptr;
  }
  std::shared_ptr<Graph> graph() const;
private:
  std::shared_ptr<GraphExecutorImpl> pImpl;
};

}}
