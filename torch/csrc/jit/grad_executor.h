#pragma once

#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/autograd/function.h>

namespace torch {
namespace jit {
namespace detail {

struct DifferentiableGraphBackward : public autograd::Node {
  DifferentiableGraphBackward(GraphExecutor executor): executor(std::move(executor)) {}
  std::shared_ptr<Graph> graph() const {
    return executor.graph();
  }

protected:
  GraphExecutor executor;
};

GraphExecutor* getGradExecutor(Operation& op);

// for debugging information we expose a way to get the last actually
// run graph. Previous approaches allowed querying the GraphExecutor
// for what graph it would run in certain circumstances (graphFor), but
// this is fragile because we sometimes change how these decisions are made.
// This interface still allows our tests to look at optimized graphs, but
// with less plumbing.

} // namespace detail
} // namespace jit
} // namespace torch