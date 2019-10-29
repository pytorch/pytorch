#pragma once

#include <ostream>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/autograd/function.h>

namespace torch {
namespace jit {
namespace detail {

struct DifferentiableGraphBackwardImpl;

struct DifferentiableGraphBackward : public autograd::Node {
  virtual std::string toString() const;
  DifferentiableGraphBackward(GraphExecutor executor, size_t input_size, size_t capture_size);
  std::shared_ptr<DifferentiableGraphBackwardImpl> pImpl;
protected:
  virtual autograd::variable_list apply(autograd::variable_list&& inputs) override;
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