#include <torch/csrc/autograd/function.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/ir.h>

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <deque>

namespace torch { namespace autograd {

/// Monotonically incrementing (thread local!) counter to supply sequence
/// numbers.
thread_local uint64_t Function_next_sequence_nr_ = 0;

uint64_t Function::peek_at_next_sequence_nr() {
  return Function_next_sequence_nr_;
}

uint64_t& Function::get_next_sequence_nr() {
  return Function_next_sequence_nr_;
}

auto Function::name() const -> std::string {
  return c10::demangle(typeid(*this).name());
}

AnomalyMetadata* Function::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

static void gatherFunctions(
    Function* func,
    std::vector<std::shared_ptr<Function>>& stack) {
  func->release_variables();

  for (auto& edge : func->next_edges()) {
    if (edge.function.use_count() == 1) {
      stack.emplace_back(std::move(edge.function));
    } else {
      edge.function.reset();
    }
  }
}

/*
  * Fix for #5534: prevent stack overflow on deletion of deep computation graph
  *
  * Sometimes one can end up with a very big computation graph of Functions
  * and Edges. Each std::shared_ptr<Function> contains a list of Edge, and
  * each Edge contains a std::shared_ptr<Function>. Deleting a
  * std::shared_ptr<Function> can trigger the recursive deletion of other
  * std::shared_ptr<Function>'s: this can stack overflow if the graph
  * is deep enough. Here is an example of such a graph:
  *
  * shared_ptr<Function> -> Edge -> shared_ptr<Function> -> Edge -> ... -> shared_ptr<Function>
  *
  * The solution here is to detect when we are decrementing away the last
  * reference to a Function, and when doing so to buffer up the Function's
  * that will be recursively decremented.  We can then decrement (and free)
  * the original Function without causing a recursive cascade, before
  * draining the buffer applying the same behavior.  This is, in effect,
  * converting recursion to a loop, using a heap buffer in place of the
  * recursive call stack.
  */
void deleteFunction(Function* function) {
  // To avoid stack overflow on large computational graphs,
  // we need to track reference decrementing and freeing
  // on the heap.
  function->release_variables();
  std::vector<std::shared_ptr<Function>> stack;
  gatherFunctions(function, stack);
  delete function;

  while (!stack.empty()) {
    auto func = std::move(stack.back());
    stack.pop_back();
    gatherFunctions(func.get(), stack);
    // Reference count is decremented on the loop backedge.
  }
}

}} // namespace torch::autograd
