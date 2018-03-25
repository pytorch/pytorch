#pragma once

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/variable.h"

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch { namespace jit {
struct Graph;
struct Value;
struct VariableFlags;
}} // namespace torch::jit

namespace torch { namespace jit { namespace tracer {

using edge_list = std::vector<autograd::Edge>;
using variable_list = std::vector<autograd::Variable>;

// TracingState tracks the necessary state when we are tracing the execution of
// autograd code; most importantly, it holds a reference to the actual IR
// graph which we are recording the trace to.
//
// The liveness of a TracingState is expected to be a superset of the region
// of code being traced; in particular, Variables do not keep a TracingState
// live.  Instead, they hold weak pointers to TracingState, to prevent leaks
// from arising when a variable that participated in a trace outlives the
// actual trace itself.

using io_variable_flags_list = std::vector<
    std::pair<std::vector<VariableFlags>, std::vector<VariableFlags>>>;

struct TracingState : public std::enable_shared_from_this<TracingState> {
  explicit TracingState(size_t num_stages);
  ~TracingState();

  std::shared_ptr<Graph> graph;
  bool active;

  // Used to free the Graph as soon as we know this trace will fail
  size_t num_stages;
  std::atomic<size_t> eval_count;

  // A pair of (input_flags, output_flags) for each stage
  io_variable_flags_list var_flags;
  std::vector<edge_list> output_edges;

  std::mutex mutex;
  variable_list inputs; // Used only for the duration of first stage

  std::unique_lock<std::mutex> lock() {
    return std::unique_lock<std::mutex>(mutex);
  }

  bool is_expired() const noexcept {
    return !graph;
  }

  bool is_complete() const;
  void push_scope(const std::string& scope_name);
  void pop_scope();
};

struct ValueTracingStateElem {
  std::weak_ptr<TracingState> state;
  // it's only valid to use this field if !state.exired()
  Value* trace = nullptr;

  void reset() {
    state.reset();
    trace = nullptr;
  }
};

using ValueTracingState = std::list<ValueTracingStateElem>;

struct FunctionTracingState {
  bool in_eval_subgraph = false;
};

}}} // namespace torch::jit::tracer
