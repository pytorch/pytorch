#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/assertions.h"

#include <memory>
#include <mutex>
#include <vector>
#include <cstdint>
#include <list>

namespace torch { namespace autograd {

struct Variable;
struct Function;

}}

namespace torch { namespace jit { namespace tracer {

using torch::autograd::Variable;
using torch::autograd::Function;
using variable_list = std::vector<Variable>;
using function_list = std::vector<std::pair<std::shared_ptr<Function>, int>>;

// TracingState tracks the necessary state when we are tracing the execution of
// autograd code; most importantly, it holds a reference to the actual IR
// graph which we are recording the trace to.
//
// The liveness of a TracingState is expected to be a superset of the region
// of code being traced; in particular, Variables do not keep a TracingState
// live.  Instead, they hold weak pointers to TracingState, to prevent leaks
// from arising when a variable that participated in a trace outlives the
// actual trace itself.

using io_variable_flags_list =
  std::vector<std::pair<std::vector<VariableFlags>, std::vector<VariableFlags>>>;

struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState(std::size_t num_stages)
    : graph(new Graph())
    , active(false)
    , num_stages(num_stages)
    , eval_count(0)
    , var_flags(num_stages)
    , output_edges(num_stages) {}

  // XXX: graph can be NULL if it's a failed trace (failed = didn't capture all
  // the stages we care about)
  std::shared_ptr<Graph> graph;
  bool active;

  // Used to free the Graph as soon as we know this trace will fail
  std::size_t num_stages;
  std::atomic<std::size_t> eval_count;

  // void* is an unsafe TH.  NON-OWNING, so it might get invalidated.
  // TODO: Perhaps, turn this into an owning reference.  The buffers
  // are persistent, so this won't lead to a leak.
  std::unordered_map<void*, Value*> buffer_map;
  // A pair of (input_flags, output_flags) for each stage
  io_variable_flags_list var_flags;
  std::vector<function_list> output_edges;

  std::mutex mutex;
  variable_list inputs; // Used only for the duration of first stage

  std::unique_lock<std::mutex> lock() { return std::unique_lock<std::mutex>(mutex); };

  bool is_expired() const {
    return !graph;
  }

  bool is_complete() const {
    return !is_expired() && graph->stage() == num_stages - 1;
  }
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

}}}
