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
}} // namespace torch::jit

namespace torch { namespace jit { namespace tracer {

// TracingState tracks the necessary state when we are tracing the execution of
// autograd code; most importantly, it holds a reference to the actual IR
// graph which we are recording the trace to.
//
// The liveness of a TracingState is expected to be a superset of the region
// of code being traced; in particular, Variables do not keep a TracingState
// live.  Instead, they hold weak pointers to TracingState, to prevent leaks
// from arising when a variable that participated in a trace outlives the
// actual trace itself.

struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  std::shared_ptr<Graph> graph;
  std::mutex mutex;
  bool active;

  std::unique_lock<std::mutex> lock() {
    return std::unique_lock<std::mutex>(mutex);
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

}}} // namespace torch::jit::tracer
