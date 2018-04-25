#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch { namespace jit { namespace tracer {
TracingState::TracingState(size_t num_stages)
    : graph(new Graph()),
      active(false),
      num_stages(num_stages),
      eval_count(0),
      var_flags(num_stages),
      output_edges(num_stages),
      creates_handles(true) {}

TracingState::~TracingState() = default;

bool TracingState::is_complete() const {
  return !is_expired() && graph->stage() == num_stages - 1;
}

void TracingState::push_scope(const std::string& scope_name) {
  graph->push_scope(scope_name);
}

void TracingState::pop_scope() {
  graph->pop_scope();
}
}}} // namespace torch::jit::tracer
