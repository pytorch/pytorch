#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/graph_fuser.h"

#include <memory>
#include <vector>
#include <iostream>
#include <unordered_map>

namespace torch { namespace jit {

struct TracingState {
  struct TracingFrame {
    TracingFrame()
      : graph(new jit::Graph())
      , value_trace() {};

    std::unique_ptr<jit::Graph> graph;
    std::unordered_map<torch::autograd::Variable*, Node*> value_trace;
  };

  jit::Graph & current() {
    JIT_ASSERT(tracing());
    return *frames.back().graph;
  }

  bool tracing() {
    return frames.size() > 0;
  }

  void enter() {
    frames.emplace_back();
  }

  void setValueTrace(torch::autograd::Variable* var, Node* trace) {
    assert(tracing());
    frames.back().value_trace[var] = trace;
  }

  Node* getValueTrace(torch::autograd::Variable* var, bool mustExist = false) {
    assert(tracing());
    auto& trace_map = frames.back().value_trace;
    if (mustExist) {
      return trace_map.at(var);
    } else {
      auto it = trace_map.find(var);
      // TODO: handle the case when var is not in the map gracefully
      if (it == trace_map.end())
        throw std::runtime_error("TracingState::getValueTrace not fully implemented yet");
      return it->second;
    }
  }

  std::unique_ptr<jit::Graph> exit() {
    JIT_ASSERT(tracing());
    auto r = std::move(frames.back());
    frames.pop_back();
    r.graph = FuseGraph(std::move(r.graph));
    return std::move(r.graph);
  }
private:
  std::vector<TracingFrame> frames;
};

extern TracingState GlobalTracingState;

}}
