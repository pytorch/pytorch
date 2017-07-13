#pragma once

#include "torch/csrc/autograd/ir.h"

#include <memory>
#include <vector>

namespace torch { namespace autograd {

class TracingState {
public:
  int next_unique;
  std::unique_ptr<Graph> graph;
  TracingState()
    : next_unique(0)
    , graph(new Graph())
    {}
  std::unique_ptr<Value> makeValue(Node * definition, size_t offset);
};

// Ugh, global state
extern std::unique_ptr<TracingState> GlobalTracingState;

void Tracer_enter();
std::unique_ptr<Graph> Tracer_exit(value_list locals);
}}
