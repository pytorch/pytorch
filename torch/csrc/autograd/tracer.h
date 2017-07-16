#pragma once

#include "torch/csrc/jit/ir.h"

#include <memory>
#include <vector>
#include <iostream>
namespace torch { namespace autograd {

struct TracingState {
  jit::Graph & current() {
    assert(tracing());
    return *graphs.back();
  }
  bool tracing() {
    return graphs.size() > 0;
  }
  void enter() {
    graphs.push_back(new jit::Graph());
  }
  std::unique_ptr<jit::Graph> exit() {
    assert(graphs.size() > 0);
    auto r = graphs.back();
    graphs.pop_back();
    return std::unique_ptr<jit::Graph>(r);
  }
private:
  std::vector<jit::Graph *> graphs;
};

// Ugh, global state
extern TracingState GlobalTracingState;

}}
