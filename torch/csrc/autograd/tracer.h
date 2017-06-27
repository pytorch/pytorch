#pragma once

#include "torch/csrc/autograd/ir.h"

#include <memory>
#include <vector>

namespace torch { namespace autograd {

class TracingState {
private:
  int next_unique;
  std::unique_ptr<LetBuilder> builder;
public:
  TracingState()
    : next_unique(0)
    , builder(std::unique_ptr<LetBuilder>(new LetBuilder()))
    {}
  std::shared_ptr<Local> makeLocal();
  void addBinding(local_list lvals, std::shared_ptr<Instruction> rval);
  std::shared_ptr<Expr> expr(local_list locals);
};

// Ugh, global state
extern std::unique_ptr<TracingState> GlobalTracingState;

void Tracer_enter();
std::shared_ptr<Expr> Tracer_exit(local_list locals);

}}
