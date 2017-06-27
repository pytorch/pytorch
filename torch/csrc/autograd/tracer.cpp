#include "torch/csrc/autograd/tracer.h"

namespace torch { namespace autograd {

std::unique_ptr<TracingState> GlobalTracingState = nullptr;

std::shared_ptr<Local> TracingState::makeLocal() {
  return std::make_shared<Local>(next_unique++);
}

void TracingState::addBinding(local_list lvals, std::shared_ptr<Expr> rval) {
  builder->add(Bind(lvals, rval));
}

std::shared_ptr<Expr> TracingState::expr(local_list locals) {
  return builder->expr(std::make_shared<Tuple>(locals));
}

void Tracer_enter() {
    if (GlobalTracingState) {
        throw std::logic_error("nested tracing not yet supported");
    } else {
        GlobalTracingState = std::unique_ptr<TracingState>(new TracingState());
    }
}
std::shared_ptr<Expr> Tracer_exit(local_list outputs) {
    if (!GlobalTracingState) {
        throw std::logic_error("attempting to exit trace, but none exist");
    } else {
        auto st = std::move(GlobalTracingState);
        GlobalTracingState = nullptr;
        return st->expr(outputs);
    }
}

}}
