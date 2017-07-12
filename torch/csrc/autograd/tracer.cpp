#include "torch/csrc/autograd/tracer.h"

namespace torch { namespace autograd {

std::unique_ptr<TracingState> GlobalTracingState = nullptr;

std::shared_ptr<Value> TracingState::makeValue() {
  return std::make_shared<Value>(next_unique++);
}

void Tracer_enter() {
    if (GlobalTracingState) {
        throw std::logic_error("nested tracing not yet supported");
    } else {
        GlobalTracingState = std::unique_ptr<TracingState>(new TracingState());
    }
}

std::unique_ptr<Graph> Tracer_exit(value_list outputs) {
    if (!GlobalTracingState) {
        throw std::logic_error("attempting to exit trace, but none exist");
    } else {
        auto st = std::move(GlobalTracingState);
        GlobalTracingState = nullptr;
        st->graph->outputs = outputs;
        return std::move(st->graph);
    }
}

}}
