#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit { namespace tracer {

TracingState::TracingState()
    : graph(new Graph())
    , active(true) {}

TracingState::~TracingState() = default;

}}} // namespace torch::jit::tracer
