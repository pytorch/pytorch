#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>

namespace torch {
namespace jit {
namespace mobile {
std::set<std::string> OperatorCallTracer::called_operators_;
} // namespace mobile
} // namespace jit
} // namespace torch
