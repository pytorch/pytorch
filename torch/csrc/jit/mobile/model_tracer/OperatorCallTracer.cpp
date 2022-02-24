#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>

namespace torch {
namespace jit {
namespace mobile {
OperatorCallTracer::OperatorCallTracer() {
  called_operators_.clear();
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    c10::optional<c10::OperatorName> op_name = fn.operator_name();
    if (op_name.has_value()) {
      called_operators_.insert(c10::toString(*op_name));
    }
    return nullptr;
  };

  handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                      .scopes({at::RecordScope::FUNCTION}));
}

std::set<std::string> OperatorCallTracer::called_operators_;
} // namespace mobile
} // namespace jit
} // namespace torch
